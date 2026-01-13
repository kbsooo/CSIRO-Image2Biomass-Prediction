#%% [markdown]
# # 🏆 Improved DINOv3 Training Pipeline v2
#
# **목표**: CV 0.70+ 달성
#
# **핵심 개선사항**:
# 1. 검증된 baseline 기반 (head=128, dropout=0.1)
# 2. 보수적 augmentation (농업 이미지 특성 고려)
# 3. Huber Loss (outlier robust)
# 4. EMA (Exponential Moving Average)
# 5. MixUp regularization
# 6. Log1p target transformation
# 7. Zero-Inflated Clover Head

#%%
import os
import gc
import json
import random
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup

import timm
import torchvision.transforms.v2 as T
from sklearn.model_selection import StratifiedGroupKFold

import warnings
warnings.filterwarnings('ignore')
tqdm.pandas()

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

#%% [markdown]
# ## 🔐 Step 1: Google Drive Mount (Colab Only)
# **중요**: 이 셀을 먼저 실행하여 Drive 권한을 승인하세요.

#%%
GDRIVE_SAVE_PATH = None

try:
    from google.colab import drive
    drive.mount('/content/drive')
    GDRIVE_SAVE_PATH = Path('/content/drive/MyDrive/kaggle_models/csiro_biomass_v2')
    GDRIVE_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    print(f"✓ Google Drive mounted: {GDRIVE_SAVE_PATH}")
except ImportError:
    print("Not in Colab - Google Drive skipped")

#%% [markdown]
# ## 🔑 Step 2: Kaggle Login (Colab Only)

#%%
import kagglehub

IS_KAGGLE = Path("/kaggle/input/csiro-biomass").exists()

if not IS_KAGGLE:
    print("🟢 Colab 환경 - Kaggle 로그인 필요")
    kagglehub.login()
else:
    print("🔵 Kaggle 환경")

#%%
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def flush():
    gc.collect()
    torch.cuda.empty_cache()

seed_everything(42)

#%% [markdown]
# ## ⚙️ Configuration

#%%
class CFG:
    # === Paths ===
    DATA_PATH = None
    OUTPUT_DIR = None
    WEIGHTS_PATH = None

    # === Model ===
    model_name = "vit_large_patch16_dinov3_qkvb.lvd1689m"
    backbone_dim = 1024
    img_size = (518, 518)  # DINOv2 optimal (14 * 37)

    # === Head Architecture ===
    head_hidden_dim = 128  # 8은 실패, 256은 CV 0.6 → 중간값 시도
    use_zero_inflated_clover = True

    # === Training ===
    n_folds = 5
    epochs = 20
    batch_size = 8
    accumulation_steps = 2  # effective batch = 16
    lr = 5e-5
    backbone_lr_mult = 0.1
    weight_decay = 1e-4
    dropout = 0.1
    warmup_ratio = 0.1

    # === Loss ===
    use_huber_loss = True
    huber_delta = 1.0

    # === EMA ===
    use_ema = True
    ema_decay = 0.999

    # === Gradient ===
    gradient_clip = 1.0

    # === Augmentation ===
    use_mixup = True
    mixup_alpha = 0.2

    # === Target Transform ===
    use_log1p = True

    # === Other ===
    seed = 42
    num_workers = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = CFG()

#%% [markdown]
# ## 📥 Step 3: Data Download

#%%
if IS_KAGGLE:
    cfg.DATA_PATH = Path("/kaggle/input/csiro-biomass")
    cfg.WEIGHTS_PATH = Path("/kaggle/input/pretrained-weights-biomass/dinov3_large/dinov3_large")
    cfg.OUTPUT_DIR = Path("/kaggle/working")
else:
    print("Downloading data via kagglehub...")
    csiro_path = kagglehub.competition_download('csiro-biomass')
    weights_path = kagglehub.dataset_download('kbsooo/pretrained-weights-biomass')

    cfg.DATA_PATH = Path(csiro_path)
    cfg.WEIGHTS_PATH = Path(weights_path) / "dinov3_large" / "dinov3_large"
    cfg.OUTPUT_DIR = Path("/content/output")

cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Data: {cfg.DATA_PATH}")
print(f"Weights: {cfg.WEIGHTS_PATH}")
print(f"Output: {cfg.OUTPUT_DIR}")

#%% [markdown]
# ## 📊 Competition Metric

#%%
TARGET_WEIGHTS = {
    'Dry_Green_g': 0.1, 'Dry_Dead_g': 0.1, 'Dry_Clover_g': 0.1,
    'GDM_g': 0.2, 'Dry_Total_g': 0.5,
}
TARGET_ORDER = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

def competition_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted R² score"""
    weighted_r2 = 0.0
    for i, target in enumerate(TARGET_ORDER):
        weight = TARGET_WEIGHTS[target]
        ss_res = np.sum((y_true[:, i] - y_pred[:, i]) ** 2)
        ss_tot = np.sum((y_true[:, i] - np.mean(y_true[:, i])) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        weighted_r2 += weight * r2
    return weighted_r2

#%% [markdown]
# ## 📁 Data Preparation

#%%
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(
        index=['image_path', 'State', 'Species', 'Sampling_Date', 'Pre_GSHH_NDVI', 'Height_Ave_cm'],
        columns='target_name',
        values='target',
        aggfunc='first'
    ).reset_index()
    pivot.columns.name = None
    return pivot

train_df = pd.read_csv(cfg.DATA_PATH / "train.csv")
train_wide = prepare_data(train_df)
train_wide['image_id'] = train_wide['image_path'].apply(lambda x: Path(x).stem)

# Stratified Group KFold
sgkf = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
train_wide['fold'] = -1
for fold, (_, val_idx) in enumerate(sgkf.split(
    train_wide,
    train_wide['State'],
    groups=train_wide['image_id']
)):
    train_wide.loc[val_idx, 'fold'] = fold

print(f"Train samples: {len(train_wide)}")
print(f"Folds: {train_wide['fold'].value_counts().sort_index().to_dict()}")

# Target 분포 확인
print(f"\nTarget statistics:")
for col in ['Dry_Green_g', 'Dry_Clover_g', 'Dry_Dead_g']:
    zeros = (train_wide[col] == 0).sum()
    print(f"  {col}: mean={train_wide[col].mean():.2f}, zeros={zeros} ({zeros/len(train_wide)*100:.1f}%)")

#%% [markdown]
# ## 🎨 Data Augmentation
#
# **전략**: 농업 이미지 특성을 고려한 보수적 augmentation
# - 색상 변환은 최소화 (녹색/갈색이 바이오매스 예측에 중요)
# - 기하학적 변환 위주 (flip, rotation)
# - MixUp으로 regularization

#%%
def get_train_transforms(cfg):
    """
    보수적 augmentation:
    - Flip: 식물 이미지에 안전
    - 작은 회전: 카메라 각도 변화 시뮬레이션
    - 가벼운 색상 변환: 조명 변화 시뮬레이션 (너무 강하면 안됨)
    """
    return T.Compose([
        T.Resize(cfg.img_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=10),  # 작은 회전만
        T.ColorJitter(
            brightness=0.1,  # 약한 밝기 변화
            contrast=0.1,
            saturation=0.1,
            hue=0.02  # 색조는 거의 안 바꿈 (녹색/갈색 보존)
        ),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms(cfg):
    return T.Compose([
        T.Resize(cfg.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

#%%
class BiomassDataset(Dataset):
    """Left/Right split dataset with log1p transform"""
    def __init__(self, df, cfg, transform=None, mode='train'):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(self.cfg.DATA_PATH / row['image_path']).convert('RGB')
        width, height = img.size
        mid = width // 2

        left_img = img.crop((0, 0, mid, height))
        right_img = img.crop((mid, 0, width, height))

        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)

        if self.mode == 'train':
            targets = torch.tensor([
                row['Dry_Green_g'],
                row['Dry_Clover_g'],
                row['Dry_Dead_g']
            ], dtype=torch.float32)

            if self.cfg.use_log1p:
                targets = torch.log1p(targets)

            return left_img, right_img, targets
        else:
            return left_img, right_img, row['image_id']

#%% [markdown]
# ## 🔀 MixUp (Simple Version)

#%%
def mixup_data(left, right, targets, alpha=0.2):
    """Simple MixUp for regression"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = left.size(0)
    index = torch.randperm(batch_size, device=left.device)

    mixed_left = lam * left + (1 - lam) * left[index]
    mixed_right = lam * right + (1 - lam) * right[index]
    mixed_targets = lam * targets + (1 - lam) * targets[index]

    return mixed_left, mixed_right, mixed_targets

#%% [markdown]
# ## 🧠 Model Architecture

#%%
class FiLM(nn.Module):
    """Feature-wise Linear Modulation"""
    def __init__(self, feat_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, feat_dim * 2)
        )

    def forward(self, context):
        out = self.mlp(context)
        gamma, beta = torch.chunk(out, 2, dim=1)
        return gamma, beta


class ZeroInflatedHead(nn.Module):
    """
    Zero-Inflated Head for Clover (38% zeros)
    Two-stage: (1) P(positive) (2) amount if positive
    """
    def __init__(self, in_features, hidden_dim, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        self.regressor = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )

    def forward(self, x):
        prob = torch.sigmoid(self.classifier(x))
        amount = self.regressor(x)
        return prob * amount


class CSIROModelV2(nn.Module):
    """
    DINOv3 + FiLM + Configurable Head

    Changes from v1:
    - Configurable head_hidden_dim
    - Optional Zero-Inflated Clover head
    - Simpler architecture
    """
    def __init__(self, cfg):
        super().__init__()

        # Backbone
        weights_file = cfg.WEIGHTS_PATH / "dinov3_vitl16_qkvb.pth"
        if weights_file.exists():
            print(f"Loading backbone from: {weights_file}")
            self.backbone = timm.create_model(
                cfg.model_name, pretrained=False, num_classes=0, global_pool='avg'
            )
            state = torch.load(weights_file, map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(state, strict=False)
            print("✓ Backbone loaded")
        else:
            print("Loading backbone from timm (online)")
            self.backbone = timm.create_model(
                cfg.model_name, pretrained=True, num_classes=0, global_pool='avg'
            )

        feat_dim = self.backbone.num_features
        combined_dim = feat_dim * 2
        hidden_dim = cfg.head_hidden_dim
        dropout = cfg.dropout

        # FiLM
        self.film = FiLM(feat_dim)

        # Heads
        def make_head():
            return nn.Sequential(
                nn.Linear(combined_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Softplus()
            )

        self.head_green = make_head()
        self.head_dead = make_head()

        if cfg.use_zero_inflated_clover:
            self.head_clover = ZeroInflatedHead(combined_dim, hidden_dim, dropout)
        else:
            self.head_clover = make_head()

        print(f"Model: head_dim={hidden_dim}, dropout={dropout}, ZI_clover={cfg.use_zero_inflated_clover}")

    def forward(self, left_img, right_img):
        left_feat = self.backbone(left_img)
        right_feat = self.backbone(right_img)

        context = (left_feat + right_feat) / 2
        gamma, beta = self.film(context)

        left_mod = left_feat * (1 + gamma) + beta
        right_mod = right_feat * (1 + gamma) + beta

        combined = torch.cat([left_mod, right_mod], dim=1)

        green = self.head_green(combined)
        clover = self.head_clover(combined)
        dead = self.head_dead(combined)

        gdm = green + clover
        total = gdm + dead

        return torch.cat([green, dead, clover, gdm, total], dim=1)

#%% [markdown]
# ## 📈 EMA (Exponential Moving Average)

#%%
class EMA:
    """Exponential Moving Average for model weights"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register()

    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply_shadow(self):
        """Apply EMA weights for evaluation"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original weights after evaluation"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

#%% [markdown]
# ## 🏋️ Training Functions

#%%
def train_one_epoch(model, loader, optimizer, scheduler, scaler, cfg, ema=None):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc="Training")
    for step, (left, right, targets) in enumerate(pbar):
        left = left.to(cfg.device)
        right = right.to(cfg.device)
        targets = targets.to(cfg.device)

        # MixUp
        if cfg.use_mixup and np.random.random() < 0.5:
            left, right, targets = mixup_data(left, right, targets, cfg.mixup_alpha)

        with autocast():
            outputs = model(left, right)
            pred = outputs[:, [0, 2, 1]]  # [Green, Clover, Dead]

            if cfg.use_huber_loss:
                loss = F.huber_loss(pred, targets, delta=cfg.huber_delta)
            else:
                loss = F.mse_loss(pred, targets)

            loss = loss / cfg.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % cfg.accumulation_steps == 0:
            if cfg.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            if ema is not None:
                ema.update()

        total_loss += loss.item() * cfg.accumulation_steps
        pbar.set_postfix({'loss': f'{loss.item() * cfg.accumulation_steps:.4f}'})

    # Flush remaining gradients
    if (step + 1) % cfg.accumulation_steps != 0:
        if cfg.gradient_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if ema is not None:
            ema.update()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, cfg):
    model.eval()
    all_preds = []
    all_targets = []

    for left, right, targets in tqdm(loader, desc="Validating"):
        left = left.to(cfg.device)
        right = right.to(cfg.device)

        outputs = model(left, right)

        if cfg.use_log1p:
            outputs = torch.expm1(outputs)

        all_preds.append(outputs.cpu().numpy())
        all_targets.append(targets.numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    if cfg.use_log1p:
        targets = np.expm1(targets)

    # Build full targets
    full_targets = np.zeros((len(targets), 5))
    full_targets[:, 0] = targets[:, 0]  # Green
    full_targets[:, 1] = targets[:, 2]  # Dead
    full_targets[:, 2] = targets[:, 1]  # Clover
    full_targets[:, 3] = targets[:, 0] + targets[:, 1]  # GDM
    full_targets[:, 4] = full_targets[:, 3] + targets[:, 2]  # Total

    score = competition_metric(full_targets, preds)
    return score

#%%
def train_fold(fold, train_df, cfg):
    print(f"\n{'='*60}")
    print(f"FOLD {fold}")
    print(f"{'='*60}")

    train_data = train_df[train_df['fold'] != fold].reset_index(drop=True)
    val_data = train_df[train_df['fold'] == fold].reset_index(drop=True)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Datasets
    train_ds = BiomassDataset(train_data, cfg, get_train_transforms(cfg), 'train')
    val_ds = BiomassDataset(val_data, cfg, get_val_transforms(cfg), 'train')

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size,
        shuffle=True, num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size * 2,
        shuffle=False, num_workers=cfg.num_workers, pin_memory=True
    )

    # Model
    model = CSIROModelV2(cfg).to(cfg.device)

    # EMA
    ema = EMA(model, cfg.ema_decay) if cfg.use_ema else None

    # Optimizer
    backbone_params = list(model.backbone.parameters())
    head_params = (
        list(model.head_green.parameters()) +
        list(model.head_clover.parameters()) +
        list(model.head_dead.parameters()) +
        list(model.film.parameters())
    )

    optimizer = AdamW([
        {'params': backbone_params, 'lr': cfg.lr * cfg.backbone_lr_mult},
        {'params': head_params, 'lr': cfg.lr}
    ], weight_decay=cfg.weight_decay)

    total_steps = len(train_loader) * cfg.epochs // cfg.accumulation_steps
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    scaler = GradScaler()

    # Training loop
    best_score = -float('inf')
    best_epoch = 0
    patience = 5
    no_improve = 0

    for epoch in range(cfg.epochs):
        print(f"\nEpoch {epoch+1}/{cfg.epochs}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, cfg, ema
        )

        # Validate with EMA weights
        if ema is not None:
            ema.apply_shadow()

        val_score = validate(model, val_loader, cfg)

        if ema is not None:
            ema.restore()

        print(f"Loss: {train_loss:.4f} | CV: {val_score:.4f}")

        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch + 1
            no_improve = 0

            # Save with EMA weights
            if ema is not None:
                ema.apply_shadow()

            torch.save(model.state_dict(), cfg.OUTPUT_DIR / f'model_fold{fold}.pth')

            if ema is not None:
                ema.restore()

            print(f"  ✓ New best! Saved.")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"\nFold {fold} Best: {best_score:.4f} (epoch {best_epoch})")

    # Backup to Google Drive
    if GDRIVE_SAVE_PATH is not None:
        src = cfg.OUTPUT_DIR / f'model_fold{fold}.pth'
        if src.exists():
            shutil.copy(src, GDRIVE_SAVE_PATH / f'model_fold{fold}.pth')
            print(f"  📁 Backed up to Drive")

    flush()
    return best_score

#%% [markdown]
# ## 🚀 Main Training Loop

#%%
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 TRAINING START")
    print("="*60)
    print(f"Config: head={cfg.head_hidden_dim}, dropout={cfg.dropout}, lr={cfg.lr}")
    print(f"        mixup={cfg.use_mixup}, huber={cfg.use_huber_loss}, ema={cfg.use_ema}")
    print(f"        log1p={cfg.use_log1p}, ZI_clover={cfg.use_zero_inflated_clover}")

    fold_scores = []

    for fold in range(cfg.n_folds):
        score = train_fold(fold, train_wide, cfg)
        fold_scores.append(score)

    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE")
    print("="*60)
    print(f"Fold scores: {[f'{s:.4f}' for s in fold_scores]}")
    print(f"Mean CV: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

    # Save to Google Drive
    if GDRIVE_SAVE_PATH is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = GDRIVE_SAVE_PATH / f"run_{timestamp}_cv{np.mean(fold_scores):.4f}"
        final_path.mkdir(parents=True, exist_ok=True)

        for f in cfg.OUTPUT_DIR.glob("model_fold*.pth"):
            shutil.copy(f, final_path / f.name)

        results = {
            'fold_scores': fold_scores,
            'mean_cv': float(np.mean(fold_scores)),
            'std_cv': float(np.std(fold_scores)),
            'config': {
                'model_name': cfg.model_name,
                'head_hidden_dim': cfg.head_hidden_dim,
                'dropout': cfg.dropout,
                'lr': cfg.lr,
                'backbone_lr_mult': cfg.backbone_lr_mult,
                'use_huber_loss': cfg.use_huber_loss,
                'use_ema': cfg.use_ema,
                'use_mixup': cfg.use_mixup,
                'use_log1p': cfg.use_log1p,
            }
        }
        with open(final_path / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Saved to: {final_path}")
