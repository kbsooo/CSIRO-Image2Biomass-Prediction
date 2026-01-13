#%% [markdown]
# # 🔧 DINOv3 Training v15 - Back to Basics
#
# **전략**: v12 baseline (CV 0.67)을 기반으로 점진적 개선
#
# **v12 → v15 변경사항** (최소한만):
# 1. Dropout 0.0 → 0.1 (약한 regularization)
# 2. More epochs: 15 → 20
# 3. Early stopping
# 4. 더 강한 augmentation (TrivialAugment 제거, 안전한 것만)

#%%
import os
import gc
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from torch.cuda.amp import GradScaler, autocast

import timm
from torchvision import transforms as T
from sklearn.model_selection import StratifiedGroupKFold

import warnings
warnings.filterwarnings('ignore')
tqdm.pandas()

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

#%% [markdown]
# ## 🔐 Step 1: Google Drive Mount (Colab Only)

#%%
GDRIVE_SAVE_PATH = None

try:
    from google.colab import drive
    drive.mount('/content/drive')
    GDRIVE_SAVE_PATH = Path('/content/drive/MyDrive/kaggle_models/csiro_biomass_v15')
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
# ## ⚙️ Configuration (v12 기반, 최소 변경)

#%%
class CFG:
    # === Paths ===
    DATA_PATH = None
    OUTPUT_DIR = None
    WEIGHTS_PATH = None
    
    # === Model (v12와 동일) ===
    model_name = "vit_large_patch16_dinov3_qkvb.lvd1689m"
    backbone_dim = 1024
    img_size = (512, 512)
    
    # === Training (v12 기반 + 약간 개선) ===
    n_folds = 5
    epochs = 20  # 15 → 20
    batch_size = 16
    lr = 1e-4  # v12와 동일
    backbone_lr_mult = 0.1  # v12와 동일
    weight_decay = 1e-4  # v12와 동일
    dropout = 0.1  # 0.0 → 0.1 (약한 regularization)
    
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
    """Weighted R² score."""
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

#%% [markdown]
# ## 🎨 Dataset & Augmentation (v12와 동일 + 약간 강화)

#%%
class BiomassDataset(Dataset):
    """v12와 동일한 Left/Right Split Dataset"""
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
        mid_point = width // 2
        
        left_img = img.crop((0, 0, mid_point, height))
        right_img = img.crop((mid_point, 0, width, height))
        
        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)
        
        if self.mode == 'train':
            # 독립 타겟 3개만 (GDM, Total은 모델에서 계산)
            targets = torch.tensor([
                row['Dry_Green_g'],
                row['Dry_Clover_g'],
                row['Dry_Dead_g']
            ], dtype=torch.float32)
            return left_img, right_img, targets
        else:
            return left_img, right_img, row['image_id']

def get_train_transforms(cfg):
    """v12 augmentation + 약간 강화"""
    return T.Compose([
        T.Resize(cfg.img_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=10),  # 추가
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms(cfg):
    return T.Compose([
        T.Resize(cfg.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

#%% [markdown]
# ## 🧠 Model (v12와 동일 구조)

#%%
class FiLM(nn.Module):
    """Feature-wise Linear Modulation (v12와 동일)"""
    def __init__(self, feat_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, feat_dim * 2)
        )
    
    def forward(self, context):
        gamma_beta = self.mlp(context)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        return gamma, beta

class CSIROModel(nn.Module):
    """
    v12와 동일한 구조 (검증됨: CV 0.67)
    변경: dropout만 추가
    """
    def __init__(self, model_name, pretrained=True, weights_path=None, dropout=0.1):
        super().__init__()
        
        # DINOv3 ViT-Large backbone
        if pretrained and weights_path and Path(weights_path).exists():
            print(f"Loading backbone from: {weights_path}")
            self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool='avg')
            state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(state_dict, strict=False)
            print("✓ Backbone loaded from local weights")
        else:
            print("Loading backbone from timm (online)")
            self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='avg')
        
        feat_dim = self.backbone.num_features
        print(f"Backbone feature dim: {feat_dim}")
        
        # FiLM for cross-region modulation
        self.film = FiLM(feat_dim)
        
        # v12와 동일한 head structure (256 hidden units)
        def make_head():
            return nn.Sequential(
                nn.Linear(feat_dim * 2, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),  # dropout 추가
                nn.Linear(256, 1)
            )
        
        self.head_green = make_head()
        self.head_clover = make_head()
        self.head_dead = make_head()
        
        # Softplus for non-negative outputs
        self.softplus = nn.Softplus(beta=1.0)
    
    def forward(self, left_img, right_img):
        # Extract features from both halves
        left_feat = self.backbone(left_img)
        right_feat = self.backbone(right_img)
        
        # Compute context as average
        context = (left_feat + right_feat) / 2
        
        # Generate modulation parameters
        gamma, beta = self.film(context)
        
        # Modulate features
        left_mod = left_feat * (1 + gamma) + beta
        right_mod = right_feat * (1 + gamma) + beta
        
        # Concatenate
        combined = torch.cat([left_mod, right_mod], dim=1)
        
        # Predict independent targets
        green = self.softplus(self.head_green(combined))
        clover = self.softplus(self.head_clover(combined))
        dead = self.softplus(self.head_dead(combined))
        
        # Physics constraints
        gdm = green + clover
        total = gdm + dead
        
        # Return: [Green, Dead, Clover, GDM, Total] (competition order)
        return torch.cat([green, dead, clover, gdm, total], dim=1)

#%% [markdown]
# ## 🏋️ Training Functions (v12와 동일)

#%%
def train_one_epoch(model, loader, optimizer, scheduler, device, scaler):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc="Training")
    for left, right, targets in pbar:
        left = left.to(device)
        right = right.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(left, right)
            # Loss on Green, Clover, Dead (indices 0, 2, 1 in output)
            pred = outputs[:, [0, 2, 1]]  # Reorder to [Green, Clover, Dead]
            loss = F.mse_loss(pred, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.2f}'})
    
    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    for left, right, targets in tqdm(loader, desc="Validating"):
        left = left.to(device)
        right = right.to(device)
        
        outputs = model(left, right)
        all_preds.append(outputs.cpu().numpy())
        all_targets.append(targets.numpy())
    
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    
    # Compute full targets for metric
    full_targets = np.zeros((len(targets), 5))
    full_targets[:, 0] = targets[:, 0]  # Green
    full_targets[:, 1] = targets[:, 2]  # Dead
    full_targets[:, 2] = targets[:, 1]  # Clover
    full_targets[:, 3] = targets[:, 0] + targets[:, 1]  # GDM = Green + Clover
    full_targets[:, 4] = full_targets[:, 3] + targets[:, 2]  # Total = GDM + Dead
    
    score = competition_metric(full_targets, preds)
    return score, preds

#%%
def train_fold(fold, train_df, cfg):
    """Train single fold"""
    print(f"\n{'='*60}")
    print(f"FOLD {fold}")
    print(f"{'='*60}")
    
    # Split data
    train_data = train_df[train_df['fold'] != fold].reset_index(drop=True)
    val_data = train_df[train_df['fold'] == fold].reset_index(drop=True)
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Datasets & Loaders
    train_ds = BiomassDataset(train_data, cfg, get_train_transforms(cfg), 'train')
    val_ds = BiomassDataset(val_data, cfg, get_val_transforms(cfg), 'train')
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, 
                              shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size * 2,
                            shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    
    # Model
    weights_path = cfg.WEIGHTS_PATH / "dinov3_vitl16_qkvb.pth"
    model = CSIROModel(
        cfg.model_name, 
        pretrained=True, 
        weights_path=weights_path,
        dropout=cfg.dropout
    )
    model = model.to(cfg.device)
    
    # Optimizer with layer-wise learning rate decay
    backbone_params = list(model.backbone.parameters())
    head_params = (list(model.head_green.parameters()) + 
                   list(model.head_clover.parameters()) + 
                   list(model.head_dead.parameters()) + 
                   list(model.film.parameters()))
    
    optimizer = AdamW([
        {'params': backbone_params, 'lr': cfg.lr * cfg.backbone_lr_mult},
        {'params': head_params, 'lr': cfg.lr}
    ], weight_decay=cfg.weight_decay)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train_loader),
        num_training_steps=len(train_loader) * cfg.epochs
    )
    
    scaler = GradScaler()
    
    # Training loop with early stopping
    best_score = -float('inf')
    best_epoch = 0
    patience = 5
    no_improve = 0
    
    for epoch in range(cfg.epochs):
        print(f"\nEpoch {epoch+1}/{cfg.epochs}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, cfg.device, scaler)
        val_score, _ = validate(model, val_loader, cfg.device)
        
        print(f"Loss: {train_loss:.4f} | CV: {val_score:.4f}")
        
        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch + 1
            no_improve = 0
            torch.save(model.state_dict(), cfg.OUTPUT_DIR / f'model_fold{fold}.pth')
            print(f"  ✓ New best! Saved.")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    print(f"\nFold {fold} Best: {best_score:.4f} (epoch {best_epoch})")
    
    # Backup to Google Drive
    if GDRIVE_SAVE_PATH is not None:
        import shutil
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
    print("🚀 TRAINING START (v15 - Back to Basics)")
    print("="*60)
    print(f"Config: dropout={cfg.dropout}, epochs={cfg.epochs}, lr={cfg.lr}")
    
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
        import shutil
        import json
        from datetime import datetime
        
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
                'dropout': cfg.dropout,
                'lr': cfg.lr,
                'epochs': cfg.epochs,
            }
        }
        with open(final_path / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Saved to: {final_path}")
