#%% [markdown]
# # 🚀 Optimized DINOv3 Training Pipeline (CV 0.85+ 목표)
#
# **핵심 개선사항**:
# 1. TrivialAugmentWide + 강화된 Augmentation
# 2. Log1p Target Transformation
# 3. C-MixUp for Regression
# 4. Deeper Head Architecture (512 → 128 → 1)
# 5. Zero-Inflated Clover Head
# 6. Optimized Hyperparameters

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
# ## Configuration (Optimized)

#%%
class CFG:
    # === Paths (Colab with kagglehub) ===
    DATA_PATH = None  # Will be set after kagglehub download
    OUTPUT_DIR = Path("/kaggle/working")
    WEIGHTS_PATH = None  # Will be set after kagglehub download
    
    # === Model ===
    model_name = "vit_large_patch16_224.dinov2.lvd142m"
    backbone_dim = 1024
    img_size = (518, 518)  # DINOv2 optimal (divisible by 14)
    
    # === Training (Optimized) ===
    n_folds = 5
    epochs = 25  # 15 → 25 (더 긴 학습)
    batch_size = 8  # 16 → 8 (gradient accumulation 사용)
    accumulation_steps = 2  # 실효 batch = 16
    lr = 3e-5  # 1e-4 → 3e-5 (더 안정적)
    backbone_lr_mult = 0.01  # 0.1 → 0.01 (backbone 더 보호)
    weight_decay = 5e-4  # 1e-4 → 5e-4 (더 강한 regularization)
    dropout = 0.3  # 0.0 → 0.3
    warmup_epochs = 2  # 1 → 2
    
    # === Augmentation ===
    use_trivial_augment = True
    use_cmixup = True
    cmixup_alpha = 0.4
    cmixup_sigma = 0.5
    
    # === Target Transform ===
    use_log1p = True
    
    # === Other ===
    seed = 42
    num_workers = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = CFG()

#%% [markdown]
# ## Kaggle Data Download (Colab)

#%%
# Colab에서 실행시 주석 해제
# import kagglehub
# kagglehub.login()
# 
# csiro_biomass_path = kagglehub.competition_download('csiro-biomass')
# kbsooo_pretrained_weights_biomass_path = kagglehub.dataset_download('kbsooo/pretrained-weights-biomass')
# 
# cfg.DATA_PATH = Path(csiro_biomass_path)
# cfg.WEIGHTS_PATH = Path(kbsooo_pretrained_weights_biomass_path) / "dinov3_large" / "dinov3_large"
# print(f"Data path: {cfg.DATA_PATH}")
# print(f"Weights path: {cfg.WEIGHTS_PATH}")

# Kaggle 환경 (기본값)
if cfg.DATA_PATH is None:
    cfg.DATA_PATH = Path("/kaggle/input/csiro-biomass")
    cfg.WEIGHTS_PATH = Path("/kaggle/input/pretrained-weights-biomass/dinov3_large/dinov3_large")

cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Device: {cfg.device}")
print(f"Config: epochs={cfg.epochs}, batch={cfg.batch_size}, lr={cfg.lr}")

#%% [markdown]
# ## Competition Metric

#%%
TARGET_WEIGHTS = {
    'Dry_Green_g': 0.1, 'Dry_Dead_g': 0.1, 'Dry_Clover_g': 0.1,
    'GDM_g': 0.2, 'Dry_Total_g': 0.5,
}
TARGET_ORDER = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

def competition_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted R² score."""
    total_weight = 0.0
    weighted_r2 = 0.0
    
    for i, target in enumerate(TARGET_ORDER):
        weight = TARGET_WEIGHTS[target]
        ss_res = np.sum((y_true[:, i] - y_pred[:, i]) ** 2)
        ss_tot = np.sum((y_true[:, i] - np.mean(y_true[:, i])) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        weighted_r2 += weight * r2
        total_weight += weight
    
    return weighted_r2 / total_weight

#%% [markdown]
# ## Data Preparation

#%%
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long format to wide format."""
    pivot = df.pivot_table(
        index=['image_path', 'State', 'Species', 'Sampling_Date', 'Pre_GSHH_NDVI', 'Height_Ave_cm'],
        columns='target_name',
        values='target',
        aggfunc='first'
    ).reset_index()
    pivot.columns.name = None
    return pivot

#%%
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

print(f"Train data shape: {train_wide.shape}")
print(f"Fold distribution:\n{train_wide['fold'].value_counts().sort_index()}")

#%% [markdown]
# ## Enhanced Dataset with TrivialAugment

#%%
def get_train_transforms(cfg):
    """TrivialAugmentWide + Enhanced Augmentation"""
    transforms_list = [
        T.Resize(cfg.img_size),
    ]
    
    if cfg.use_trivial_augment:
        transforms_list.append(T.TrivialAugmentWide())
    
    transforms_list.extend([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        T.RandomPerspective(distortion_scale=0.15, p=0.3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])
    
    return T.Compose(transforms_list)

def get_val_transforms(cfg):
    return T.Compose([
        T.Resize(cfg.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

#%%
class BiomassDataset(Dataset):
    """Enhanced Dataset with Left/Right Split"""
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
            targets = torch.tensor([
                row['Dry_Green_g'],
                row['Dry_Clover_g'],
                row['Dry_Dead_g']
            ], dtype=torch.float32)
            
            # Log1p transform
            if self.cfg.use_log1p:
                targets = torch.log1p(targets)
            
            return left_img, right_img, targets
        else:
            return left_img, right_img, row['image_id']

#%% [markdown]
# ## C-MixUp for Regression

#%%
def c_mixup(left1, right1, targets1, left2, right2, targets2, sigma=0.5, alpha=0.4):
    """
    C-MixUp: Continuous target-aware MixUp for regression
    비슷한 target끼리 더 강하게 mixing
    """
    # Target 거리 계산
    target_dist = torch.abs(targets1 - targets2).mean()
    
    # 거리 기반 mixing probability
    mix_weight = torch.exp(-target_dist / sigma)
    
    # Beta distribution으로 lambda 샘플링
    if mix_weight > 0.1:  # 거리가 가까울 때만 mixing
        lam = np.random.beta(alpha * mix_weight, alpha * mix_weight)
    else:
        lam = 1.0  # 거리가 멀면 mixing 안함
    
    # Mix images and targets
    mixed_left = lam * left1 + (1 - lam) * left2
    mixed_right = lam * right1 + (1 - lam) * right2
    mixed_targets = lam * targets1 + (1 - lam) * targets2
    
    return mixed_left, mixed_right, mixed_targets

#%% [markdown]
# ## Model Components

#%%
class FiLM(nn.Module):
    """Feature-wise Linear Modulation"""
    def __init__(self, feat_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Linear(feat_dim // 2, feat_dim * 2)
        )
    
    def forward(self, context):
        gamma_beta = self.mlp(context)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        return gamma, beta

#%%
class ZeroInflatedHead(nn.Module):
    """
    Zero-Inflated Head for Clover
    두 단계 예측: (1) is_positive? (2) amount if positive
    """
    def __init__(self, in_features, dropout=0.3):
        super().__init__()
        
        # Binary classifier (is positive?)
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        # Amount regressor
        self.regressor = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Softplus()
        )
    
    def forward(self, x):
        prob = torch.sigmoid(self.classifier(x))
        amount = self.regressor(x)
        return prob * amount  # Expected value

#%%
class CSIROModelV2(nn.Module):
    """
    Optimized DINOv2 Model with:
    - Deeper Head (512 → 128 → 1)
    - LayerNorm + GELU
    - Dropout regularization
    - Zero-Inflated Clover Head
    """
    def __init__(self, model_name, pretrained=True, weights_path=None, dropout=0.3):
        super().__init__()
        
        # DINOv3 ViT-Large backbone
        if pretrained and weights_path and Path(weights_path).exists():
            print(f"Loading backbone from: {weights_path}")
            self.backbone = timm.create_model(
                "vit_large_patch16_224.dinov2.lvd142m",
                pretrained=False,
                num_classes=0,
                global_pool='avg'
            )
            state_dict = torch.load(weights_path / "dinov3_vitl16_qkvb.pth", map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(state_dict, strict=False)
            print("✓ Backbone loaded from local weights")
        else:
            print("Loading backbone from timm (online)")
            self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='avg')
        
        feat_dim = self.backbone.num_features
        print(f"Backbone feature dim: {feat_dim}")
        
        # FiLM for cross-region modulation
        self.film = FiLM(feat_dim)
        
        # Deeper head architecture
        def make_head(in_dim, dropout):
            return nn.Sequential(
                nn.Linear(in_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(512, 128),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(128, 1),
                nn.Softplus()
            )
        
        combined_dim = feat_dim * 2
        self.head_green = make_head(combined_dim, dropout)
        self.head_dead = make_head(combined_dim, dropout)
        
        # Zero-inflated head for Clover (38% zeros)
        self.head_clover = ZeroInflatedHead(combined_dim, dropout)
    
    def forward(self, left_img, right_img):
        # Extract features
        left_feat = self.backbone(left_img)
        right_feat = self.backbone(right_img)
        
        # FiLM modulation
        context = (left_feat + right_feat) / 2
        gamma, beta = self.film(context)
        
        left_mod = left_feat * (1 + gamma) + beta
        right_mod = right_feat * (1 + gamma) + beta
        
        # Concatenate
        combined = torch.cat([left_mod, right_mod], dim=1)
        
        # Predict
        green = self.head_green(combined)
        clover = self.head_clover(combined)
        dead = self.head_dead(combined)
        
        # Physics constraints
        gdm = green + clover
        total = gdm + dead
        
        # Return: [Green, Dead, Clover, GDM, Total]
        return torch.cat([green, dead, clover, gdm, total], dim=1)

#%% [markdown]
# ## Training Functions with C-MixUp

#%%
def train_one_epoch(model, loader, optimizer, scheduler, device, scaler, cfg):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc="Training")
    for step, (left, right, targets) in enumerate(pbar):
        left = left.to(device)
        right = right.to(device)
        targets = targets.to(device)
        
        # C-MixUp
        if cfg.use_cmixup and np.random.random() < 0.5:
            indices = torch.randperm(len(left))
            left, right, targets = c_mixup(
                left, right, targets,
                left[indices], right[indices], targets[indices],
                sigma=cfg.cmixup_sigma, alpha=cfg.cmixup_alpha
            )
        
        with autocast():
            outputs = model(left, right)
            # Loss on Green, Clover, Dead (indices 0, 2, 1 in output)
            pred = outputs[:, [0, 2, 1]]
            loss = F.mse_loss(pred, targets)
            loss = loss / cfg.accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (step + 1) % cfg.accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        
        total_loss += loss.item() * cfg.accumulation_steps
        pbar.set_postfix({'loss': f'{loss.item() * cfg.accumulation_steps:.2f}'})
    
    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, device, cfg):
    model.eval()
    all_preds = []
    all_targets = []
    
    for left, right, targets in tqdm(loader, desc="Validating"):
        left = left.to(device)
        right = right.to(device)
        
        outputs = model(left, right)
        
        # Inverse log1p if used
        if cfg.use_log1p:
            outputs = torch.expm1(outputs)
        
        all_preds.append(outputs.cpu().numpy())
        all_targets.append(targets.numpy())
    
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    
    # Inverse log1p for targets
    if cfg.use_log1p:
        targets = np.expm1(targets)
    
    # Compute full targets for metric
    full_targets = np.zeros((len(targets), 5))
    full_targets[:, 0] = targets[:, 0]  # Green
    full_targets[:, 1] = targets[:, 2]  # Dead
    full_targets[:, 2] = targets[:, 1]  # Clover
    full_targets[:, 3] = targets[:, 0] + targets[:, 1]  # GDM
    full_targets[:, 4] = full_targets[:, 3] + targets[:, 2]  # Total
    
    score = competition_metric(full_targets, preds)
    return score, preds

#%%
def train_fold(fold, train_df, cfg):
    """Train single fold with all optimizations"""
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
    
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size,
        shuffle=True, num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size * 2,
        shuffle=False, num_workers=cfg.num_workers, pin_memory=True
    )
    
    # Model
    model = CSIROModelV2(
        cfg.model_name,
        pretrained=True,
        weights_path=cfg.WEIGHTS_PATH,
        dropout=cfg.dropout
    )
    model = model.to(cfg.device)
    
    # Optimizer with layer-wise LR
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
    warmup_steps = len(train_loader) * cfg.warmup_epochs // cfg.accumulation_steps
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    scaler = GradScaler()
    
    # Training loop
    best_score = -float('inf')
    best_epoch = 0
    patience = 5
    no_improve = 0
    
    for epoch in range(cfg.epochs):
        print(f"\nEpoch {epoch+1}/{cfg.epochs}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, cfg.device, scaler, cfg)
        val_score, _ = validate(model, val_loader, cfg.device, cfg)
        
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
    
    flush()
    return best_score

#%% [markdown]
# ## Main Training Loop

#%%
if __name__ == "__main__":
    fold_scores = []
    
    for fold in range(cfg.n_folds):
        score = train_fold(fold, train_wide, cfg)
        fold_scores.append(score)
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE")
    print("="*60)
    print(f"Fold scores: {fold_scores}")
    print(f"Mean CV: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    
    # List saved models
    print("\nSaved models:")
    for f in sorted(cfg.OUTPUT_DIR.glob("model_fold*.pth")):
        print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")
