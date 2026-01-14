#%% [markdown]
# # 🏆 v17: Optuna-Optimized Training
#
# **최적 하이퍼파라미터** (Optuna 탐색 결과):
# - Head: 512 hidden, 3 layers, LayerNorm, dropout=0.1
# - LR: 2.33e-4, backbone_mult=0.084, warmup=7.8%
# - Augmentation: color_focus
# - Loss: MSE
# - Batch: 8

#%%
import os
import gc
import json
import random
import shutil
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
from torchvision import transforms as T
from sklearn.model_selection import StratifiedGroupKFold

import warnings
warnings.filterwarnings('ignore')
tqdm.pandas()

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

#%% [markdown]
# ## 🔐 Setup

#%%
GDRIVE_SAVE_PATH = None

try:
    from google.colab import drive
    drive.mount('/content/drive')
    GDRIVE_SAVE_PATH = Path('/content/drive/MyDrive/kaggle_models/csiro_biomass_v17')
    GDRIVE_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    print(f"✓ Drive: {GDRIVE_SAVE_PATH}")
except ImportError:
    print("Not in Colab")

#%%
import kagglehub

IS_KAGGLE = Path("/kaggle/input/csiro-biomass").exists()
if not IS_KAGGLE:
    kagglehub.login()

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
# ## ⚙️ Configuration (Optuna Best)

#%%
class CFG:
    # === Paths ===
    DATA_PATH = None
    OUTPUT_DIR = None
    WEIGHTS_PATH = None
    
    # === Model ===
    model_name = "vit_large_patch16_dinov3_qkvb.lvd1689m"
    img_size = (512, 512)
    
    # === Head Architecture (Optuna Best) ===
    hidden_dim = 512
    num_layers = 3
    dropout = 0.1
    use_layernorm = True
    
    # === Training (Optuna Best) ===
    n_folds = 5
    epochs = 25
    batch_size = 8
    lr = 2.33e-4  # 0.0002325906067455594
    backbone_lr_mult = 0.084  # 0.08439262136026988
    warmup_ratio = 0.078  # 0.07812989471500509
    weight_decay = 6.37e-5  # 6.370374992582331e-05
    
    # === Augmentation (Optuna Best) ===
    aug_strategy = "color_focus"
    
    # === Loss (Optuna Best) ===
    loss_type = "mse"
    
    # === Other ===
    patience = 7
    seed = 42
    num_workers = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = CFG()

#%%
# Data paths
if IS_KAGGLE:
    cfg.DATA_PATH = Path("/kaggle/input/csiro-biomass")
    cfg.WEIGHTS_PATH = Path("/kaggle/input/pretrained-weights-biomass/dinov3_large/dinov3_large")
    cfg.OUTPUT_DIR = Path("/kaggle/working")
else:
    csiro_path = kagglehub.competition_download('csiro-biomass')
    weights_path = kagglehub.dataset_download('kbsooo/pretrained-weights-biomass')
    cfg.DATA_PATH = Path(csiro_path)
    cfg.WEIGHTS_PATH = Path(weights_path) / "dinov3_large" / "dinov3_large"
    cfg.OUTPUT_DIR = Path("/content/output")

cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Data: {cfg.DATA_PATH}")
print(f"Output: {cfg.OUTPUT_DIR}")

#%% [markdown]
# ## 📊 Data

#%%
TARGET_WEIGHTS = {'Dry_Green_g': 0.1, 'Dry_Dead_g': 0.1, 'Dry_Clover_g': 0.1, 'GDM_g': 0.2, 'Dry_Total_g': 0.5}
TARGET_ORDER = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

def competition_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    weighted_r2 = 0.0
    for i, target in enumerate(TARGET_ORDER):
        weight = TARGET_WEIGHTS[target]
        ss_res = np.sum((y_true[:, i] - y_pred[:, i]) ** 2)
        ss_tot = np.sum((y_true[:, i] - np.mean(y_true[:, i])) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        weighted_r2 += weight * r2
    return weighted_r2

#%%
def prepare_data(df):
    pivot = df.pivot_table(
        index=['image_path', 'State', 'Species', 'Sampling_Date', 'Pre_GSHH_NDVI', 'Height_Ave_cm'],
        columns='target_name', values='target', aggfunc='first'
    ).reset_index()
    pivot.columns.name = None
    return pivot

train_df = pd.read_csv(cfg.DATA_PATH / "train.csv")
train_wide = prepare_data(train_df)
train_wide['image_id'] = train_wide['image_path'].apply(lambda x: Path(x).stem)

sgkf = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
train_wide['fold'] = -1
for fold, (_, val_idx) in enumerate(sgkf.split(train_wide, train_wide['State'], groups=train_wide['image_id'])):
    train_wide.loc[val_idx, 'fold'] = fold

print(f"Train samples: {len(train_wide)}")
print(f"Folds: {train_wide['fold'].value_counts().sort_index().to_dict()}")

#%% [markdown]
# ## 🎨 Augmentation (color_focus)

#%%
def get_train_transforms(cfg):
    """Color-focused augmentation (Optuna best)"""
    return T.Compose([
        T.Resize(cfg.img_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
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
# ## 📦 Dataset

#%%
class BiomassDataset(Dataset):
    def __init__(self, df, cfg, transform=None):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.transform = transform
    
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
        
        targets = torch.tensor([
            row['Dry_Green_g'], row['Dry_Clover_g'], row['Dry_Dead_g']
        ], dtype=torch.float32)
        
        return left_img, right_img, targets

#%% [markdown]
# ## 🧠 Model (Optuna Best Architecture)

#%%
class FiLM(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, feat_dim * 2)
        )
    
    def forward(self, context):
        gamma_beta = self.mlp(context)
        return torch.chunk(gamma_beta, 2, dim=1)


def make_head(in_dim: int, hidden_dim: int, num_layers: int, dropout: float, use_layernorm: bool):
    """동적 head 생성 - v16 Optuna와 동일"""
    layers = []
    current_dim = in_dim
    
    for i in range(num_layers):
        out_dim = hidden_dim if i < num_layers - 1 else 1
        layers.append(nn.Linear(current_dim, out_dim if i < num_layers - 1 else hidden_dim))
        
        if i < num_layers - 1:
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim
    
    layers.append(nn.Linear(hidden_dim, 1))
    return nn.Sequential(*layers)


class CSIROModelV17(nn.Module):
    """
    Optuna-optimized architecture (v16과 동일한 make_head 사용):
    - hidden_dim: 512
    - num_layers: 3
    - LayerNorm: True
    - dropout: 0.1
    """
    def __init__(self, cfg):
        super().__init__()
        
        # Backbone
        weights_file = cfg.WEIGHTS_PATH / "dinov3_vitl16_qkvb.pth"
        if weights_file.exists():
            print(f"Loading backbone from: {weights_file}")
            self.backbone = timm.create_model(cfg.model_name, pretrained=False, num_classes=0, global_pool='avg')
            state = torch.load(weights_file, map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(state, strict=False)
            print("✓ Backbone loaded")
        else:
            self.backbone = timm.create_model(cfg.model_name, pretrained=True, num_classes=0, global_pool='avg')
        
        feat_dim = self.backbone.num_features
        combined_dim = feat_dim * 2
        
        self.film = FiLM(feat_dim)
        
        # v16 Optuna와 동일한 make_head 함수 사용
        self.head_green = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout, cfg.use_layernorm)
        self.head_clover = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout, cfg.use_layernorm)
        self.head_dead = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout, cfg.use_layernorm)
        
        self.softplus = nn.Softplus(beta=1.0)
        
        print(f"Model: hidden={cfg.hidden_dim}, layers={cfg.num_layers}, LayerNorm={cfg.use_layernorm}, dropout={cfg.dropout}")
    
    def forward(self, left_img, right_img):
        left_feat = self.backbone(left_img)
        right_feat = self.backbone(right_img)
        
        context = (left_feat + right_feat) / 2
        gamma, beta = self.film(context)
        
        left_mod = left_feat * (1 + gamma) + beta
        right_mod = right_feat * (1 + gamma) + beta
        
        combined = torch.cat([left_mod, right_mod], dim=1)
        
        green = self.softplus(self.head_green(combined))
        clover = self.softplus(self.head_clover(combined))
        dead = self.softplus(self.head_dead(combined))
        
        gdm = green + clover
        total = gdm + dead
        
        return torch.cat([green, dead, clover, gdm, total], dim=1)

#%% [markdown]
# ## 🏋️ Training

#%%
def train_one_epoch(model, loader, optimizer, scheduler, scaler, cfg):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc="Training")
    for left, right, targets in pbar:
        left = left.to(cfg.device)
        right = right.to(cfg.device)
        targets = targets.to(cfg.device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(left, right)
            pred = outputs[:, [0, 2, 1]]  # Green, Clover, Dead
            loss = F.mse_loss(pred, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, cfg):
    model.eval()
    all_preds, all_targets = [], []
    
    for left, right, targets in tqdm(loader, desc="Validating"):
        left = left.to(cfg.device)
        right = right.to(cfg.device)
        
        outputs = model(left, right)
        all_preds.append(outputs.cpu().numpy())
        all_targets.append(targets.numpy())
    
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    
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
    
    train_ds = BiomassDataset(train_data, cfg, get_train_transforms(cfg))
    val_ds = BiomassDataset(val_data, cfg, get_val_transforms(cfg))
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, 
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size*2, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    
    model = CSIROModelV17(cfg).to(cfg.device)
    
    # Optimizer
    backbone_params = list(model.backbone.parameters())
    head_params = (list(model.head_green.parameters()) + 
                   list(model.head_clover.parameters()) + 
                   list(model.head_dead.parameters()) + 
                   list(model.film.parameters()))
    
    optimizer = AdamW([
        {'params': backbone_params, 'lr': cfg.lr * cfg.backbone_lr_mult},
        {'params': head_params, 'lr': cfg.lr}
    ], weight_decay=cfg.weight_decay)
    
    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    scaler = GradScaler()
    
    best_score = -float('inf')
    best_epoch = 0
    no_improve = 0
    
    for epoch in range(cfg.epochs):
        print(f"\nEpoch {epoch+1}/{cfg.epochs}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, cfg)
        val_score = validate(model, val_loader, cfg)
        
        print(f"Loss: {train_loss:.4f} | CV: {val_score:.4f}")
        
        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch + 1
            no_improve = 0
            torch.save(model.state_dict(), cfg.OUTPUT_DIR / f'model_fold{fold}.pth')
            print(f"  ✓ New best! Saved.")
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    print(f"\nFold {fold} Best: {best_score:.4f} (epoch {best_epoch})")
    
    # Backup to Drive
    if GDRIVE_SAVE_PATH:
        src = cfg.OUTPUT_DIR / f'model_fold{fold}.pth'
        if src.exists():
            shutil.copy(src, GDRIVE_SAVE_PATH / f'model_fold{fold}.pth')
            print(f"  📁 Backed up to Drive")
    
    flush()
    return best_score

#%% [markdown]
# ## 🚀 Main

#%%
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🏆 v17 TRAINING START (Optuna-Optimized)")
    print("="*60)
    print(f"Config:")
    print(f"  Head: {cfg.hidden_dim} hidden, 3 layers, LayerNorm")
    print(f"  LR: {cfg.lr:.2e}, backbone_mult: {cfg.backbone_lr_mult:.3f}")
    print(f"  Augmentation: {cfg.aug_strategy}")
    print(f"  Batch: {cfg.batch_size}, Epochs: {cfg.epochs}")
    
    fold_scores = []
    
    for fold in range(cfg.n_folds):
        score = train_fold(fold, train_wide, cfg)
        fold_scores.append(score)
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE")
    print("="*60)
    print(f"Fold scores: {[f'{s:.4f}' for s in fold_scores]}")
    print(f"Mean CV: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    
    # Save results
    results = {
        'fold_scores': fold_scores,
        'mean_cv': float(np.mean(fold_scores)),
        'std_cv': float(np.std(fold_scores)),
        'config': {
            'hidden_dim': cfg.hidden_dim,
            'num_layers': 3,
            'dropout': cfg.dropout,
            'use_layernorm': cfg.use_layernorm,
            'lr': cfg.lr,
            'backbone_lr_mult': cfg.backbone_lr_mult,
            'warmup_ratio': cfg.warmup_ratio,
            'weight_decay': cfg.weight_decay,
            'aug_strategy': cfg.aug_strategy,
            'loss_type': cfg.loss_type,
            'batch_size': cfg.batch_size,
            'epochs': cfg.epochs,
        }
    }
    
    # Save to Drive
    if GDRIVE_SAVE_PATH:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = GDRIVE_SAVE_PATH / f"run_{timestamp}_cv{np.mean(fold_scores):.4f}"
        final_path.mkdir(parents=True, exist_ok=True)
        
        for f in cfg.OUTPUT_DIR.glob("model_fold*.pth"):
            shutil.copy(f, final_path / f.name)
        
        with open(final_path / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Saved to: {final_path}")
