#%% [markdown]
# # 🚀 CV5: ConvNeXt-Base Multi-Backbone Ensemble
#
# **Phase A 전략**: DINOv3 (Transformer) + ConvNeXt (CNN) 혼합 앙상블
#
# **ConvNeXt-Base 특징**:
# - CNN 기반 (로컬 피처 강점)
# - ImageNet-22k pretrained
# - Feature dim: 1024
# - DINOv3와 다른 특성 → 앙상블 다양성 ↑
#
# **Expected**: CV ~0.60-0.68, 앙상블 시 LB +0.03

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

#%% [markdown]
# ## 📊 WandB Setup

#%%
import wandb

wandb.login()

WANDB_ENTITY = "kbsoo0620-"
WANDB_PROJECT = "csiro"

print(f"✓ WandB: {WANDB_ENTITY}/{WANDB_PROJECT}")

#%% [markdown]
# ## 🔐 Setup

#%%
GDRIVE_SAVE_PATH = None

try:
    from google.colab import drive
    drive.mount('/content/drive')
    GDRIVE_SAVE_PATH = Path('/content/drive/MyDrive/kaggle_models/csiro_biomass_cv5')
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
# ## ⚙️ Configuration

#%%
class CFG:
    # Image
    img_size = (560, 560)  # 16의 배수
    
    # Model - ConvNeXt-Base
    backbone_name = "convnext_base.fb_in22k_ft_in1k"  # ImageNet-22k pretrained
    hidden_dim = 512
    num_layers = 3
    dropout = 0.1
    use_layernorm = True
    
    # Training
    lr = 5e-5  # ConvNeXt는 backbone도 학습하므로 낮은 lr
    weight_decay = 1e-4
    warmup_ratio = 0.1
    
    batch_size = 8
    epochs = 25
    patience = 7
    
    # Augmentation
    hue_jitter = 0.02
    aux_weight = 0.2
    
cfg = CFG()

print("=== CV5 Configuration: ConvNeXt-Base ===")
print(f"Backbone: {cfg.backbone_name}")
print(f"Image size: {cfg.img_size}")
print(f"LR: {cfg.lr}")

#%%
if IS_KAGGLE:
    DATA_PATH = Path("/kaggle/input/csiro-biomass")
    OUTPUT_DIR = Path("/kaggle/working")
else:
    csiro_path = kagglehub.competition_download('csiro-biomass')
    DATA_PATH = Path(csiro_path)
    OUTPUT_DIR = Path("/content/output")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Data: {DATA_PATH}")

#%% [markdown]
# ## 📊 Data Loading

#%%
TARGET_WEIGHTS = {'Dry_Green_g': 0.1, 'Dry_Dead_g': 0.1, 'Dry_Clover_g': 0.1, 'GDM_g': 0.2, 'Dry_Total_g': 0.5}
TARGET_ORDER = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

def competition_metric(y_true, y_pred):
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

train_df = pd.read_csv(DATA_PATH / "train.csv")
train_wide = prepare_data(train_df)
train_wide['image_id'] = train_wide['image_path'].apply(lambda x: Path(x).stem)
train_wide['Month'] = pd.to_datetime(train_wide['Sampling_Date']).dt.month

print(f"Train samples: {len(train_wide)}")

#%% [markdown]
# ## 🎯 Sampling_Date 기반 CV Split

#%%
def create_proper_folds(df, n_splits=5):
    """Sampling_Date 기반 GroupKFold (data leakage 방지)"""
    df = df.copy()
    df['date_group'] = pd.to_datetime(df['Sampling_Date']).dt.strftime('%Y-%m-%d')
    df['strat_key'] = df['State'] + '_' + df['Month'].astype(str)
    
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    df['fold'] = -1
    for fold, (_, val_idx) in enumerate(sgkf.split(
        df, 
        df['strat_key'], 
        groups=df['date_group']
    )):
        df.loc[val_idx, 'fold'] = fold
    
    print("=== Fold Distribution ===")
    for fold in range(n_splits):
        fold_data = df[df['fold'] == fold]
        n_samples = len(fold_data)
        n_dates = fold_data['date_group'].nunique()
        print(f"  Fold {fold}: {n_samples} samples, {n_dates} unique dates")
    
    return df

train_wide = create_proper_folds(train_wide)

#%% [markdown]
# ## 🎨 Augmentation

#%%
def get_train_transforms(cfg):
    return T.Compose([
        T.Resize(cfg.img_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=cfg.hue_jitter),
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
    def __init__(self, df, data_path, transform=None, 
                 height_mean=None, height_std=None,
                 ndvi_mean=None, ndvi_std=None,
                 return_idx=False):
        self.df = df.reset_index(drop=True)
        self.data_path = data_path
        self.transform = transform
        self.return_idx = return_idx
        
        self.height_mean = height_mean if height_mean else df['Height_Ave_cm'].mean()
        self.height_std = height_std if height_std else df['Height_Ave_cm'].std()
        self.ndvi_mean = ndvi_mean if ndvi_mean else df['Pre_GSHH_NDVI'].mean()
        self.ndvi_std = ndvi_std if ndvi_std else df['Pre_GSHH_NDVI'].std()
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.data_path / row['image_path']).convert('RGB')
        width, height = img.size
        mid = width // 2
        
        left_img = img.crop((0, 0, mid, height))
        right_img = img.crop((mid, 0, width, height))
        
        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)
        
        main_targets = torch.tensor([
            row['Dry_Green_g'], row['Dry_Clover_g'], row['Dry_Dead_g']
        ], dtype=torch.float32)
        
        height_norm = (row['Height_Ave_cm'] - self.height_mean) / (self.height_std + 1e-8)
        ndvi_norm = (row['Pre_GSHH_NDVI'] - self.ndvi_mean) / (self.ndvi_std + 1e-8)
        aux_targets = torch.tensor([height_norm, ndvi_norm], dtype=torch.float32)
        
        if self.return_idx:
            return left_img, right_img, main_targets, aux_targets, idx
        return left_img, right_img, main_targets, aux_targets
    
    def get_stats(self):
        return {
            'height_mean': self.height_mean,
            'height_std': self.height_std,
            'ndvi_mean': self.ndvi_mean,
            'ndvi_std': self.ndvi_std
        }

#%% [markdown]
# ## 🧠 Model

#%%
def make_head(in_dim, hidden_dim, num_layers, dropout, use_layernorm):
    layers = []
    current_dim = in_dim
    
    for i in range(num_layers):
        layers.append(nn.Linear(current_dim, hidden_dim))
        if i < num_layers - 1:
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim
    
    layers.append(nn.Linear(hidden_dim, 1))
    return nn.Sequential(*layers)


class CSIROModelCV5(nn.Module):
    """
    CV5 모델: ConvNeXt-Base Backbone
    
    ConvNeXt는 CNN 기반이므로 FiLM fusion 제거
    (이미 로컬 context를 잘 캡처하므로)
    """
    def __init__(self, cfg):
        super().__init__()
        
        # ConvNeXt-Base backbone (ImageNet-22k pretrained)
        self.backbone = timm.create_model(
            cfg.backbone_name,
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        
        feat_dim = self.backbone.num_features  # 1024 for ConvNeXt-Base
        combined_dim = feat_dim * 2  # Left + Right concatenation
        
        print(f"✓ Backbone: {cfg.backbone_name}")
        print(f"  Feature dim: {feat_dim}")
        print(f"  Combined dim: {combined_dim}")
        
        # Main heads
        self.head_green = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers, 
                                    cfg.dropout, cfg.use_layernorm)
        self.head_clover = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers,
                                     cfg.dropout, cfg.use_layernorm)
        self.head_dead = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers,
                                   cfg.dropout, cfg.use_layernorm)
        
        # Auxiliary heads
        self.head_height = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
        self.head_ndvi = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
        self.softplus = nn.Softplus(beta=1.0)
    
    def forward(self, left_img, right_img):
        # Extract features
        left_feat = self.backbone(left_img)   # [B, 1024]
        right_feat = self.backbone(right_img)  # [B, 1024]
        
        # Simple concatenation (no FiLM for CNN)
        combined = torch.cat([left_feat, right_feat], dim=1)  # [B, 2048]
        
        # Main predictions
        green = self.softplus(self.head_green(combined))
        clover = self.softplus(self.head_clover(combined))
        dead = self.softplus(self.head_dead(combined))
        
        # Physics constraints
        gdm = green + clover
        total = gdm + dead
        
        main_output = torch.cat([green, dead, clover, gdm, total], dim=1)
        
        # Auxiliary predictions
        height_pred = self.head_height(combined)
        ndvi_pred = self.head_ndvi(combined)
        aux_output = torch.cat([height_pred, ndvi_pred], dim=1)
        
        return main_output, aux_output

#%% [markdown]
# ## 🏋️ Training with OOF Collection

#%%
def train_fold(fold, train_df, cfg, device="cuda"):
    """학습 + OOF 예측 저장"""
    train_data = train_df[train_df['fold'] != fold].reset_index(drop=True)
    val_data = train_df[train_df['fold'] == fold].reset_index(drop=True)
    
    print(f"\n  Train: {len(train_data)} | Val: {len(val_data)}")
    print(f"  Val dates: {val_data['date_group'].nunique()} unique")
    
    # Dataset
    train_ds = BiomassDataset(train_data, DATA_PATH, get_train_transforms(cfg))
    stats = train_ds.get_stats()
    val_ds = BiomassDataset(val_data, DATA_PATH, get_val_transforms(cfg), 
                            return_idx=True, **stats)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size*2, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    # Model
    model = CSIROModelCV5(cfg).to(device)
    
    # Optimizer - ConvNeXt는 전체 backbone 학습
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    scaler = GradScaler()
    
    best_score = -float('inf')
    no_improve = 0
    best_oof = None
    
    for epoch in range(cfg.epochs):
        # Train
        model.train()
        train_loss = 0
        
        for left, right, main_targets, aux_targets in train_loader:
            left = left.to(device)
            right = right.to(device)
            main_targets = main_targets.to(device)
            aux_targets = aux_targets.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                main_output, aux_output = model(left, right)
                pred = main_output[:, [0, 2, 1]]  # Green, Clover, Dead
                main_loss = F.mse_loss(pred, main_targets)
                aux_loss = F.mse_loss(aux_output, aux_targets)
                loss = main_loss + cfg.aux_weight * aux_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate & Collect OOF
        model.eval()
        all_preds, all_targets, all_indices = [], [], []
        
        with torch.no_grad():
            for left, right, main_targets, _, indices in val_loader:
                left, right = left.to(device), right.to(device)
                main_output, _ = model(left, right)
                all_preds.append(main_output.cpu().numpy())
                all_targets.append(main_targets.numpy())
                all_indices.extend(indices.numpy().tolist())
        
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        
        # 5개 타겟으로 확장
        full_targets = np.zeros((len(targets), 5))
        full_targets[:, 0] = targets[:, 0]  # Green
        full_targets[:, 1] = targets[:, 2]  # Dead
        full_targets[:, 2] = targets[:, 1]  # Clover
        full_targets[:, 3] = targets[:, 0] + targets[:, 1]  # GDM
        full_targets[:, 4] = full_targets[:, 3] + targets[:, 2]  # Total
        
        val_score = competition_metric(full_targets, preds)
        
        wandb.log({
            f"fold{fold}/train_loss": train_loss,
            f"fold{fold}/val_score": val_score,
            f"fold{fold}/epoch": epoch + 1,
        })
        
        print(f"  Epoch {epoch+1}: loss={train_loss:.4f}, CV={val_score:.4f}")
        
        if val_score > best_score:
            best_score = val_score
            no_improve = 0
            torch.save(model.state_dict(), OUTPUT_DIR / f'model_fold{fold}.pth')
            
            best_oof = {
                'predictions': preds.copy(),
                'targets': full_targets.copy(),
                'indices': np.array(all_indices),
                'fold': fold,
                'val_score': val_score
            }
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # OOF 저장
    np.save(OUTPUT_DIR / f'oof_fold{fold}.npy', best_oof)
    print(f"  ✓ OOF saved: {len(best_oof['predictions'])} samples, score={best_score:.4f}")
    
    wandb.log({f"fold{fold}/best_score": best_score})
    
    flush()
    return best_score, best_oof

#%% [markdown]
# ## 🚀 Run Training

#%%
run = wandb.init(
    entity=WANDB_ENTITY,
    project=WANDB_PROJECT,
    name=f"cv5_convnext_base",
    config={
        "version": "cv5",
        "backbone": cfg.backbone_name,
        "strategy": "Multi-Backbone Ensemble (Phase A)",
        "img_size": cfg.img_size,
        "hidden_dim": cfg.hidden_dim,
        "num_layers": cfg.num_layers,
        "lr": cfg.lr,
    }
)

#%%
print("\n" + "="*60)
print("🚀 CV5 Training: ConvNeXt-Base Multi-Backbone Ensemble")
print("="*60)
print(f"Backbone: {cfg.backbone_name}")
print(f"Image size: {cfg.img_size}")
print(f"Strategy: Phase A - CNN + Transformer 앙상블 다양성")

fold_scores = []
all_oof = []

for fold in range(5):
    print(f"\n--- Fold {fold} ---")
    score, oof = train_fold(fold, train_wide, cfg)
    fold_scores.append(score)
    all_oof.append(oof)

#%%
mean_cv = np.mean(fold_scores)
std_cv = np.std(fold_scores)

print("\n" + "="*60)
print("🎉 CV5 RESULTS: ConvNeXt-Base")
print("="*60)
print(f"Folds: {[f'{s:.4f}' for s in fold_scores]}")
print(f"Mean CV: {mean_cv:.4f} ± {std_cv:.4f}")

#%% [markdown]
# ## 📊 OOF Score Verification

#%%
# 전체 OOF score 계산
all_predictions = []
all_targets = []

for fold in range(5):
    oof = np.load(OUTPUT_DIR / f'oof_fold{fold}.npy', allow_pickle=True).item()
    all_predictions.append(oof['predictions'])
    all_targets.append(oof['targets'])

oof_predictions = np.concatenate(all_predictions)
oof_targets = np.concatenate(all_targets)

total_oof_score = competition_metric(oof_targets, oof_predictions)
print(f"\n✓ Total OOF Score: {total_oof_score:.4f}")
print(f"  (이 OOF를 Stacking에 활용!)")

#%%
# Google Drive에 저장
if GDRIVE_SAVE_PATH:
    for f in OUTPUT_DIR.glob("model_fold*.pth"):
        shutil.copy(f, GDRIVE_SAVE_PATH / f.name)
    for f in OUTPUT_DIR.glob("oof_fold*.npy"):
        shutil.copy(f, GDRIVE_SAVE_PATH / f.name)
    
    with open(GDRIVE_SAVE_PATH / 'results.json', 'w') as f:
        json.dump({
            'backbone': cfg.backbone_name,
            'fold_scores': fold_scores,
            'mean_cv': float(mean_cv),
            'std_cv': float(std_cv),
            'total_oof_score': float(total_oof_score),
            'strategy': 'Phase A - Multi-Backbone Ensemble',
        }, f, indent=2)
    print(f"\n✓ All saved to: {GDRIVE_SAVE_PATH}")

wandb.log({
    "final/mean_cv": mean_cv,
    "final/std_cv": std_cv,
    "final/oof_score": total_oof_score,
})

wandb.finish()

print("\n" + "="*60)
print("✅ CV5 Complete!")
print(f"   Backbone: ConvNeXt-Base")
print(f"   Mean CV: {mean_cv:.4f}")
print(f"   Next: cv5_infer.py로 제출 또는 DINOv3와 앙상블")
print("="*60)
