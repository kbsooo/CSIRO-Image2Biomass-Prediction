#%% [markdown]
# # 🚀 CV3: Image Preprocessing + WA Postprocessing
#
# **CV1 기반 + 핵심 변경사항**:
# 1. ⭐ 이미지 전처리: Bottom crop + Orange timestamp 제거
# 2. ⭐ WA Dead=0 후처리 (Infer에서)
# 3. CV1 설정 유지 (검증됨)

#%%
import os
import gc
import json
import random
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
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
# ## ⚙️ Configuration (CV1 기반)

#%%
class CFG:
    # === Model Architecture (CV1과 동일) ===
    hidden_dim = 256
    num_layers = 2
    dropout = 0.3
    
    # === Backbone ===
    freeze_backbone = False  # Fine-tune이 더 좋았음
    
    # === Training ===
    lr = 2e-4
    weight_decay = 1e-4
    warmup_ratio = 0.1
    
    batch_size = 16
    epochs = 30
    patience = 7
    
    # === Augmentation ===
    hue_jitter = 0.02
    
    # === Loss ===
    aux_weight = 0.2
    
    # === Resolution ===
    img_size = (560, 560)
    
    # === CV3 전처리 ===
    use_clean_image = True
    bottom_crop_ratio = 0.90  # Bottom 10% crop

cfg = CFG()

print("="*60)
print("🔧 CV3 Configuration")
print("="*60)
print(f"  use_clean_image: {cfg.use_clean_image}")
print(f"  bottom_crop_ratio: {cfg.bottom_crop_ratio}")
print(f"  hidden_dim: {cfg.hidden_dim}")
print(f"  lr: {cfg.lr}")
print("="*60)

#%% [markdown]
# ## 📊 WandB Setup

#%%
import wandb

wandb.login()

WANDB_ENTITY = "kbsoo0620-"
WANDB_PROJECT = "csiro"

#%% [markdown]
# ## 🔐 Setup

#%%
GDRIVE_SAVE_PATH = None

try:
    from google.colab import drive
    drive.mount('/content/drive')
    GDRIVE_SAVE_PATH = Path('/content/drive/MyDrive/kaggle_models/csiro_biomass_cv3')
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

#%%
if IS_KAGGLE:
    DATA_PATH = Path("/kaggle/input/csiro-biomass")
    WEIGHTS_PATH = Path("/kaggle/input/pretrained-weights-biomass/dinov3_large/dinov3_large")
    OUTPUT_DIR = Path("/kaggle/working")
else:
    csiro_path = kagglehub.competition_download('csiro-biomass')
    weights_path = kagglehub.dataset_download('kbsooo/pretrained-weights-biomass')
    DATA_PATH = Path(csiro_path)
    WEIGHTS_PATH = Path(weights_path) / "dinov3_large" / "dinov3_large"
    OUTPUT_DIR = Path("/content/output")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Data: {DATA_PATH}")

#%% [markdown]
# ## 📊 Data & Metrics

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

# WA State 통계
wa_samples = train_wide[train_wide['State'] == 'WA']
print(f"WA samples: {len(wa_samples)}")
print(f"WA Dead=0 samples: {len(wa_samples[wa_samples['Dry_Dead_g'] == 0])}")

#%%
def create_proper_folds(df, n_splits=5):
    """Sampling_Date 기반 CV split"""
    df = df.copy()
    df['date_group'] = pd.to_datetime(df['Sampling_Date']).dt.strftime('%Y-%m-%d')
    df['strat_key'] = df['State'] + '_' + df['Month'].astype(str)
    
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    df['fold'] = -1
    for fold, (_, val_idx) in enumerate(sgkf.split(
        df, df['strat_key'], groups=df['date_group']
    )):
        df.loc[val_idx, 'fold'] = fold
    
    print("✓ Date-based CV folds created")
    return df

train_wide = create_proper_folds(train_wide)

#%% [markdown]
# ## ⭐ Image Preprocessing Functions

#%%
def clean_image(img_array, bottom_crop_ratio=0.90):
    """
    이미지 전처리: timestamp 제거 + bottom crop
    
    Args:
        img_array: numpy array (H, W, 3)
        bottom_crop_ratio: 유지할 비율 (0.90 = 하단 10% 제거)
    
    Returns:
        cleaned image array
    """
    img = img_array.copy()
    h, w = img.shape[:2]
    
    # 1. Bottom crop (하단 artifacts 제거)
    new_h = int(h * bottom_crop_ratio)
    img = img[0:new_h, :]
    
    # 2. Orange timestamp inpainting
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Orange color range (HSV)
    lower_orange = np.array([5, 150, 150])
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Dilate mask to cover text edges
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    
    # Inpaint if orange pixels found
    if np.sum(mask) > 100:  # 최소 100 픽셀 이상일 때만
        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    
    return img

#%% [markdown]
# ## 🎨 Augmentation & Dataset

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

#%%
class BiomassDatasetCV3(Dataset):
    """CV3 Dataset with image preprocessing"""
    def __init__(self, df, data_path, cfg, transform=None, 
                 height_mean=None, height_std=None,
                 ndvi_mean=None, ndvi_std=None,
                 return_idx=False):
        self.df = df.reset_index(drop=True)
        self.data_path = data_path
        self.cfg = cfg
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
        img_array = np.array(img)
        
        # ⭐ CV3: 이미지 전처리 적용
        if self.cfg.use_clean_image:
            img_array = clean_image(img_array, self.cfg.bottom_crop_ratio)
        
        h, w = img_array.shape[:2]
        mid = w // 2
        
        # Left/Right split
        left_array = img_array[:, :mid, :]
        right_array = img_array[:, mid:, :]
        
        left_img = Image.fromarray(left_array)
        right_img = Image.fromarray(right_array)
        
        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)
        
        # 3개 컴포넌트 타겟
        targets = torch.tensor([
            row['Dry_Green_g'],
            row['Dry_Clover_g'],
            row['Dry_Dead_g']
        ], dtype=torch.float32)
        
        height_norm = (row['Height_Ave_cm'] - self.height_mean) / (self.height_std + 1e-8)
        ndvi_norm = (row['Pre_GSHH_NDVI'] - self.ndvi_mean) / (self.ndvi_std + 1e-8)
        aux_targets = torch.tensor([height_norm, ndvi_norm], dtype=torch.float32)
        
        if self.return_idx:
            return left_img, right_img, targets, aux_targets, idx
        return left_img, right_img, targets, aux_targets
    
    def get_stats(self):
        return {
            'height_mean': self.height_mean,
            'height_std': self.height_std,
            'ndvi_mean': self.ndvi_mean,
            'ndvi_std': self.ndvi_std
        }

#%% [markdown]
# ## 🧠 Model (CV1과 동일)

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
        return torch.chunk(self.mlp(context), 2, dim=1)


def make_head(in_dim, hidden_dim, num_layers, dropout):
    if num_layers == 1:
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    else:
        layers = []
        current_dim = in_dim
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if i < num_layers - 1:
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        return nn.Sequential(*layers)


class CSIROModelCV3(nn.Module):
    """CV3 모델 (CV1과 동일 구조)"""
    def __init__(self, cfg):
        super().__init__()
        
        self.backbone = timm.create_model(
            "vit_large_patch16_dinov3_qkvb.lvd1689m", 
            pretrained=False, num_classes=0, global_pool='avg'
        )
        weights_file = WEIGHTS_PATH / "dinov3_vitl16_qkvb.pth"
        if weights_file.exists():
            state = torch.load(weights_file, map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(state, strict=False)
        
        if cfg.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("✓ Backbone frozen")
        
        feat_dim = self.backbone.num_features
        combined_dim = feat_dim * 2
        
        self.film = FiLM(feat_dim)
        
        self.head_green = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout)
        self.head_clover = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout)
        self.head_dead = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout)
        
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
        
        main_output = torch.cat([green, dead, clover, gdm, total], dim=1)
        
        height_pred = self.head_height(combined)
        ndvi_pred = self.head_ndvi(combined)
        aux_output = torch.cat([height_pred, ndvi_pred], dim=1)
        
        return main_output, aux_output

#%% [markdown]
# ## 🏋️ Training

#%%
def train_fold(fold, train_df, cfg, device="cuda"):
    train_data = train_df[train_df['fold'] != fold].reset_index(drop=True)
    val_data = train_df[train_df['fold'] == fold].reset_index(drop=True)
    
    print(f"\n  Train: {len(train_data)} | Val: {len(val_data)}")
    
    train_ds = BiomassDatasetCV3(train_data, DATA_PATH, cfg, get_train_transforms(cfg))
    stats = train_ds.get_stats()
    val_ds = BiomassDatasetCV3(val_data, DATA_PATH, cfg, get_val_transforms(cfg), 
                               return_idx=True, **stats)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size*2, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    model = CSIROModelCV3(cfg).to(device)
    
    if cfg.freeze_backbone:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        backbone_params = list(model.backbone.parameters())
        head_params = [p for n, p in model.named_parameters() if 'backbone' not in n]
        optimizer = AdamW([
            {'params': backbone_params, 'lr': cfg.lr * 0.1},
            {'params': head_params, 'lr': cfg.lr}
        ], weight_decay=cfg.weight_decay)
    
    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    scaler = GradScaler()
    
    best_score = -float('inf')
    no_improve = 0
    best_oof = None
    
    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0
        
        for left, right, targets, aux_targets in train_loader:
            left = left.to(device)
            right = right.to(device)
            targets = targets.to(device)
            aux_targets = aux_targets.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                main_output, aux_output = model(left, right)
                pred = main_output[:, [0, 2, 1]]  # Green, Clover, Dead
                main_loss = F.mse_loss(pred, targets)
                aux_loss = F.mse_loss(aux_output, aux_targets)
                loss = main_loss + cfg.aux_weight * aux_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        all_preds, all_targets, all_indices = [], [], []
        
        with torch.no_grad():
            for left, right, targets, _, indices in val_loader:
                left, right = left.to(device), right.to(device)
                main_output, _ = model(left, right)
                all_preds.append(main_output.cpu().numpy())
                all_targets.append(targets.numpy())
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
                'val_score': val_score
            }
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    np.save(OUTPUT_DIR / f'oof_fold{fold}.npy', best_oof)
    wandb.log({f"fold{fold}/best_score": best_score})
    
    flush()
    return best_score, best_oof

#%% [markdown]
# ## 🚀 Run Training

#%%
run = wandb.init(
    entity=WANDB_ENTITY,
    project=WANDB_PROJECT,
    name="cv3_clean_image",
    config={
        "version": "cv3",
        "use_clean_image": cfg.use_clean_image,
        "bottom_crop_ratio": cfg.bottom_crop_ratio,
        "hidden_dim": cfg.hidden_dim,
        "lr": cfg.lr,
    }
)

#%%
print("\n" + "="*60)
print("🚀 CV3 Training: Image Preprocessing + CV1 Base")
print("="*60)

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
print("🎉 CV3 RESULTS")
print("="*60)
print(f"Folds: {[f'{s:.4f}' for s in fold_scores]}")
print(f"Mean CV: {mean_cv:.4f} ± {std_cv:.4f}")

#%% [markdown]
# ## 📊 OOF Score

#%%
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

#%% [markdown]
# ## 💾 Save to Google Drive

#%%
if GDRIVE_SAVE_PATH:
    save_dir = GDRIVE_SAVE_PATH
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for f in OUTPUT_DIR.glob("model_fold*.pth"):
        shutil.copy(f, save_dir / f.name)
    
    for f in OUTPUT_DIR.glob("oof_fold*.npy"):
        shutil.copy(f, save_dir / f.name)
    
    with open(save_dir / 'results.json', 'w') as f:
        json.dump({
            'version': 'cv3',
            'use_clean_image': cfg.use_clean_image,
            'fold_scores': [float(s) for s in fold_scores],
            'mean_cv': float(mean_cv),
            'std_cv': float(std_cv),
            'total_oof_score': float(total_oof_score),
        }, f, indent=2)
    
    zip_path = save_dir / 'models_cv3.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fold in range(5):
            model_path = save_dir / f'model_fold{fold}.pth'
            if model_path.exists():
                zf.write(model_path, f'model_fold{fold}.pth')
    
    print(f"\n✓ All saved to: {save_dir}")
    print(f"✓ ZIP file: {zip_path}")

wandb.log({
    "final/mean_cv": mean_cv,
    "final/oof_score": total_oof_score,
})

wandb.finish()
print(f"\n🎉 CV3 Training complete!")
