#%% [markdown]
# # 🚀 v25: Vegetation Index Late Fusion
#
# **핵심 아이디어**:
# - RGB → DINOv3 (기존)
# - Vegetation Index (ExG, GR_ratio) → 별도 CNN
# - Late Fusion으로 결합
#
# **기대 효과**: Location-invariant 특성 학습으로 일반화 향상

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
    GDRIVE_SAVE_PATH = Path('/content/drive/MyDrive/kaggle_models/csiro_biomass_v25')
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
    # Data
    img_size = (512, 512)
    
    # Model (v20 베이스)
    hidden_dim = 512
    num_layers = 3
    dropout = 0.1
    use_layernorm = True
    
    # Training
    lr = 2.33e-4
    backbone_lr_mult = 0.084
    warmup_ratio = 0.078
    weight_decay = 6.37e-5
    
    batch_size = 8
    epochs = 25
    patience = 7
    
    hue_jitter = 0.02
    aux_weight = 0.2
    
    # === NEW: Vegetation Index 설정 ===
    veg_feat_dim = 128  # Veg encoder 출력 차원
    
cfg = CFG()

#%%
# Data paths
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
# ## 🎯 Multi-Stratified Fold Split

#%%
def create_stratified_folds(df, n_splits=5):
    df = df.copy()
    df['strat_key'] = df['State'] + '_' + df['Month'].astype(str)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    df['fold'] = -1
    for fold, (_, val_idx) in enumerate(sgkf.split(df, df['strat_key'], groups=df['image_id'])):
        df.loc[val_idx, 'fold'] = fold
    
    return df

train_wide = create_stratified_folds(train_wide)

#%% [markdown]
# ## 🌱 Vegetation Index 계산

#%%
def compute_vegetation_indices(img_array):
    """
    RGB 이미지에서 Vegetation Index 계산
    
    Args:
        img_array: [H, W, 3] numpy array, 0-255 범위
    
    Returns:
        veg_indices: [H, W, 2] numpy array (ExG, GR_ratio)
    """
    # 0-1 범위로 정규화
    img = img_array.astype(np.float32) / 255.0
    
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    
    # Excess Green Index: 녹색이 강할수록 높음
    exg = 2*g - r - b  # 범위: -2 ~ 2
    exg = (exg + 2) / 4  # 0~1로 정규화
    
    # Green-Red Ratio: 녹색/빨강 비율
    gr_ratio = g / (r + 1e-8)
    gr_ratio = np.clip(gr_ratio, 0, 3) / 3  # 0~1로 클리핑 및 정규화
    
    return np.stack([exg, gr_ratio], axis=-1)

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

# Vegetation Index용 간단한 transform
def get_veg_transforms(cfg, is_train=True):
    if is_train:
        return T.Compose([
            T.Resize(cfg.img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ])
    else:
        return T.Compose([
            T.Resize(cfg.img_size),
        ])

#%% [markdown]
# ## 📦 Dataset with Vegetation Index

#%%
class BiomassDatasetV25(Dataset):
    """
    v25 Dataset: RGB + Vegetation Index 동시 반환
    """
    def __init__(self, df, data_path, rgb_transform=None, is_train=True,
                 height_mean=None, height_std=None, ndvi_mean=None, ndvi_std=None):
        self.df = df.reset_index(drop=True)
        self.data_path = data_path
        self.rgb_transform = rgb_transform
        self.is_train = is_train
        self.img_size = (512, 512)
        
        self.height_mean = height_mean if height_mean else df['Height_Ave_cm'].mean()
        self.height_std = height_std if height_std else df['Height_Ave_cm'].std()
        self.ndvi_mean = ndvi_mean if ndvi_mean else df['Pre_GSHH_NDVI'].mean()
        self.ndvi_std = ndvi_std if ndvi_std else df['Pre_GSHH_NDVI'].std()
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 이미지 로드
        img = Image.open(self.data_path / row['image_path']).convert('RGB')
        width, height = img.size
        mid = width // 2
        
        # Left/Right 분할 (PIL)
        left_pil = img.crop((0, 0, mid, height))
        right_pil = img.crop((mid, 0, width, height))
        
        # RGB Transform 적용 전에 numpy로 변환 (Veg Index 계산용)
        left_np = np.array(left_pil.resize(self.img_size))
        right_np = np.array(right_pil.resize(self.img_size))
        
        # Vegetation Index 계산
        left_veg = compute_vegetation_indices(left_np)  # [H, W, 2]
        right_veg = compute_vegetation_indices(right_np)
        
        # Augmentation (flip) - RGB와 동기화 필요
        if self.is_train:
            # 랜덤 플립 결정
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            
            if hflip:
                left_pil = T.functional.hflip(left_pil)
                right_pil = T.functional.hflip(right_pil)
                left_veg = np.flip(left_veg, axis=1).copy()
                right_veg = np.flip(right_veg, axis=1).copy()
            
            if vflip:
                left_pil = T.functional.vflip(left_pil)
                right_pil = T.functional.vflip(right_pil)
                left_veg = np.flip(left_veg, axis=0).copy()
                right_veg = np.flip(right_veg, axis=0).copy()
        
        # RGB Transform (resize + color jitter + normalize)
        rgb_base_transform = T.Compose([
            T.Resize(self.img_size),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.02) if self.is_train else T.Lambda(lambda x: x),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        left_rgb = rgb_base_transform(left_pil)
        right_rgb = rgb_base_transform(right_pil)
        
        # Veg Index to Tensor
        left_veg = torch.from_numpy(left_veg).permute(2, 0, 1).float()  # [2, H, W]
        right_veg = torch.from_numpy(right_veg).permute(2, 0, 1).float()
        
        # 메인 타겟
        main_targets = torch.tensor([
            row['Dry_Green_g'], row['Dry_Clover_g'], row['Dry_Dead_g']
        ], dtype=torch.float32)
        
        # Auxiliary 타겟
        height_norm = (row['Height_Ave_cm'] - self.height_mean) / (self.height_std + 1e-8)
        ndvi_norm = (row['Pre_GSHH_NDVI'] - self.ndvi_mean) / (self.ndvi_std + 1e-8)
        aux_targets = torch.tensor([height_norm, ndvi_norm], dtype=torch.float32)
        
        return left_rgb, right_rgb, left_veg, right_veg, main_targets, aux_targets
    
    def get_stats(self):
        return {
            'height_mean': self.height_mean,
            'height_std': self.height_std,
            'ndvi_mean': self.ndvi_mean,
            'ndvi_std': self.ndvi_std
        }

#%% [markdown]
# ## 🧠 Model with Vegetation Index Encoder

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


class VegetationEncoder(nn.Module):
    """Vegetation Index 인코더 (가벼운 CNN)"""
    def __init__(self, in_channels=2, out_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.encoder(x)


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


class CSIROModelV25(nn.Module):
    """
    v25 모델: RGB (DINOv3) + Vegetation Index (CNN) Late Fusion
    """
    def __init__(self, cfg):
        super().__init__()
        
        # RGB 브랜치 (DINOv3)
        self.backbone = timm.create_model(
            "vit_large_patch16_dinov3_qkvb.lvd1689m", 
            pretrained=False, num_classes=0, global_pool='avg'
        )
        weights_file = WEIGHTS_PATH / "dinov3_vitl16_qkvb.pth"
        if weights_file.exists():
            state = torch.load(weights_file, map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(state, strict=False)
        
        feat_dim = self.backbone.num_features  # 1024
        
        # Vegetation Index 브랜치
        self.veg_encoder = VegetationEncoder(in_channels=2, out_dim=cfg.veg_feat_dim)
        
        self.film = FiLM(feat_dim)
        
        # 결합 차원: RGB(1024*2) + Veg(128*2) = 2048 + 256 = 2304
        combined_dim = feat_dim * 2 + cfg.veg_feat_dim * 2
        
        # 메인 Heads
        self.head_green = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers, 
                                    cfg.dropout, cfg.use_layernorm)
        self.head_clover = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers,
                                     cfg.dropout, cfg.use_layernorm)
        self.head_dead = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers,
                                   cfg.dropout, cfg.use_layernorm)
        
        # Auxiliary Heads
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
    
    def forward(self, left_rgb, right_rgb, left_veg, right_veg):
        # RGB 특징 추출
        left_feat = self.backbone(left_rgb)
        right_feat = self.backbone(right_rgb)
        
        # Vegetation Index 특징 추출
        left_veg_feat = self.veg_encoder(left_veg)
        right_veg_feat = self.veg_encoder(right_veg)
        
        # FiLM (RGB 기반)
        context = (left_feat + right_feat) / 2
        gamma, beta = self.film(context)
        
        left_mod = left_feat * (1 + gamma) + beta
        right_mod = right_feat * (1 + gamma) + beta
        
        # Late Fusion: RGB + Veg 결합
        combined = torch.cat([left_mod, right_mod, left_veg_feat, right_veg_feat], dim=1)
        
        # 예측
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
    
    # Dataset
    train_ds = BiomassDatasetV25(train_data, DATA_PATH, is_train=True)
    stats = train_ds.get_stats()
    val_ds = BiomassDatasetV25(val_data, DATA_PATH, is_train=False, **stats)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size*2, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    # Model
    model = CSIROModelV25(cfg).to(device)
    
    # Optimizer
    backbone_params = list(model.backbone.parameters())
    other_params = (list(model.head_green.parameters()) + 
                   list(model.head_clover.parameters()) +
                   list(model.head_dead.parameters()) + 
                   list(model.head_height.parameters()) +
                   list(model.head_ndvi.parameters()) +
                   list(model.film.parameters()) +
                   list(model.veg_encoder.parameters()))
    
    optimizer = AdamW([
        {'params': backbone_params, 'lr': cfg.lr * cfg.backbone_lr_mult},
        {'params': other_params, 'lr': cfg.lr}
    ], weight_decay=cfg.weight_decay)
    
    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    scaler = GradScaler()
    
    best_score = -float('inf')
    no_improve = 0
    
    for epoch in range(cfg.epochs):
        # Train
        model.train()
        train_loss = 0
        
        for left_rgb, right_rgb, left_veg, right_veg, main_targets, aux_targets in train_loader:
            left_rgb = left_rgb.to(device)
            right_rgb = right_rgb.to(device)
            left_veg = left_veg.to(device)
            right_veg = right_veg.to(device)
            main_targets = main_targets.to(device)
            aux_targets = aux_targets.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                main_output, aux_output = model(left_rgb, right_rgb, left_veg, right_veg)
                
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
        
        # Validate
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for left_rgb, right_rgb, left_veg, right_veg, main_targets, _ in val_loader:
                left_rgb = left_rgb.to(device)
                right_rgb = right_rgb.to(device)
                left_veg = left_veg.to(device)
                right_veg = right_veg.to(device)
                
                main_output, _ = model(left_rgb, right_rgb, left_veg, right_veg)
                all_preds.append(main_output.cpu().numpy())
                all_targets.append(main_targets.numpy())
        
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        
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
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    wandb.log({f"fold{fold}/best_score": best_score})
    
    flush()
    return best_score

#%% [markdown]
# ## 🚀 Run Training

#%%
run = wandb.init(
    entity=WANDB_ENTITY,
    project=WANDB_PROJECT,
    name=f"v25_vegidx_fusion",
    config={
        "version": "v25",
        "veg_feat_dim": cfg.veg_feat_dim,
        "hidden_dim": cfg.hidden_dim,
        "num_layers": cfg.num_layers,
        "lr": cfg.lr,
    }
)

#%%
print("\n" + "="*60)
print("🚀 v25 Training: Vegetation Index Late Fusion")
print("="*60)
print(f"Veg feature dim: {cfg.veg_feat_dim}")
print(f"Aux weight: {cfg.aux_weight}")

fold_scores = []

for fold in range(5):
    print(f"\n--- Fold {fold} ---")
    score = train_fold(fold, train_wide, cfg)
    fold_scores.append(score)
    
    wandb.log({
        "current_fold": fold,
        "running_mean": np.mean(fold_scores),
    })

#%%
mean_cv = np.mean(fold_scores)
std_cv = np.std(fold_scores)

wandb.log({
    "final/mean_cv": mean_cv,
    "final/std_cv": std_cv,
})

print("\n" + "="*60)
print("🎉 v25 RESULTS")
print("="*60)
print(f"Folds: {[f'{s:.4f}' for s in fold_scores]}")
print(f"Mean CV: {mean_cv:.4f} ± {std_cv:.4f}")

#%%
if GDRIVE_SAVE_PATH:
    for f in OUTPUT_DIR.glob("model_fold*.pth"):
        shutil.copy(f, GDRIVE_SAVE_PATH / f.name)
    
    with open(GDRIVE_SAVE_PATH / 'results.json', 'w') as f:
        json.dump({
            'fold_scores': fold_scores,
            'mean_cv': float(mean_cv),
            'std_cv': float(std_cv),
        }, f, indent=2)
    print(f"\n✓ Saved to: {GDRIVE_SAVE_PATH}")

wandb.finish()
