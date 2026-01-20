#%% [markdown]
# # 🚀 CV2: Weighted Loss + Frozen Backbone
#
# **핵심 변경사항**:
# 1. ✅ Date-based CV (Sampling_Date 그룹핑)
# 2. ✅ 해상도 560x560
# 3. ⭐ **Weighted Loss**: 대회 metric에 맞춘 가중치 (Total 50%!)
# 4. ⭐ **Frozen Backbone**: DINOv2 고정, Head만 학습
# 5. Grid Search 용이한 구조

#%%
import os
import gc
import json
import random
import shutil
import zipfile
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
# ## ⚙️ Grid Search Configuration
#
# **여기서 파라미터 변경하세요!**

#%%
# ========================================
# ⚠️ GRID SEARCH: 여기서 파라미터 변경!
# ========================================

class CFG:
    # === Model Architecture ===
    hidden_dim = 256       # 시도: 64, 128, 256, 512
    num_layers = 2         # 시도: 1, 2, 3
    dropout = 0.3          # 시도: 0.1, 0.2, 0.3, 0.4, 0.5
    
    # === Backbone ===
    freeze_backbone = True  # ⭐ Backbone 동결
    
    # === Training ===
    lr = 1e-3              # Frozen backbone이면 높은 lr 가능
    weight_decay = 1e-3
    warmup_ratio = 0.1
    
    batch_size = 16        # Frozen이면 더 큰 batch 가능
    epochs = 30
    patience = 7
    
    # === Augmentation ===
    hue_jitter = 0.02
    
    # === Loss ===
    use_weighted_loss = True  # ⭐ Weighted Loss 사용
    aux_weight = 0.2
    
    # === Resolution ===
    img_size = (560, 560)

cfg = CFG()

# 설정 출력
print("="*60)
print("🔧 CV2 Configuration")
print("="*60)
print(f"  hidden_dim: {cfg.hidden_dim}")
print(f"  num_layers: {cfg.num_layers}")
print(f"  dropout: {cfg.dropout}")
print(f"  freeze_backbone: {cfg.freeze_backbone}")
print(f"  use_weighted_loss: {cfg.use_weighted_loss}")
print(f"  lr: {cfg.lr}")
print(f"  batch_size: {cfg.batch_size}")
print("="*60)

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
    GDRIVE_SAVE_PATH = Path('/content/drive/MyDrive/kaggle_models/csiro_biomass_cv2')
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
    
    # 검증
    date_fold_counts = df.groupby('date_group')['fold'].nunique()
    if (date_fold_counts > 1).any():
        print("⚠️ WARNING: Some dates are in multiple folds!")
    else:
        print("✓ CV split verified: dates are properly grouped")
    
    return df

train_wide = create_proper_folds(train_wide)

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
        
        # 5개 타겟 모두 반환 (Weighted Loss용)
        targets = torch.tensor([
            row['Dry_Green_g'],
            row['Dry_Dead_g'],
            row['Dry_Clover_g'],
            row['Dry_Green_g'] + row['Dry_Clover_g'],  # GDM
            row['Dry_Green_g'] + row['Dry_Clover_g'] + row['Dry_Dead_g'],  # Total
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
# ## ⚡ Weighted Loss

#%%
class WeightedMSELoss(nn.Module):
    """
    대회 평가 지표에 맞춘 Weighted MSE Loss
    
    가중치:
    - Dry_Green_g: 0.1
    - Dry_Dead_g: 0.1
    - Dry_Clover_g: 0.1
    - GDM_g: 0.2
    - Dry_Total_g: 0.5  ← 가장 중요!
    """
    def __init__(self):
        super().__init__()
        # [Green, Dead, Clover, GDM, Total]
        self.register_buffer('weights', torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5]))
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, 5] - [Green, Dead, Clover, GDM, Total]
            target: [B, 5] - [Green, Dead, Clover, GDM, Total]
        """
        mse = (pred - target) ** 2  # [B, 5]
        weighted_mse = mse * self.weights  # 가중치 적용
        return weighted_mse.mean()

#%% [markdown]
# ## 🧠 Model with Frozen Backbone

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
    """유연한 Head 생성"""
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


class CSIROModelCV2(nn.Module):
    """CV2 모델: Frozen Backbone + Weighted Loss"""
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
        
        # ⭐ Backbone 동결
        if cfg.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("✓ Backbone frozen")
        
        feat_dim = self.backbone.num_features  # 1024
        combined_dim = feat_dim * 2  # 2048
        
        self.film = FiLM(feat_dim)
        
        self.head_green = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout)
        self.head_clover = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout)
        self.head_dead = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout)
        
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
        # Backbone (frozen이면 no_grad)
        if not any(p.requires_grad for p in self.backbone.parameters()):
            with torch.no_grad():
                left_feat = self.backbone(left_img)
                right_feat = self.backbone(right_img)
        else:
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
        
        # [Green, Dead, Clover, GDM, Total] 순서
        main_output = torch.cat([green, dead, clover, gdm, total], dim=1)
        
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
    model = CSIROModelCV2(cfg).to(device)
    
    # Optimizer (Frozen backbone이면 head만)
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
    
    # Loss
    if cfg.use_weighted_loss:
        main_criterion = WeightedMSELoss().to(device)
        print("  ✓ Using Weighted MSE Loss")
    else:
        main_criterion = nn.MSELoss()
    
    best_score = -float('inf')
    no_improve = 0
    best_oof = None
    
    for epoch in range(cfg.epochs):
        # Train
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
                main_loss = main_criterion(main_output, targets)
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
            for left, right, targets, _, indices in val_loader:
                left, right = left.to(device), right.to(device)
                main_output, _ = model(left, right)
                all_preds.append(main_output.cpu().numpy())
                all_targets.append(targets.numpy())
                all_indices.extend(indices.numpy().tolist())
        
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        
        val_score = competition_metric(targets, preds)
        
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
                'targets': targets.copy(),
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
    print(f"  ✓ Best score: {best_score:.4f}")
    
    wandb.log({f"fold{fold}/best_score": best_score})
    
    flush()
    return best_score, best_oof

#%% [markdown]
# ## 🚀 Run Training

#%%
# Config 저장 (Grid Search 추적용)
config_name = f"h{cfg.hidden_dim}_l{cfg.num_layers}_d{int(cfg.dropout*10)}"
print(f"\n=== Config: {config_name} ===")

run = wandb.init(
    entity=WANDB_ENTITY,
    project=WANDB_PROJECT,
    name=f"cv2_{config_name}",
    config={
        "version": "cv2",
        "hidden_dim": cfg.hidden_dim,
        "num_layers": cfg.num_layers,
        "dropout": cfg.dropout,
        "freeze_backbone": cfg.freeze_backbone,
        "use_weighted_loss": cfg.use_weighted_loss,
        "lr": cfg.lr,
        "batch_size": cfg.batch_size,
        "img_size": cfg.img_size,
    }
)

#%%
print("\n" + "="*60)
print("🚀 CV2 Training: Weighted Loss + Frozen Backbone")
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
print("🎉 CV2 RESULTS")
print("="*60)
print(f"Config: {config_name}")
print(f"Folds: {[f'{s:.4f}' for s in fold_scores]}")
print(f"Mean CV: {mean_cv:.4f} ± {std_cv:.4f}")

#%% [markdown]
# ## 📊 OOF Score

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

#%% [markdown]
# ## 💾 Save to Google Drive

#%%
if GDRIVE_SAVE_PATH:
    # Config별 폴더 생성
    save_dir = GDRIVE_SAVE_PATH / config_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 모델 복사
    for f in OUTPUT_DIR.glob("model_fold*.pth"):
        shutil.copy(f, save_dir / f.name)
    
    # OOF 복사
    for f in OUTPUT_DIR.glob("oof_fold*.npy"):
        shutil.copy(f, save_dir / f.name)
    
    # 결과 저장
    with open(save_dir / 'results.json', 'w') as f:
        json.dump({
            'config_name': config_name,
            'hidden_dim': cfg.hidden_dim,
            'num_layers': cfg.num_layers,
            'dropout': cfg.dropout,
            'freeze_backbone': cfg.freeze_backbone,
            'fold_scores': fold_scores,
            'mean_cv': float(mean_cv),
            'std_cv': float(std_cv),
            'total_oof_score': float(total_oof_score),
        }, f, indent=2)
    
    # ⭐ ZIP 파일 생성
    zip_path = save_dir / f'models_{config_name}.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fold in range(5):
            model_path = save_dir / f'model_fold{fold}.pth'
            if model_path.exists():
                zf.write(model_path, f'model_fold{fold}.pth')
    
    print(f"\n✓ All saved to: {save_dir}")
    print(f"✓ ZIP file: {zip_path}")

#%%
wandb.log({
    "final/mean_cv": mean_cv,
    "final/std_cv": std_cv,
    "final/oof_score": total_oof_score,
})

wandb.finish()

print(f"\n🎉 Training complete!")
print(f"   Config: {config_name}")
print(f"   Mean CV: {mean_cv:.4f}")
