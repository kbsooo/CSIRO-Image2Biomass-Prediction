#%% [markdown]
# # 🎯 v23: Optimized Leave-One-Group-Out CV
#
# **핵심 발견**: State와 Season이 심하게 confounded
# - NSW: Summer 집중 (41/75), Winter 없음
# - Tas: Spring/Autumn/Winter, Summer 없음
# - Vic: Spring/Winter만, Summer/Autumn 없음
# - WA: 32개뿐, Spring/Winter만
#
# **전략**:
# 1. State-LOGO: 새로운 지역 일반화 테스트 (현실적 시나리오)
# 2. 강한 정규화: 일반화 능력 향상
# 3. Data Augmentation: LOGO에서도 필수!

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

import warnings
warnings.filterwarnings('ignore')
tqdm.pandas()

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

#%% [markdown]
# ## 🔐 Setup

#%%
GDRIVE_SAVE_PATH = None

try:
    from google.colab import drive
    drive.mount('/content/drive')
    GDRIVE_SAVE_PATH = Path('/content/drive/MyDrive/kaggle_models/csiro_biomass_v23')
    GDRIVE_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Drive: {GDRIVE_SAVE_PATH}")
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
#
# **LOGO에 맞춘 보수적 설정**:
# - 더 작은 모델 (overfitting 방지)
# - 더 강한 dropout
# - 더 강한 augmentation

#%%
class CFG:
    # === Paths ===
    DATA_PATH = None
    WEIGHTS_PATH = None
    OUTPUT_DIR = None

    # === Model (더 단순하게!) ===
    model_name = "vit_large_patch16_dinov3_qkvb.lvd1689m"
    img_size = (512, 512)

    # 일반화를 위해 단순화
    hidden_dim = 256      # 512 → 256
    num_layers = 2        # 3 → 2
    dropout = 0.3         # 0.1 → 0.3 (강한 정규화)
    use_layernorm = True

    # === Training ===
    lr = 1e-4             # 더 낮은 LR
    backbone_lr_mult = 0.05  # backbone 더 천천히
    warmup_ratio = 0.1
    weight_decay = 1e-3   # 더 강한 weight decay

    batch_size = 8
    epochs = 30
    patience = 10

    # === LOGO Settings ===
    split_mode = "state"  # "state" or "season"

    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = CFG()

#%%
# Data paths setup
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
# ## 📊 Data Analysis & Loading

#%%
TARGET_WEIGHTS = {
    'Dry_Green_g': 0.1,
    'Dry_Dead_g': 0.1,
    'Dry_Clover_g': 0.1,
    'GDM_g': 0.2,
    'Dry_Total_g': 0.5
}
TARGET_ORDER = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

def competition_metric(y_true, y_pred):
    """Weighted R² metric"""
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

# 월/계절 추출
train_wide['Month'] = pd.to_datetime(train_wide['Sampling_Date']).dt.month

def get_season(month):
    """호주 남반구 계절"""
    if month in [12, 1, 2]:
        return 'Summer'
    elif month in [3, 4, 5]:
        return 'Autumn'
    elif month in [6, 7, 8]:
        return 'Winter'
    else:
        return 'Spring'

train_wide['Season'] = train_wide['Month'].apply(get_season)

print(f"총 샘플: {len(train_wide)}")

#%%
# 데이터 분포 분석
print("\n" + "="*60)
print("📊 데이터 분포 분석")
print("="*60)

print("\n[State 분포]")
state_counts = train_wide['State'].value_counts()
for state, count in state_counts.items():
    print(f"  {state}: {count} ({100*count/len(train_wide):.1f}%)")

print("\n[Season 분포]")
season_counts = train_wide['Season'].value_counts()
for season, count in season_counts.items():
    print(f"  {season}: {count} ({100*count/len(train_wide):.1f}%)")

print("\n[State x Season 교차표]")
cross = pd.crosstab(train_wide['State'], train_wide['Season'])
print(cross)

print("\n[Target 통계 by State]")
state_stats = train_wide.groupby('State')['Dry_Total_g'].agg(['mean', 'std', 'count'])
print(state_stats.round(2))

#%% [markdown]
# ## 🎨 Augmentation (LOGO에서도 중요!)
#
# LOGO에서는 일반화가 더 어려우므로 **강한 augmentation** 사용

#%%
def get_train_transforms(cfg):
    """강한 augmentation - 일반화 향상"""
    return T.Compose([
        T.Resize(cfg.img_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        # 강한 색상 변환 (다른 지역/계절 시뮬레이션)
        T.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.05  # hue는 적당히
        ),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
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
    def __init__(self, df, data_path, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_path = data_path
        self.transform = transform

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

        # Green, Clover, Dead 순서
        targets = torch.tensor([
            row['Dry_Green_g'], row['Dry_Clover_g'], row['Dry_Dead_g']
        ], dtype=torch.float32)

        return left_img, right_img, targets

#%% [markdown]
# ## 🧠 Model (단순화된 버전)

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
        return torch.chunk(self.mlp(context), 2, dim=1)


def make_head(in_dim, hidden_dim, num_layers, dropout, use_layernorm):
    """단순화된 head"""
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


class CSIROModelV23(nn.Module):
    """v23: LOGO에 최적화된 단순 모델"""
    def __init__(self, cfg):
        super().__init__()

        # Backbone
        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=False,
            num_classes=0,
            global_pool='avg'
        )

        weights_file = cfg.WEIGHTS_PATH / "dinov3_vitl16_qkvb.pth"
        if weights_file.exists():
            state = torch.load(weights_file, map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(state, strict=False)
            print("Backbone loaded")

        feat_dim = self.backbone.num_features  # 1024
        combined_dim = feat_dim * 2

        self.film = FiLM(feat_dim)

        # 단순화된 heads
        self.head_green = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers,
                                    cfg.dropout, cfg.use_layernorm)
        self.head_clover = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers,
                                     cfg.dropout, cfg.use_layernorm)
        self.head_dead = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers,
                                   cfg.dropout, cfg.use_layernorm)

        self.softplus = nn.Softplus(beta=1.0)

        # Parameter count
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Parameters: {trainable_params:,} trainable / {total_params:,} total")

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

        # [Green, Dead, Clover, GDM, Total]
        return torch.cat([green, dead, clover, gdm, total], dim=1)

#%% [markdown]
# ## 🎯 LOGO Split Setup

#%%
STATES = ['Tas', 'Vic', 'NSW', 'WA']
SEASONS = ['Spring', 'Summer', 'Autumn', 'Winter']

if cfg.split_mode == "state":
    GROUPS = STATES
    GROUP_COL = "State"
else:
    GROUPS = SEASONS
    GROUP_COL = "Season"

print(f"\n{'='*60}")
print(f"🎯 Leave-One-{GROUP_COL}-Out CV")
print(f"{'='*60}")

for i, group in enumerate(GROUPS):
    val_data = train_wide[train_wide[GROUP_COL] == group]
    train_data = train_wide[train_wide[GROUP_COL] != group]

    print(f"\nFold {i}: Val = {group}")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val: {len(val_data)} samples")

    # Train에서 빠지는 정보 분석
    if cfg.split_mode == "state":
        val_seasons = val_data['Season'].unique()
        train_seasons = train_data['Season'].unique()
        missing = set(val_seasons) - set(train_seasons)
        if missing:
            print(f"  ⚠️ Val의 Season 중 Train에 없는 것: {missing}")

#%% [markdown]
# ## 🏋️ Training

#%%
def train_one_epoch(model, loader, optimizer, scheduler, scaler, cfg):
    model.train()
    total_loss = 0

    for left, right, targets in loader:
        left = left.to(cfg.device)
        right = right.to(cfg.device)
        targets = targets.to(cfg.device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(left, right)
            # outputs: [Green, Dead, Clover, GDM, Total]
            # targets: [Green, Clover, Dead]
            pred = outputs[:, [0, 2, 1]]  # [Green, Clover, Dead]
            loss = F.mse_loss(pred, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, cfg):
    model.eval()
    all_preds, all_targets = [], []

    for left, right, targets in loader:
        left = left.to(cfg.device)
        right = right.to(cfg.device)

        outputs = model(left, right)
        all_preds.append(outputs.cpu().numpy())
        all_targets.append(targets.numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    # 5개 타겟으로 확장
    full_targets = np.zeros((len(targets), 5))
    full_targets[:, 0] = targets[:, 0]  # Green
    full_targets[:, 1] = targets[:, 2]  # Dead
    full_targets[:, 2] = targets[:, 1]  # Clover
    full_targets[:, 3] = targets[:, 0] + targets[:, 1]  # GDM
    full_targets[:, 4] = full_targets[:, 3] + targets[:, 2]  # Total

    score = competition_metric(full_targets, preds)
    return score, preds, full_targets


def train_fold(fold_idx, val_group, train_df, cfg):
    """Leave-One-Group-Out 학습"""
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx}: Val = {val_group}")
    print(f"{'='*60}")

    # Split
    train_data = train_df[train_df[GROUP_COL] != val_group].reset_index(drop=True)
    val_data = train_df[train_df[GROUP_COL] == val_group].reset_index(drop=True)

    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    if len(val_data) < 10:
        print(f"⚠️ Val 샘플이 너무 적음 ({len(val_data)}), 스킵")
        return None, None

    # Datasets
    train_ds = BiomassDataset(train_data, cfg.DATA_PATH, get_train_transforms(cfg))
    val_ds = BiomassDataset(val_data, cfg.DATA_PATH, get_val_transforms(cfg))

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size * 2, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # Model
    model = CSIROModelV23(cfg).to(cfg.device)

    # Optimizer with different LR for backbone
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

    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = GradScaler()

    best_score = -float('inf')
    best_preds = None
    no_improve = 0

    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, cfg)
        val_score, preds, targets = validate(model, val_loader, cfg)

        print(f"  Epoch {epoch+1:2d}: loss={train_loss:.4f}, CV={val_score:.4f}", end="")

        if val_score > best_score:
            best_score = val_score
            best_preds = preds
            no_improve = 0
            torch.save(model.state_dict(), cfg.OUTPUT_DIR / f'model_fold{fold_idx}.pth')
            print(" ✓ (saved)")
        else:
            no_improve += 1
            print()
            if no_improve >= cfg.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"\n  Best CV: {best_score:.4f}")

    # Cleanup
    del model
    flush()

    return best_score, best_preds

#%% [markdown]
# ## 🚀 Run LOGO Training

#%%
print("\n" + "="*60)
print(f"🚀 v23: Leave-One-{GROUP_COL}-Out Training")
print("="*60)
print(f"Config:")
print(f"  hidden_dim: {cfg.hidden_dim}")
print(f"  num_layers: {cfg.num_layers}")
print(f"  dropout: {cfg.dropout}")
print(f"  lr: {cfg.lr}, backbone_mult: {cfg.backbone_lr_mult}")
print(f"  weight_decay: {cfg.weight_decay}")

fold_scores = []
fold_results = {}

for fold_idx, val_group in enumerate(GROUPS):
    score, preds = train_fold(fold_idx, val_group, train_wide, cfg)

    if score is not None:
        fold_scores.append(score)
        fold_results[val_group] = {
            'score': score,
            'n_samples': len(train_wide[train_wide[GROUP_COL] == val_group])
        }

#%%
# Results Summary
print("\n" + "="*60)
print(f"🎉 v23 RESULTS (Leave-One-{GROUP_COL}-Out)")
print("="*60)

if fold_scores:
    print("\nPer-fold scores:")
    for group, result in fold_results.items():
        print(f"  {group}: {result['score']:.4f} (n={result['n_samples']})")

    mean_cv = np.mean(fold_scores)
    std_cv = np.std(fold_scores)

    # Weighted mean (by sample count)
    total_samples = sum(r['n_samples'] for r in fold_results.values())
    weighted_cv = sum(r['score'] * r['n_samples'] for r in fold_results.values()) / total_samples

    print(f"\nMean CV: {mean_cv:.4f} ± {std_cv:.4f}")
    print(f"Weighted CV: {weighted_cv:.4f}")
    print(f"Min: {min(fold_scores):.4f}, Max: {max(fold_scores):.4f}")

    print("\n💡 해석:")
    print(f"  - 이 CV ({mean_cv:.4f})가 LB와 비슷하면 일반화 잘 됨")
    print(f"  - LB가 이것보다 높으면 Test 분포가 Train과 유사")
    print(f"  - LB가 이것보다 낮으면 Test에 더 어려운 케이스 포함")

#%%
# Save results
if fold_scores and GDRIVE_SAVE_PATH:
    results = {
        'version': 'v23',
        'split_mode': cfg.split_mode,
        'fold_results': {k: {'score': float(v['score']), 'n_samples': v['n_samples']}
                         for k, v in fold_results.items()},
        'mean_cv': float(mean_cv),
        'std_cv': float(std_cv),
        'weighted_cv': float(weighted_cv),
        'config': {
            'hidden_dim': cfg.hidden_dim,
            'num_layers': cfg.num_layers,
            'dropout': cfg.dropout,
            'lr': cfg.lr,
            'backbone_lr_mult': cfg.backbone_lr_mult,
            'weight_decay': cfg.weight_decay,
        }
    }

    # Save models
    for f in cfg.OUTPUT_DIR.glob("model_fold*.pth"):
        shutil.copy(f, GDRIVE_SAVE_PATH / f.name)

    # Save results
    with open(GDRIVE_SAVE_PATH / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved to: {GDRIVE_SAVE_PATH}")
