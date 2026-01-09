#%% [markdown]
# # CSIRO Image2Biomass - Simple Baseline for Submission
#
# **Goal**: Get a valid submission.csv first, then iterate
#
# **Strategy**:
# - EfficientNet-B0 (torchvision, no external weights needed)
# - Simple regression head
# - Fast training (5 epochs)
# - Single fold for speed

#%%
# === Kaggle 커널 재시작 필요 시 이 셀만 먼저 실행 ===
# 런타임 -> 세션 다시 시작 후 실행

import os
import gc
import random
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from tqdm.auto import tqdm
from sklearn.model_selection import KFold

# PyTorch imports (커널 재시작 후 정상 작동)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.models as models

import albumentations as A
from albumentations.pytorch import ToTensorV2

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

#%%
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

seed_everything(42)

#%% [markdown]
# ## Config

#%%
class CFG:
    # === Kaggle Paths ===
    DATA_PATH = Path("/kaggle/input/csiro-biomass")
    OUTPUT_DIR = Path("/kaggle/working")

    # === Model ===
    model_name = "efficientnet_b0"  # torchvision 기본 제공
    input_size = 384  # EfficientNet-B0 default

    # === Training ===
    epochs = 5  # 빠른 학습
    batch_size = 8
    lr = 1e-3
    weight_decay = 1e-4

    # === Misc ===
    seed = 42
    num_workers = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === Targets ===
    targets = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

cfg = CFG()
cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {cfg.device}")
print(f"Data path: {cfg.DATA_PATH}")

#%% [markdown]
# ## Constants

#%%
TARGET_WEIGHTS = {
    'Dry_Green_g': 0.1,
    'Dry_Dead_g': 0.1,
    'Dry_Clover_g': 0.1,
    'GDM_g': 0.2,
    'Dry_Total_g': 0.5,
}
TARGET_ORDER = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

def competition_metric(y_true, y_pred):
    """Globally weighted R²."""
    weights = np.array([TARGET_WEIGHTS[t] for t in TARGET_ORDER])
    y_weighted_mean = sum(y_true[:, i].mean() * weights[i] for i in range(5))
    ss_res = sum(((y_true[:, i] - y_pred[:, i]) ** 2).mean() * weights[i] for i in range(5))
    ss_tot = sum(((y_true[:, i] - y_weighted_mean) ** 2).mean() * weights[i] for i in range(5))
    return 1 - ss_res / (ss_tot + 1e-8)

#%% [markdown]
# ## Data Preparation

#%%
def pivot_table(df):
    if 'target' in df.columns:
        df_pt = pd.pivot_table(
            df, values='target',
            index=['image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm'],
            columns='target_name', aggfunc='mean'
        ).reset_index()
    else:
        df['target'] = 0
        df_pt = pd.pivot_table(
            df, values='target', index='image_path',
            columns='target_name', aggfunc='mean'
        ).reset_index()
    return df_pt

def melt_table(df):
    melted = df.melt(
        id_vars='image_path', value_vars=TARGET_ORDER,
        var_name='target_name', value_name='target'
    )
    melted['sample_id'] = (
        melted['image_path']
        .str.replace(r'^.*/', '', regex=True)
        .str.replace('.jpg', '', regex=False)
        + '__' + melted['target_name']
    )
    return melted[['sample_id', 'image_path', 'target_name', 'target']]

#%%
# Load train data
train_df = pd.read_csv(cfg.DATA_PATH / "train.csv")
train_wide = pivot_table(train_df)
train_wide['image_id'] = train_wide['image_path'].apply(lambda x: Path(x).stem)

# Simple KFold
kf = KFold(n_splits=5, shuffle=True, random_state=cfg.seed)
train_wide['fold'] = -1
for fold, (_, val_idx) in enumerate(kf.split(train_wide)):
    train_wide.loc[val_idx, 'fold'] = fold

print(f"Train data shape: {train_wide.shape}")
print(train_wide.head())

#%% [markdown]
# ## Dataset

#%%
def get_transforms(mode='train', size=384):
    if mode == 'train':
        return A.Compose([
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(size, size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

#%%
class BiomassDataset(Dataset):
    def __init__(self, df, cfg, transforms=None, mode='train'):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        img_path = self.cfg.DATA_PATH / row['image_path']
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms:
            img = self.transforms(image=img)['image']

        # Targets
        targets = torch.tensor([row[t] for t in TARGET_ORDER], dtype=torch.float32)

        return img, targets

#%% [markdown]
# ## Model

#%%
class SimpleBaseline(nn.Module):
    """
    EfficientNet-B0 + Simple regression head.
    torchvision 기본 제공 모델 사용 (인터넷 OFF에서도 작동).
    """
    def __init__(self, num_targets=5, pretrained=True):
        super().__init__()

        # EfficientNet-B0 backbone
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        # Get feature dimension
        in_features = self.backbone.classifier[1].in_features

        # Replace classifier with regression head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_targets)
        )

        # Softplus for non-negative outputs
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.backbone(x)
        x = self.softplus(x)
        return x

#%% [markdown]
# ## Training

#%%
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for imgs, targets in tqdm(loader, desc='Train', leave=False):
        imgs = imgs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []

    for imgs, targets in tqdm(loader, desc='Valid', leave=False):
        imgs = imgs.to(device)
        targets = targets.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, targets)

        total_loss += loss.item()
        all_preds.append(outputs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    cv_score = competition_metric(all_targets, all_preds)

    return total_loss / len(loader), cv_score

#%%
def train_fold(fold, train_df, cfg):
    print(f"\n{'='*50}")
    print(f"Training Fold {fold}")
    print(f"{'='*50}")

    # Split
    train_data = train_df[train_df['fold'] != fold].reset_index(drop=True)
    val_data = train_df[train_df['fold'] == fold].reset_index(drop=True)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Datasets
    train_dataset = BiomassDataset(train_data, cfg, get_transforms('train', cfg.input_size), 'train')
    val_dataset = BiomassDataset(val_data, cfg, get_transforms('val', cfg.input_size), 'val')

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size * 2, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

    # Model
    model = SimpleBaseline(num_targets=5, pretrained=True).to(cfg.device)

    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    # Loss - weighted MSE
    weights = torch.tensor([TARGET_WEIGHTS[t] for t in TARGET_ORDER]).to(cfg.device)
    def weighted_mse(pred, target):
        return ((pred - target) ** 2 * weights).mean()

    # Training loop
    best_score = -float('inf')

    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, weighted_mse, cfg.device)
        val_loss, cv_score = validate(model, val_loader, weighted_mse, cfg.device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{cfg.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | CV: {cv_score:.4f}")

        if cv_score > best_score:
            best_score = cv_score
            torch.save(model.state_dict(), cfg.OUTPUT_DIR / f'best_model_fold{fold}.pt')
            print(f"  -> New best! Saved.")

    flush()
    return best_score

#%%
# Train only fold 0 for speed
best_score = train_fold(0, train_wide, cfg)
print(f"\nBest CV Score: {best_score:.4f}")

#%% [markdown]
# ## Inference & Submission

#%%
@torch.no_grad()
def inference(model, loader, device):
    model.eval()
    all_preds = []

    for imgs, _ in tqdm(loader, desc='Inference'):
        imgs = imgs.to(device)
        outputs = model(imgs)
        all_preds.append(outputs.cpu().numpy())

    return np.concatenate(all_preds)

#%%
# Load test data
test_df = pd.read_csv(cfg.DATA_PATH / "test.csv")
test_wide = test_df.drop_duplicates(subset=['image_path']).reset_index(drop=True)
test_wide['image_id'] = test_wide['image_path'].apply(lambda x: Path(x).stem)

# Add dummy targets
for t in TARGET_ORDER:
    if t not in test_wide.columns:
        test_wide[t] = 0.0

print(f"Test data: {len(test_wide)} images")

#%%
# Load best model
model = SimpleBaseline(num_targets=5, pretrained=False).to(cfg.device)
model.load_state_dict(torch.load(cfg.OUTPUT_DIR / 'best_model_fold0.pt', weights_only=True))

# Test dataset
test_dataset = BiomassDataset(test_wide, cfg, get_transforms('val', cfg.input_size), 'test')
test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False,
                         num_workers=cfg.num_workers, pin_memory=True)

# Inference
preds = inference(model, test_loader, cfg.device)
print(f"Predictions shape: {preds.shape}")

#%%
# Create submission
test_wide[TARGET_ORDER] = preds
submission = melt_table(test_wide)
submission = submission[['sample_id', 'target']]

# Clip to non-negative
submission['target'] = submission['target'].clip(lower=0)

# Save
submission.to_csv(cfg.OUTPUT_DIR / 'submission.csv', index=False)
print(f"\nSubmission saved: {len(submission)} rows")
print(submission.head(10))

#%%
# Verify submission format
print("\n=== Submission Verification ===")
print(f"Shape: {submission.shape}")
print(f"Columns: {submission.columns.tolist()}")
print(f"Null values: {submission.isnull().sum().sum()}")
print(f"Target range: [{submission['target'].min():.2f}, {submission['target'].max():.2f}]")

#%% [markdown]
# ## Done!
#
# submission.csv가 `/kaggle/working/submission.csv`에 저장되었습니다.

#%%
print(f"""
{'='*50}
Baseline Complete!
{'='*50}

Output: {cfg.OUTPUT_DIR / 'submission.csv'}
CV Score: {best_score:.4f}

Next steps:
1. Submit this notebook to verify pipeline
2. Improve model architecture
3. Add more epochs, folds, TTA
{'='*50}
""")
