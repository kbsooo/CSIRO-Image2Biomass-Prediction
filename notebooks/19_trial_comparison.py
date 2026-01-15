#%% [markdown]
# # 🔬 v19: Trial 22 vs 27 Comparison with WandB
#
# **목적**: 안정성 높은 config 찾기
#
# **테스트**:
# - Trial 22: hidden=256, layers=1, dropout=0.0, smooth_l1
# - Trial 27: hidden=256, layers=1, dropout=0.3, mse

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

# wandb 로그인 (API key 필요)
# Colab에서: wandb.login() 실행 후 API key 입력
wandb.login()

PROJECT_NAME = "csiro-biomass-v19"
print(f"✓ WandB project: {PROJECT_NAME}")

#%% [markdown]
# ## 🔐 Setup

#%%
GDRIVE_SAVE_PATH = None

try:
    from google.colab import drive
    drive.mount('/content/drive')
    GDRIVE_SAVE_PATH = Path('/content/drive/MyDrive/kaggle_models/csiro_biomass_v19')
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
# ## ⚙️ Trial Configs

#%%
TRIAL_CONFIGS = {
    "trial_22": {
        "name": "Trial22_h256_L1_d0.0_smooth_l1",
        "hidden_dim": 256,
        "num_layers": 1,
        "dropout": 0.0,
        "use_layernorm": True,
        "lr": 2.63e-4,
        "backbone_lr_mult": 0.07,
        "warmup_ratio": 0.067,
        "weight_decay": 8.6e-5,
        "aug_strategy": "color_focus",
        "loss_type": "smooth_l1",
        "batch_size": 8,
    },
    "trial_27": {
        "name": "Trial27_h256_L1_d0.3_mse",
        "hidden_dim": 256,
        "num_layers": 1,
        "dropout": 0.3,
        "use_layernorm": True,
        "lr": 2.18e-4,
        "backbone_lr_mult": 0.14,
        "warmup_ratio": 0.067,
        "weight_decay": 3.52e-3,
        "aug_strategy": "color_focus",
        "loss_type": "mse",
        "batch_size": 8,
    }
}

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
# ## 📊 Data

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

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_wide['fold'] = -1
for fold, (_, val_idx) in enumerate(sgkf.split(train_wide, train_wide['State'], groups=train_wide['image_id'])):
    train_wide.loc[val_idx, 'fold'] = fold

print(f"Train samples: {len(train_wide)}")

#%% [markdown]
# ## 🎨 Augmentation

#%%
def get_train_transforms(img_size=(512, 512)):
    return T.Compose([
        T.Resize(img_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms(img_size=(512, 512)):
    return T.Compose([
        T.Resize(img_size),
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
        
        targets = torch.tensor([
            row['Dry_Green_g'], row['Dry_Clover_g'], row['Dry_Dead_g']
        ], dtype=torch.float32)
        
        return left_img, right_img, targets

#%% [markdown]
# ## 🧠 Model

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


class CSIROModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.backbone = timm.create_model(
            "vit_large_patch16_dinov3_qkvb.lvd1689m", 
            pretrained=False, num_classes=0, global_pool='avg'
        )
        weights_file = WEIGHTS_PATH / "dinov3_vitl16_qkvb.pth"
        if weights_file.exists():
            state = torch.load(weights_file, map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(state, strict=False)
        
        feat_dim = self.backbone.num_features
        combined_dim = feat_dim * 2
        
        self.film = FiLM(feat_dim)
        
        self.head_green = make_head(combined_dim, config['hidden_dim'], config['num_layers'], 
                                    config['dropout'], config['use_layernorm'])
        self.head_clover = make_head(combined_dim, config['hidden_dim'], config['num_layers'],
                                     config['dropout'], config['use_layernorm'])
        self.head_dead = make_head(combined_dim, config['hidden_dim'], config['num_layers'],
                                   config['dropout'], config['use_layernorm'])
        
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
        
        return torch.cat([green, dead, clover, gdm, total], dim=1)

#%% [markdown]
# ## 🏋️ Training with WandB

#%%
def get_loss_fn(loss_type):
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "smooth_l1":
        return nn.SmoothL1Loss()
    else:
        return nn.MSELoss()


def train_fold(fold, train_df, config, device="cuda"):
    train_data = train_df[train_df['fold'] != fold].reset_index(drop=True)
    val_data = train_df[train_df['fold'] == fold].reset_index(drop=True)
    
    train_ds = BiomassDataset(train_data, DATA_PATH, get_train_transforms())
    val_ds = BiomassDataset(val_data, DATA_PATH, get_val_transforms())
    
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size']*2, shuffle=False, num_workers=4, pin_memory=True)
    
    model = CSIROModel(config).to(device)
    
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.head_green.parameters()) + list(model.head_clover.parameters()) + \
                  list(model.head_dead.parameters()) + list(model.film.parameters())
    
    optimizer = AdamW([
        {'params': backbone_params, 'lr': config['lr'] * config['backbone_lr_mult']},
        {'params': head_params, 'lr': config['lr']}
    ], weight_decay=config['weight_decay'])
    
    epochs = 25
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * config['warmup_ratio'])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    loss_fn = get_loss_fn(config['loss_type'])
    scaler = GradScaler()
    
    best_score = -float('inf')
    patience = 7
    no_improve = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for left, right, targets in train_loader:
            left, right, targets = left.to(device), right.to(device), targets.to(device)
            
            optimizer.zero_grad()
            with autocast():
                outputs = model(left, right)
                pred = outputs[:, [0, 2, 1]]
                loss = loss_fn(pred, targets)
            
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
            for left, right, targets in val_loader:
                left, right = left.to(device), right.to(device)
                outputs = model(left, right)
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.numpy())
        
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        
        full_targets = np.zeros((len(targets), 5))
        full_targets[:, 0] = targets[:, 0]
        full_targets[:, 1] = targets[:, 2]
        full_targets[:, 2] = targets[:, 1]
        full_targets[:, 3] = targets[:, 0] + targets[:, 1]
        full_targets[:, 4] = full_targets[:, 3] + targets[:, 2]
        
        val_score = competition_metric(full_targets, preds)
        
        # WandB logging
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
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # Log best score for this fold
    wandb.log({f"fold{fold}/best_score": best_score})
    
    flush()
    return best_score


def run_trial(trial_key, config, train_df):
    """Run full 5-fold training for a trial config"""
    
    print(f"\n{'='*60}")
    print(f"🚀 {config['name']}")
    print(f"{'='*60}")
    
    # Start WandB run
    run = wandb.init(
        project=PROJECT_NAME,
        name=config['name'],
        config=config,
        reinit=True
    )
    
    fold_scores = []
    
    for fold in range(5):
        print(f"\n--- Fold {fold} ---")
        score = train_fold(fold, train_df, config)
        fold_scores.append(score)
        
        # Log running stats
        wandb.log({
            "current_fold": fold,
            "running_mean": np.mean(fold_scores),
            "running_std": np.std(fold_scores) if len(fold_scores) > 1 else 0,
        })
    
    mean_cv = np.mean(fold_scores)
    std_cv = np.std(fold_scores)
    
    # Log final results
    wandb.log({
        "final/mean_cv": mean_cv,
        "final/std_cv": std_cv,
        "final/min_fold": np.min(fold_scores),
        "final/max_fold": np.max(fold_scores),
        "final/stability_score": mean_cv - std_cv,  # Higher = better
    })
    
    print(f"\n🎉 {config['name']} Results:")
    print(f"  Folds: {[f'{s:.4f}' for s in fold_scores]}")
    print(f"  Mean: {mean_cv:.4f} ± {std_cv:.4f}")
    
    # Save to Drive
    if GDRIVE_SAVE_PATH:
        trial_path = GDRIVE_SAVE_PATH / trial_key
        trial_path.mkdir(parents=True, exist_ok=True)
        
        for f in OUTPUT_DIR.glob("model_fold*.pth"):
            shutil.copy(f, trial_path / f.name)
        
        with open(trial_path / 'results.json', 'w') as f:
            json.dump({
                'config': config,
                'fold_scores': fold_scores,
                'mean_cv': float(mean_cv),
                'std_cv': float(std_cv),
            }, f, indent=2)
        
        print(f"  Saved to: {trial_path}")
    
    wandb.finish()
    
    return {
        'name': config['name'],
        'fold_scores': fold_scores,
        'mean_cv': mean_cv,
        'std_cv': std_cv,
    }

#%% [markdown]
# ## 🚀 Run Comparison

#%%
results = {}

for trial_key, config in TRIAL_CONFIGS.items():
    results[trial_key] = run_trial(trial_key, config, train_wide)

#%% [markdown]
# ## 📊 Final Comparison

#%%
print("\n" + "="*60)
print("🏆 FINAL COMPARISON")
print("="*60)

for trial_key, result in results.items():
    print(f"\n{result['name']}:")
    print(f"  Mean CV: {result['mean_cv']:.4f} ± {result['std_cv']:.4f}")
    print(f"  Stability Score: {result['mean_cv'] - result['std_cv']:.4f}")
    print(f"  Min/Max: {np.min(result['fold_scores']):.4f} / {np.max(result['fold_scores']):.4f}")

# Recommend best
best_trial = max(results.keys(), key=lambda k: results[k]['mean_cv'] - results[k]['std_cv'])
print(f"\n✅ Recommended: {best_trial}")
print(f"   (Highest stability score = mean - std)")
