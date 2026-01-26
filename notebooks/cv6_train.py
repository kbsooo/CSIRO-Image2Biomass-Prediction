#%% [markdown]
# # 🚀 CV6: Rect-Frame DINOv3 + SSF
#
# **Goal**: Single full-frame approach (no left/right split)
#
# **Key Ideas**:
# - **Full-frame rectangular input (784x336)**: preserve 70×30 aspect ratio
# - **CLS + Patch Mean pooling**: richer global + local aggregation
# - **SSF adapters**: lightweight backbone adaptation (scale/shift)
# - **Zero-Inflated Clover head** + physics constraints
# - **5-Fold CV** with OOF collection

#%%
import os
import gc
import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List

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
    GDRIVE_SAVE_PATH = Path('/content/drive/MyDrive/kaggle_models/csiro_biomass_cv6')
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
    # Model settings
    model_name = "vit_large_patch16_dinov3_qkvb.lvd1689m"
    # Full-frame: 70x30 aspect → (H=336, W=784), H/W divisible by 16
    img_size = (336, 784)
    backbone_dim = 1024
    freeze_backbone = True
    use_ssf = True
    ssf_per_block = True
    
    # Head
    head_dim = 256
    head_layers = 2
    dropout = 0.2
    
    # Training
    lr = 2e-4
    weight_decay = 1e-4
    warmup_ratio = 0.1
    grad_clip = 1.0
    
    # High-res full-frame: reduce batch to avoid OOM
    batch_size = 4
    epochs = 25
    patience = 7
    
    # Loss
    zi_weight = 0.3
    
    # Augmentation
    hue_jitter = 0.02

cfg = CFG()

print("=== CV6 Configuration: Rect-Frame + SSF ===")
print(f"Image size (H, W): {cfg.img_size}")
print(f"SSF adapters: {cfg.use_ssf}, per_block: {cfg.ssf_per_block}")

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
print(f"Weights: {WEIGHTS_PATH}")

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
        T.Resize(cfg.img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=cfg.hue_jitter),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms(cfg):
    return T.Compose([
        T.Resize(cfg.img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

#%% [markdown]
# ## 📦 Dataset

#%%
class BiomassRectDataset(Dataset):
    """Full-frame dataset (no left/right split)"""
    def __init__(self, df, data_path, transform=None, return_idx=False):
        self.df = df.reset_index(drop=True)
        self.data_path = data_path
        self.transform = transform
        self.return_idx = return_idx
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.data_path / row['image_path']).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        # Independent targets: [Green, Clover, Dead]
        main_targets = torch.tensor([
            row['Dry_Green_g'],
            row['Dry_Clover_g'],
            row['Dry_Dead_g']
        ], dtype=torch.float32)
        
        if self.return_idx:
            return img, main_targets, idx
        return img, main_targets

#%% [markdown]
# ## 🧠 Model

#%%
class SSFAdapter(nn.Module):
    """Feature-wise scale & shift (lightweight PEFT)"""
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, 1, dim))
        self.shift = nn.Parameter(torch.zeros(1, 1, dim))
    
    def forward(self, x):
        return x * self.scale + self.shift


class DINOv3Backbone(nn.Module):
    def __init__(self, cfg, weights_path=None):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=False,
            num_classes=0,
            global_pool=""  # keep tokens
        )
        
        # Load weights
        if weights_path:
            weights_file = weights_path / "dinov3_vitl16_qkvb.pth"
            if weights_file.exists():
                state = torch.load(weights_file, map_location='cpu', weights_only=True)
                self.backbone.load_state_dict(state, strict=False)
                print(f"✓ Loaded backbone weights from {weights_file}")
        
        self.num_features = self.backbone.num_features
        
        # Freeze backbone
        if cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("✓ Backbone frozen")
        
        # SSF adapters
        self.use_ssf = cfg.use_ssf
        self.ssf_per_block = cfg.use_ssf and cfg.ssf_per_block
        self.ssf_blocks = None
        self._hooks = []
        
        if self.use_ssf:
            if self.ssf_per_block:
                self._init_ssf_per_block()
            else:
                self.ssf_out = SSFAdapter(self.num_features)
    
    def _init_ssf_per_block(self):
        assert hasattr(self.backbone, 'blocks'), "Backbone has no blocks attribute"
        self.ssf_blocks = nn.ModuleList([
            SSFAdapter(self.num_features) for _ in range(len(self.backbone.blocks))
        ])
        
        def make_hook(i):
            def hook(_module, _input, output):
                return self.ssf_blocks[i](output)
            return hook
        
        for i, blk in enumerate(self.backbone.blocks):
            self._hooks.append(blk.register_forward_hook(make_hook(i)))
        
        print(f"✓ SSF hooks attached: {len(self.ssf_blocks)} blocks")
    
    def forward_tokens(self, x):
        tokens = self.backbone.forward_features(x)
        if isinstance(tokens, (tuple, list)):
            tokens = tokens[0]
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(1)
        
        if self.use_ssf and not self.ssf_per_block:
            tokens = self.ssf_out(tokens)
        
        return tokens


class ZeroInflatedHead(nn.Module):
    """Two-stage head for zero-inflated targets (Clover)
    
    Note: classifier outputs logits (no sigmoid) for AMP compatibility
    """
    def __init__(self, in_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        # Classifier outputs logits (no sigmoid for AMP compatibility)
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.regressor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )
    
    def forward(self, x):
        logits = self.classifier(x)
        p_pos = torch.sigmoid(logits)  # Convert to probability for prediction
        amount = self.regressor(x)
        pred = p_pos * amount
        return logits, amount, pred  # Return logits for loss, not p_pos


class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        layers = []
        cur = in_dim
        for i in range(num_layers):
            layers.append(nn.Linear(cur, hidden_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
            cur = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class CV6Model(nn.Module):
    """Rect-Frame + CLS/Patch pooling + SSF"""
    def __init__(self, cfg, weights_path=None):
        super().__init__()
        self.backbone = DINOv3Backbone(cfg, weights_path)
        feat_dim = self.backbone.num_features
        
        # CLS + mean pooling → 2x features
        self.out_dim = feat_dim * 2
        
        self.green_head = MLPHead(self.out_dim, cfg.head_dim, cfg.head_layers, cfg.dropout)
        self.dead_head = MLPHead(self.out_dim, cfg.head_dim, cfg.head_layers, cfg.dropout)
        self.clover_head = ZeroInflatedHead(self.out_dim, hidden_dim=cfg.head_dim, dropout=cfg.dropout)
        
        self.softplus = nn.Softplus(beta=1.0)
    
    def forward(self, x):
        tokens = self.backbone.forward_tokens(x)  # [B, N, C]
        B, N, C = tokens.shape
        
        cls = tokens[:, 0]
        patch_mean = tokens[:, 1:].mean(dim=1) if N > 1 else cls
        
        feat = torch.cat([cls, patch_mean], dim=1)
        
        green = self.softplus(self.green_head(feat))
        dead = self.softplus(self.dead_head(feat))
        clover_prob, clover_amount, clover = self.clover_head(feat)
        
        # Physics constraints
        gdm = green + clover
        total = gdm + dead
        
        main_output = torch.cat([green, dead, clover, gdm, total], dim=1)
        
        return main_output, clover_prob, clover_amount

#%% [markdown]
# ## 🔧 Loss Functions

#%%
class ZeroInflatedLoss(nn.Module):
    """Zero-inflated loss using BCEWithLogits for AMP compatibility"""
    def __init__(self, cls_weight=0.5):
        super().__init__()
        self.cls_weight = cls_weight
    
    def forward(self, logits, amount, targets):
        # logits: classifier output (before sigmoid)
        is_pos = (targets > 0).float()
        # Use binary_cross_entropy_with_logits for AMP compatibility
        cls_loss = F.binary_cross_entropy_with_logits(logits, is_pos)
        
        pos_mask = targets > 0
        if pos_mask.sum() > 0:
            reg_loss = F.mse_loss(amount[pos_mask], targets[pos_mask])
        else:
            reg_loss = torch.tensor(0.0, device=targets.device)
        
        return self.cls_weight * cls_loss + (1 - self.cls_weight) * reg_loss

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
    train_ds = BiomassRectDataset(train_data, DATA_PATH, get_train_transforms(cfg))
    val_ds = BiomassRectDataset(val_data, DATA_PATH, get_val_transforms(cfg), return_idx=True)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size*2, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    # Model
    model = CV6Model(cfg, WEIGHTS_PATH).to(device)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"  Trainable params: {sum(p.numel() for p in trainable_params) / 1e6:.2f}M")
    
    optimizer = AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    zi_loss_fn = ZeroInflatedLoss()
    scaler = GradScaler()
    
    best_score = -float('inf')
    no_improve = 0
    best_oof = None
    
    for epoch in range(cfg.epochs):
        # Train
        model.train()
        train_loss = 0
        
        for img, main_targets in train_loader:
            img = img.to(device)
            main_targets = main_targets.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                main_output, clover_prob, clover_amount = model(img)
                pred = main_output[:, [0, 2, 1]]  # Green, Clover, Dead
                main_loss = F.mse_loss(pred, main_targets)
                
                # Zero-inflated clover loss
                clover_targets = main_targets[:, 1:2]
                zi_loss = zi_loss_fn(clover_prob, clover_amount, clover_targets)
                
                loss = main_loss + cfg.zi_weight * zi_loss
            
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate & Collect OOF
        model.eval()
        all_preds, all_targets, all_indices = [], [], []
        
        with torch.no_grad():
            for img, main_targets, indices in val_loader:
                img = img.to(device)
                main_output, _, _ = model(img)
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
    name=f"cv6_rectframe_ssf",
    config={
        "version": "cv6",
        "strategy": "Rect-Frame + SSF",
        "img_size": cfg.img_size,
        "freeze_backbone": cfg.freeze_backbone,
        "use_ssf": cfg.use_ssf,
        "ssf_per_block": cfg.ssf_per_block,
        "head_dim": cfg.head_dim,
        "head_layers": cfg.head_layers,
        "lr": cfg.lr,
    }
)

#%%
print("\n" + "="*60)
print("🚀 CV6 Training: Rect-Frame + SSF Adapters")
print("="*60)
print(f"Image size: {cfg.img_size} (H, W)")
print(f"SSF: {cfg.use_ssf}, per_block: {cfg.ssf_per_block}")
print(f"Backbone frozen: {cfg.freeze_backbone}")

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
print("🎉 CV6 RESULTS: Rect-Frame + SSF")
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

#%%
# Google Drive에 저장
if GDRIVE_SAVE_PATH:
    for f in OUTPUT_DIR.glob("model_fold*.pth"):
        shutil.copy(f, GDRIVE_SAVE_PATH / f.name)
    for f in OUTPUT_DIR.glob("oof_fold*.npy"):
        shutil.copy(f, GDRIVE_SAVE_PATH / f.name)
    
    with open(GDRIVE_SAVE_PATH / 'results.json', 'w') as f:
        json.dump({
            'version': 'cv6',
            'strategy': 'Rect-Frame + SSF',
            'fold_scores': fold_scores,
            'mean_cv': float(mean_cv),
            'std_cv': float(std_cv),
            'total_oof_score': float(total_oof_score),
        }, f, indent=2)
    print(f"\n✓ All saved to: {GDRIVE_SAVE_PATH}")

wandb.log({
    "final/mean_cv": mean_cv,
    "final/std_cv": std_cv,
    "final/oof_score": total_oof_score,
})

wandb.finish()

print("\n" + "="*60)
print("✅ CV6 Complete!")
print(f"   Strategy: Rect-Frame + SSF Adapters")
print(f"   Mean CV: {mean_cv:.4f}")
print("="*60)
