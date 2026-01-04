#%% [markdown]
# # CSIRO Image2Biomass - Improved Training Pipeline v2
#
# **Key Improvements over 0.69 baseline:**
# - DINOv2-Large backbone (1024d vs 768d)
# - Left/Right image split with Bidirectional Cross-Attention
# - 3 predictions → 2 derived (physics constraint)
# - Competition-weighted loss (Total=0.5, GDM=0.2)
# - Post-processing with constraint projection
#
# **Target**: 0.69 → 0.78+

#%% [markdown]
# ## Section 0: Setup

#%%
# !pip install -q timm albumentations

#%%
import os
import gc
import math
import random
from copy import deepcopy
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold, GroupKFold

#%%
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

#%% [markdown]
# ## Section 1: Configuration

#%%
def get_paths():
    """Auto-detect paths based on environment."""
    # Colab
    colab_data = Path("/content/drive/MyDrive/kaggle/csiro-biomass")
    colab_output = Path("/content/drive/MyDrive/kaggle/outputs")

    # Local
    local_data = Path("./data")
    local_output = Path("./outputs")

    # Check which environment
    if colab_data.exists():
        return colab_data, colab_output
    elif local_data.exists():
        return local_data, local_output
    else:
        # Fallback to relative path
        return local_data, local_output

@dataclass
class CFG:
    # Paths (auto-detect: Colab or Local)
    DATA_PATH: Path = None
    OUTPUT_DIR: Path = None

    # Model
    backbone: str = "vit_large_patch14_dinov2.lvd142m"
    input_size: int = 518
    embed_dim: int = 1024  # DINOv2-large
    num_heads: int = 16
    dropout: float = 0.1

    # Training
    n_folds: int = 5
    train_folds: List[int] = None  # [0] for quick test, None for all
    epochs: int = 20
    batch_size: int = 4
    gradient_accumulation: int = 2
    effective_batch_size: int = 8  # batch_size * gradient_accumulation

    # Optimizer
    lr: float = 1e-4
    backbone_lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_epochs: int = 2

    # Training phases
    freeze_backbone_epochs: int = 3  # Freeze backbone for first N epochs

    # Target
    targets: List[str] = None
    num_targets: int = 5

    # Misc
    seed: int = 42
    num_workers: int = 2
    mixed_precision: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        # Auto-detect paths
        if self.DATA_PATH is None or self.OUTPUT_DIR is None:
            self.DATA_PATH, self.OUTPUT_DIR = get_paths()

        if self.targets is None:
            self.targets = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
        if self.train_folds is None:
            self.train_folds = list(range(self.n_folds))
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cfg = CFG()
seed_everything(cfg.seed)

print(f"Device: {cfg.device}")
print(f"PyTorch: {torch.__version__}")
print(f"Data path: {cfg.DATA_PATH}")
print(f"Output dir: {cfg.OUTPUT_DIR}")

#%% [markdown]
# ## Section 2: Competition Metric & Target Weights

#%%
# Competition weights - Total이 50%로 가장 중요!
TARGET_WEIGHTS = {
    'Dry_Green_g': 0.1,
    'Dry_Dead_g': 0.1,
    'Dry_Clover_g': 0.1,
    'GDM_g': 0.2,
    'Dry_Total_g': 0.5,
}

# Target order: Green, Dead, Clover, GDM, Total
TARGET_ORDER = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

# Max values for normalization (from train data)
TARGET_MAX = {
    'Dry_Green_g': 157.9836,
    'Dry_Dead_g': 83.8407,
    'Dry_Clover_g': 71.7865,
    'GDM_g': 157.9836,
    'Dry_Total_g': 185.70,
}

#%%
def competition_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate competition weighted R² score.

    Args:
        y_true: (N, 5) ground truth [Green, Dead, Clover, GDM, Total]
        y_pred: (N, 5) predictions
    """
    weights = np.array([TARGET_WEIGHTS[t] for t in TARGET_ORDER])

    # Weighted mean for baseline
    y_weighted_mean = sum(
        y_true[:, i].mean() * weights[i]
        for i in range(5)
    )

    # Weighted SS_res and SS_tot
    ss_res = sum(
        ((y_true[:, i] - y_pred[:, i]) ** 2).mean() * weights[i]
        for i in range(5)
    )
    ss_tot = sum(
        ((y_true[:, i] - y_weighted_mean) ** 2).mean() * weights[i]
        for i in range(5)
    )

    return 1 - ss_res / (ss_tot + 1e-8)

#%% [markdown]
# ## Section 3: Data Preparation

#%%
def pivot_table(df: pd.DataFrame) -> pd.DataFrame:
    """Convert long format to wide format."""
    if 'target' in df.columns:
        df_pt = pd.pivot_table(
            df,
            values='target',
            index=['image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm'],
            columns='target_name',
            aggfunc='mean'
        ).reset_index()
    else:
        df['target'] = 0
        df_pt = pd.pivot_table(
            df,
            values='target',
            index='image_path',
            columns='target_name',
            aggfunc='mean'
        ).reset_index()
    return df_pt

def melt_table(df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide format to submission format."""
    melted = df.melt(
        id_vars='image_path',
        value_vars=TARGET_ORDER,
        var_name='target_name',
        value_name='target'
    )
    melted['sample_id'] = (
        melted['image_path']
        .str.replace(r'^.*/', '', regex=True)
        .str.replace('.jpg', '', regex=False)
        + '__' + melted['target_name']
    )
    return melted[['sample_id', 'image_path', 'target_name', 'target']]

#%%
def prepare_data(cfg: CFG) -> pd.DataFrame:
    """Load and prepare training data."""
    train_df = pd.read_csv(cfg.DATA_PATH / "train.csv")
    train_wide = pivot_table(train_df)

    # Add image_id
    train_wide['image_id'] = train_wide['image_path'].apply(lambda x: Path(x).stem)

    # Create folds (KFold - random split)
    # Insight: Random split이 State-based GroupKFold보다 CV 점수가 높음
    kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    train_wide['fold'] = -1
    for fold, (_, val_idx) in enumerate(kf.split(train_wide)):
        train_wide.loc[val_idx, 'fold'] = fold

    print(f"Data shape: {train_wide.shape}")
    print(f"Fold distribution:\n{train_wide['fold'].value_counts().sort_index()}")

    return train_wide

#%%
# Mount Google Drive (Colab only)
import sys
if 'google.colab' in sys.modules:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    print("Google Drive mounted")
else:
    print("Running locally, skipping Drive mount")

# Load data
train_df = prepare_data(cfg)
train_df.head()

#%% [markdown]
# ## Section 4: Dataset & Transforms

#%%
def get_transforms(cfg: CFG, mode: str = 'train') -> A.Compose:
    """Get augmentation transforms."""
    if mode == 'train':
        return A.Compose([
            A.Resize(cfg.input_size, cfg.input_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            # Insight: 색상 변환은 약하게 (녹색/갈색 구분이 중요)
            A.ColorJitter(
                brightness=0.1, contrast=0.1,
                saturation=0.05, hue=0.02,
                p=0.3
            ),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT, p=0.5
            ),
            # Regularization
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(cfg.input_size, cfg.input_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

#%%
class BiomassDataset(Dataset):
    """
    Dataset for biomass prediction.
    Splits image into left/right halves for separate processing.
    """
    def __init__(self, df: pd.DataFrame, cfg: CFG, transforms=None, mode: str = 'train'):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # Load image
        img_path = self.cfg.DATA_PATH / row['image_path']
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Split into left/right halves (2000x1000 → 1000x1000 x2)
        h, w = img.shape[:2]
        mid = w // 2
        left_img = img[:, :mid]
        right_img = img[:, mid:]

        # Apply transforms
        if self.transforms:
            left_aug = self.transforms(image=left_img)['image']
            right_aug = self.transforms(image=right_img)['image']
        else:
            left_aug = torch.from_numpy(left_img).permute(2, 0, 1).float() / 255.0
            right_aug = torch.from_numpy(right_img).permute(2, 0, 1).float() / 255.0

        # Targets: [Green, Dead, Clover, GDM, Total]
        targets = torch.tensor([
            row[t] for t in TARGET_ORDER
        ], dtype=torch.float32)

        return {
            'left': left_aug,
            'right': right_aug,
            'targets': targets,
            'image_id': row['image_id']
        }

#%% [markdown]
# ## Section 5: Model Architecture

#%%
class FeedForward(nn.Module):
    """FFN block with GELU activation."""
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

#%%
class BiDirectionalCrossAttention(nn.Module):
    """
    Bidirectional cross-attention for left/right image fusion.

    ### [Architectural Insight]
    Unlike simple concatenation, cross-attention allows:
    1. Left features to query relevant context from right
    2. Right features to query relevant context from left
    3. Learnable [FUSE] tokens aggregate cross-view information

    This captures spatial correspondence between left/right views.
    """
    def __init__(self, dim: int = 1024, num_heads: int = 16,
                 num_fuse_tokens: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_fuse_tokens = num_fuse_tokens

        # Learnable fusion tokens
        self.fuse_tokens = nn.Parameter(torch.randn(1, num_fuse_tokens, dim) * 0.02)

        # Cross-attention layers
        self.cross_layers = nn.ModuleList([
            nn.ModuleDict({
                'l2r_attn': nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True),
                'r2l_attn': nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True),
                'l2r_norm': nn.LayerNorm(dim),
                'r2l_norm': nn.LayerNorm(dim),
                'ffn_l': FeedForward(dim, mlp_ratio=4.0, dropout=dropout),
                'ffn_r': FeedForward(dim, mlp_ratio=4.0, dropout=dropout),
                'ffn_norm_l': nn.LayerNorm(dim),
                'ffn_norm_r': nn.LayerNorm(dim),
            })
            for _ in range(num_layers)
        ])

        # Final fusion
        fusion_input_dim = dim * 2 + dim * num_fuse_tokens
        self.fusion_proj = nn.Sequential(
            nn.LayerNorm(fusion_input_dim),
            nn.Linear(fusion_input_dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim * 2)
        )

    def forward(self, feat_left: torch.Tensor, feat_right: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_left: (B, D) - left image CLS token
            feat_right: (B, D) - right image CLS token

        Returns:
            (B, D*2) - fused features
        """
        B = feat_left.shape[0]

        # Expand to sequence (B, 1, D)
        feat_left = feat_left.unsqueeze(1)
        feat_right = feat_right.unsqueeze(1)

        # Expand fuse tokens
        fuse = self.fuse_tokens.expand(B, -1, -1)  # (B, num_fuse, D)

        for layer in self.cross_layers:
            # Left attends to Right (with fuse tokens)
            l_query = torch.cat([feat_left, fuse], dim=1)  # (B, 1+num_fuse, D)
            l2r_out, _ = layer['l2r_attn'](
                layer['l2r_norm'](l_query),
                feat_right, feat_right,
                need_weights=False
            )
            feat_left = feat_left + l2r_out[:, :1]
            fuse_l = l2r_out[:, 1:]

            # Right attends to Left (with fuse tokens)
            r_query = torch.cat([feat_right, fuse], dim=1)
            r2l_out, _ = layer['r2l_attn'](
                layer['r2l_norm'](r_query),
                feat_left, feat_left,
                need_weights=False
            )
            feat_right = feat_right + r2l_out[:, :1]
            fuse_r = r2l_out[:, 1:]

            # FFN
            feat_left = feat_left + layer['ffn_l'](layer['ffn_norm_l'](feat_left))
            feat_right = feat_right + layer['ffn_r'](layer['ffn_norm_r'](feat_right))

            # Merge fuse tokens
            fuse = (fuse_l + fuse_r) / 2

        # Final fusion
        left_pool = feat_left.squeeze(1)   # (B, D)
        right_pool = feat_right.squeeze(1) # (B, D)
        fuse_flat = fuse.flatten(1)        # (B, num_fuse * D)

        combined = torch.cat([left_pool, right_pool, fuse_flat], dim=1)
        return self.fusion_proj(combined)  # (B, D*2)

#%%
class BiomassModel(nn.Module):
    """
    DINOv2-Large based model for biomass prediction.

    Architecture:
    1. DINOv2-Large backbone (shared for left/right)
    2. Bidirectional cross-attention fusion
    3. 3 prediction heads (Green, Dead, Clover)
    4. Derived outputs (GDM = Green + Clover, Total = GDM + Dead)
    """
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg

        # DINOv2-Large backbone
        # ### [SOTA Alert] Using DINOv2-large for richer representations
        self.backbone = timm.create_model(
            cfg.backbone,
            pretrained=True,
            num_classes=0,  # Remove classification head
        )
        self.embed_dim = self.backbone.embed_dim  # 1024 for large

        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(True)

        # Bidirectional cross-attention fusion
        self.cross_attn = BiDirectionalCrossAttention(
            dim=self.embed_dim,
            num_heads=cfg.num_heads,
            num_fuse_tokens=4,
            num_layers=2,
            dropout=cfg.dropout
        )

        # Prediction heads (3 only - Green, Dead, Clover)
        # Insight: 3개만 예측하고 나머지는 물리 법칙으로 계산
        head_dim = self.embed_dim * 2
        hidden_dim = head_dim // 2

        def make_head():
            return nn.Sequential(
                nn.Linear(head_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(hidden_dim, 1)
            )

        self.head_green = make_head()
        self.head_dead = make_head()
        self.head_clover = make_head()

        # Softplus for non-negative outputs
        self.softplus = nn.Softplus(beta=1.0)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            left: (B, 3, H, W) - left half image
            right: (B, 3, H, W) - right half image

        Returns:
            dict with 'green', 'dead', 'clover', 'gdm', 'total', 'all' (B, 5)
        """
        # Extract features (CLS token)
        feat_left = self.backbone(left)   # (B, embed_dim)
        feat_right = self.backbone(right) # (B, embed_dim)

        # Cross-attention fusion
        fused = self.cross_attn(feat_left, feat_right)  # (B, embed_dim*2)

        # Predict 3 components
        green = self.softplus(self.head_green(fused))   # (B, 1)
        dead = self.softplus(self.head_dead(fused))     # (B, 1)
        clover = self.softplus(self.head_clover(fused)) # (B, 1)

        # Derive GDM and Total (physics constraint)
        gdm = green + clover        # GDM = Green + Clover
        total = gdm + dead          # Total = GDM + Dead

        # Stack all: [Green, Dead, Clover, GDM, Total]
        all_preds = torch.cat([green, dead, clover, gdm, total], dim=1)

        return {
            'green': green,
            'dead': dead,
            'clover': clover,
            'gdm': gdm,
            'total': total,
            'all': all_preds  # (B, 5)
        }

    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True

#%% [markdown]
# ## Section 6: Loss Function

#%%
class WeightedBiomassLoss(nn.Module):
    """
    Competition-aligned weighted loss function.

    ### [Insight]
    Competition metric weights Total at 0.5, so we align training loss.
    Also adds physics constraint to ensure GDM = Green + Clover.
    """
    def __init__(self, physics_weight: float = 0.1):
        super().__init__()

        # Weights: [Green, Dead, Clover, GDM, Total]
        self.register_buffer('weights', torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5]))
        self.physics_weight = physics_weight
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none', beta=1.0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 5) - [Green, Dead, Clover, GDM, Total]
            target: (B, 5)
        """
        # Per-target SmoothL1 loss
        loss_per_target = self.smooth_l1(pred, target)  # (B, 5)

        # Weighted mean
        weighted_loss = (loss_per_target * self.weights.unsqueeze(0)).sum(dim=1).mean()

        # Physics constraint loss (should already be satisfied by architecture)
        # But we add it for extra regularization
        pred_gdm_check = pred[:, 0] + pred[:, 2]   # green + clover
        pred_total_check = pred[:, 3] + pred[:, 1] # gdm + dead

        physics_loss = (
            F.smooth_l1_loss(pred_gdm_check, pred[:, 3]) +  # GDM consistency
            F.smooth_l1_loss(pred_total_check, pred[:, 4])   # Total consistency
        )

        return weighted_loss + self.physics_weight * physics_loss

#%% [markdown]
# ## Section 7: Training Functions

#%%
def train_one_epoch(model, loader, optimizer, scheduler, criterion, cfg, scaler=None, epoch=0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f'Epoch {epoch+1} Training')

    for step, batch in enumerate(pbar):
        left = batch['left'].to(cfg.device)
        right = batch['right'].to(cfg.device)
        targets = batch['targets'].to(cfg.device)

        # Mixed precision forward
        with torch.amp.autocast('cuda', enabled=cfg.mixed_precision):
            outputs = model(left, right)
            loss = criterion(outputs['all'], targets)
            loss = loss / cfg.gradient_accumulation

        # Backward
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation
        if (step + 1) % cfg.gradient_accumulation == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * cfg.gradient_accumulation
        pbar.set_postfix({'loss': f'{loss.item() * cfg.gradient_accumulation:.4f}'})

    return total_loss / len(loader)

#%%
@torch.no_grad()
def validate(model, loader, criterion, cfg):
    """Validate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    for batch in tqdm(loader, desc='Validation'):
        left = batch['left'].to(cfg.device)
        right = batch['right'].to(cfg.device)
        targets = batch['targets'].to(cfg.device)

        with torch.amp.autocast('cuda', enabled=cfg.mixed_precision):
            outputs = model(left, right)
            loss = criterion(outputs['all'], targets)

        total_loss += loss.item()
        all_preds.append(outputs['all'].cpu())
        all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    # Compute competition metric
    cv_score = competition_metric(all_targets, all_preds)

    # Per-target R²
    r2_scores = {}
    for i, name in enumerate(TARGET_ORDER):
        ss_res = np.sum((all_targets[:, i] - all_preds[:, i]) ** 2)
        ss_tot = np.sum((all_targets[:, i] - all_targets[:, i].mean()) ** 2)
        r2_scores[name] = 1 - ss_res / (ss_tot + 1e-8)

    return total_loss / len(loader), cv_score, r2_scores, all_preds

#%% [markdown]
# ## Section 8: Post-Processing

#%%
def post_process_biomass(preds: np.ndarray) -> np.ndarray:
    """
    Project predictions to satisfy physics constraints.

    Constraints:
    1) GDM = Green + Clover
    2) Total = GDM + Dead

    Uses linear algebra projection to find closest point satisfying constraints.
    """
    # Order: Green, Dead, Clover, GDM, Total
    # Constraints in matrix form: C @ Y = 0
    # [1, 0, 1, -1, 0] @ [G, D, C, GDM, T]^T = G + C - GDM = 0
    # [0, 1, 0, 1, -1] @ [G, D, C, GDM, T]^T = D + GDM - T = 0

    C = np.array([
        [1, 0, 1, -1,  0],   # Green + Clover - GDM = 0
        [0, 1, 0,  1, -1]    # Dead + GDM - Total = 0
    ], dtype=np.float64)

    # Projection matrix: P = I - C^T @ (C @ C^T)^{-1} @ C
    C_T = C.T
    inv_CCt = np.linalg.inv(C @ C_T)
    P = np.eye(5) - C_T @ inv_CCt @ C

    # Project each prediction
    Y = preds.T  # (5, N)
    Y_proj = P @ Y
    Y_proj = Y_proj.T  # (N, 5)

    # Clip to non-negative
    Y_proj = np.clip(Y_proj, 0, None)

    return Y_proj

#%% [markdown]
# ## Section 9: Main Training Loop

#%%
def train_fold(fold: int, train_df: pd.DataFrame, cfg: CFG):
    """Train one fold."""
    print(f"\n{'='*60}")
    print(f"Training Fold {fold}")
    print(f"{'='*60}")

    # Split data
    train_data = train_df[train_df['fold'] != fold].reset_index(drop=True)
    val_data = train_df[train_df['fold'] == fold].reset_index(drop=True)

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Datasets
    train_dataset = BiomassDataset(
        train_data, cfg,
        transforms=get_transforms(cfg, 'train'),
        mode='train'
    )
    val_dataset = BiomassDataset(
        val_data, cfg,
        transforms=get_transforms(cfg, 'val'),
        mode='val'
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    # Model
    model = BiomassModel(cfg).to(cfg.device)

    # Freeze backbone initially
    if cfg.freeze_backbone_epochs > 0:
        model.freeze_backbone()
        print(f"Backbone frozen for first {cfg.freeze_backbone_epochs} epochs")

    # Optimizer with differential learning rates
    backbone_params = list(model.backbone.parameters())
    other_params = [p for n, p in model.named_parameters() if 'backbone' not in n]

    optimizer = AdamW([
        {'params': backbone_params, 'lr': cfg.backbone_lr},
        {'params': other_params, 'lr': cfg.lr}
    ], weight_decay=cfg.weight_decay)

    # Scheduler
    num_training_steps = len(train_loader) * cfg.epochs // cfg.gradient_accumulation
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[cfg.backbone_lr * 10, cfg.lr],
        total_steps=num_training_steps,
        pct_start=cfg.warmup_epochs / cfg.epochs,
        anneal_strategy='cos'
    )

    # Loss & Scaler
    criterion = WeightedBiomassLoss(physics_weight=0.1).to(cfg.device)
    scaler = torch.amp.GradScaler('cuda') if cfg.mixed_precision else None

    # Training history
    best_score = -float('inf')
    history = {'train_loss': [], 'val_loss': [], 'cv_score': []}

    for epoch in range(cfg.epochs):
        # Unfreeze backbone after initial epochs
        if epoch == cfg.freeze_backbone_epochs and cfg.freeze_backbone_epochs > 0:
            model.unfreeze_backbone()
            print(f"\nBackbone unfrozen at epoch {epoch + 1}")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, cfg, scaler, epoch
        )

        # Validate
        val_loss, cv_score, r2_scores, val_preds = validate(
            model, val_loader, criterion, cfg
        )

        # Apply post-processing and re-evaluate
        val_preds_pp = post_process_biomass(val_preds)
        val_targets = np.array([
            val_dataset[i]['targets'].numpy() for i in range(len(val_dataset))
        ])
        cv_score_pp = competition_metric(val_targets, val_preds_pp)

        # Log
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['cv_score'].append(cv_score_pp)

        print(f"\nEpoch {epoch + 1}/{cfg.epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"CV Score (raw): {cv_score:.4f}")
        print(f"CV Score (post-processed): {cv_score_pp:.4f}")
        print("Per-target R²:")
        for name, r2 in r2_scores.items():
            weight = TARGET_WEIGHTS[name]
            print(f"  {name} (w={weight}): {r2:.4f}")

        # Save best model
        if cv_score_pp > best_score:
            best_score = cv_score_pp
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_score': best_score,
                'r2_scores': r2_scores,
                'config': {
                    'backbone': cfg.backbone,
                    'embed_dim': cfg.embed_dim,
                    'num_heads': cfg.num_heads,
                    'dropout': cfg.dropout,
                }
            }, cfg.OUTPUT_DIR / f'best_model_fold{fold}.pt')
            print(f"✓ Saved best model (CV: {best_score:.4f})")

    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_title('Loss Curve')

    axes[1].plot(history['cv_score'], label='CV Score', color='green')
    axes[1].axhline(y=best_score, color='r', linestyle='--', label=f'Best: {best_score:.4f}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Competition Metric')
    axes[1].legend()
    axes[1].set_title('CV Score (Post-processed)')

    plt.tight_layout()
    plt.savefig(cfg.OUTPUT_DIR / f'training_history_fold{fold}.png', dpi=150)
    plt.show()

    flush()
    return best_score

#%%
# Train all folds
all_scores = []
for fold in cfg.train_folds:
    best_score = train_fold(fold, train_df, cfg)
    all_scores.append(best_score)

print(f"\n{'='*60}")
print(f"Training Complete!")
print(f"Mean CV Score: {np.mean(all_scores):.4f} ± {np.std(all_scores):.4f}")
print(f"Scores per fold: {all_scores}")
print(f"{'='*60}")

#%% [markdown]
# ## Section 10: Inference & Submission (Local Test)

#%%
@torch.no_grad()
def inference_with_tta(model, loader, cfg, n_tta: int = 4) -> np.ndarray:
    """Inference with Test-Time Augmentation."""
    model.eval()
    all_preds = []

    tta_transforms = [
        lambda x: x,                                    # Original
        lambda x: torch.flip(x, dims=[3]),             # HFlip
        lambda x: torch.flip(x, dims=[2]),             # VFlip
        lambda x: torch.flip(x, dims=[2, 3]),          # HFlip + VFlip
    ][:n_tta]

    for batch in tqdm(loader, desc='Inference'):
        left = batch['left'].to(cfg.device)
        right = batch['right'].to(cfg.device)

        batch_preds = []

        for tta in tta_transforms:
            left_aug = tta(left)
            right_aug = tta(right)

            with torch.amp.autocast('cuda', enabled=cfg.mixed_precision):
                outputs = model(left_aug, right_aug)
                batch_preds.append(outputs['all'].cpu())

        # Average TTA predictions
        batch_pred = torch.stack(batch_preds).mean(dim=0)
        all_preds.append(batch_pred.numpy())

    return np.concatenate(all_preds, axis=0)

#%%
def create_submission(model, test_df: pd.DataFrame, cfg: CFG, fold: int = 0) -> pd.DataFrame:
    """Create submission from test data."""

    # Test dataset
    test_wide = test_df.drop_duplicates(subset=['image_path']).reset_index(drop=True)
    test_wide['image_id'] = test_wide['image_path'].apply(lambda x: Path(x).stem)

    # Add dummy targets for dataset
    for t in TARGET_ORDER:
        if t not in test_wide.columns:
            test_wide[t] = 0.0

    test_dataset = BiomassDataset(
        test_wide, cfg,
        transforms=get_transforms(cfg, 'val'),
        mode='test'
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    # Inference
    preds = inference_with_tta(model, test_loader, cfg, n_tta=4)

    # Post-process
    preds = post_process_biomass(preds)

    # Create submission
    test_wide[TARGET_ORDER] = preds
    submission = melt_table(test_wide)

    return submission[['sample_id', 'target']]

#%%
# Create submission using best fold 0 model
if (cfg.OUTPUT_DIR / 'best_model_fold0.pt').exists():
    # Load model
    checkpoint = torch.load(cfg.OUTPUT_DIR / 'best_model_fold0.pt', weights_only=False)
    model = BiomassModel(cfg).to(cfg.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model with CV score: {checkpoint['best_score']:.4f}")

    # Load test data
    test_df = pd.read_csv(cfg.DATA_PATH / 'test.csv')

    # Create submission
    submission = create_submission(model, test_df, cfg, fold=0)
    submission.to_csv(cfg.OUTPUT_DIR / 'submission.csv', index=False)
    print(f"Submission saved: {len(submission)} rows")
    print(submission.head(10))
else:
    print("No trained model found. Run training first.")

#%% [markdown]
# ## Section 11: Save for Kaggle Upload
#
# After training, upload the following to Kaggle Datasets:
# - `best_model_fold0.pt` (and other folds if trained)
#
# Then use `04_inference_kaggle.py` for submission.

#%%
print(f"""
{'='*60}
Training Pipeline Complete!
{'='*60}

Files saved in {cfg.OUTPUT_DIR}:
- best_model_fold*.pt (model weights)
- training_history_fold*.png (loss curves)
- submission.csv (local test submission)

Next steps:
1. Upload model weights to Kaggle Datasets
2. Use 04_inference_kaggle.py for final submission
{'='*60}
""")
