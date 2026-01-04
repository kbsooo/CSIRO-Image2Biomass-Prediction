#%% [markdown]
# # CSIRO Image2Biomass - Kaggle Inference
#
# **Kaggle 환경 제약:**
# - 인터넷 OFF (weights는 Dataset으로 사전 업로드)
# - GPU 9시간 제한
# - 메모리 제약 → batch_size=1
#
# **Features:**
# - DINOv2-Large backbone
# - Bidirectional Cross-Attention (Left/Right fusion)
# - TTA (Test Time Augmentation)
# - Post-processing with physics constraints

#%% [markdown]
# ## Section 0: Setup

#%%
import os
import gc
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

#%%
def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

#%% [markdown]
# ## Section 1: Configuration

#%%
def get_paths():
    """Auto-detect paths based on environment."""
    # Kaggle
    kaggle_data = Path("/kaggle/input/csiro-crop-biomass-prediction")
    kaggle_weights = Path("/kaggle/input/csiro-biomass-weights")  # 공개 weights dataset
    kaggle_output = Path("/kaggle/working")

    # Local (for testing)
    local_data = Path("./data")
    local_weights = Path("./outputs")  # Local trained weights
    local_output = Path("./outputs")

    if kaggle_data.exists():
        return kaggle_data, kaggle_weights, kaggle_output
    else:
        return local_data, local_weights, local_output

@dataclass
class CFG:
    # Paths (auto-detect)
    DATA_PATH: Path = None
    WEIGHTS_PATH: Path = None
    OUTPUT_DIR: Path = None

    # Model (must match training config)
    backbone: str = "vit_large_patch14_dinov2.lvd142m"
    input_size: int = 518
    embed_dim: int = 1024
    num_heads: int = 16
    dropout: float = 0.0  # No dropout during inference

    # Inference
    batch_size: int = 1  # Memory-safe
    num_workers: int = 2
    use_tta: bool = True
    tta_scales: List[float] = None
    mixed_precision: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensemble folds
    ensemble_folds: List[int] = None

    def __post_init__(self):
        if self.DATA_PATH is None:
            self.DATA_PATH, self.WEIGHTS_PATH, self.OUTPUT_DIR = get_paths()
        if self.tta_scales is None:
            self.tta_scales = [0.9, 1.0, 1.1]
        if self.ensemble_folds is None:
            self.ensemble_folds = [0, 1, 2, 3, 4]

cfg = CFG()
print(f"Device: {cfg.device}")
print(f"Data path: {cfg.DATA_PATH}")
print(f"Weights path: {cfg.WEIGHTS_PATH}")

#%% [markdown]
# ## Section 2: Competition Constants

#%%
TARGET_ORDER = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

TARGET_WEIGHTS = {
    'Dry_Green_g': 0.1,
    'Dry_Dead_g': 0.1,
    'Dry_Clover_g': 0.1,
    'GDM_g': 0.2,
    'Dry_Total_g': 0.5,
}

# For denormalization
TARGET_MAX = {
    'Dry_Green_g': 157.9836,
    'Dry_Dead_g': 83.8407,
    'Dry_Clover_g': 71.7865,
    'GDM_g': 157.9836,
    'Dry_Total_g': 185.70,
}

#%% [markdown]
# ## Section 3: Model Architecture
#
# 학습 코드와 동일한 모델 정의 (weights 호환성)

#%%
class BiDirectionalCrossAttention(nn.Module):
    """
    Left ↔ Right 양방향 Cross-Attention.

    Left image features와 Right image features가 서로를 attend하여
    두 이미지 간의 상관관계를 학습.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Cross-attention: Left → Right
        self.q_left = nn.Linear(dim, dim)
        self.kv_right = nn.Linear(dim, dim * 2)

        # Cross-attention: Right → Left
        self.q_right = nn.Linear(dim, dim)
        self.kv_left = nn.Linear(dim, dim * 2)

        # Output projections
        self.proj_left = nn.Linear(dim, dim)
        self.proj_right = nn.Linear(dim, dim)

        # Learnable fusion tokens
        self.fuse_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """
        Args:
            left: (B, dim) left image features (CLS token)
            right: (B, dim) right image features (CLS token)
        Returns:
            fused: (B, dim) fused features
        """
        B = left.shape[0]

        # Reshape to sequence: (B, 1, dim)
        left = left.unsqueeze(1)
        right = right.unsqueeze(1)

        # Add fuse token: (B, 2, dim) each
        fuse = self.fuse_token.expand(B, -1, -1)
        left_seq = torch.cat([left, fuse], dim=1)
        right_seq = torch.cat([right, fuse.clone()], dim=1)

        # Cross-attention: Left attends to Right
        q_l = self.q_left(left_seq).reshape(B, 2, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv_r = self.kv_right(right_seq).reshape(B, 2, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k_r, v_r = kv_r[0], kv_r[1]

        attn_l = F.scaled_dot_product_attention(q_l, k_r, v_r, scale=self.scale)
        attn_l = attn_l.permute(0, 2, 1, 3).reshape(B, 2, self.dim)
        left_out = self.norm1(left_seq + self.dropout(self.proj_left(attn_l)))

        # Cross-attention: Right attends to Left
        q_r = self.q_right(right_seq).reshape(B, 2, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv_l = self.kv_left(left_seq).reshape(B, 2, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k_l, v_l = kv_l[0], kv_l[1]

        attn_r = F.scaled_dot_product_attention(q_r, k_l, v_l, scale=self.scale)
        attn_r = attn_r.permute(0, 2, 1, 3).reshape(B, 2, self.dim)
        right_out = self.norm2(right_seq + self.dropout(self.proj_right(attn_r)))

        # Fuse: average of both fuse tokens
        fused = (left_out[:, 1, :] + right_out[:, 1, :]) / 2

        return fused


class BiomassModel(nn.Module):
    """
    DINOv2-Large + Bidirectional Cross-Attention + 3-head prediction.

    Architecture:
    1. DINOv2-Large backbone (shared for left/right)
    2. Bidirectional cross-attention fusion
    3. 3 prediction heads (Green, Dead, Clover)
    4. Derive 2 targets (GDM = Green + Clover, Total = GDM + Dead)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # DINOv2-Large backbone
        self.backbone = timm.create_model(
            cfg.backbone,
            pretrained=True,
            num_classes=0,  # Remove classifier
        )

        # Get feature dimension
        self.feat_dim = self.backbone.num_features  # 1024 for large

        # Bidirectional cross-attention for L/R fusion
        self.cross_attn = BiDirectionalCrossAttention(
            dim=self.feat_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
        )

        # Prediction heads: only 3 (Green, Dead, Clover)
        # GDM and Total are derived
        self.head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Dropout(cfg.dropout),
            nn.Linear(self.feat_dim, self.feat_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(self.feat_dim // 2, 3),  # Green, Dead, Clover
            nn.Softplus(),  # Ensure positive outputs
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """
        Args:
            left: (B, 3, H, W) left image
            right: (B, 3, H, W) right image
        Returns:
            preds: (B, 5) [Green, Dead, Clover, GDM, Total]
        """
        # Extract features
        feat_left = self.backbone(left)   # (B, 1024)
        feat_right = self.backbone(right) # (B, 1024)

        # Fuse with cross-attention
        fused = self.cross_attn(feat_left, feat_right)  # (B, 1024)

        # Predict 3 targets
        pred_3 = self.head(fused)  # (B, 3) [Green, Dead, Clover]

        # Derive 2 targets
        green, dead, clover = pred_3[:, 0], pred_3[:, 1], pred_3[:, 2]
        gdm = green + clover
        total = gdm + dead

        # Stack all 5: [Green, Dead, Clover, GDM, Total]
        preds = torch.stack([green, dead, clover, gdm, total], dim=1)

        return preds

#%% [markdown]
# ## Section 4: Dataset

#%%
class TestDataset(Dataset):
    """Test dataset for inference."""

    def __init__(self, df: pd.DataFrame, cfg, transform=None):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        img_path = self.cfg.DATA_PATH / "test" / row['filename']
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Split left/right (2000x1000 → 1000x1000 x2)
        h, w = img.shape[:2]
        mid = w // 2
        left = img[:, :mid]
        right = img[:, mid:]

        # Apply transforms
        if self.transform:
            left = self.transform(image=left)['image']
            right = self.transform(image=right)['image']

        return {
            'left': left,
            'right': right,
            'index': row['index'],
        }


def get_test_transform(cfg, scale: float = 1.0):
    """Get test-time transform with optional scaling."""
    size = int(cfg.input_size * scale)
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

#%% [markdown]
# ## Section 5: TTA (Test Time Augmentation)

#%%
def apply_tta(model, left: torch.Tensor, right: torch.Tensor, cfg) -> torch.Tensor:
    """
    Apply TTA: flips + rotations + multi-scale.

    Returns:
        preds: (B, 5) averaged predictions
    """
    model.eval()
    all_preds = []

    with torch.inference_mode():
        # Original
        all_preds.append(model(left, right))

        if cfg.use_tta:
            # Horizontal flip
            all_preds.append(model(
                torch.flip(left, dims=[-1]),
                torch.flip(right, dims=[-1])
            ))

            # Vertical flip
            all_preds.append(model(
                torch.flip(left, dims=[-2]),
                torch.flip(right, dims=[-2])
            ))

            # H+V flip
            all_preds.append(model(
                torch.flip(left, dims=[-2, -1]),
                torch.flip(right, dims=[-2, -1])
            ))

            # Swap left/right (symmetry)
            all_preds.append(model(right, left))

            # Swap + H flip
            all_preds.append(model(
                torch.flip(right, dims=[-1]),
                torch.flip(left, dims=[-1])
            ))

    # Average all predictions
    return torch.stack(all_preds).mean(dim=0)

#%% [markdown]
# ## Section 6: Post-Processing

#%%
def post_process_biomass(preds: np.ndarray) -> np.ndarray:
    """
    Post-process predictions to satisfy physics constraints:
    - GDM = Green + Clover
    - Total = GDM + Dead = Green + Clover + Dead

    Uses linear algebra projection to nearest valid point.

    Args:
        preds: (N, 5) [Green, Dead, Clover, GDM, Total]
    Returns:
        corrected: (N, 5) satisfying constraints
    """
    # Constraint matrix C:
    # C @ x = 0 where x = [Green, Dead, Clover, GDM, Total]
    # GDM - Green - Clover = 0  → [1, 0, 1, -1, 0]
    # Total - Green - Dead - Clover = 0 → [1, 1, 1, 0, -1]
    C = np.array([
        [1, 0, 1, -1, 0],   # Green + Clover - GDM = 0
        [1, 1, 1, 0, -1],   # Green + Dead + Clover - Total = 0
    ], dtype=np.float64)

    # Projection matrix: P = I - C.T @ (C @ C.T)^{-1} @ C
    CTC_inv = np.linalg.inv(C @ C.T)
    P = np.eye(5) - C.T @ CTC_inv @ C

    # Project to constraint manifold
    corrected = (P @ preds.T).T

    # Clip to non-negative
    corrected = np.clip(corrected, 0, None)

    return corrected

#%% [markdown]
# ## Section 7: Inference Pipeline

#%%
@torch.inference_mode()
def run_inference(cfg) -> pd.DataFrame:
    """Run full inference pipeline with ensemble and TTA."""

    # Load test data
    sample_sub = pd.read_csv(cfg.DATA_PATH / "sample_submission.csv")

    # Get unique test images
    test_indices = sample_sub['index'].str.split('_').str[0].unique()
    test_df = pd.DataFrame({'index': test_indices})
    test_df['filename'] = test_df['index'] + '.jpg'

    print(f"Test samples: {len(test_df)}")

    # Prepare dataloader
    test_transform = get_test_transform(cfg)
    test_dataset = TestDataset(test_df, cfg, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # Load models for ensemble
    all_fold_preds = []

    for fold in cfg.ensemble_folds:
        weight_path = cfg.WEIGHTS_PATH / f"fold{fold}_best.pth"

        if not weight_path.exists():
            print(f"Skipping fold {fold}: weights not found at {weight_path}")
            continue

        print(f"\nLoading fold {fold} weights...")

        # Create model
        model = BiomassModel(cfg)

        # Load weights
        state_dict = torch.load(weight_path, map_location=cfg.device, weights_only=False)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        model.load_state_dict(state_dict)
        model.to(cfg.device)
        model.eval()

        # Inference
        fold_preds = []
        fold_indices = []

        # Use mixed precision
        with torch.amp.autocast(cfg.device, enabled=cfg.mixed_precision):
            for batch in tqdm(test_loader, desc=f"Fold {fold}"):
                left = batch['left'].to(cfg.device)
                right = batch['right'].to(cfg.device)
                indices = batch['index']

                # TTA
                preds = apply_tta(model, left, right, cfg)

                fold_preds.append(preds.cpu().numpy())
                fold_indices.extend(indices)

        fold_preds = np.concatenate(fold_preds, axis=0)
        all_fold_preds.append(fold_preds)

        # Cleanup
        del model
        flush()

    if len(all_fold_preds) == 0:
        raise ValueError("No fold weights found!")

    # Ensemble: average across folds
    ensemble_preds = np.mean(all_fold_preds, axis=0)

    # Post-process
    ensemble_preds = post_process_biomass(ensemble_preds)

    # Create submission dataframe
    result_df = pd.DataFrame({
        'index': fold_indices,
        **{target: ensemble_preds[:, i] for i, target in enumerate(TARGET_ORDER)}
    })

    return result_df

#%% [markdown]
# ## Section 8: Create Submission

#%%
def create_submission(result_df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Convert wide format to long format for submission.

    Input: index, Green, Dead, Clover, GDM, Total
    Output: index (e.g., 'IMG001_Dry_Green_g'), value
    """
    # Melt to long format
    submission = result_df.melt(
        id_vars=['index'],
        value_vars=TARGET_ORDER,
        var_name='target',
        value_name='value'
    )

    # Create submission index
    submission['index'] = submission['index'] + '_' + submission['target']
    submission = submission[['index', 'value']]

    # Sort to match sample submission
    sample_sub = pd.read_csv(cfg.DATA_PATH / "sample_submission.csv")
    submission = submission.set_index('index').loc[sample_sub['index']].reset_index()

    return submission

#%% [markdown]
# ## Section 9: Main Execution

#%%
# Run inference
print("=" * 50)
print("Starting inference pipeline...")
print("=" * 50)

result_df = run_inference(cfg)
print(f"\nPredictions shape: {result_df.shape}")
print(result_df.head())

#%%
# Create submission
submission = create_submission(result_df, cfg)
print(f"\nSubmission shape: {submission.shape}")
print(submission.head(10))

#%%
# Save submission
submission_path = cfg.OUTPUT_DIR / "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"\nSubmission saved to: {submission_path}")

#%%
# Summary statistics
print("\n" + "=" * 50)
print("Prediction Summary")
print("=" * 50)
for target in TARGET_ORDER:
    values = result_df[target]
    print(f"{target:15s}: mean={values.mean():7.2f}, std={values.std():6.2f}, "
          f"min={values.min():6.2f}, max={values.max():7.2f}")

#%%
# Verify constraints
print("\n" + "=" * 50)
print("Constraint Verification")
print("=" * 50)
green = result_df['Dry_Green_g'].values
dead = result_df['Dry_Dead_g'].values
clover = result_df['Dry_Clover_g'].values
gdm = result_df['GDM_g'].values
total = result_df['Dry_Total_g'].values

gdm_error = np.abs(gdm - (green + clover)).max()
total_error = np.abs(total - (green + dead + clover)).max()

print(f"GDM = Green + Clover: max error = {gdm_error:.6f}")
print(f"Total = Green + Dead + Clover: max error = {total_error:.6f}")

if gdm_error < 1e-5 and total_error < 1e-5:
    print("All constraints satisfied!")
else:
    print("WARNING: Constraint violations detected!")

print("\nDone!")
