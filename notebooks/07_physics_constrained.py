#%% [markdown]
# # 🏆 Physics-Constrained Biomass Prediction
#
# **Breakthrough Ideas:**
# 1. Physics-Constrained Head: Only predict 3 targets (Green, Clover, Dead)
#    - GDM = Green + Clover (calculated)
#    - Total = GDM + Dead (calculated)
# 2. Multi-Modal: Image + Tabular features (NDVI, Height)
# 3. Pretrained EfficientNet-B4 backbone

#%%
import os
import gc
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import cv2
from tqdm.auto import tqdm
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

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
# ## Configuration

#%%
class CFG:
    # === Kaggle Paths ===
    DATA_PATH = Path("/kaggle/input/csiro-biomass")
    OUTPUT_DIR = Path("/kaggle/working")
    
    # Pretrained weights path (Kaggle Dataset)
    # 📌 이 경로는 본인의 Dataset 경로로 변경하세요!
    WEIGHTS_PATH = Path("/kaggle/input/pretrained-weights-biomass")
    
    # === Model ===
    backbone = "efficientnet_b4"  # 또는 "dinov2_vitb14"
    input_size = 384
    
    # === Training ===
    n_folds = 5
    epochs = 15
    # GPU 개수에 따라 batch_size 자동 조정
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    batch_size = 8 * n_gpus  # T4 2개면 16
    lr = 1e-4 * n_gpus  # Linear scaling rule
    weight_decay = 1e-4
    
    # === Multi-Modal ===
    use_tabular = True  # Tabular features 사용 여부
    tabular_cols = ['Pre_GSHH_NDVI', 'Height_Ave_cm']
    
    # === Misc ===
    seed = 42
    num_workers = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # === Targets (물리적 제약 조건 기반) ===
    # 예측할 독립 변수: Green, Clover, Dead
    # 계산되는 변수: GDM = Green + Clover, Total = GDM + Dead
    independent_targets = ['Dry_Green_g', 'Dry_Clover_g', 'Dry_Dead_g']
    all_targets = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

cfg = CFG()
cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {cfg.device}")
print(f"Use tabular: {cfg.use_tabular}")

#%% [markdown]
# ## Competition Metric

#%%
TARGET_WEIGHTS = {
    'Dry_Green_g': 0.1,
    'Dry_Dead_g': 0.1,
    'Dry_Clover_g': 0.1,
    'GDM_g': 0.2,
    'Dry_Total_g': 0.5,  # 이것이 50%!
}
TARGET_ORDER = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

def competition_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Globally weighted R² (대회 평가 지표)"""
    weights = np.array([TARGET_WEIGHTS[t] for t in TARGET_ORDER])
    
    # Weighted mean
    y_weighted_mean = sum(y_true[:, i].mean() * weights[i] for i in range(5))
    
    # SS_res and SS_tot
    ss_res = sum(((y_true[:, i] - y_pred[:, i]) ** 2).mean() * weights[i] for i in range(5))
    ss_tot = sum(((y_true[:, i] - y_weighted_mean) ** 2).mean() * weights[i] for i in range(5))
    
    return 1 - ss_res / (ss_tot + 1e-8)

#%% [markdown]
# ## Data Preparation

#%%
def prepare_data(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """CSV를 wide format으로 변환"""
    if 'target' in df.columns:
        # Train data: pivot to wide
        df_wide = pd.pivot_table(
            df, values='target',
            index=['image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm'],
            columns='target_name', aggfunc='mean'
        ).reset_index()
    else:
        # Test data
        df['target'] = 0
        cols = ['image_path']
        # Check if tabular columns exist
        for col in ['Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm']:
            if col in df.columns:
                cols.append(col)
        
        df_wide = df.drop_duplicates(subset=['image_path'])[cols].reset_index(drop=True)
        
        # Add dummy targets
        for t in TARGET_ORDER:
            df_wide[t] = 0.0
    
    return df_wide

#%%
# Load data
train_df = pd.read_csv(cfg.DATA_PATH / "train.csv")
train_wide = prepare_data(train_df, is_train=True)
train_wide['image_id'] = train_wide['image_path'].apply(lambda x: Path(x).stem)

# KFold split
kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
train_wide['fold'] = -1
for fold, (_, val_idx) in enumerate(kf.split(train_wide)):
    train_wide.loc[val_idx, 'fold'] = fold

print(f"Train data shape: {train_wide.shape}")
print(f"Columns: {train_wide.columns.tolist()}")

# Check tabular features
if cfg.use_tabular:
    for col in cfg.tabular_cols:
        if col in train_wide.columns:
            print(f"  {col}: min={train_wide[col].min():.2f}, max={train_wide[col].max():.2f}")
        else:
            print(f"  ⚠️ {col} not found!")
            cfg.use_tabular = False

#%% [markdown]
# ## Augmentations

#%%
def get_transforms(mode: str = 'train', size: int = 384) -> A.Compose:
    if mode == 'train':
        return A.Compose([
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            # 도메인 특화 augmentation
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.3,
                hue=0.05,
                p=0.7
            ),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(size, size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

#%% [markdown]
# ## Dataset

#%%
class BiomassDataset(Dataset):
    def __init__(
        self, 
        df: pd.DataFrame, 
        cfg, 
        transforms=None, 
        mode: str = 'train',
        tabular_scaler: Optional[StandardScaler] = None
    ):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.transforms = transforms
        self.mode = mode
        self.tabular_scaler = tabular_scaler
        
        # Tabular features
        self.use_tabular = cfg.use_tabular and all(col in df.columns for col in cfg.tabular_cols)
        if self.use_tabular:
            tabular_data = df[cfg.tabular_cols].values.astype(np.float32)
            if self.tabular_scaler is not None:
                if mode == 'train':
                    self.tabular_data = self.tabular_scaler.fit_transform(tabular_data)
                else:
                    self.tabular_data = self.tabular_scaler.transform(tabular_data)
            else:
                self.tabular_data = tabular_data
        else:
            self.tabular_data = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self.cfg.DATA_PATH / row['image_path']
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            img = self.transforms(image=img)['image']
        
        # Targets (독립 변수만: Green, Clover, Dead)
        # 순서: [Green, Clover, Dead]
        targets = torch.tensor([
            row['Dry_Green_g'],
            row['Dry_Clover_g'],
            row['Dry_Dead_g']
        ], dtype=torch.float32)
        
        # Tabular features
        if self.use_tabular and self.tabular_data is not None:
            tabular = torch.tensor(self.tabular_data[idx], dtype=torch.float32)
            return img, tabular, targets
        else:
            return img, targets

#%% [markdown]
# ## 🔑 Physics-Constrained Model

#%%
class PhysicsConstrainedHead(nn.Module):
    """
    물리적 제약 조건을 만족하는 예측 헤드
    
    독립 변수: Green, Clover, Dead (3개)
    종속 변수: GDM = Green + Clover, Total = GDM + Dead
    
    ➡️ 5개 타겟 모두 물리적으로 일관성 있음!
    """
    def __init__(self, in_features: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        
        # 3개 독립 변수 예측
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)  # Green, Clover, Dead
        )
        
        # Softplus for non-negative outputs
        self.softplus = nn.Softplus()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            independent: [B, 3] - Green, Clover, Dead
            full: [B, 5] - Green, Dead, Clover, GDM, Total (대회 순서)
        """
        raw = self.head(x)
        independent = self.softplus(raw)  # 비음수 보장
        
        green = independent[:, 0:1]
        clover = independent[:, 1:2]
        dead = independent[:, 2:3]
        
        # 물리 법칙 적용 (Hard Constraint)
        gdm = green + clover
        total = gdm + dead
        
        # 대회 순서: [Green, Dead, Clover, GDM, Total]
        full = torch.cat([green, dead, clover, gdm, total], dim=1)
        
        return independent, full

#%%
class MultiModalBiomassModel(nn.Module):
    """
    Multi-Modal Model: Image + Tabular
    
    Features:
    1. Pretrained CNN backbone
    2. Tabular feature encoder with FiLM conditioning
    3. Physics-Constrained prediction head
    """
    def __init__(
        self, 
        backbone_name: str = "efficientnet_b4",
        n_tabular: int = 2,
        use_tabular: bool = True,
        dropout: float = 0.3,
        pretrained: bool = True,
        weights_path: Optional[str] = None
    ):
        super().__init__()
        
        self.use_tabular = use_tabular
        
        # === Image Backbone ===
        if pretrained and weights_path and Path(weights_path).exists():
            # Load from local weights
            self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
            weights = torch.load(weights_path, weights_only=True)
            # Remove classifier weights if present
            weights = {k: v for k, v in weights.items() if not k.startswith('classifier')}
            self.backbone.load_state_dict(weights, strict=False)
            print(f"✓ Loaded pretrained weights from {weights_path}")
        elif pretrained:
            # Try to load from timm (requires internet)
            try:
                self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
                print("✓ Loaded pretrained weights from timm")
            except:
                self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
                print("⚠️ Using random initialization (no pretrained weights)")
        else:
            self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        
        self.feat_dim = self.backbone.num_features
        
        # === Tabular Encoder (FiLM conditioning) ===
        if use_tabular:
            self.tabular_encoder = nn.Sequential(
                nn.Linear(n_tabular, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 128),
                nn.ReLU()
            )
            # FiLM: Feature-wise Linear Modulation
            self.film_gamma = nn.Linear(128, self.feat_dim)
            self.film_beta = nn.Linear(128, self.feat_dim)
        
        # === Physics-Constrained Head ===
        self.head = PhysicsConstrainedHead(
            in_features=self.feat_dim, 
            hidden_dim=256, 
            dropout=dropout
        )
    
    def forward(
        self, 
        image: torch.Tensor, 
        tabular: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image: [B, C, H, W]
            tabular: [B, n_tabular] (optional)
        
        Returns:
            independent: [B, 3] - Green, Clover, Dead
            full: [B, 5] - All 5 targets
        """
        # Image features
        img_feat = self.backbone(image)  # [B, feat_dim]
        
        # Tabular conditioning (FiLM)
        if self.use_tabular and tabular is not None:
            tab_feat = self.tabular_encoder(tabular)  # [B, 128]
            gamma = self.film_gamma(tab_feat)  # [B, feat_dim]
            beta = self.film_beta(tab_feat)    # [B, feat_dim]
            
            # FiLM modulation
            img_feat = img_feat * (1 + gamma) + beta
        
        # Physics-Constrained prediction
        independent, full = self.head(img_feat)
        
        return independent, full

#%% [markdown]
# ## Training

#%%
def train_one_epoch(
    model: nn.Module, 
    loader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    device: str,
    use_tabular: bool = False
) -> float:
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc='Train', leave=False):
        if use_tabular:
            imgs, tabular, targets = batch
            imgs = imgs.to(device)
            tabular = tabular.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            independent, _ = model(imgs, tabular)
        else:
            imgs, targets = batch
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            independent, _ = model(imgs)
        
        # MSE loss on independent variables
        loss = F.mse_loss(independent, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

@torch.no_grad()
def validate(
    model: nn.Module, 
    loader: DataLoader, 
    device: str,
    use_tabular: bool = False
) -> Tuple[float, float]:
    model.eval()
    all_preds, all_targets = [], []
    
    for batch in tqdm(loader, desc='Valid', leave=False):
        if use_tabular:
            imgs, tabular, targets = batch
            imgs = imgs.to(device)
            tabular = tabular.to(device)
            
            _, full_pred = model(imgs, tabular)
        else:
            imgs, targets = batch
            imgs = imgs.to(device)
            
            _, full_pred = model(imgs)
        
        all_preds.append(full_pred.cpu().numpy())
        
        # Reconstruct full targets from independent
        green = targets[:, 0:1]
        clover = targets[:, 1:2]
        dead = targets[:, 2:3]
        gdm = green + clover
        total = gdm + dead
        full_targets = torch.cat([green, dead, clover, gdm, total], dim=1)
        all_targets.append(full_targets.numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Competition metric
    cv_score = competition_metric(all_targets, all_preds)
    
    # MSE for logging
    mse = np.mean((all_preds - all_targets) ** 2)
    
    return mse, cv_score

#%%
def train_fold(fold: int, train_df: pd.DataFrame, cfg) -> float:
    print(f"\n{'='*60}")
    print(f"🚀 Training Fold {fold}")
    print(f"{'='*60}")
    
    # Split
    train_data = train_df[train_df['fold'] != fold].reset_index(drop=True)
    val_data = train_df[train_df['fold'] == fold].reset_index(drop=True)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Tabular scaler
    tabular_scaler = StandardScaler() if cfg.use_tabular else None
    
    # Datasets
    train_dataset = BiomassDataset(
        train_data, cfg, 
        get_transforms('train', cfg.input_size), 
        'train',
        tabular_scaler
    )
    val_dataset = BiomassDataset(
        val_data, cfg, 
        get_transforms('val', cfg.input_size), 
        'val',
        tabular_scaler
    )
    
    # Check if tabular is available
    use_tabular = train_dataset.use_tabular
    print(f"Using tabular features: {use_tabular}")
    
    # Loaders
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size * 2, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )
    
    # Model
    weights_path = None
    if cfg.WEIGHTS_PATH.exists():
        weights_path = str(cfg.WEIGHTS_PATH / cfg.backbone / f"{cfg.backbone}.pth")
    
    model = MultiModalBiomassModel(
        backbone_name=cfg.backbone,
        n_tabular=len(cfg.tabular_cols),
        use_tabular=use_tabular,
        dropout=0.3,
        pretrained=True,
        weights_path=weights_path
    )
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"🚀 Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    model = model.to(cfg.device)
    
    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    
    # Training loop
    best_score = -float('inf')
    
    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, cfg.device, use_tabular
        )
        val_mse, cv_score = validate(model, val_loader, cfg.device, use_tabular)
        scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{cfg.epochs} | LR: {lr:.6f} | "
              f"Train Loss: {train_loss:.4f} | Val MSE: {val_mse:.4f} | CV: {cv_score:.4f}")
        
        if cv_score > best_score:
            best_score = cv_score
            # Save model (handle DataParallel)
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save({
                'model_state_dict': model_to_save.state_dict(),
                'fold': fold,
                'score': best_score,
                'tabular_scaler': tabular_scaler,
                'use_tabular': use_tabular
            }, cfg.OUTPUT_DIR / f'best_model_fold{fold}.pt')
            print(f"  ✓ New best! Saved.")
    
    flush()
    return best_score

#%%
# Train all folds
fold_scores = []
for fold in range(cfg.n_folds):
    score = train_fold(fold, train_wide, cfg)
    fold_scores.append(score)
    print(f"Fold {fold} Best CV: {score:.4f}")

print(f"\n{'='*60}")
print(f"📊 Overall CV: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
print(f"{'='*60}")

#%% [markdown]
# ## Inference & Submission

#%%
@torch.no_grad()
def inference(
    models: list, 
    loader: DataLoader, 
    device: str,
    use_tabular: bool = False
) -> np.ndarray:
    all_preds = []
    
    for batch in tqdm(loader, desc='Inference'):
        if use_tabular:
            imgs, tabular, _ = batch
            imgs = imgs.to(device)
            tabular = tabular.to(device)
        else:
            imgs, _ = batch
            imgs = imgs.to(device)
            tabular = None
        
        # Ensemble prediction
        batch_preds = []
        for model in models:
            model.eval()
            if use_tabular:
                _, full_pred = model(imgs, tabular)
            else:
                _, full_pred = model(imgs)
            batch_preds.append(full_pred.cpu().numpy())
        
        # Average
        avg_pred = np.mean(batch_preds, axis=0)
        all_preds.append(avg_pred)
    
    return np.concatenate(all_preds)

#%%
# Load test data
test_df = pd.read_csv(cfg.DATA_PATH / "test.csv")
test_wide = prepare_data(test_df, is_train=False)

print(f"Test data: {len(test_wide)} images")
print(f"Test columns: {test_wide.columns.tolist()}")

#%%
# Load all fold models
models = []
use_tabular = False
tabular_scaler = None

for fold in range(cfg.n_folds):
    ckpt_path = cfg.OUTPUT_DIR / f'best_model_fold{fold}.pt'
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False)
        
        # Get config from first checkpoint
        if fold == 0:
            use_tabular = ckpt.get('use_tabular', False)
            tabular_scaler = ckpt.get('tabular_scaler', None)
        
        model = MultiModalBiomassModel(
            backbone_name=cfg.backbone,
            n_tabular=len(cfg.tabular_cols),
            use_tabular=use_tabular,
            pretrained=False
        ).to(cfg.device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        models.append(model)
        print(f"✓ Loaded fold {fold} (CV: {ckpt['score']:.4f})")

print(f"\nLoaded {len(models)} models")

#%%
# Test dataset
test_dataset = BiomassDataset(
    test_wide, cfg, 
    get_transforms('val', cfg.input_size), 
    'test',
    tabular_scaler
)
test_loader = DataLoader(
    test_dataset, batch_size=cfg.batch_size, shuffle=False,
    num_workers=cfg.num_workers, pin_memory=True
)

# Inference
preds = inference(models, test_loader, cfg.device, use_tabular)
print(f"Predictions shape: {preds.shape}")

#%%
# Create submission
def melt_table(df: pd.DataFrame) -> pd.DataFrame:
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

# Apply predictions
test_wide[TARGET_ORDER] = preds

# Clip to non-negative (should already be non-negative due to Softplus)
test_wide[TARGET_ORDER] = test_wide[TARGET_ORDER].clip(lower=0)

submission = melt_table(test_wide)
submission = submission[['sample_id', 'target']]

# Verify physics constraints
test_gdm = test_wide['Dry_Green_g'] + test_wide['Dry_Clover_g']
test_total = test_gdm + test_wide['Dry_Dead_g']
gdm_match = np.allclose(test_wide['GDM_g'], test_gdm)
total_match = np.allclose(test_wide['Dry_Total_g'], test_total)
print(f"\n✓ Physics constraint check:")
print(f"  GDM = Green + Clover: {gdm_match}")
print(f"  Total = GDM + Dead: {total_match}")

# Save
submission.to_csv(cfg.OUTPUT_DIR / 'submission.csv', index=False)
print(f"\n📄 Submission saved: {len(submission)} rows")
print(submission.head(10))

#%%
# Verification
print("\n=== Submission Verification ===")
print(f"Shape: {submission.shape}")
print(f"Columns: {submission.columns.tolist()}")
print(f"Null values: {submission.isnull().sum().sum()}")
print(f"Target range: [{submission['target'].min():.2f}, {submission['target'].max():.2f}]")

#%%
print(f"""
{'='*60}
🏆 Physics-Constrained Baseline Complete!
{'='*60}

Output: {cfg.OUTPUT_DIR / 'submission.csv'}
CV Score: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}

Key Features:
1. ✅ Physics-Constrained Head (3 independent, 2 derived)
2. ✅ {cfg.n_folds}-Fold Cross-Validation
3. ✅ Pretrained {cfg.backbone} backbone
4. ✅ Multi-Modal (Image + Tabular): {use_tabular}
5. ✅ Domain-specific augmentations

Next steps:
- Try DINOv2 backbone for potentially higher performance
- Add TTA (Test-Time Augmentation)
- Experiment with different loss functions
{'='*60}
""")
