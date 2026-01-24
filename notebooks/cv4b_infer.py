#%% [markdown]
# # 🚀 CV4b: 3-Model Ensemble + TTA + WA Postprocessing
#
# **Phase 1b 목표**: Timeout 해결 + 0.72
#
# **핵심 변경사항**:
# 1. ⭐ 3-모델 앙상블 (v20, v23, v26만) - Timeout 방지
# 2. 4-fold TTA (Original, HFlip, VFlip, Both)
# 3. ⭐ WA State Dead=0 후처리

#%%
import warnings
warnings.filterwarnings('ignore')

import os
import gc
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

import timm

tqdm.pandas()

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

#%%
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

seed_everything(42)

#%% [markdown]
# ## ⚙️ Configuration

#%%
class CFG:
    DATA_PATH = Path("/kaggle/input/csiro-biomass")
    BACKBONE_WEIGHTS = Path("/kaggle/input/pretrained-weights-biomass/dinov3_large/dinov3_large/dinov3_vitl16_qkvb.pth")
    
    # 각 버전 모델 경로 (핵심 3개만 - timeout 방지)
    MODELS = {
        'v20': Path("/kaggle/input/csiro-v20-models"),
        # 'v22': Path("/kaggle/input/csiro-v22-models"),  # 제외 (다른 구조)
        'v23': Path("/kaggle/input/csiro-v23-models"),
        # 'v25': Path("/kaggle/input/csiro-v25-models"),  # 제외 (VegEncoder)
        'v26': Path("/kaggle/input/csiro-v26-models"),
    }
    
    model_name = "vit_large_patch16_dinov3_qkvb.lvd1689m"
    img_size = (512, 512)
    
    # Base model config
    hidden_dim = 512
    num_layers = 3
    dropout = 0.1
    use_layernorm = True
    veg_feat_dim = 128  # v25용
    
    batch_size = 8  # TTA용으로 줄임
    num_workers = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ⭐ TTA 설정
    use_tta = True
    n_tta = 4
    
    # ⭐ WA 후처리
    use_wa_postprocess = True

cfg = CFG()

TARGET_ORDER = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

print(f"Device: {cfg.device}")
print(f"TTA: {cfg.use_tta} ({cfg.n_tta}-fold)")
print(f"WA Postprocess: {cfg.use_wa_postprocess}")

#%% [markdown]
# ## ⭐ WA Postprocessing

#%%
def postprocess_wa(preds, test_df):
    """WA State 후처리: Dead = 0 강제"""
    preds = preds.copy()
    wa_count = 0
    
    for idx in range(len(test_df)):
        row = test_df.iloc[idx]
        state = row.get('State', None)
        
        if state == 'WA':
            wa_count += 1
            preds[idx, 1] = 0.0  # Dead = 0
            
            # GDM과 Total 재계산
            green = preds[idx, 0]
            clover = preds[idx, 2]
            preds[idx, 3] = green + clover
            preds[idx, 4] = green + clover
    
    if wa_count > 0:
        print(f"✓ WA samples: {wa_count} (Dead forced to 0)")
    
    return preds

#%% [markdown]
# ## 📊 Dataset

#%%
class TestDataset(Dataset):
    def __init__(self, df, cfg, transform=None):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.cfg.DATA_PATH / row['image_path']).convert('RGB')
        width, height = img.size
        mid = width // 2
        
        left_img = img.crop((0, 0, mid, height))
        right_img = img.crop((mid, 0, width, height))
        
        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)
        
        return left_img, right_img, row['sample_id_prefix']


def compute_vegetation_indices(img_array):
    """RGB 이미지에서 Vegetation Index 계산"""
    img = img_array.astype(np.float32) / 255.0
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    
    exg = 2*g - r - b
    exg = (exg + 2) / 4
    
    gr_ratio = g / (r + 1e-8)
    gr_ratio = np.clip(gr_ratio, 0, 3) / 3
    
    return np.stack([exg, gr_ratio], axis=-1)


class TestDatasetV25(Dataset):
    """v25용 Dataset (Vegetation Index 이미지 포함)"""
    def __init__(self, df, cfg, transform=None):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.cfg.DATA_PATH / row['image_path']).convert('RGB')
        width, height = img.size
        mid = width // 2
        
        left_pil = img.crop((0, 0, mid, height))
        right_pil = img.crop((mid, 0, width, height))
        
        # Resize & numpy
        left_pil = left_pil.resize(self.cfg.img_size)
        right_pil = right_pil.resize(self.cfg.img_size)
        
        left_np = np.array(left_pil)
        right_np = np.array(right_pil)
        
        # Vegetation Index
        left_veg = compute_vegetation_indices(left_np)
        right_veg = compute_vegetation_indices(right_np)
        
        # RGB Transform
        if self.transform:
            left_rgb = self.transform(left_pil)
            right_rgb = self.transform(right_pil)
        
        # Veg to Tensor
        left_veg = torch.from_numpy(left_veg).permute(2, 0, 1).float()
        right_veg = torch.from_numpy(right_veg).permute(2, 0, 1).float()
        
        return left_rgb, right_rgb, left_veg, right_veg, row['sample_id_prefix']


def get_test_transform(cfg):
    return T.Compose([
        T.Resize(cfg.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_test_transform_v25():
    """v25용 Transform (Resize 없음 - Dataset에서 이미 resize)"""
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

#%% [markdown]
# ## 🧠 Models

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


class CSIROModelBase(nn.Module):
    """v20/v23/v26 모델"""
    def __init__(self, cfg, backbone_weights_path=None):
        super().__init__()
        
        if backbone_weights_path and Path(backbone_weights_path).exists():
            self.backbone = timm.create_model(cfg.model_name, pretrained=False, 
                                               num_classes=0, global_pool='avg')
            state = torch.load(backbone_weights_path, map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(state, strict=False)
        else:
            self.backbone = timm.create_model(cfg.model_name, pretrained=True, 
                                               num_classes=0, global_pool='avg')
        
        feat_dim = self.backbone.num_features
        combined_dim = feat_dim * 2
        
        self.film = FiLM(feat_dim)
        
        self.head_green = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers, 
                                    cfg.dropout, cfg.use_layernorm)
        self.head_clover = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers,
                                     cfg.dropout, cfg.use_layernorm)
        self.head_dead = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers,
                                   cfg.dropout, cfg.use_layernorm)
        
        self.head_height = nn.Sequential(
            nn.Linear(combined_dim, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(256, 1)
        )
        self.head_ndvi = nn.Sequential(
            nn.Linear(combined_dim, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(256, 1)
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
        
        return torch.cat([green, dead, clover, gdm, total], dim=1)


class CSIROModelV22(nn.Module):
    """v22 모델 (hidden_dim=256, num_layers=2, no aux)"""
    def __init__(self, cfg, backbone_weights_path=None):
        super().__init__()
        
        if backbone_weights_path and Path(backbone_weights_path).exists():
            self.backbone = timm.create_model(cfg.model_name, pretrained=False, 
                                               num_classes=0, global_pool='avg')
            state = torch.load(backbone_weights_path, map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(state, strict=False)
        else:
            self.backbone = timm.create_model(cfg.model_name, pretrained=True, 
                                               num_classes=0, global_pool='avg')
        
        feat_dim = self.backbone.num_features
        combined_dim = feat_dim * 2
        
        self.film = FiLM(feat_dim)
        
        # v22 specific: hidden_dim=256, num_layers=2
        hidden_dim_v22 = 256
        num_layers_v22 = 2
        
        self.head_green = make_head(combined_dim, hidden_dim_v22, num_layers_v22, 
                                    cfg.dropout, cfg.use_layernorm)
        self.head_clover = make_head(combined_dim, hidden_dim_v22, num_layers_v22,
                                     cfg.dropout, cfg.use_layernorm)
        self.head_dead = make_head(combined_dim, hidden_dim_v22, num_layers_v22,
                                   cfg.dropout, cfg.use_layernorm)
        
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


class VegetationEncoder(nn.Module):
    """v25 Vegetation Encoder (Conv2d 기반)"""
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


class CSIROModelV25(nn.Module):
    """v25 모델 (Vegetation Index Late Fusion - Conv2d 기반)"""
    def __init__(self, cfg, backbone_weights_path=None):
        super().__init__()
        
        if backbone_weights_path and Path(backbone_weights_path).exists():
            self.backbone = timm.create_model(cfg.model_name, pretrained=False, 
                                               num_classes=0, global_pool='avg')
            state = torch.load(backbone_weights_path, map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(state, strict=False)
        else:
            self.backbone = timm.create_model(cfg.model_name, pretrained=True, 
                                               num_classes=0, global_pool='avg')
        
        feat_dim = self.backbone.num_features
        
        self.veg_encoder = VegetationEncoder(in_channels=2, out_dim=cfg.veg_feat_dim)
        self.film = FiLM(feat_dim)
        
        combined_dim = feat_dim * 2 + cfg.veg_feat_dim * 2
        
        self.head_green = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers,
                                    cfg.dropout, cfg.use_layernorm)
        self.head_clover = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers,
                                     cfg.dropout, cfg.use_layernorm)
        self.head_dead = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers,
                                   cfg.dropout, cfg.use_layernorm)
        
        self.head_height = nn.Sequential(
            nn.Linear(combined_dim, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(256, 1)
        )
        self.head_ndvi = nn.Sequential(
            nn.Linear(combined_dim, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(256, 1)
        )
        
        self.softplus = nn.Softplus(beta=1.0)
    
    def forward(self, left_rgb, right_rgb, left_veg, right_veg):
        left_feat = self.backbone(left_rgb)
        right_feat = self.backbone(right_rgb)
        
        left_veg_feat = self.veg_encoder(left_veg)
        right_veg_feat = self.veg_encoder(right_veg)
        
        context = (left_feat + right_feat) / 2
        gamma, beta = self.film(context)
        
        left_mod = left_feat * (1 + gamma) + beta
        right_mod = right_feat * (1 + gamma) + beta
        
        combined = torch.cat([left_mod, right_mod, left_veg_feat, right_veg_feat], dim=1)
        
        green = self.softplus(self.head_green(combined))
        clover = self.softplus(self.head_clover(combined))
        dead = self.softplus(self.head_dead(combined))
        
        gdm = green + clover
        total = gdm + dead
        
        return torch.cat([green, dead, clover, gdm, total], dim=1)

#%% [markdown]
# ## ⭐ TTA Prediction Functions

#%%
@torch.no_grad()
def apply_tta(img, transform_type):
    """TTA 변환 적용"""
    if transform_type == 'original':
        return img
    elif transform_type == 'hflip':
        return torch.flip(img, [3])
    elif transform_type == 'vflip':
        return torch.flip(img, [2])
    elif transform_type == 'both':
        return torch.flip(img, [2, 3])
    return img


@torch.no_grad()
def predict_base_with_tta(model, loader, device):
    """Base 모델 TTA 예측"""
    model.eval()
    all_outputs, all_ids = [], []
    
    tta_transforms = ['original', 'hflip', 'vflip', 'both']
    
    for left, right, ids in tqdm(loader, desc="TTA Prediction"):
        batch_preds = []
        
        for t in tta_transforms:
            l = apply_tta(left, t).to(device)
            r = apply_tta(right, t).to(device)
            pred = model(l, r)
            batch_preds.append(pred.cpu())
        
        # TTA 평균
        mean_pred = torch.stack(batch_preds).mean(0)
        all_outputs.append(mean_pred.numpy())
        all_ids.extend(ids)
    
    return np.concatenate(all_outputs), all_ids


@torch.no_grad()
def predict_v25_with_tta(model, loader, device):
    """v25 모델 TTA 예측 (4-input)"""
    model.eval()
    all_outputs, all_ids = [], []
    
    tta_transforms = ['original', 'hflip', 'vflip', 'both']
    
    for left_rgb, right_rgb, left_veg, right_veg, ids in tqdm(loader, desc="v25 TTA Prediction"):
        batch_preds = []
        
        for t in tta_transforms:
            l_rgb = apply_tta(left_rgb, t).to(device)
            r_rgb = apply_tta(right_rgb, t).to(device)
            l_veg = apply_tta(left_veg, t).to(device)
            r_veg = apply_tta(right_veg, t).to(device)
            pred = model(l_rgb, r_rgb, l_veg, r_veg)
            batch_preds.append(pred.cpu())
        
        mean_pred = torch.stack(batch_preds).mean(0)
        all_outputs.append(mean_pred.numpy())
        all_ids.extend(ids)
    
    return np.concatenate(all_outputs), all_ids


@torch.no_grad()
def predict_base_no_tta(model, loader, device):
    """Base 모델 TTA 없이 예측"""
    model.eval()
    all_outputs, all_ids = [], []
    
    for left, right, ids in tqdm(loader, desc="Prediction"):
        left, right = left.to(device), right.to(device)
        pred = model(left, right)
        all_outputs.append(pred.cpu().numpy())
        all_ids.extend(ids)
    
    return np.concatenate(all_outputs), all_ids


@torch.no_grad()
def predict_v25_no_tta(model, loader, device):
    """v25 모델 TTA 없이 예측 (4-input)"""
    model.eval()
    all_outputs, all_ids = [], []
    
    for left_rgb, right_rgb, left_veg, right_veg, ids in tqdm(loader, desc="v25 Prediction"):
        left_rgb = left_rgb.to(device)
        right_rgb = right_rgb.to(device)
        left_veg = left_veg.to(device)
        right_veg = right_veg.to(device)
        pred = model(left_rgb, right_rgb, left_veg, right_veg)
        all_outputs.append(pred.cpu().numpy())
        all_ids.extend(ids)
    
    return np.concatenate(all_outputs), all_ids

#%% [markdown]
# ## 🔮 Version-specific Prediction

#%%
def predict_version(version, model_dir, test_df, cfg, device, use_tta=True):
    """버전별 앙상블 예측 (TTA 옵션)"""
    model_files = sorted(model_dir.glob("model_fold*.pth"))
    if not model_files:
        print(f"⚠️ No models for {version}")
        return None, None
    
    print(f"\n=== {version}: {len(model_files)} models (TTA={use_tta}) ===")
    
    all_fold_preds = []
    final_ids = None
    
    transform = get_test_transform(cfg)
    
    if version == 'v25':
        transform_v25 = get_test_transform_v25()
        dataset = TestDatasetV25(test_df, cfg, transform_v25)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                           num_workers=cfg.num_workers, pin_memory=True)
        
        for model_file in model_files:
            print(f"  Loading {model_file.name}...")
            model = CSIROModelV25(cfg, cfg.BACKBONE_WEIGHTS).to(device)
            model.load_state_dict(torch.load(model_file, map_location=device))
            
            if use_tta:
                preds, ids = predict_v25_with_tta(model, loader, device)
            else:
                preds, ids = predict_v25_no_tta(model, loader, device)
            
            all_fold_preds.append(preds)
            if final_ids is None:
                final_ids = ids
            
            del model
            gc.collect()
            torch.cuda.empty_cache()
    
    elif version == 'v22':
        dataset = TestDataset(test_df, cfg, transform)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                           num_workers=cfg.num_workers, pin_memory=True)
        
        for model_file in model_files:
            print(f"  Loading {model_file.name}...")
            model = CSIROModelV22(cfg, cfg.BACKBONE_WEIGHTS).to(device)
            model.load_state_dict(torch.load(model_file, map_location=device))
            
            if use_tta:
                preds, ids = predict_base_with_tta(model, loader, device)
            else:
                preds, ids = predict_base_no_tta(model, loader, device)
            
            all_fold_preds.append(preds)
            if final_ids is None:
                final_ids = ids
            
            del model
            gc.collect()
            torch.cuda.empty_cache()
    
    else:  # v20, v23, v26
        dataset = TestDataset(test_df, cfg, transform)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                           num_workers=cfg.num_workers, pin_memory=True)
        
        for model_file in model_files:
            print(f"  Loading {model_file.name}...")
            model = CSIROModelBase(cfg, cfg.BACKBONE_WEIGHTS).to(device)
            model.load_state_dict(torch.load(model_file, map_location=device))
            
            if use_tta:
                preds, ids = predict_base_with_tta(model, loader, device)
            else:
                preds, ids = predict_base_no_tta(model, loader, device)
            
            all_fold_preds.append(preds)
            if final_ids is None:
                final_ids = ids
            
            del model
            gc.collect()
            torch.cuda.empty_cache()
    
    return np.mean(all_fold_preds, axis=0), final_ids

#%% [markdown]
# ## 📋 Main

#%%
# 데이터 로드
test_df = pd.read_csv(cfg.DATA_PATH / "test.csv")
test_df['sample_id_prefix'] = test_df['sample_id'].str.split('__').str[0]
test_wide = test_df.drop_duplicates(subset=['image_path']).reset_index(drop=True)
print(f"Test samples: {len(test_wide)}")

# State 확인
if 'State' in test_wide.columns:
    print(f"✓ State column found")
    wa_count = len(test_wide[test_wide['State'] == 'WA'])
    print(f"  WA samples: {wa_count}")
else:
    print("⚠️ State column not found")

#%%
print("\n" + "="*60)
print("🚀 CV4: Multi-Model Ensemble + TTA + WA Postprocessing")
print("="*60)
print(f"TTA: {cfg.n_tta}-fold")
print(f"WA Postprocess: {cfg.use_wa_postprocess}")

#%%
# 각 버전별 예측
predictions = {}

for version, model_dir in cfg.MODELS.items():
    if model_dir.exists():
        pred, ids = predict_version(
            version, model_dir, test_wide, cfg, cfg.device, 
            use_tta=cfg.use_tta
        )
        if pred is not None:
            predictions[version] = pred
    else:
        print(f"⚠️ {version} models not found: {model_dir}")

print(f"\n✓ Loaded {len(predictions)} versions")

#%%
# Simple Average 앙상블
valid_preds = [v for v in predictions.values() if v is not None]
ensemble_pred = np.mean(valid_preds, axis=0)
print(f"Ensemble shape: {ensemble_pred.shape}")

#%%
# WA 후처리
if cfg.use_wa_postprocess and 'State' in test_wide.columns:
    ensemble_pred = postprocess_wa(ensemble_pred, test_wide)

#%%
# 예측 통계
print("\n=== Prediction Statistics ===")
print(f"{'Target':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
for idx, target in enumerate(TARGET_ORDER):
    vals = ensemble_pred[:, idx]
    print(f"{target:<15} {vals.mean():>10.2f} {vals.std():>10.2f} {vals.min():>10.2f} {vals.max():>10.2f}")

#%%
# Submission 생성
sample_ids = [test_wide.iloc[i]['sample_id_prefix'] for i in range(len(test_wide))]

pred_df = pd.DataFrame(ensemble_pred, columns=TARGET_ORDER)
pred_df['sample_id_prefix'] = sample_ids

sub_df = pred_df.melt(
    id_vars=['sample_id_prefix'],
    value_vars=TARGET_ORDER,
    var_name='target_name',
    value_name='target'
)
sub_df['sample_id'] = sub_df['sample_id_prefix'] + '__' + sub_df['target_name']

submission = sub_df[['sample_id', 'target']]
submission.to_csv('submission.csv', index=False)

# 검증
sample_sub = pd.read_csv(cfg.DATA_PATH / "sample_submission.csv")
assert len(submission) == len(sample_sub), "Format mismatch!"

print(f"\n✅ submission.csv saved")
print(f"   {len(submission)} rows")
print(f"   Models: {list(predictions.keys())}")
print(f"   TTA: {cfg.n_tta}-fold")
print(f"   WA Postprocess: {cfg.use_wa_postprocess}")
