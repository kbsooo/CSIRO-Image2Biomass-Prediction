#%% [markdown]
# # v26 Calibrated Inference
#
# **목적**: OOF Calibrator를 적용하여 Test 예측 보정
# **필요 파일**: 
# - model_fold{0-4}.pth (학습된 모델)
# - calibrators.pkl (Ridge calibrators)

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
import joblib

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
    
    # ⚠️ v26 모델 및 calibrator 경로
    MODELS_DIR = Path("/kaggle/input/csiro-v26-models")
    CALIBRATORS_PATH = Path("/kaggle/input/csiro-v26-models/calibrators.pkl")
    
    model_name = "vit_large_patch16_dinov3_qkvb.lvd1689m"
    img_size = (560, 560)
    
    hidden_dim = 512
    num_layers = 3
    dropout = 0.1
    use_layernorm = True
    
    batch_size = 16
    num_workers = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = CFG()

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


def get_tta_dataloaders(df, cfg):
    loaders = []
    base = [
        T.Resize(cfg.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    transforms_list = [
        T.Compose(base),
        T.Compose([T.Resize(cfg.img_size), T.RandomHorizontalFlip(p=1.0)] + base[1:]),
        T.Compose([T.Resize(cfg.img_size), T.RandomVerticalFlip(p=1.0)] + base[1:]),
    ]
    
    for transform in transforms_list:
        dataset = TestDataset(df, cfg, transform)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, 
                           num_workers=cfg.num_workers, pin_memory=True)
        loaders.append(loader)
    
    return loaders

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
# ## 🔮 Inference

#%%
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_outputs, all_ids = [], []
    
    for left, right, ids in tqdm(loader, desc="Predicting"):
        left = left.to(device)
        right = right.to(device)
        outputs = model(left, right)
        all_outputs.append(outputs.cpu().numpy())
        all_ids.extend(ids)
    
    return np.concatenate(all_outputs), all_ids


def predict_with_tta(model, tta_loaders, device):
    all_preds = []
    final_ids = None
    
    for loader in tta_loaders:
        preds, ids = predict(model, loader, device)
        all_preds.append(preds)
        if final_ids is None:
            final_ids = ids
    
    return np.mean(all_preds, axis=0), final_ids


def predict_ensemble(cfg, tta_loaders):
    all_fold_preds = []
    final_ids = None
    
    model_files = sorted(cfg.MODELS_DIR.glob("model_fold*.pth"))
    print(f"Found {len(model_files)} models")
    
    for model_file in model_files:
        print(f"\nLoading {model_file.name}...")
        
        model = CSIROModel(cfg, cfg.BACKBONE_WEIGHTS).to(cfg.device)
        model.load_state_dict(torch.load(model_file, map_location=cfg.device))
        print("✓ Loaded")
        
        preds, ids = predict_with_tta(model, tta_loaders, cfg.device)
        all_fold_preds.append(preds)
        
        if final_ids is None:
            final_ids = ids
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    return np.mean(all_fold_preds, axis=0), final_ids

#%% [markdown]
# ## 🔧 Apply Calibration

#%%
TARGET_ORDER = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

def apply_calibration(raw_preds, calibrators):
    """
    Raw predictions에 calibrator 적용
    
    Args:
        raw_preds: [N, 5] 원본 예측
        calibrators: dict of Ridge calibrators
    
    Returns:
        calibrated_preds: [N, 5] 보정된 예측
    """
    calibrated = np.zeros_like(raw_preds)
    
    for idx, target in enumerate(TARGET_ORDER):
        if target in calibrators:
            X = raw_preds[:, idx].reshape(-1, 1)
            calibrated[:, idx] = calibrators[target].predict(X)
        else:
            calibrated[:, idx] = raw_preds[:, idx]
    
    # 음수 값 클리핑 (biomass는 음수 불가)
    calibrated = np.maximum(calibrated, 0)
    
    return calibrated

#%% [markdown]
# ## 📋 Main

#%%
test_df = pd.read_csv(cfg.DATA_PATH / "test.csv")
test_df['sample_id_prefix'] = test_df['sample_id'].str.split('__').str[0]
test_wide = test_df.drop_duplicates(subset=['image_path']).reset_index(drop=True)
print(f"Test samples: {len(test_wide)}")

#%%
# Calibrator 로드
calibrators = joblib.load(cfg.CALIBRATORS_PATH)
print(f"\n✓ Loaded calibrators for: {list(calibrators.keys())}")

for target, cal in calibrators.items():
    print(f"  {target}: coef={cal.coef_[0]:.4f}, intercept={cal.intercept_:.4f}")

#%%
print("\n" + "="*60)
print("🔧 v26 Calibrated Inference")
print("="*60)

# 1. Raw predictions
tta_loaders = get_tta_dataloaders(test_wide, cfg)
raw_preds, sample_ids = predict_ensemble(cfg, tta_loaders)
print(f"\nRaw predictions shape: {raw_preds.shape}")

#%%
# 2. Apply calibration
calibrated_preds = apply_calibration(raw_preds, calibrators)
print(f"Calibrated predictions shape: {calibrated_preds.shape}")

# 예측 통계 비교
print("\n=== Prediction Statistics ===")
print(f"{'Target':<15} {'Raw Mean':>10} {'Cal Mean':>10} {'Raw Std':>10} {'Cal Std':>10}")
for idx, target in enumerate(TARGET_ORDER):
    print(f"{target:<15} {raw_preds[:, idx].mean():>10.2f} {calibrated_preds[:, idx].mean():>10.2f} "
          f"{raw_preds[:, idx].std():>10.2f} {calibrated_preds[:, idx].std():>10.2f}")

#%%
# 3. Create submission
pred_df = pd.DataFrame(calibrated_preds, columns=TARGET_ORDER)
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

print(f"\n✅ Saved: {len(submission)} rows")

#%%
sample_sub = pd.read_csv(cfg.DATA_PATH / "sample_submission.csv")
assert len(submission) == len(sample_sub)
print("✓ Format verified!")
