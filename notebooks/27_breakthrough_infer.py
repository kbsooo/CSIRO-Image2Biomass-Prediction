#%% [markdown]
# # 🚀 v27: Breakthrough Multi-Model Inference
#
# **목적**: v20, v22, v23, v25, v26 모델 최적 결합으로 0.69 벽 돌파
#
# **전략**:
# 1. 모든 모델 예측 수집
# 2. 다양한 앙상블 방법 적용
# 3. 최적 조합 찾기

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
from scipy.optimize import minimize
from scipy.stats import rankdata

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
    
    # 각 버전 모델 경로 (Kaggle Dataset으로 업로드)
    MODELS = {
        'v20': Path("/kaggle/input/csiro-v20-models"),
        'v22': Path("/kaggle/input/csiro-v22-models"),
        'v23': Path("/kaggle/input/csiro-v23-models"),
        'v25': Path("/kaggle/input/csiro-v25-models"),
        'v26': Path("/kaggle/input/csiro-v26-models"),
    }
    
    model_name = "vit_large_patch16_dinov3_qkvb.lvd1689m"
    img_size = (512, 512)
    
    hidden_dim = 512
    num_layers = 3
    dropout = 0.1
    use_layernorm = True
    veg_feat_dim = 128  # v25용
    
    batch_size = 16
    num_workers = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = CFG()

TARGET_ORDER = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

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


def get_test_transform(cfg):
    return T.Compose([
        T.Resize(cfg.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

#%% [markdown]
# ## 🧠 Models (v20/v22/v23/v26 호환)

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
    """v20/v22/v23/v26 호환 모델"""
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
    """v22 전용 모델 (Frozen backbone, 더 작은 head)"""
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
        
        # v22: hidden_dim=256, num_layers=2
        v22_hidden = 256
        v22_layers = 2
        v22_dropout = 0.3
        
        self.head_green = self._make_head_v22(combined_dim, v22_hidden, v22_layers, v22_dropout)
        self.head_clover = self._make_head_v22(combined_dim, v22_hidden, v22_layers, v22_dropout)
        self.head_dead = self._make_head_v22(combined_dim, v22_hidden, v22_layers, v22_dropout)
        
        self.softplus = nn.Softplus(beta=1.0)
    
    def _make_head_v22(self, in_dim, hidden_dim, num_layers, dropout):
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
    """v25용 Vegetation Index Encoder"""
    def __init__(self, in_channels=2, out_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(128, out_dim), nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.encoder(x)


class CSIROModelV25(nn.Module):
    """v25 전용 모델 (VegIdx Fusion)"""
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
# ## 🔮 Prediction Functions

#%%
@torch.no_grad()
def predict_base_model(model, loader, device):
    """v20/v22/v23/v26 예측"""
    model.eval()
    all_outputs, all_ids = [], []
    
    for left, right, ids in tqdm(loader, desc="Predicting"):
        left = left.to(device)
        right = right.to(device)
        outputs = model(left, right)
        all_outputs.append(outputs.cpu().numpy())
        all_ids.extend(ids)
    
    return np.concatenate(all_outputs), all_ids


def compute_vegetation_indices(img_array):
    """RGB에서 Vegetation Index 계산"""
    img = img_array.astype(np.float32) / 255.0
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    exg = (2*g - r - b + 2) / 4
    gr_ratio = np.clip(g / (r + 1e-8), 0, 3) / 3
    return np.stack([exg, gr_ratio], axis=-1)


@torch.no_grad()
def predict_v25_model(model, test_df, cfg, device):
    """v25 전용 예측 (VegIdx 포함)"""
    model.eval()
    all_outputs, all_ids = [], []
    
    transform = get_test_transform(cfg)
    
    for idx in tqdm(range(len(test_df)), desc="Predicting v25"):
        row = test_df.iloc[idx]
        img = Image.open(cfg.DATA_PATH / row['image_path']).convert('RGB')
        width, height = img.size
        mid = width // 2
        
        left_pil = img.crop((0, 0, mid, height)).resize(cfg.img_size)
        right_pil = img.crop((mid, 0, width, height)).resize(cfg.img_size)
        
        # RGB
        left_rgb = transform(left_pil).unsqueeze(0).to(device)
        right_rgb = transform(right_pil).unsqueeze(0).to(device)
        
        # VegIdx
        left_np = np.array(left_pil)
        right_np = np.array(right_pil)
        left_veg = torch.from_numpy(compute_vegetation_indices(left_np)).permute(2, 0, 1).unsqueeze(0).float().to(device)
        right_veg = torch.from_numpy(compute_vegetation_indices(right_np)).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
        outputs = model(left_rgb, right_rgb, left_veg, right_veg)
        all_outputs.append(outputs.cpu().numpy())
        all_ids.append(row['sample_id_prefix'])
    
    return np.concatenate(all_outputs), all_ids


def predict_version(version, model_dir, test_df, cfg, device):
    """버전별 앙상블 예측"""
    model_files = sorted(model_dir.glob("model_fold*.pth"))
    if not model_files:
        print(f"⚠️ No models found for {version}")
        return None, None
    
    print(f"\n=== {version}: {len(model_files)} models ===")
    
    all_fold_preds = []
    final_ids = None
    
    if version == 'v25':
        # v25는 별도 처리 (VegIdx 포함)
        for model_file in model_files:
            print(f"Loading {model_file.name}...")
            model = CSIROModelV25(cfg, cfg.BACKBONE_WEIGHTS).to(device)
            model.load_state_dict(torch.load(model_file, map_location=device))
            
            preds, ids = predict_v25_model(model, test_df, cfg, device)
            all_fold_preds.append(preds)
            if final_ids is None:
                final_ids = ids
            
            del model
            gc.collect()
            torch.cuda.empty_cache()
    elif version == 'v22':
        # v22는 별도 모델 구조 (hidden_dim=256, num_layers=2)
        transform = get_test_transform(cfg)
        dataset = TestDataset(test_df, cfg, transform)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                           num_workers=cfg.num_workers, pin_memory=True)
        
        for model_file in model_files:
            print(f"Loading {model_file.name}...")
            model = CSIROModelV22(cfg, cfg.BACKBONE_WEIGHTS).to(device)
            model.load_state_dict(torch.load(model_file, map_location=device))
            
            preds, ids = predict_base_model(model, loader, device)
            all_fold_preds.append(preds)
            if final_ids is None:
                final_ids = ids
            
            del model
            gc.collect()
            torch.cuda.empty_cache()
    else:
        # v20/v23/v26
        transform = get_test_transform(cfg)
        dataset = TestDataset(test_df, cfg, transform)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                           num_workers=cfg.num_workers, pin_memory=True)
        
        for model_file in model_files:
            print(f"Loading {model_file.name}...")
            model = CSIROModelBase(cfg, cfg.BACKBONE_WEIGHTS).to(device)
            model.load_state_dict(torch.load(model_file, map_location=device))
            
            preds, ids = predict_base_model(model, loader, device)
            all_fold_preds.append(preds)
            if final_ids is None:
                final_ids = ids
            
            del model
            gc.collect()
            torch.cuda.empty_cache()
    
    return np.mean(all_fold_preds, axis=0), final_ids

#%% [markdown]
# ## 🔧 Ensemble Methods

#%%
def simple_average(predictions_dict):
    """단순 평균"""
    preds = [v for v in predictions_dict.values() if v is not None]
    return np.mean(preds, axis=0)


def weighted_average(predictions_dict, weights):
    """가중 평균"""
    result = np.zeros_like(list(predictions_dict.values())[0])
    total_weight = 0
    
    for version, pred in predictions_dict.items():
        if pred is not None and version in weights:
            result += weights[version] * pred
            total_weight += weights[version]
    
    return result / total_weight


def rank_average(predictions_dict):
    """순위 기반 평균 (outlier에 강함)"""
    preds_list = [v for v in predictions_dict.values() if v is not None]
    n_samples, n_targets = preds_list[0].shape
    
    # 각 모델별, 타겟별로 순위 계산
    ranks = []
    for pred in preds_list:
        rank = np.zeros_like(pred)
        for t in range(n_targets):
            rank[:, t] = rankdata(pred[:, t])
        ranks.append(rank)
    
    # 순위 평균
    avg_rank = np.mean(ranks, axis=0)
    
    # 순위를 값으로 변환 (전체 예측의 중앙값 기준)
    all_preds = np.concatenate(preds_list, axis=0)
    result = np.zeros((n_samples, n_targets))
    
    for t in range(n_targets):
        sorted_vals = np.sort(all_preds[:, t])
        # 평균 순위에 해당하는 값 추정
        for i in range(n_samples):
            idx = int(avg_rank[i, t] / len(preds_list)) - 1
            idx = max(0, min(idx, len(sorted_vals) - 1))
            result[i, t] = sorted_vals[idx]
    
    return result


def target_wise_best(predictions_dict, oof_scores=None):
    """타겟별 최적 모델 선택 (OOF 기반)"""
    if oof_scores is None:
        # OOF 없으면 단순 평균
        return simple_average(predictions_dict)
    
    versions = [v for v in predictions_dict.keys() if predictions_dict[v] is not None]
    n_samples, n_targets = list(predictions_dict.values())[0].shape
    result = np.zeros((n_samples, n_targets))
    
    for t_idx, target in enumerate(TARGET_ORDER):
        # 해당 타겟에서 가장 좋은 모델 선택
        best_version = max(versions, key=lambda v: oof_scores.get(v, {}).get(target, 0))
        result[:, t_idx] = predictions_dict[best_version][:, t_idx]
        print(f"  {target}: {best_version}")
    
    return result

#%% [markdown]
# ## 📋 Main Execution

#%%
# 데이터 로드
test_df = pd.read_csv(cfg.DATA_PATH / "test.csv")
test_df['sample_id_prefix'] = test_df['sample_id'].str.split('__').str[0]
test_wide = test_df.drop_duplicates(subset=['image_path']).reset_index(drop=True)
print(f"Test samples: {len(test_wide)}")

#%%
print("\n" + "="*60)
print("🚀 v27: Breakthrough Multi-Model Inference")
print("="*60)

# 각 버전 예측 수집
predictions = {}
sample_ids = None

for version, model_dir in cfg.MODELS.items():
    if model_dir.exists():
        preds, ids = predict_version(version, model_dir, test_wide, cfg, cfg.device)
        if preds is not None:
            predictions[version] = preds
            if sample_ids is None:
                sample_ids = ids
    else:
        print(f"⚠️ {version} not found: {model_dir}")

print(f"\n✓ Collected predictions from: {list(predictions.keys())}")

#%% [markdown]
# ## 🎯 Ensemble & Submission

#%%
# ⚠️ 여기서 앙상블 방법 선택!
# 옵션: "simple", "rank", "weighted"
ENSEMBLE_METHOD = "simple"

# 가중 평균용 가중치 (ENSEMBLE_METHOD = "weighted" 선택 시 사용)
WEIGHTS = {'v20': 1.0, 'v22': 0.8, 'v23': 1.0, 'v25': 0.9, 'v26': 1.0}

#%%
print("\n=== Generating Ensemble ===")

if ENSEMBLE_METHOD == "simple":
    final_preds = simple_average(predictions)
    print("✓ Method: Simple Average")
elif ENSEMBLE_METHOD == "rank":
    final_preds = rank_average(predictions)
    print("✓ Method: Rank Average")
elif ENSEMBLE_METHOD == "weighted":
    final_preds = weighted_average(predictions, WEIGHTS)
    print(f"✓ Method: Weighted Average")
    print(f"  Weights: {WEIGHTS}")
else:
    raise ValueError(f"Unknown method: {ENSEMBLE_METHOD}")

print(f"  Shape: {final_preds.shape}")

#%%
# 예측 통계
print("\n=== Prediction Statistics ===")
print(f"{'Target':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
for idx, target in enumerate(TARGET_ORDER):
    vals = final_preds[:, idx]
    print(f"{target:<15} {vals.mean():>10.2f} {vals.std():>10.2f} {vals.min():>10.2f} {vals.max():>10.2f}")

#%%
# Submission 생성
pred_df = pd.DataFrame(final_preds, columns=TARGET_ORDER)
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

print(f"\n✅ submission.csv saved ({ENSEMBLE_METHOD} method)")
print(f"   {len(submission)} rows")

