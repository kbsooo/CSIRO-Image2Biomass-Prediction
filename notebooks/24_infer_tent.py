#%% [markdown]
# # v24 Inference with Test-Time Adaptation (TENT)
#
# **핵심 아이디어**:
# - Inference 중 LayerNorm 파라미터를 업데이트하여 Test 분포에 적응
# - Regression에 맞게 변형: Prediction Variance Minimization
#
# **과정**:
# 1. v20/v23 모델 로드
# 2. Test 배치마다 TENT 적응 (LayerNorm만 업데이트)
# 3. 적응된 모델로 최종 예측

#%%
import warnings
warnings.filterwarnings('ignore')

import os
import gc
import copy
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    
    # ⚠️ 기존 v20 또는 v23 모델 경로
    MODELS_DIR = Path("/kaggle/input/csiro-v20-models")  # v20 또는 v23
    
    model_name = "vit_large_patch16_dinov3_qkvb.lvd1689m"
    img_size = (512, 512)
    
    # Model 구조
    hidden_dim = 512
    num_layers = 3
    dropout = 0.1
    use_layernorm = True
    
    batch_size = 8  # TENT는 작은 배치 사용
    num_workers = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # === TENT 설정 ===
    tent_lr = 1e-5  # LayerNorm 업데이트 lr (매우 작게)
    tent_steps = 1  # 배치당 적응 스텝 수
    tent_augmentations = 4  # variance 계산용 augmentation 수

cfg = CFG()

#%% [markdown]
# ## 📊 Dataset (TENT용 Augmentation 포함)

#%%
class TENTDataset(Dataset):
    """TENT용 Dataset: augmentation 적용된 여러 버전 반환"""
    def __init__(self, df, cfg, base_transform, aug_transform, num_augs=4):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.base_transform = base_transform
        self.aug_transform = aug_transform
        self.num_augs = num_augs
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.cfg.DATA_PATH / row['image_path']).convert('RGB')
        width, height = img.size
        mid = width // 2
        
        left_pil = img.crop((0, 0, mid, height))
        right_pil = img.crop((mid, 0, width, height))
        
        # 기본 변환 (최종 예측용)
        left_base = self.base_transform(left_pil)
        right_base = self.base_transform(right_pil)
        
        # Augmentation 버전들 (TENT 적응용)
        left_augs = [self.aug_transform(left_pil) for _ in range(self.num_augs)]
        right_augs = [self.aug_transform(right_pil) for _ in range(self.num_augs)]
        
        return (left_base, right_base, 
                torch.stack(left_augs), torch.stack(right_augs),
                row['sample_id_prefix'])


def get_transforms(cfg):
    base = T.Compose([
        T.Resize(cfg.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    aug = T.Compose([
        T.Resize(cfg.img_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return base, aug

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


class CSIROModelV20(nn.Module):
    """v20/v23 호환 모델"""
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
# ## 🔧 TENT: Test-Time Adaptation

#%%
def configure_tent(model, cfg):
    """
    TENT를 위한 모델 설정
    - LayerNorm 파라미터만 학습 가능하게 설정
    - 나머지는 고정
    """
    model.eval()
    model.requires_grad_(False)
    
    # LayerNorm 파라미터만 학습 가능하게
    trainable_params = []
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            module.requires_grad_(True)
            trainable_params.extend(module.parameters())
    
    print(f"TENT: {len(trainable_params)} LayerNorm parameters trainable")
    
    optimizer = torch.optim.Adam(trainable_params, lr=cfg.tent_lr)
    
    return model, optimizer


def tent_adapt_batch(model, optimizer, left_augs, right_augs, cfg):
    """
    단일 배치에 대해 TENT 적응 수행
    
    Args:
        left_augs: [B, num_augs, C, H, W]
        right_augs: [B, num_augs, C, H, W]
    """
    B, N, C, H, W = left_augs.shape
    
    for step in range(cfg.tent_steps):
        all_preds = []
        
        # 각 augmentation에 대해 예측
        for i in range(N):
            left = left_augs[:, i]  # [B, C, H, W]
            right = right_augs[:, i]
            
            pred = model(left, right)  # [B, 5]
            all_preds.append(pred)
        
        # [N, B, 5] -> variance 계산
        preds_stack = torch.stack(all_preds, dim=0)  # [N, B, 5]
        
        # Prediction variance를 loss로 사용 (variance가 작을수록 confident)
        variance = torch.var(preds_stack, dim=0).mean()  # scalar
        
        optimizer.zero_grad()
        variance.backward()
        optimizer.step()
    
    return variance.item()

#%% [markdown]
# ## 🔮 Inference with TENT

#%%
def predict_with_tent(model, loader, cfg, device):
    """TENT 적응 + 예측"""
    model, optimizer = configure_tent(model, cfg)
    model = model.to(device)
    
    all_outputs = []
    all_ids = []
    all_variances = []
    
    for left_base, right_base, left_augs, right_augs, ids in tqdm(loader, desc="TENT"):
        left_base = left_base.to(device)
        right_base = right_base.to(device)
        left_augs = left_augs.to(device)
        right_augs = right_augs.to(device)
        
        # 1. TENT 적응 (이 배치의 분포에 맞게 LayerNorm 업데이트)
        var = tent_adapt_batch(model, optimizer, left_augs, right_augs, cfg)
        all_variances.append(var)
        
        # 2. 최종 예측 (기본 transform 사용)
        with torch.no_grad():
            outputs = model(left_base, right_base)
        
        all_outputs.append(outputs.cpu().numpy())
        all_ids.extend(ids)
    
    print(f"Average variance: {np.mean(all_variances):.6f}")
    
    return np.concatenate(all_outputs), all_ids


def predict_ensemble_tent(cfg, test_df):
    """5-fold 앙상블 + TENT"""
    base_transform, aug_transform = get_transforms(cfg)
    
    dataset = TENTDataset(test_df, cfg, base_transform, aug_transform, 
                          num_augs=cfg.tent_augmentations)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                       num_workers=cfg.num_workers, pin_memory=True)
    
    all_fold_preds = []
    model_files = sorted(cfg.MODELS_DIR.glob("model_fold*.pth"))
    print(f"Found {len(model_files)} models")
    
    for model_file in model_files:
        print(f"\n=== {model_file.name} ===")
        
        # 매 fold마다 새 모델 로드 (TENT가 파라미터를 수정하므로)
        model = CSIROModelV20(cfg, cfg.BACKBONE_WEIGHTS)
        model.load_state_dict(torch.load(model_file, map_location='cpu'))
        print("✓ Loaded")
        
        preds, ids = predict_with_tent(model, loader, cfg, cfg.device)
        all_fold_preds.append(preds)
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    return np.mean(all_fold_preds, axis=0), ids

#%% [markdown]
# ## 📋 Main

#%%
test_df = pd.read_csv(cfg.DATA_PATH / "test.csv")
test_df['sample_id_prefix'] = test_df['sample_id'].str.split('__').str[0]
test_wide = test_df.drop_duplicates(subset=['image_path']).reset_index(drop=True)
print(f"Test samples: {len(test_wide)}")

#%%
print("\n" + "="*60)
print("🔧 v24: Test-Time Adaptation (TENT)")
print("="*60)
print(f"TENT lr: {cfg.tent_lr}")
print(f"TENT steps: {cfg.tent_steps}")
print(f"TENT augmentations: {cfg.tent_augmentations}")

preds, sample_ids = predict_ensemble_tent(cfg, test_wide)
print(f"\nPredictions: {preds.shape}")

#%%
TARGET_ORDER = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

pred_df = pd.DataFrame(preds, columns=TARGET_ORDER)
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
