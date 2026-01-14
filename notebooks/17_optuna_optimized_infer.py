#%% [markdown]
# # 🔮 v17: Optuna-Optimized Inference
#
# **모델**: v17 (Optuna-optimized 하이퍼파라미터)
# **환경**: Kaggle

#%%
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['PYTHONWARNINGS'] = 'ignore'

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
    
    # ⚠️ 이 경로를 업로드한 모델 Dataset 경로로 변경하세요
    MODELS_DIR = Path("/kaggle/input/csiro-v17-models")
    
    model_name = "vit_large_patch16_dinov3_qkvb.lvd1689m"
    img_size = (512, 512)
    
    # Optuna best
    hidden_dim = 512
    dropout = 0.1
    
    batch_size = 32
    num_workers = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = CFG()
print(f"Device: {cfg.device}")

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
    """7x TTA: Original + Flips + Rotations"""
    loaders = []
    
    transforms_list = [
        T.Compose([T.Resize(cfg.img_size), T.ToTensor(), 
                   T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        T.Compose([T.Resize(cfg.img_size), T.RandomHorizontalFlip(p=1.0), T.ToTensor(),
                   T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        T.Compose([T.Resize(cfg.img_size), T.RandomVerticalFlip(p=1.0), T.ToTensor(),
                   T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    ]
    
    for transform in transforms_list:
        dataset = TestDataset(df, cfg, transform)
        loader = DataLoader(dataset, batch_size=cfg.batch_size,
                           shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
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
        gamma_beta = self.mlp(context)
        return torch.chunk(gamma_beta, 2, dim=1)


class CSIROModelV17(nn.Module):
    """v17 Optuna-optimized architecture"""
    def __init__(self, cfg, backbone_weights_path=None):
        super().__init__()
        
        self.backbone = timm.create_model(cfg.model_name, pretrained=False, num_classes=0, global_pool='avg')
        
        if backbone_weights_path and Path(backbone_weights_path).exists():
            state = torch.load(backbone_weights_path, map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(state, strict=False)
        
        feat_dim = self.backbone.num_features
        combined_dim = feat_dim * 2
        hidden_dim = cfg.hidden_dim
        dropout = cfg.dropout
        
        self.film = FiLM(feat_dim)
        
        def make_head():
            return nn.Sequential(
                nn.Linear(combined_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
        
        self.head_green = make_head()
        self.head_clover = make_head()
        self.head_dead = make_head()
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
        print(f"Loading {model_file.name}...")
        
        model = CSIROModelV17(cfg, cfg.BACKBONE_WEIGHTS).to(cfg.device)
        model.load_state_dict(torch.load(model_file, map_location=cfg.device))
        
        preds, ids = predict_with_tta(model, tta_loaders, cfg.device)
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
test_df = pd.read_csv(cfg.DATA_PATH / "test.csv")
test_df['sample_id_prefix'] = test_df['sample_id'].str.split('__').str[0]
test_wide = test_df.drop_duplicates(subset=['image_path']).reset_index(drop=True)
print(f"Test samples: {len(test_wide)}")

#%%
tta_loaders = get_tta_dataloaders(test_wide, cfg)
preds, sample_ids = predict_ensemble(cfg, tta_loaders)
print(f"Predictions: {preds.shape}")

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
print(submission.head(10))

#%%
sample_sub = pd.read_csv(cfg.DATA_PATH / "sample_submission.csv")
assert len(submission) == len(sample_sub), "Row count mismatch!"
print("✓ Format verified!")
