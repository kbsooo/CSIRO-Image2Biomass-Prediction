#%% [markdown]
# # 🔧 DINOv3 Inference v15
#
# **모델**: v15 (15_back_to_basics.py로 학습된 모델)
# **환경**: Kaggle (학습된 모델을 Dataset으로 업로드 후 사용)
#
# **필요 Datasets**:
# 1. `csiro-biomass` (competition data)
# 2. `pretrained-weights-biomass` (DINOv3 backbone weights)
# 3. 학습된 모델 Dataset (직접 업로드)

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

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

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
    # === 경로 (Kaggle) ===
    DATA_PATH = Path("/kaggle/input/csiro-biomass")
    
    # DINOv3 backbone weights (timm이 이 이름을 인식하도록 등록)
    BACKBONE_WEIGHTS = Path("/kaggle/input/pretrained-weights-biomass/dinov3_large/dinov3_large/dinov3_vitl16_qkvb.pth")
    
    # ⚠️ 이 경로를 업로드한 모델 Dataset 경로로 변경하세요
    MODELS_DIR = Path("/kaggle/input/csiro-v15-models")
    
    # === Model ===
    # pretrained-weights-biomass Dataset에 등록된 모델명
    model_name = "vit_large_patch16_dinov3_qkvb"
    img_size = (512, 512)
    dropout = 0.1
    
    # === Inference ===
    batch_size = 32
    num_workers = 0  # Kaggle에서 multiprocessing 에러 방지
    device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = CFG()
print(f"Device: {cfg.device}")
print(f"Models: {cfg.MODELS_DIR}")
print(f"Backbone: {cfg.BACKBONE_WEIGHTS}")

#%% [markdown]
# ## 📊 Dataset

#%%
class TestDataset(Dataset):
    """Test dataset with Left/Right split"""
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
    """3x TTA: Original, HFlip, VFlip"""
    loaders = []
    
    transforms_list = [
        # Original
        T.Compose([
            T.Resize(cfg.img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        # Horizontal Flip
        T.Compose([
            T.Resize(cfg.img_size),
            T.RandomHorizontalFlip(p=1.0),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        # Vertical Flip
        T.Compose([
            T.Resize(cfg.img_size),
            T.RandomVerticalFlip(p=1.0),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    ]
    
    for transform in transforms_list:
        dataset = TestDataset(df, cfg, transform)
        loader = DataLoader(dataset, batch_size=cfg.batch_size,
                           shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
        loaders.append(loader)
    
    return loaders

#%% [markdown]
# ## 🧠 Model Definition (v15와 동일)

#%%
class FiLM(nn.Module):
    """Feature-wise Linear Modulation"""
    def __init__(self, feat_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, feat_dim * 2)
        )
    
    def forward(self, context):
        gamma_beta = self.mlp(context)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        return gamma, beta


class CSIROModel(nn.Module):
    """v15 Model - v12 기반"""
    def __init__(self, model_name, backbone_weights_path=None, dropout=0.1):
        super().__init__()
        
        # Backbone - hf_hub prefix로 HuggingFace에서 architecture 로드
        # 그 후 local backbone weights로 덮어씀
        print(f"Creating backbone: {model_name}")
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool='avg')
        
        if backbone_weights_path and Path(backbone_weights_path).exists():
            print(f"Loading backbone weights from: {backbone_weights_path}")
            backbone_state = torch.load(backbone_weights_path, map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(backbone_state, strict=False)
            print("✓ Backbone weights loaded")
        else:
            print(f"WARNING: Backbone weights not found, using random init")
        
        feat_dim = self.backbone.num_features  # 1024
        print(f"Feature dim: {feat_dim}")
        
        self.film = FiLM(feat_dim)
        
        # v12/v15와 동일한 head (256 hidden units)
        def make_head():
            return nn.Sequential(
                nn.Linear(feat_dim * 2, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, 1)
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
        
        # [Green, Dead, Clover, GDM, Total]
        return torch.cat([green, dead, clover, gdm, total], dim=1)

#%% [markdown]
# ## 🔮 Inference Functions

#%%
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_outputs = []
    all_ids = []
    
    for left, right, ids in tqdm(loader, desc="Predicting"):
        left = left.to(device)
        right = right.to(device)
        
        outputs = model(left, right)
        all_outputs.append(outputs.cpu().numpy())
        all_ids.extend(ids)
    
    return np.concatenate(all_outputs), all_ids


def predict_with_tta(model, tta_loaders, device):
    """Predict with TTA (average across augmentations)"""
    all_preds = []
    final_ids = None
    
    for i, loader in enumerate(tta_loaders):
        preds, ids = predict(model, loader, device)
        all_preds.append(preds)
        if final_ids is None:
            final_ids = ids
    
    avg_preds = np.mean(all_preds, axis=0)
    return avg_preds, final_ids


def predict_ensemble(cfg, tta_loaders):
    """Ensemble prediction: N folds × 3 TTA"""
    all_fold_preds = []
    final_ids = None
    
    model_files = sorted(cfg.MODELS_DIR.glob("model_fold*.pth"))
    print(f"Found {len(model_files)} model files")
    
    for model_file in model_files:
        print(f"\nLoading {model_file.name}...")
        
        # 1. Backbone architecture 생성 + backbone weights 로드
        model = CSIROModel(
            cfg.model_name, 
            backbone_weights_path=cfg.BACKBONE_WEIGHTS,
            dropout=cfg.dropout
        ).to(cfg.device)
        
        # 2. 학습된 fold weights로 전체 덮어쓰기
        state_dict = torch.load(model_file, map_location=cfg.device)
        model.load_state_dict(state_dict)
        print("✓ Fold weights loaded")
        
        preds, ids = predict_with_tta(model, tta_loaders, cfg.device)
        all_fold_preds.append(preds)
        
        if final_ids is None:
            final_ids = ids
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    # Average across folds
    final_preds = np.mean(all_fold_preds, axis=0)
    return final_preds, final_ids

#%% [markdown]
# ## 📋 Main Inference

#%%
# Load test data
test_df = pd.read_csv(cfg.DATA_PATH / "test.csv")

# Prepare test data
test_df['sample_id_prefix'] = test_df['sample_id'].str.split('__').str[0]
test_wide = test_df.drop_duplicates(subset=['image_path']).reset_index(drop=True)

print(f"Test samples: {len(test_wide)}")

#%%
# Get TTA dataloaders
tta_loaders = get_tta_dataloaders(test_wide, cfg)

# Run ensemble prediction
preds, sample_ids = predict_ensemble(cfg, tta_loaders)

print(f"Predictions shape: {preds.shape}")

#%%
# Create submission
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

print(f"\n✅ Submission saved: {len(submission)} rows")
print(submission.head(10))

#%%
# Verify submission format
sample_submission = pd.read_csv(cfg.DATA_PATH / "sample_submission.csv")
print(f"\nExpected rows: {len(sample_submission)}")
print(f"Actual rows: {len(submission)}")
assert len(submission) == len(sample_submission), "Row count mismatch!"
print("✓ Submission format verified!")
