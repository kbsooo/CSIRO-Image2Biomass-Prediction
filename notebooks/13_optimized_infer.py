#%% [markdown]
# # 🚀 Optimized DINOv3 Inference Pipeline
#
# **Model**: CSIROModelV2 (trained with 13_optimized_train.py)
# **Features**: TTA (Test Time Augmentation)

#%%
import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torchvision.transforms.v2 as T

import timm

import warnings
warnings.filterwarnings('ignore')
tqdm.pandas()

#%% [markdown]
# ## Configuration

#%%
class CFG:
    DATA_PATH = Path("/kaggle/input/csiro-biomass")
    MODELS_DIR = Path("/kaggle/input/csiro-optimized-models")  # 학습된 모델 경로
    WEIGHTS_PATH = Path("/kaggle/input/pretrained-weights-biomass/dinov3_large/dinov3_large")
    
    model_name = "vit_large_patch16_dinov3_qkvb.lvd1689m"  # DINOv3 Large
    img_size = (518, 518)
    dropout = 0.3
    
    use_tta = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = CFG()

#%% [markdown]
# ## Model Definition (Same as training)

#%%
class FiLM(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Linear(feat_dim // 2, feat_dim * 2)
        )
    
    def forward(self, context):
        gamma_beta = self.mlp(context)
        return torch.chunk(gamma_beta, 2, dim=1)

class ZeroInflatedHead(nn.Module):
    def __init__(self, in_features, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        self.regressor = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Softplus()
        )
    
    def forward(self, x):
        prob = torch.sigmoid(self.classifier(x))
        amount = self.regressor(x)
        return prob * amount

class CSIROModelV2(nn.Module):
    def __init__(self, model_name, dropout=0.3):
        super().__init__()
        
        self.backbone = timm.create_model(
            model_name, pretrained=False, num_classes=0, global_pool='avg'
        )
        
        feat_dim = self.backbone.num_features
        self.film = FiLM(feat_dim)
        
        def make_head(in_dim, dropout):
            return nn.Sequential(
                nn.Linear(in_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(512, 128),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(128, 1),
                nn.Softplus()
            )
        
        combined_dim = feat_dim * 2
        self.head_green = make_head(combined_dim, dropout)
        self.head_dead = make_head(combined_dim, dropout)
        self.head_clover = ZeroInflatedHead(combined_dim, dropout)
    
    def forward(self, left_img, right_img):
        left_feat = self.backbone(left_img)
        right_feat = self.backbone(right_img)
        
        context = (left_feat + right_feat) / 2
        gamma, beta = self.film(context)
        
        left_mod = left_feat * (1 + gamma) + beta
        right_mod = right_feat * (1 + gamma) + beta
        
        combined = torch.cat([left_mod, right_mod], dim=1)
        
        green = self.head_green(combined)
        clover = self.head_clover(combined)
        dead = self.head_dead(combined)
        
        gdm = green + clover
        total = gdm + dead
        
        return torch.cat([green, dead, clover, gdm, total], dim=1)

#%% [markdown]
# ## TTA Transforms

#%%
def get_tta_transforms(cfg):
    """Test Time Augmentation: original + hflip + vflip"""
    base = T.Compose([
        T.Resize(cfg.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    hflip = T.Compose([
        T.Resize(cfg.img_size),
        T.RandomHorizontalFlip(p=1.0),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    vflip = T.Compose([
        T.Resize(cfg.img_size),
        T.RandomVerticalFlip(p=1.0),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return [base, hflip, vflip]

#%% [markdown]
# ## Inference

#%%
@torch.no_grad()
def predict_single(model, img_path, transforms, cfg):
    """Single image prediction with TTA"""
    img = Image.open(cfg.DATA_PATH / img_path).convert('RGB')
    width, height = img.size
    mid_point = width // 2
    
    left_img = img.crop((0, 0, mid_point, height))
    right_img = img.crop((mid_point, 0, width, height))
    
    all_preds = []
    for transform in transforms:
        left_t = transform(left_img).unsqueeze(0).to(cfg.device)
        right_t = transform(right_img).unsqueeze(0).to(cfg.device)
        
        outputs = model(left_t, right_t)
        all_preds.append(outputs.cpu().numpy())
    
    return np.mean(all_preds, axis=0)

def predict_all_folds(test_df, cfg):
    """Ensemble prediction across all folds"""
    transforms = get_tta_transforms(cfg) if cfg.use_tta else [get_tta_transforms(cfg)[0]]
    
    all_fold_preds = []
    
    for model_path in sorted(cfg.MODELS_DIR.glob("model_fold*.pth")):
        print(f"Loading {model_path.name}...")
        
        model = CSIROModelV2(cfg.model_name, cfg.dropout)
        model.load_state_dict(torch.load(model_path, map_location=cfg.device))
        model = model.to(cfg.device)
        model.eval()
        
        fold_preds = []
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            pred = predict_single(model, row['image_path'], transforms, cfg)
            fold_preds.append(pred)
        
        all_fold_preds.append(np.concatenate(fold_preds))
    
    # Average across folds
    return np.mean(all_fold_preds, axis=0)

#%% [markdown]
# ## Main

#%%
if __name__ == "__main__":
    # Load test data
    test_df = pd.read_csv(cfg.DATA_PATH / "test.csv")
    test_df['target'] = 0.0
    test_df[['sample_id_prefix', 'sample_id_suffix']] = test_df.sample_id.str.split('__', expand=True)
    
    test_data = test_df.groupby(['sample_id_prefix', 'image_path']).apply(
        lambda df: df.set_index('target_name').target
    ).reset_index()
    test_data.columns.name = None
    
    print(f"Test samples: {len(test_data)}")
    
    # Predict
    preds = predict_all_folds(test_data, cfg)
    
    # Format predictions
    test_data[['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']] = preds
    
    # Create submission
    cols = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
    sub_df = test_data.set_index('sample_id_prefix')[cols].stack().reset_index()
    sub_df.columns = ['sample_id_prefix', 'target_name', 'target']
    sub_df['sample_id'] = sub_df.sample_id_prefix + '__' + sub_df.target_name
    
    sub_df[['sample_id', 'target']].to_csv('submission.csv', index=False)
    print("\n✓ submission.csv created!")
    print(sub_df[['sample_id', 'target']].head(10))
