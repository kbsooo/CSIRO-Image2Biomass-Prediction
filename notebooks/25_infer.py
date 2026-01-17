#%% [markdown]
# # v25 Inference: Vegetation Index Late Fusion
#
# RGB + Vegetation Index Late Fusion 모델 추론

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
    
    # ⚠️ v25 모델 경로
    MODELS_DIR = Path("/kaggle/input/csiro-v25-models")
    
    model_name = "vit_large_patch16_dinov3_qkvb.lvd1689m"
    img_size = (512, 512)
    
    hidden_dim = 512
    num_layers = 3
    dropout = 0.1
    use_layernorm = True
    veg_feat_dim = 128
    
    batch_size = 16
    num_workers = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = CFG()

#%% [markdown]
# ## 🌱 Vegetation Index 계산

#%%
def compute_vegetation_indices(img_array):
    """RGB 이미지에서 Vegetation Index 계산"""
    img = img_array.astype(np.float32) / 255.0
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    
    exg = 2*g - r - b
    exg = (exg + 2) / 4
    
    gr_ratio = g / (r + 1e-8)
    gr_ratio = np.clip(gr_ratio, 0, 3) / 3
    
    return np.stack([exg, gr_ratio], axis=-1)

#%% [markdown]
# ## 📊 Dataset

#%%
class TestDatasetV25(Dataset):
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
        
        # Resize & numpy 변환
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
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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


class VegetationEncoder(nn.Module):
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


class CSIROModelV25(nn.Module):
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
# ## 🔮 Inference

#%%
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_outputs, all_ids = [], []
    
    for left_rgb, right_rgb, left_veg, right_veg, ids in tqdm(loader, desc="Predicting"):
        left_rgb = left_rgb.to(device)
        right_rgb = right_rgb.to(device)
        left_veg = left_veg.to(device)
        right_veg = right_veg.to(device)
        
        outputs = model(left_rgb, right_rgb, left_veg, right_veg)
        all_outputs.append(outputs.cpu().numpy())
        all_ids.extend(ids)
    
    return np.concatenate(all_outputs), all_ids


def predict_ensemble(cfg, test_df):
    transform = get_test_transform(cfg)
    dataset = TestDatasetV25(test_df, cfg, transform)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                       num_workers=cfg.num_workers, pin_memory=True)
    
    all_fold_preds = []
    model_files = sorted(cfg.MODELS_DIR.glob("model_fold*.pth"))
    print(f"Found {len(model_files)} models")
    
    for model_file in model_files:
        print(f"\nLoading {model_file.name}...")
        
        model = CSIROModelV25(cfg, cfg.BACKBONE_WEIGHTS).to(cfg.device)
        model.load_state_dict(torch.load(model_file, map_location=cfg.device))
        print("✓ Loaded")
        
        preds, ids = predict(model, loader, cfg.device)
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
print("🌱 v25 Inference: Vegetation Index Late Fusion")
print("="*60)

preds, sample_ids = predict_ensemble(cfg, test_wide)
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
