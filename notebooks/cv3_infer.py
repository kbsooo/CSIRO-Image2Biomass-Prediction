#%% [markdown]
# # CV3 Inference: TTA + WA Postprocessing
#
# **핵심 변경사항**:
# 1. ⭐ 이미지 전처리 (학습과 동일)
# 2. ⭐ WA Dead=0 후처리
# 3. 4-fold TTA

#%%
import warnings
warnings.filterwarnings('ignore')

import os
import gc
import random
import cv2
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
    
    # CV3 모델 경로
    MODELS_DIR = Path("/kaggle/input/csiro-cv3-models")
    
    model_name = "vit_large_patch16_dinov3_qkvb.lvd1689m"
    img_size = (560, 560)
    
    # Model (CV1/CV3와 동일)
    hidden_dim = 256
    num_layers = 2
    dropout = 0.3
    
    # Inference
    batch_size = 16
    num_workers = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # TTA
    use_tta = True
    n_tta = 4
    
    # CV3 전처리
    use_clean_image = True
    bottom_crop_ratio = 0.90
    
    # WA 후처리
    use_wa_postprocess = True

cfg = CFG()

print(f"Device: {cfg.device}")
print(f"use_clean_image: {cfg.use_clean_image}")
print(f"use_wa_postprocess: {cfg.use_wa_postprocess}")

TARGET_ORDER = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

#%% [markdown]
# ## ⭐ Image Preprocessing

#%%
def clean_image(img_array, bottom_crop_ratio=0.90):
    """이미지 전처리: timestamp 제거 + bottom crop"""
    img = img_array.copy()
    h, w = img.shape[:2]
    
    # 1. Bottom crop
    new_h = int(h * bottom_crop_ratio)
    img = img[0:new_h, :]
    
    # 2. Orange timestamp inpainting
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_orange = np.array([5, 150, 150])
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    
    if np.sum(mask) > 100:
        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    
    return img

#%% [markdown]
# ## ⭐ WA Postprocessing

#%%
def postprocess_wa(preds, test_df):
    """
    WA State 후처리: Dead = 0 강제
    Discussion에서 발견된 100% 패턴
    """
    preds = preds.copy()
    wa_count = 0
    
    for idx in range(len(test_df)):
        row = test_df.iloc[idx]
        
        # State 확인 (test.csv에 State가 없을 수 있음)
        state = row.get('State', None)
        
        if state == 'WA':
            wa_count += 1
            
            # Dry_Dead_g = 0 강제
            preds[idx, 1] = 0.0  # Dead index
            
            # GDM과 Total 재계산
            green = preds[idx, 0]
            clover = preds[idx, 2]
            preds[idx, 3] = green + clover         # GDM
            preds[idx, 4] = green + clover         # Total (Dead=0)
    
    if wa_count > 0:
        print(f"✓ WA samples processed: {wa_count} (Dead forced to 0)")
    else:
        print("⚠️ No WA samples found (State column may not exist)")
    
    return preds

#%% [markdown]
# ## 📊 Dataset

#%%
class TestDatasetCV3(Dataset):
    def __init__(self, df, cfg, transform=None):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.cfg.DATA_PATH / row['image_path']).convert('RGB')
        img_array = np.array(img)
        
        # CV3: 이미지 전처리
        if self.cfg.use_clean_image:
            img_array = clean_image(img_array, self.cfg.bottom_crop_ratio)
        
        h, w = img_array.shape[:2]
        mid = w // 2
        
        left_array = img_array[:, :mid, :]
        right_array = img_array[:, mid:, :]
        
        left_img = Image.fromarray(left_array)
        right_img = Image.fromarray(right_array)
        
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


def make_head(in_dim, hidden_dim, num_layers, dropout):
    if num_layers == 1:
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    else:
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


class CSIROModelCV3(nn.Module):
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
        
        self.head_green = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout)
        self.head_clover = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout)
        self.head_dead = make_head(combined_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout)
        
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

#%% [markdown]
# ## 🔮 Inference with TTA

#%%
@torch.no_grad()
def predict_with_tta(model, left, right, device, n_tta=4):
    """4-fold TTA: HFlip x VFlip"""
    preds = []
    
    for hflip in [False, True]:
        for vflip in [False, True]:
            l = torch.flip(left, [3]) if hflip else left
            r = torch.flip(right, [3]) if hflip else right
            l = torch.flip(l, [2]) if vflip else l
            r = torch.flip(r, [2]) if vflip else r
            
            pred = model(l.to(device), r.to(device))
            preds.append(pred.cpu())
            
            if len(preds) >= n_tta:
                break
        if len(preds) >= n_tta:
            break
    
    return torch.stack(preds).mean(0)


def predict_batch(model, loader, cfg):
    model.eval()
    device = cfg.device
    all_outputs, all_ids = [], []
    
    for left, right, ids in tqdm(loader, desc="Predicting"):
        if cfg.use_tta:
            outputs = predict_with_tta(model, left, right, device, cfg.n_tta)
        else:
            outputs = model(left.to(device), right.to(device)).cpu()
        
        all_outputs.append(outputs.numpy())
        all_ids.extend(ids)
    
    return np.concatenate(all_outputs), all_ids


def predict_ensemble(cfg, loader):
    """5-fold 앙상블"""
    model_files = sorted(cfg.MODELS_DIR.glob("model_fold*.pth"))
    print(f"\nFound {len(model_files)} models")
    
    all_fold_preds = []
    final_ids = None
    
    for model_file in model_files:
        print(f"\nLoading {model_file.name}...")
        
        model = CSIROModelCV3(cfg, cfg.BACKBONE_WEIGHTS).to(cfg.device)
        model.load_state_dict(torch.load(model_file, map_location=cfg.device))
        
        preds, ids = predict_batch(model, loader, cfg)
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

# State 컬럼 확인
if 'State' in test_wide.columns:
    print(f"✓ State column exists")
    print(f"  WA samples: {len(test_wide[test_wide['State'] == 'WA'])}")
else:
    print("⚠️ State column not found - WA postprocessing will be skipped")

#%%
print("\n" + "="*60)
print("🚀 CV3 Inference: Clean Image + TTA + WA Postprocessing")
print("="*60)

transform = get_test_transform(cfg)
dataset = TestDatasetCV3(test_wide, cfg, transform)
loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                   num_workers=cfg.num_workers, pin_memory=True)

predictions, sample_ids = predict_ensemble(cfg, loader)
print(f"\nPredictions: {predictions.shape}")

#%%
# WA 후처리
if cfg.use_wa_postprocess and 'State' in test_wide.columns:
    predictions = postprocess_wa(predictions, test_wide)

#%%
# 예측 통계
print("\n=== Prediction Statistics ===")
print(f"{'Target':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
for idx, target in enumerate(TARGET_ORDER):
    vals = predictions[:, idx]
    print(f"{target:<15} {vals.mean():>10.2f} {vals.std():>10.2f} {vals.min():>10.2f} {vals.max():>10.2f}")

#%%
# Submission 생성
pred_df = pd.DataFrame(predictions, columns=TARGET_ORDER)
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

sample_sub = pd.read_csv(cfg.DATA_PATH / "sample_submission.csv")
assert len(submission) == len(sample_sub), "Format mismatch!"

print(f"\n✅ submission.csv saved")
print(f"   {len(submission)} rows")
print(f"   Clean image: {cfg.use_clean_image}")
print(f"   WA postprocess: {cfg.use_wa_postprocess}")
