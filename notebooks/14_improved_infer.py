#%% [markdown]
# # 🚀 Improved DINOv3 Inference Pipeline v2
#
# **Model**: CSIROModelV2 (14_improved_train.py와 동일 구조)
# **TTA**: Original + HFlip + VFlip

#%%
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['PYTHONWARNINGS'] = 'ignore'

import gc
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from torch.utils.data import Dataset, DataLoader

import timm

tqdm.pandas()

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

#%% [markdown]
# ## Configuration

#%%
class CFG:
    DATA_PATH = Path("/kaggle/input/csiro-biomass")
    WEIGHTS_PATH = Path("/kaggle/input/pretrained-weights-biomass/dinov3_large/dinov3_large")
    MODELS_DIR = Path("/kaggle/input/csiro-improved-models")  # 학습된 모델

    model_name = "vit_large_patch16_dinov3_qkvb.lvd1689m"
    img_size = (512, 512)  # patch16 모델

    # Must match training config
    head_hidden_dim = 128
    dropout = 0.1
    use_zero_inflated_clover = True
    use_log1p = False  # Physics constraint와 충돌하므로 비활성화

    batch_size = 16
    num_workers = 0  # Avoid multiprocessing errors
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
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, feat_dim * 2)
        )

    def forward(self, context):
        out = self.mlp(context)
        gamma, beta = torch.chunk(out, 2, dim=1)
        return gamma, beta


class ZeroInflatedHead(nn.Module):
    def __init__(self, in_features, hidden_dim, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        self.regressor = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )

    def forward(self, x):
        prob = torch.sigmoid(self.classifier(x))
        amount = self.regressor(x)
        return prob * amount


class CSIROModelV2(nn.Module):
    def __init__(self, cfg, load_backbone=True):
        super().__init__()

        if load_backbone:
            self.backbone = timm.create_model(
                cfg.model_name, pretrained=False, num_classes=0, global_pool='avg'
            )
        else:
            self.backbone = timm.create_model(
                cfg.model_name, pretrained=False, num_classes=0, global_pool='avg'
            )

        feat_dim = self.backbone.num_features
        combined_dim = feat_dim * 2
        hidden_dim = cfg.head_hidden_dim
        dropout = cfg.dropout

        self.film = FiLM(feat_dim)

        def make_head():
            return nn.Sequential(
                nn.Linear(combined_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Softplus()
            )

        self.head_green = make_head()
        self.head_dead = make_head()

        if cfg.use_zero_inflated_clover:
            self.head_clover = ZeroInflatedHead(combined_dim, hidden_dim, dropout)
        else:
            self.head_clover = make_head()

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
# ## Dataset & TTA

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
        w, h = img.size
        mid = w // 2

        left = img.crop((0, 0, mid, h))
        right = img.crop((mid, 0, w, h))

        if self.transform:
            left = self.transform(left)
            right = self.transform(right)

        return left, right, row['sample_id_prefix']


def get_tta_loaders(df, cfg):
    """3x TTA: Original, HFlip, VFlip"""
    transforms = [
        T.Compose([
            T.Resize(cfg.img_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        T.Compose([
            T.Resize(cfg.img_size),
            T.RandomHorizontalFlip(p=1.0),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        T.Compose([
            T.Resize(cfg.img_size),
            T.RandomVerticalFlip(p=1.0),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    ]

    loaders = []
    for t in transforms:
        ds = TestDataset(df, cfg, t)
        loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=True)
        loaders.append(loader)

    return loaders

#%% [markdown]
# ## Inference

#%%
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    outputs = []
    ids = []

    for left, right, sample_ids in tqdm(loader, desc="Predicting"):
        left = left.to(device)
        right = right.to(device)

        out = model(left, right)
        outputs.append(out.cpu().numpy())
        ids.extend(sample_ids)

    return np.concatenate(outputs), ids


def predict_with_tta(model, loaders, device):
    """TTA: average across augmentations"""
    all_preds = []
    final_ids = None

    for loader in loaders:
        preds, ids = predict(model, loader, device)
        all_preds.append(preds)
        if final_ids is None:
            final_ids = ids

    return np.mean(all_preds, axis=0), final_ids


def predict_ensemble(models_dir, loaders, cfg):
    """Ensemble: N folds × 3 TTA"""
    model_files = sorted(Path(models_dir).glob("model_fold*.pth"))
    print(f"Found {len(model_files)} models")

    all_preds = []
    final_ids = None

    for mf in model_files:
        print(f"\nLoading {mf.name}...")

        model = CSIROModelV2(cfg, load_backbone=False).to(cfg.device)
        model.load_state_dict(torch.load(mf, map_location=cfg.device))

        preds, ids = predict_with_tta(model, loaders, cfg.device)
        all_preds.append(preds)

        if final_ids is None:
            final_ids = ids

        del model
        gc.collect()
        torch.cuda.empty_cache()

    return np.mean(all_preds, axis=0), final_ids

#%% [markdown]
# ## Main

#%%
# Load test data
test_df = pd.read_csv(cfg.DATA_PATH / "test.csv")
test_df['sample_id_prefix'] = test_df['sample_id'].str.split('__').str[0]
test_wide = test_df.drop_duplicates(subset=['image_path']).reset_index(drop=True)

print(f"Test samples: {len(test_wide)}")

#%%
# TTA loaders
loaders = get_tta_loaders(test_wide, cfg)

# Predict
preds, sample_ids = predict_ensemble(cfg.MODELS_DIR, loaders, cfg)
print(f"Predictions: {preds.shape}")

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

print(f"\n✓ Submission: {len(submission)} rows")
print(submission.head(10))

#%%
# Verify
sample_sub = pd.read_csv(cfg.DATA_PATH / "sample_submission.csv")
assert len(submission) == len(sample_sub), "Row count mismatch!"
print("✓ Format verified!")
