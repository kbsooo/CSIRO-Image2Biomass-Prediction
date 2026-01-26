#%% [markdown]
# # CV6 Inference (Single Model)
#
# - Full-frame rectangular input
# - CLS + patch mean pooling
# - SSF adapters enabled
# - Zero-inflated clover + physics constraints

#%%
import warnings
warnings.filterwarnings('ignore')

import os
import gc
import random
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

import timm

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

tqdm.pandas()

#%%
# Reproducibility

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def flush() -> None:
    gc.collect()
    torch.cuda.empty_cache()


#%%
class CFG:
    DATA_PATH = Path("/kaggle/input/csiro-biomass")
    WEIGHTS_PATH = Path("/kaggle/input/pretrained-weights-biomass/dinov3_large/dinov3_large")
    # Update this to your uploaded model dataset
    MODEL_PATH = Path("/kaggle/input/csiro-cv6-models/cv6_model_best.pth")

    model_name = "vit_large_patch16_dinov3_qkvb.lvd1689m"
    img_size = (336, 784)  # (H, W)

    head_dim = 256
    head_layers = 2
    dropout = 0.2

    freeze_backbone = True
    use_ssf = True
    ssf_per_block = True

    batch_size = 16
    num_workers = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    use_tta = False  # keep false for deterministic single-pass


cfg = CFG()
seed_everything(42)

TARGET_ORDER = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

#%%
# Transforms

def get_test_transform(cfg: CFG) -> T.Compose:
    return T.Compose([
        T.Resize(cfg.img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

#%%
class TestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: CFG, transform: T.Compose):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(self.cfg.DATA_PATH / row['image_path']).convert('RGB')
        img = self.transform(img)
        assert img.ndim == 3 and img.shape[0] == 3, f"Unexpected image shape: {img.shape}"
        assert img.shape[1] % 16 == 0 and img.shape[2] % 16 == 0, (
            f"Image H/W must be divisible by 16, got {img.shape[1:]}"
        )
        return img, row['sample_id_prefix']

#%%
class SSFAdapter(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, 1, dim))
        self.shift = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.shift


class DINOv3Backbone(nn.Module):
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=False,
            num_classes=0,
            global_pool=""
        )

        weights_file = cfg.WEIGHTS_PATH / "dinov3_vitl16_qkvb.pth"
        if weights_file.exists():
            state = torch.load(weights_file, map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(state, strict=False)
            print(f"✓ Loaded backbone weights from {weights_file}")
        else:
            print("⚠️ Backbone weights not found; using random init")

        self.num_features = self.backbone.num_features

        if cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.ssf_per_block = cfg.use_ssf and cfg.ssf_per_block
        self.use_ssf = cfg.use_ssf
        self.ssf_blocks = None
        self._hooks = []

        if self.use_ssf:
            if self.ssf_per_block:
                self._init_ssf_per_block()
            else:
                self.ssf_out = SSFAdapter(self.num_features)

    def _init_ssf_per_block(self) -> None:
        assert hasattr(self.backbone, 'blocks'), "Backbone has no blocks attribute"
        self.ssf_blocks = nn.ModuleList([
            SSFAdapter(self.num_features) for _ in range(len(self.backbone.blocks))
        ])

        def make_hook(i: int):
            def hook(_module, _input, output):
                return self.ssf_blocks[i](output)
            return hook

        for i, blk in enumerate(self.backbone.blocks):
            self._hooks.append(blk.register_forward_hook(make_hook(i)))

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone.forward_features(x)
        if isinstance(tokens, (tuple, list)):
            tokens = tokens[0]
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(1)
        if self.use_ssf and (not self.ssf_per_block):
            tokens = self.ssf_out(tokens)
        return tokens


class ZeroInflatedHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor):
        p_pos = self.classifier(x)
        amount = self.regressor(x)
        pred = p_pos * amount
        return p_pos, amount, pred


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        layers = []
        cur = in_dim
        for i in range(num_layers):
            layers.append(nn.Linear(cur, hidden_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
            cur = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CV6Model(nn.Module):
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg
        self.backbone = DINOv3Backbone(cfg)
        feat_dim = self.backbone.num_features
        self.out_dim = feat_dim * 2

        self.green_head = MLPHead(self.out_dim, cfg.head_dim, cfg.head_layers, cfg.dropout)
        self.dead_head = MLPHead(self.out_dim, cfg.head_dim, cfg.head_layers, cfg.dropout)
        self.clover_head = ZeroInflatedHead(self.out_dim, hidden_dim=cfg.head_dim, dropout=cfg.dropout)

        self.softplus = nn.Softplus(beta=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone.forward_tokens(x)
        B, N, C = tokens.shape
        cls = tokens[:, 0]
        if N > 1:
            patch_mean = tokens[:, 1:].mean(dim=1)
        else:
            patch_mean = cls

        feat = torch.cat([cls, patch_mean], dim=1)

        green = self.softplus(self.green_head(feat))
        dead = self.softplus(self.dead_head(feat))
        _, _, clover = self.clover_head(feat)

        gdm = green + clover
        total = gdm + dead

        return torch.cat([green, dead, clover, gdm, total], dim=1)

#%%
@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, list]:
    model.eval()
    preds = []
    ids = []
    for images, sample_ids in tqdm(loader, desc="Predict"):
        images = images.to(device)
        out = model(images)
        preds.append(out.cpu().numpy())
        ids.extend(sample_ids)
    return np.concatenate(preds, axis=0), ids


#%%
# Load test data

test_df = pd.read_csv(cfg.DATA_PATH / "test.csv")

test_df['sample_id_prefix'] = test_df['sample_id'].str.split('__').str[0]
# Unique images only
test_wide = test_df.drop_duplicates(subset=['image_path']).reset_index(drop=True)
print(f"Test images: {len(test_wide)}")

transform = get_test_transform(cfg)

test_ds = TestDataset(test_wide, cfg, transform)
loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

#%%
# Load model

model = CV6Model(cfg).to(cfg.device)
state = torch.load(cfg.MODEL_PATH, map_location=cfg.device)
model.load_state_dict(state, strict=True)
print(f"✓ Loaded model: {cfg.MODEL_PATH}")

preds, sample_ids = predict(model, loader, cfg.device)
print(f"Pred shape: {preds.shape}")

#%%
# Build submission

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

print(f"Saved submission.csv with {len(submission)} rows")

#%%
# Sanity check
sample_sub = pd.read_csv(cfg.DATA_PATH / "sample_submission.csv")
assert len(submission) == len(sample_sub)
print("✓ Format verified")

#%%
# Cleanup

del model
flush()
