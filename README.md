<img width="4800" height="2960" alt="Byungsoo Kang - CSIRO - Image2Biomass Prediction" src="https://github.com/user-attachments/assets/d64e4f33-1359-4d61-b707-83b9c494deb2" />

# CSIRO Image2Biomass Prediction

> **Kaggle competition** hosted by CSIRO, MLA, and Google Australia.  
> Predict pasture biomass (dry weight, grams) from top-view quadrat images.

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Table of Contents

1. [Competition Overview](#-competition-overview)
2. [Dataset](#-dataset)
3. [Architecture](#-architecture)
4. [Experiment History](#-experiment-history)
5. [Directory Structure](#-directory-structure)
6. [Environment Setup](#-environment-setup)
7. [Training & Inference](#-training--inference)
8. [Key Techniques](#-key-techniques)
9. [Citation](#-citation)

---

## 🏆 Competition Overview

| Item | Detail |
|------|--------|
| **Host** | CSIRO · MLA · Google Australia |
| **Platform** | Kaggle |
| **Task** | Multi-output regression (5 biomass targets) |
| **Metric** | Globally weighted R² |
| **Deadline** | 2026-01-21 11:59 UTC |

### Target Variables

| Target | Description | Competition Weight |
|--------|-------------|-------------------|
| `Dry_Green_g` | Green vegetation dry weight (excl. clover) | 0.10 |
| `Dry_Dead_g` | Dead material dry weight | 0.10 |
| `Dry_Clover_g` | Clover (legume) dry weight | 0.10 |
| `GDM_g` | Green Dry Matter (= Green + Clover) | 0.20 |
| `Dry_Total_g` | Total dry weight (= GDM + Dead) | **0.50** |

`Dry_Total_g` dominates the metric with a 50% weight.

### Evaluation Metric

The competition uses a **globally weighted R²**:

$$R^2_w = 1 - \frac{\sum_j w_j (y_j - \hat{y}_j)^2}{\sum_j w_j (y_j - \bar{y}_w)^2}, \quad \bar{y}_w = \frac{\sum_j w_j y_j}{\sum_j w_j}$$

where $w_j$ is the per-row weight defined by the target type above.

---

## 📊 Dataset

- **Training images**: 357 images (70 cm × 30 cm top-view quadrats)
- **Test images**: 800+ images
- **Locations**: 19 sites across Australia (NSW, Vic, Tas, WA)
- **Year**: 2015 (multi-season)
- **Tabular features**: `State`, `Species`, `Pre_GSHH_NDVI` (GreenSeeker), `Height_Ave_cm`

### State × Season Distribution

```
        Autumn  Spring  Summer  Winter
NSW       23      11      41       0     ← No Winter
Tas       29      71       0      38     ← No Summer
Vic        0      39       0      73     ← No Autumn/Summer
WA         0      12       0      20     ← No Autumn/Summer
```

This confounding means standard K-Fold CV overestimates performance; the repository uses **StratifiedGroupKFold by State+Month** throughout.

### Clover Zero-Inflation

~37.8 % of training samples have `Dry_Clover_g = 0`, so clover uses a zero-inflated two-stage head.

---

## 🏗️ Architecture

```
Image (518×518)
    │
    ▼
DINOv2 ViT-B/14 ──────────────────────────────────────┐
(frozen backbone, 768-dim CLS token)                    │
    │                                                   │
    │           Tabular features                        │
    │           (NDVI, Height, State, Species)          │
    │                    │                              │
    │                    ▼                              │
    │          TabularFeatureEncoder                    │
    │          (embeddings + MLP → 128-dim)             │
    │                    │                              │
    │                    ▼                              │
    │               FiLM Fusion ←──────────────────────┘
    │          (γ, β from tabular modulate image feat)
    │
    ▼
Multi-Task Head
    ├── head_green  → softplus → Dry_Green_g
    ├── head_clover → zero-inflated (p_pos × amount) → Dry_Clover_g
    ├── head_dead   → softplus → Dry_Dead_g
    │
    │ Physics constraints (no extra parameters)
    ├── GDM_g   = Dry_Green_g + Dry_Clover_g
    └── Dry_Total_g = GDM_g + Dry_Dead_g
```

### Knowledge Distillation (Teacher → Student)

A **Teacher** model that receives both image and tabular features trains first (~20 epochs). A **Student** model (image only) then learns from both the ground-truth labels (hard targets) and the Teacher's soft predictions via KL-divergence loss, controlled by `kd_alpha`.

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Derive GDM / Total from independent heads | Prevents physics violations |
| Softplus activation for all raw heads | Enforces non-negativity |
| Zero-inflated head for Clover | 37.8 % zero samples |
| FiLM modulation (tabular → image) | More expressive than concat |
| `hue_jitter ≤ 0.02` | Colour is diagnostic for biomass |
| StratifiedGroupKFold by State+Month | Avoids spatial/temporal leakage |

---

## 📈 Experiment History

| Version | CV Score | LB Score | Key Changes |
|---------|----------|----------|-------------|
| v02–v11 | — | 0.50 | Baselines: ResNet/EfficientNet, LUPI, Knowledge Distillation |
| v12 | — | — | DINOv2 ViT-B/14 backbone introduced |
| v15 | 0.67 | 0.61 | Back-to-basics; simple regression head |
| v16–v17 | 0.79 | 0.69 | Optuna hyperparameter tuning (512 hidden, 3 layers) |
| v20 | 0.79 | 0.69 | Best stable baseline; used as reference |
| v22 | ~0.65 | 0.64 | Frozen backbone, simpler head |
| v25–v26 | — | — | OOF predictions + post-hoc calibration |
| cv1–cv8 | — | — | Iterative CV experiments; gap reduction |

> **Current best LB**: **0.69** (v20 / v17).  
> **Gold medal threshold**: LB ≥ 0.76.

---

## 📁 Directory Structure

```
CSIRO-Image2Biomass-Prediction/
├── src/                         # Core ML library
│   ├── config.py                # Hyperparameters & paths (CFG dataclass)
│   ├── dataset.py               # Dataset, transforms, fold creation
│   ├── models.py                # DINOv2Backbone, FiLM, Physics head
│   ├── losses.py                # ZeroInflatedLoss, competition_metric
│   └── trainer.py               # Teacher-Student KD training loop
│
├── notebooks/                   # Experiments (Jupytext .py + .ipynb pairs)
│   ├── 01_eda.py                # Exploratory data analysis
│   ├── 12_dinov3_{train,infer}  # DINOv2 baseline
│   ├── 16_hyperparameter_tuning # Optuna search
│   ├── 20_{train,infer}         # Best stable baseline
│   ├── 26_{train_oof,infer_calibrated} # OOF + calibration (latest numbered)
│   └── cv{1-8}_{train,infer}   # CV gap-reduction experiments
│
├── docs/                        # Strategy & analysis documents
│   ├── DINOV3_GOLD_STRATEGY.md  # Gold medal road-map
│   ├── DIAGNOSTIC_ANALYSIS.md   # CV-LB gap diagnosis
│   ├── HYBRID_APPROACH_DESIGN.md
│   └── data_augmentation.md
│
├── data/                        # Local data (gitignored)
├── pyproject.toml               # uv/pip dependencies
├── test_environment.py          # Quick environment sanity-check
├── data_description.md          # Official competition data description
├── README_COLOB_SETUP.md        # Colab quick-start guide
└── TORCHVISION_FIX.md           # torchvision.transforms.v2 compat notes
```

---

## ⚙️ Environment Setup

### Local (uv recommended)

```bash
# Clone
git clone https://github.com/kbsooo/CSIRO-Image2Biomass-Prediction.git
cd CSIRO-Image2Biomass-Prediction

# Create env & install
uv sync
# or: pip install -e .
```

### Colab / Kaggle

```python
!pip install torch torchvision timm transformers albumentations \
             scikit-learn pandas pillow tqdm kagglehub wandb
```

### Verify installation

```bash
python test_environment.py
```

---

## 🚀 Training & Inference

### Run latest experiment (v26 – OOF + Calibration)

```bash
# Step 1 – Train 5-fold models and save OOF predictions
python notebooks/26_train_oof.py

# Step 2 – Calibrated inference → submission.csv
python notebooks/26_infer_calibrated.py
```

### Run best stable baseline (v20)

```bash
python notebooks/20_train.py
python notebooks/20_infer.py
```

### Configuration

All hyperparameters live in `src/config.py` (`CFG` dataclass):

```python
from src.config import CFG

cfg = CFG()
cfg.n_folds = 5
cfg.batch_size = 16
cfg.teacher_epochs = 20
cfg.student_epochs = 25
cfg.freeze_backbone = True
```

### Pretrained DINOv2 weights (offline / Kaggle)

```python
import kagglehub
weights_path = kagglehub.dataset_download('kbsooo/pretrained-weights-biomass')
```

---

## 🔑 Key Techniques

| Technique | Location | Notes |
|-----------|----------|-------|
| DINOv2 ViT-B/14 backbone | `src/models.py` | Self-supervised, 768-dim CLS token |
| FiLM modulation | `src/models.py` | Tabular features modulate image features |
| Physics constraints | `src/models.py` | GDM = Green + Clover; Total = GDM + Dead |
| Zero-inflated Clover head | `src/models.py` | Classifier × regressor for sparse target |
| Teacher-Student KD | `src/trainer.py` | Teacher: img+tab; Student: img only |
| StratifiedGroupKFold | `src/dataset.py` | Stratify by State to reduce leakage |
| Conservative colour jitter | notebooks | `hue_jitter ≤ 0.02` |
| TTA (4×) | `src/config.py` | Original + h-flip + v-flip + both |
| OOF + post-hoc calibration | `notebooks/26_*` | Reduces CV-LB gap |
| Optuna hyperparameter search | `notebooks/16_*`, `17_*` | TPE sampler, 50 trials |

---

## 📝 Citation

```bibtex
@misc{liao2025estimatingpasturebiomasstopview,
  title   = {Estimating Pasture Biomass from Top-View Images: A Dataset for Precision Agriculture},
  author  = {Qiyu Liao and Dadong Wang and Rebecca Haling and Jiajun Liu and Xun Li
             and Martyna Plomecka and Andrew Robson and Matthew Pringle and Rhys Pirie
             and Megan Walker and Joshua Whelan},
  year    = {2025},
  eprint  = {2510.22916},
  archivePrefix = {arXiv},
  primaryclass  = {cs.CV},
  url     = {https://arxiv.org/abs/2510.22916},
}
```

---

*Experiment tracking: [WandB – kbsoo0620-/csiro](https://wandb.ai/kbsoo0620-/csiro)*
