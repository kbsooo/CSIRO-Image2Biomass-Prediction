# CSIRO - Image2Biomass Prediction

**Competition**: [Kaggle CSIRO Biomass](https://www.kaggle.com/competitions/csiro-biomass/overview)
**Task**: 목초지(Pasture) top-view 이미지로 바이오매스(건조 중량) 예측
**Metric**: Globally Weighted R² (Dry_Total_g 50%, GDM_g 20%, 나머지 각 10%)
**Data**: 357개 이미지 (70cm × 30cm quadrat), 19개 호주 사이트

---

## 🏆 최종 결과 요약

| 버전 | CV Score | Public LB | 핵심 특징 |
|------|----------|-----------|-----------|
| v27 (5-model ensemble) | ~0.80 ⚠️leakage | **0.70** | 현재 최고 Public |
| CV1 (honest CV) | 0.6366 | **0.68** | 정직한 CV 기준점 |
| CV3 (preprocessing) | 0.5253 | **0.65** | 전처리가 역효과 |
| 1위 | ? | **0.79** | 목표 |

> ⚠️ v27의 CV ~0.80은 data leakage가 있는 허위 수치. Private LB에서 하락 위험 있음.

---

## 📁 파일 구조

```
notebooks/
├── 01_eda.py                   # 초기 EDA
├── 02~11_*.py                  # 초기 실험 (DINOv2 기반, LB ~0.50)
├── 12~19_*.py                  # DINOv3 전환 (LB ~0.70 달성)
├── 20~27_*.py                  # v20 계열 (v27 최고점 0.70)
├── cv1_train/infer.py          # 정직한 CV 기준점 (LB 0.68)
├── cv2_train/infer.py          # Weighted Loss + Frozen backbone (실패)
├── cv3_train/infer.py          # 이미지 전처리 실험 (역효과)
├── cv4_infer.py                # v27 + TTA + WA 후처리
├── cv4a/cv4b_infer.py          # Kaggle timeout 대응 경량 버전
├── cv5_train/infer.py          # ConvNeXt-Base 멀티 백본
├── cv5e_infer.py               # DINOv3 + ConvNeXt 앙상블
├── cv6_train/infer.py          # 직사각형 full-frame + SSF
├── cv7_train/infer.py          # v26 OOF training
├── cv7a_train.py               # EMA 적용
├── cv7b_train.py               # EMA variant 2
└── cv8_train/infer.py          # LLRD (Layer-wise LR Decay)

data/
├── EDA_Report_CSIRO_Biomass.md
├── Full_Analysis_Report.md
├── Breakthrough_Strategy_0.70_to_0.79.md
├── CV3_Strategy_0.72_Target.md
├── CSIRO_Competition_Strategy_DINOv2.md
└── Strategy_0.75_Execution_Plan.md

docs/
├── DINOV3_GOLD_STRATEGY.md     # 핵심 아키텍처 설계 문서
├── DIAGNOSTIC_ANALYSIS.md
└── HYBRID_APPROACH_DESIGN.md
```

---

## 🗺️ 실험 연대기

### Phase 0: EDA & 데이터 파악

**파일**: `01_eda.py`
**주요 발견**:
- Train: 357개 고유 이미지, 각 이미지당 5개 샘플 (타겟별) → 총 1,785행
- Test: 메타데이터 없음 → **이미지만으로 예측해야 함**
- 타겟 분포:
  - `Dry_Clover_g`: 37.8%가 0 (심한 sparsity)
  - `Dry_Total_g` ≈ `Dry_Green_g` + `Dry_Dead_g` + `Dry_Clover_g` (물리적 관계)
  - `GDM_g` ≈ `Dry_Green_g` + `Dry_Clover_g`
- `Height_Ave_cm`과 `Dry_Green_g` 상관계수 0.648 (가장 강함)
- WA(Western Australia) 32개 샘플 전부 `Dry_Dead_g = 0` (100% 패턴)

---

### Phase 1: 초기 실험 — DINOv2 ViT-Base 시대 (v02~v11)

**파일**: `02~11_*.py`
**접근법**: DINOv2 ViT-Base(86M) → Frozen backbone → 다양한 Head 실험

| 실험 | 특징 |
|------|------|
| 02~06 | ResNet/EfficientNet baseline, Kaggle/Colab 환경 설정 |
| 07 | Physics-constrained head (GDM=G+C, Total=GDM+D) |
| 08 | Auxiliary task learning |
| 09~10 | Pseudo labeling |
| 11 | LUPI/Knowledge Distillation hybrid approach |

**결과**: LB ~0.50에서 정체
**원인 분석**:
- ViT-Base는 feature 표현력이 부족
- Frozen backbone → 도메인 적응 불가
- LUPI/KD는 Teacher ceiling(0.62)이 Student 상한선이 됨

---

### Phase 2: DINOv3 ViT-Large 전환 (v12~v19)

**핵심 전환**: Public notebook `070.py` 분석으로 핵심 구조 파악

```
핵심 인사이트:
1. DINOv3 ViT-Large (~300M) >> ViT-Base (86M)
2. 이미지를 Left/Right 절반으로 분할 후 FiLM fusion
3. Physics constraints (GDM=G+C, Total=GDM+D)
4. 5-Fold Ensemble + TTA
```

**아키텍처 (v12~ 이후 표준)**:

```
Input Image (70cm × 30cm)
        │
   ┌────┴────┐
   ▼         ▼
Left Half  Right Half (각각 512×512 resize)
   │         │
   └────┬────┘
        │ DINOv3 ViT-Large (공유 backbone)
        │ output: 1024-dim
        ▼
      FiLM Module (cross-region context sharing)
      γ = Tanh(MLP(left+right)/2)
      β = Tanh(MLP(left+right)/2)
      left_mod = left × (1+γ) + β
        │
   Concat(left_mod, right_mod) → 2048-dim
        │
   ┌────┼────┐
Head_Green  Head_Clover  Head_Dead
   │           │           │
   └─────Softplus (non-negative)
        │
   Physics Layer: GDM = G+C, Total = GDM+D
        │
   [Green, Dead, Clover, GDM, Total]
```

**실험 흐름**:

| 버전 | 핵심 변경 | 결과 |
|------|-----------|------|
| v12 | DINOv3 ViT-Large + FiLM 첫 구현 | 기준점 |
| v13 | 최적화 (AMP, gradient clipping) | - |
| v14 | 개선 | - |
| v15 | 기본으로 복귀 | - |
| v16 | Optuna HPO 추가 | - |
| v17 | Optuna 최적화 적용 | - |
| v17b | CV-LB gap 감소 시도 | - |
| v18 | Simple model tuning | - |
| v19 | Trial 22 vs 27 비교 | - |

---

### Phase 3: v20 계열 — 최고점 달성 (v20~v27)

#### v20: 핵심 baseline 확립

**파일**: `20_train.py`
**주요 특징**:
- `hidden_dim=512, num_layers=3, dropout=0.1`
- Sampling_Date 기반 `StratifiedGroupKFold` ← 이후 발견적으로 중요
- AMP(자동 혼합 정밀도) 학습
- Cosine scheduler with warmup

#### v21: LOGO (Leave-One-Group-Out)

**파일**: `21_train.py`
**아이디어**: Location(Site) 기반 leave-one-out CV로 더 엄격한 validation
**결과**: CV 하락 (데이터가 너무 작음)

#### v22: Frozen backbone + 강한 정규화

**파일**: `22_train.py`
**특징**: `hidden_dim=256, num_layers=2`, backbone 고정
**결과**: CV 하락 - fine-tuning이 필요함을 확인

#### v23: 최적화된 LOGO

**파일**: `23_train.py`
**결과**: 개선 없음

#### v24: TENT (Test-Time Entropy minimization)

**파일**: `24_infer_tent.py`
**아이디어**: Test 데이터의 BN 통계로 모델 adapt
**결과**: Hidden test set에서 오류 발생으로 실패

#### v25: Vegetation Index Late Fusion

**파일**: `25_train.py`
**아이디어**: NDVI/Height 등 tabular feature를 추가 입력으로
**결과**: Test에 메타데이터 없어 효과 제한적

#### v26: OOF (Out-of-Fold) 저장

**파일**: `26_train_oof.py`
**목적**: 앙상블 가중치 최적화를 위한 OOF 예측 수집
**특징**: v20 구조 유지, OOF 파일 추가 저장

#### v27: 5-모델 앙상블 — **현재 최고점 LB 0.70**

**파일**: `27_train.py`, `27_infer.py`

```python
# v27의 5개 모델 조합
MODELS = {
    'v20': (hidden_dim=512, layers=3),  # 기본 구조
    'v22': (hidden_dim=256, layers=2),  # 작은 Head, frozen backbone
    'v23': (hidden_dim=512, layers=3),  # v20과 동일, 다른 seed
    'v25': VegetationEncoder + FiLM,    # VegIdx late fusion
    'v26': (hidden_dim=512, layers=3),  # OOF 버전
}
# 앙상블: Simple average (5 × 5-fold = 25개 모델)
```

**왜 0.70을 달성했나**:
- 모델 다양성: 3가지 구조 × 다른 seed → 앙상블 효과
- 5개 모델 × 5-fold = 25개 예측의 평균

**약점**:
- CV leakage (image_id로 grouping → 같은 날짜 이미지들이 섞임)
- CV ~0.80은 허위 — 과적합된 것
- TTA 없음

---

### Phase 4: CV 개혁 — 정직한 Cross-Validation

**배경**: Discussion 126 votes 인사이트
> "반드시 `Sampling_Date`로 GroupKFold 해야 함. 같은 날짜 이미지들은 동일한 날씨/조명 조건 공유. image_id로 하면 심각한 data leakage."

#### CV1: 정직한 CV 기준점 — **LB 0.68**

**파일**: `cv1_train.py`, `cv1_infer.py`
**핵심 변경**:

```python
# ❌ 이전 (leakage)
groups = df['image_id']

# ✅ CV1 (정직)
groups = df['Sampling_Date']  # 날짜별 완전 분리
sgkf = StratifiedGroupKFold(n_splits=5)
# + State_Month stratification key
```

```
변경사항:
- CV Split: image_id → Sampling_Date (data leakage 제거)
- 해상도: 512 → 560 (14와 16 모두의 배수)
- TTA: 4-fold (Original × HFlip × VFlip)
- Head: hidden_dim=256, num_layers=2, dropout=0.3
```

**결과**:
- CV Score: 0.6366 (정직, fold별 0.71, 0.65, 0.64, 0.66, 0.63)
- Public LB: **0.68**
- CV-LB gap: ~0.04 (이전 v27의 gap ~0.10보다 훨씬 작음)

**의미**: 정직한 CV 확보 → 이후 실험의 신뢰 가능한 기준점

---

#### CV2: Weighted Loss + Frozen backbone

**파일**: `cv2_train.py`
**실험 목적**: 대회 평가지표(Weighted R²)에 맞춘 loss function

```python
# Competition weights: Total=0.5, GDM=0.2, 나머지=0.1
loss = weighted_mse(pred, target, weights=[0.1, 0.1, 0.1, 0.2, 0.5])
+ frozen backbone (fine-tuning 없이 head만 학습)
```

**결과**: CV 0.5966 → **실패**
**원인**:
- Frozen backbone: 357개 소규모 데이터에서 도메인 갭 해결 불가
- Weighted Loss: MSE 대비 학습 불안정

**서브 실험**: Optuna HPO (`cv2_optuna.py`)
- 작은 Head 탐색: `hidden_dim=[64,128,256,512]`, `num_layers=[1,2,3]`
- 결론: `hidden_dim=256, num_layers=2`가 최적 (큰 head는 357개에 과적합)

---

#### CV3: 이미지 전처리 — 역효과

**파일**: `cv3_train.py`, `cv3_infer.py`
**실험 목적**: Discussion 보고 (0.60 → 0.62 개선)

```python
def clean_image(img):
    # 1. Bottom 10% crop (color chart, cardboard artifacts)
    img = img[0:int(h*0.90), :]

    # 2. Orange timestamp inpainting (HSV mask + cv2.inpaint)
    mask = cv2.inRange(hsv, [5,150,150], [25,255,255])
    img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
```

**결과**: CV 0.5253 → Public LB **0.65** → **역효과**

**원인 분석**:
```
0.60 수준: 노이즈 제거 → 신호 대비 향상 → 효과적
0.70 수준: 정보 손실이 더 큼 → 오히려 해로움
결론: 고득점 구간에서는 "노이즈 제거"보다 "정보 보존"이 중요
```

**병행 실험**: WA State Dead=0 후처리 개발
```python
# WA 32개 샘플 모두 Dead=0 → 100% 패턴 → 강제 후처리
if state == 'WA':
    pred[dead_idx] = 0.0
    pred[gdm_idx] = green + clover
    pred[total_idx] = green + clover  # dead=0이므로
```

---

### Phase 5: 앙상블 전략 강화

#### CV4: v27 + TTA + WA 후처리

**파일**: `cv4_infer.py`
**목표**: 기존 v27 체크포인트 활용, 빠른 점수 향상

```
v27 5개 모델 × 5-fold = 25 체크포인트
+ 4-fold TTA (Original, HFlip, VFlip, Both)
+ WA State Dead=0 강제 후처리
= 총 100개 예측 평균
```

**파생 버전**:
- `cv4a_infer.py`: 2-fold TTA만 (Kaggle 시간 제한 대응)
- `cv4b_infer.py`: v20/v22/v26 3개 모델만 (timeout 문제 해결)

---

#### CV5: ConvNeXt-Base 멀티 백본 앙상블

**파일**: `cv5_train.py`, `cv5_infer.py`
**아이디어**: Transformer + CNN의 이중 관점 앙상블

```
DINOv3 ViT-Large: Global attention, long-range dependency
ConvNeXt-Base (ImageNet-22k): Local patterns, hierarchical features
→ 서로 다른 inductive bias → 앙상블 다양성 극대화
```

```python
# cv5e (앙상블 inference)
CV5E_WEIGHT = 0.3   # ConvNeXt: 다양성 목적
V27_WEIGHT = 0.7    # DINOv3: 메인 모델
```

**특징**:
- ConvNeXt: `convnext_base.fb_in22k_ft_in1k`, feat_dim=1024
- 이미지 크기: 560×560 (DINOv3와 동일)
- WandB 실험 추적

---

### Phase 6: 아키텍처 실험 (CV6~CV8)

#### CV6: 직사각형 full-frame + SSF Adapters

**파일**: `cv6_train.py`, `cv6_infer.py`
**핵심 아이디어**: 이미지를 Left/Right로 자르지 말고, 원래 비율 유지

```
기존: Left(512×512) + Right(512×512) → FiLM fusion
CV6:  Full-frame (784×336)           → 단일 입력
     ↑ 70cm×30cm 원래 비율 근사
```

```python
class CFG:
    img_size = (336, 784)      # Height × Width (비율 유지)
    freeze_backbone = True     # Frozen + SSF만 학습
    use_ssf = True             # Scale-Shift Feature adapters

# SSF: 각 Transformer block 출력에 학습 가능한 scale/shift 적용
# γ, β만 학습 → Frozen backbone에서 도메인 적응 가능
class SSFAdapter(nn.Module):
    # scale: ξ * x + γ (per-feature)
```

**추가 특징**:
- `ZeroInflatedHead`: Clover의 37.8% zero 처리 (분류 + 회귀 2-stage)
- `CLS + Patch Mean` pooling (richer aggregation)
- `ZeroInflatedLoss`: BCE(zero/nonzero) + MSE(nonzero only)

**목표**: SSF로 frozen backbone의 한계 극복하면서 overfitting 방지

---

#### CV7: OOF 기반 재학습

**파일**: `cv7_train.py`, `cv7_infer.py`
**기반**: v26 구조
**목적**: 다른 해상도/날짜 split 설정으로 OOF 품질 개선

---

#### CV7a: EMA (Exponential Moving Average)

**파일**: `cv7a_train.py`
**핵심 추가**:

```python
class EMA:
    """Shadow weights: θ_ema = decay * θ_ema + (1-decay) * θ"""
    # 학습 중 weight의 지수 이동 평균 유지
    # Inference: EMA weights 사용 → 안정적, 일반화 우수

class CFG:
    use_ema = True
    ema_decay = 0.999
    backbone_lr_mult = 0.084  # LLRD: backbone lr = lr × 0.084
```

**왜 EMA인가**: 소규모 데이터(357개)에서 weight 진동 완화, 더 부드러운 loss landscape

---

#### CV7b: EMA variant 2

**파일**: `cv7b_train.py`
EMA 설정 변형 실험

---

#### CV8: LLRD (Layer-wise Learning Rate Decay)

**파일**: `cv8_train.py`
**핵심 아이디어**:

```python
# Transformer 레이어별 LR 감쇠 적용
# 깊은 레이어 (하위 레이어): 작은 LR → 사전학습 보존
# 상위 레이어 + Head: 큰 LR → 태스크 적응
layer_lr_decay = 0.9  # 레이어당 10% 감쇠
# Layer 0 lr = base_lr × decay^n_layers
# Layer n lr = base_lr × decay^0 = base_lr
```

---

## 🔑 핵심 인사이트 및 교훈

### 1. CV 전략이 가장 중요

```
image_id grouping → CV 0.80 (허위) → LB 0.70, CV-LB gap 0.10
Sampling_Date grouping → CV 0.64 (정직) → LB 0.68, CV-LB gap 0.04
```

→ 정직한 CV 없이는 아무것도 신뢰할 수 없다.

### 2. 이미지 전처리의 역설

```
낮은 점수(0.60): 전처리 노이즈 제거 → +0.02 효과 있음 (Discussion 보고)
높은 점수(0.70): 전처리 → 오히려 -0.05 (CV3)
```

→ 정보 보존 vs 노이즈 제거의 trade-off는 점수 수준에 따라 다름

### 3. 앙상블 다양성 > 단순 정확도

```
단일 모델 (CV1): LB 0.68
5-모델 앙상블 (v27): LB 0.70
차이: +0.02 (모델 다양성만으로)
```

### 4. 소규모 데이터 + 강력한 Backbone

```
데이터: 357개 이미지 (매우 작음)
최적 Head: hidden_dim=128~256, num_layers=1~2, dropout=0.3~0.5
큰 Head(hidden_dim=512, layers=3) → 과적합
DINOv3 ViT-Large가 이미 1024-dim 풍부한 feature 제공
```

### 5. Physics Constraints는 무료 성능 향상

```python
# 모델이 물리 법칙을 자동으로 만족하도록 강제
GDM = Green + Clover      # 별도 학습 불필요
Total = GDM + Dead        # 항상 consistent
```

---

## 🏗️ 핵심 아키텍처 (표준 v20 계열)

```python
class CSIROModel(nn.Module):
    backbone = timm.create_model(
        "vit_large_patch16_dinov3_qkvb.lvd1689m",
        pretrained=False  # 별도 weights 로드
    )  # DINOv3 ViT-Large, 1024-dim output

    film = FiLM(feat_dim=1024)           # Cross-region modulation

    head_green  = MLP(2048 → 512 → 1)   # Dry_Green_g
    head_clover = MLP(2048 → 512 → 1)   # Dry_Clover_g
    head_dead   = MLP(2048 → 512 → 1)   # Dry_Dead_g

    # Physics: GDM = G+C, Total = GDM+D (computation, not learning)
```

**학습 설정**:
- Optimizer: AdamW
- LR: `1e-4` (head), `1e-5` (backbone, ×0.1)
- Scheduler: Cosine with warmup
- Loss: MSE on [Green, Dead, Clover] (GDM, Total은 계산으로 자동)
- Image size: 560×560 (14와 16 모두의 배수)
- Augmentation: HFlip, VFlip, ColorJitter
- AMP (Mixed Precision)

---

## 🚀 현재 도전 과제 및 다음 전략

### 문제: 0.70 → 0.79 gap (0.09)

**예상 효과별 전략**:

| 전략 | 예상 향상 | 상태 |
|------|----------|------|
| CV 수정 (Sampling_Date) | +0.03~0.05 | ✅ CV1에서 완료 |
| 해상도 560 + TTA | +0.02~0.03 | ✅ CV1에서 완료 |
| Optuna HPO (작은 Head) | +0.02~0.03 | 🔜 진행 예정 |
| EMA (CV7a) | +0.01~0.02 | 🔜 실험 중 |
| LLRD (CV8) | +0.01~0.02 | 🔜 실험 중 |
| Multi-backbone (DINOv3+ConvNeXt) | +0.02~0.03 | 🔜 CV5E |
| Weighted Loss alignment | +0.01~0.02 | 실험 필요 |
| Multi-seed ensemble | +0.01 | 실험 필요 |
| OOF 기반 앙상블 최적화 | +0.01 | 실험 필요 |

**주의**: Private LB는 전체의 47%. v27(leakage CV) 기반 제출은 Private에서 하락 위험.

---

## 🛠️ 환경 설정

```bash
# 주요 의존성
torch>=2.0  timm  albumentations  transformers  wandb  optuna

# DINOv3 weights (Kaggle dataset: pretrained-weights-biomass)
# /kaggle/input/pretrained-weights-biomass/dinov3_large/dinov3_large/dinov3_vitl16_qkvb.pth
```

**학습/추론 분리 전략**:
```
1. [학습] cv*_train.py → Colab/Kaggle GPU에서 실행 (~80분)
                       → fold*.pth 저장 → Google Drive 또는 Kaggle Dataset
2. [추론] cv*_infer.py → Kaggle 제출 노트북 (가중치만 로드, ~1분)
```

---

*마지막 업데이트: 2026-02-26*
*현재 최고 Public LB: 0.70 (v27)*
*정직한 CV 기준점: 0.68 (CV1)*
