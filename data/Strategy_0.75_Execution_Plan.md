# 🎯 CSIRO Biomass Competition - 0.75 달성 실행 계획서

## 📊 현재 상황

| 모델 | CV | Public LB | 상태 |
|------|-----|-----------|------|
| v27 (5-모델 앙상블) | ~0.80 (leakage) | **0.70** | ✅ 현재 최고 |
| CV1 (단일 모델) | 0.6366 | **0.68** | ✅ 정직한 CV |
| 1위 | ? | **0.79** | 🎯 목표 |

**목표**: Public LB **0.75** 이상 달성

---

## 🚀 Phase 1: Quick Win (v27 + TTA)

### 목표: 0.70 → 0.72~0.73

### 1.1 TTA (Test Time Augmentation) 추가

**원리**: 테스트 이미지에 여러 변환을 적용하고 예측 평균

```python
# TTA 변환 (4가지)
# 1. Original
# 2. Horizontal Flip
# 3. Vertical Flip
# 4. Both Flip

def predict_with_tta(model, left, right):
    predictions = []

    # Original
    pred = model(left, right)
    predictions.append(pred)

    # Horizontal flip
    pred = model(torch.flip(left, [3]), torch.flip(right, [3]))
    predictions.append(pred)

    # Vertical flip
    pred = model(torch.flip(left, [2]), torch.flip(right, [2]))
    predictions.append(pred)

    # Both flip
    pred = model(torch.flip(left, [2,3]), torch.flip(right, [2,3]))
    predictions.append(pred)

    return torch.stack(predictions).mean(0)
```

**예상 효과**: +0.01 ~ 0.02

### 1.2 WA State 후처리

**발견**: WA 주의 모든 샘플(32개)에서 Dry_Dead_g = 0

```python
# WA State 후처리
def apply_wa_postprocess(predictions, test_df):
    for idx, row in test_df.iterrows():
        if row['State'] == 'WA':
            predictions[idx, 1] = 0.0  # Dry_Dead_g index
    return predictions
```

**예상 효과**: +0.005 ~ 0.01

### 1.3 구현 파일

**파일명**: `v28_tta_infer.py`

```python
"""
v28: v27 + TTA + WA Postprocessing
목표: 0.70 → 0.72
"""

# === 변경 사항 ===
# 1. TTA 4x (original, hflip, vflip, both)
# 2. WA State Dead=0 후처리
# 3. 기존 5-모델 앙상블 유지

TTA_TRANSFORMS = ['original', 'hflip', 'vflip', 'both']

def apply_tta(img, transform):
    if transform == 'original':
        return img
    elif transform == 'hflip':
        return torch.flip(img, [3])
    elif transform == 'vflip':
        return torch.flip(img, [2])
    elif transform == 'both':
        return torch.flip(img, [2, 3])

def predict_single_model_tta(model, left, right):
    preds = []
    for t in TTA_TRANSFORMS:
        l = apply_tta(left, t)
        r = apply_tta(right, t)
        pred = model(l, r)
        preds.append(pred)
    return torch.stack(preds).mean(0)
```

### 1.4 Phase 1 예상 결과

| 개선 사항 | 예상 효과 |
|----------|----------|
| TTA 4x | +0.01 ~ 0.02 |
| WA 후처리 | +0.005 ~ 0.01 |
| **총합** | **+0.015 ~ 0.03** |

**예상 Public LB**: 0.70 + 0.02 = **0.72**

---

## 📈 Phase 2: Multi-Seed Ensemble

### 목표: CV1 0.68 → 0.71~0.72

### 2.1 전략

CV1을 여러 seed로 학습하여 다양성 확보

```python
SEEDS = [42, 123, 456, 789, 999]

# 각 seed로 5-fold 학습
# 총 25개 모델 (5 seeds × 5 folds)
```

### 2.2 학습 계획

| Seed | 학습 시간 (예상) | 모델 수 |
|------|-----------------|--------|
| 42 | 이미 완료 | 5개 |
| 123 | ~2시간 | 5개 |
| 456 | ~2시간 | 5개 |
| 789 | ~2시간 (선택) | 5개 |
| 999 | ~2시간 (선택) | 5개 |

### 2.3 앙상블 방법

```python
# 방법 1: 단순 평균
final_pred = np.mean([seed_preds for seed_preds in all_preds], axis=0)

# 방법 2: OOF 기반 가중치 최적화
from scipy.optimize import minimize

def optimize_weights(oof_preds_list, oof_targets):
    def loss(weights):
        weights = weights / weights.sum()
        blended = sum(w * p for w, p in zip(weights, oof_preds_list))
        return -competition_metric(oof_targets, blended)

    result = minimize(loss, np.ones(len(oof_preds_list)) / len(oof_preds_list))
    return result.x / result.x.sum()
```

### 2.4 구현 파일

**파일명**: `cv1_seed{N}_train.py` (N = 123, 456, ...)

```python
# 변경 사항: seed만 변경
def seed_everything(seed=123):  # 42 → 123
    random.seed(seed)
    ...
```

**파일명**: `cv1_multi_seed_infer.py`

```python
"""
CV1 Multi-Seed Ensemble
Seeds: 42, 123, 456
"""

SEEDS = [42, 123, 456]
MODEL_PATHS = {
    42: '/path/to/cv1_seed42/',
    123: '/path/to/cv1_seed123/',
    456: '/path/to/cv1_seed456/',
}
```

### 2.5 Phase 2 예상 결과

| Seed 수 | 예상 LB |
|---------|---------|
| 1 (현재) | 0.68 |
| 3 seeds | 0.70~0.71 |
| 5 seeds | 0.71~0.72 |

**예상 Public LB**: **0.71~0.72**

---

## 🔥 Phase 3: Hybrid Ensemble

### 목표: 0.73 → 0.75

### 3.1 전략

v27 (leakage CV)와 CV1 (honest CV)를 결합

```
v27의 장점: 모델 다양성, 과적합 (LB에 유리)
CV1의 장점: 정직한 CV, Private LB 안정성
```

### 3.2 앙상블 구성

```python
# v27: 5개 모델 × 5 fold × 4 TTA = 100개 예측
# CV1: 1개 모델 × 5 fold × 3 seeds × 4 TTA = 60개 예측

# 최종 앙상블
hybrid_pred = alpha * v27_pred + (1-alpha) * cv1_pred
# alpha는 OOF 기반 최적화 또는 0.5로 시작
```

### 3.3 가중치 최적화

```python
# CV1의 OOF만 있으므로, CV1 OOF로 v27 가중치 간접 추정
# 또는 단순히 0.5:0.5로 시작

def optimize_hybrid(v27_pred, cv1_pred, cv1_oof, cv1_oof_targets):
    """
    v27은 OOF가 없으므로, public LB 피드백으로 튜닝
    """
    alphas = [0.3, 0.4, 0.5, 0.6, 0.7]
    # 각 alpha로 제출하여 최적 값 찾기
    pass
```

### 3.4 구현 파일

**파일명**: `v29_hybrid_infer.py`

```python
"""
v29: Hybrid Ensemble (v27 + CV1)
목표: 0.73 → 0.75
"""

# v27 예측 로드
v27_pred = load_v27_predictions()

# CV1 multi-seed 예측 로드
cv1_pred = load_cv1_predictions()

# 하이브리드 앙상블
ALPHA = 0.5  # 시작점
final_pred = ALPHA * v27_pred + (1 - ALPHA) * cv1_pred

# WA 후처리
final_pred = apply_wa_postprocess(final_pred, test_df)
```

### 3.5 Phase 3 예상 결과

| 조합 | 예상 LB |
|------|---------|
| v27 alone | 0.70 |
| v27 + TTA | 0.72 |
| v27 + CV1 (0.5:0.5) | 0.73~0.74 |
| 최적화된 가중치 | **0.74~0.75** |

---

## 📋 전체 실행 타임라인

### Week 1

| 일차 | 작업 | 목표 LB |
|-----|------|---------|
| Day 1 | Phase 1: v28 (TTA + WA) 구현 및 제출 | 0.72 |
| Day 2 | Phase 2: CV1 seed 123 학습 | - |
| Day 3 | Phase 2: CV1 seed 456 학습 | - |
| Day 4 | Phase 2: Multi-seed 앙상블 제출 | 0.71 |
| Day 5 | Phase 3: Hybrid 앙상블 구현 | 0.73 |
| Day 6 | Phase 3: 가중치 튜닝 | 0.74 |
| Day 7 | 최종 제출 및 검증 | **0.75** |

### 제출 계획

| 버전 | 설명 | 예상 LB |
|------|------|---------|
| v28 | v27 + TTA + WA | 0.72 |
| v29 | CV1 × 3 seeds | 0.71 |
| v30 | Hybrid (v27 + CV1) | 0.74 |
| v31 | Hybrid + 최적화 | **0.75** |

---

## ⚠️ 리스크 관리

### Public vs Private LB

```
Public LB: 53% 샘플
Private LB: 47% 샘플
```

| 모델 | Public 예상 | Private 리스크 |
|------|------------|---------------|
| v27 (leakage) | 높음 | ⚠️ 하락 가능 |
| CV1 (honest) | 보통 | ✅ 안정적 |
| Hybrid | 높음 | ✅ 균형적 |

### 권장 최종 제출

1. **Best Public**: v30/v31 (Hybrid, 0.74~0.75)
2. **Safe Backup**: CV1 multi-seed (0.71~0.72)

---

## 🔧 필요한 리소스

### 컴퓨팅

| 작업 | GPU 시간 | 플랫폼 |
|------|---------|--------|
| CV1 seed 123 | ~2시간 | Colab/Kaggle |
| CV1 seed 456 | ~2시간 | Colab/Kaggle |
| Inference | ~30분 | Kaggle |

### 파일 구조

```
/kaggle/working/
├── v28_tta_infer.py          # Phase 1
├── cv1_seed123_train.py      # Phase 2
├── cv1_seed456_train.py      # Phase 2
├── cv1_multi_seed_infer.py   # Phase 2
├── v29_hybrid_infer.py       # Phase 3
└── models/
    ├── v27/                  # 기존 모델
    ├── cv1_seed42/           # 기존 CV1
    ├── cv1_seed123/          # 새로 학습
    └── cv1_seed456/          # 새로 학습
```

---

## ✅ 체크리스트

### Phase 1
- [ ] v28_tta_infer.py 작성
- [ ] TTA 구현 (4x)
- [ ] WA 후처리 구현
- [ ] 제출 및 LB 확인 (목표: 0.72)

### Phase 2
- [ ] cv1_seed123_train.py 작성
- [ ] Seed 123 학습 완료
- [ ] cv1_seed456_train.py 작성
- [ ] Seed 456 학습 완료
- [ ] Multi-seed 앙상블 inference
- [ ] 제출 및 LB 확인 (목표: 0.71)

### Phase 3
- [ ] v29_hybrid_infer.py 작성
- [ ] v27 + CV1 앙상블
- [ ] 가중치 최적화
- [ ] 최종 제출 (목표: 0.75)

---

## 🎯 성공 기준

| 단계 | 목표 LB | 상태 |
|------|---------|------|
| Phase 1 | 0.72 | ⬜ 대기 |
| Phase 2 | 0.71 | ⬜ 대기 |
| Phase 3 | 0.75 | ⬜ 대기 |

**최종 목표**: Public LB **0.75** 이상

---

*Plan Created: 2025-01-23*
*Current Best: Public LB 0.70*
*Target: Public LB 0.75+*
