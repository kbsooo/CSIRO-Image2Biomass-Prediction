# 🚀 Breakthrough Strategy: 0.70 → 0.79+

## 📊 현재 상황 분석

### 현재 점수
- **Your Best Public LB**: 0.70
- **1위 Public LB**: 0.79
- **Gap**: 0.09 (상당히 큰 차이)

### 현재 코드 분석

| Version | 특징 | 문제점 |
|---------|------|--------|
| v20/v26 | DINOv2 Large + FiLM + Dual View | 기본 베이스라인 |
| v22 | Frozen backbone + 작은 Head | 제한된 학습 |
| v25 | VegIdx Late Fusion | 추가 정보지만 효과 제한적 |
| v27 | 단순 앙상블 (Simple/Rank Average) | 최적화되지 않은 앙상블 |

### 🔴 핵심 문제점 발견

#### 1. **CV 전략 오류** ⚠️ (가장 심각) - ✅ 해결됨
```python
# ❌ 이전 코드 (잘못된 방법)
groups = df['image_id']  # image_id로 그룹핑

# ✅ cv1에서 수정됨
groups = df['Sampling_Date']  # 날짜별 그룹핑!
```

**문제**: Discussion에서 126 votes를 받은 핵심 인사이트는 **Sampling_Date로 그룹핑**해야 한다는 것!
- 같은 날짜에 촬영된 이미지들은 비슷한 조건 공유
- `image_id`로 그룹핑하면 같은 날짜의 다른 이미지가 train/val에 분리됨
- **심각한 data leakage → overfitting**

#### 2. **이미지 해상도 제한** - ✅ 해결됨
```python
# ❌ 이전
img_size = (512, 512)

# ✅ cv1에서 수정됨
img_size = (560, 560)  # 14와 16 모두의 배수
```

#### 3. **TTA 미사용** - ✅ 해결됨
- cv1_infer.py에서 4-fold TTA (HFlip x VFlip) 구현됨

#### 4. **앙상블 최적화 부족**
- 가중치 최적화 없음
- 모델 다양성 부족 (모두 같은 backbone)

#### 5. **Loss Function**
```python
main_loss = F.mse_loss(pred, main_targets)  # 단순 MSE
```
- 대회 평가 지표(Weighted R²)와 다른 loss 사용
- Dry_Total_g가 50% 가중치인데 동일하게 취급

---

## 🎯 Breakthrough 전략 (우선순위 순)

### ✅ Priority 1: CV 전략 수정 (예상 +0.03~0.05) - 완료!

**cv1_train.py에서 구현됨**

```python
def create_proper_folds(df, n_splits=5):
    """Sampling_Date 기반 올바른 CV split"""
    df = df.copy()
    df['date_group'] = pd.to_datetime(df['Sampling_Date']).dt.strftime('%Y-%m-%d')
    df['strat_key'] = df['State'] + '_' + df['Month'].astype(str)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    df['fold'] = -1
    for fold, (_, val_idx) in enumerate(sgkf.split(
        df,
        df['strat_key'],
        groups=df['date_group']  # ⚠️ 핵심: date로 그룹핑!
    )):
        df.loc[val_idx, 'fold'] = fold

    return df
```

### ✅ Priority 2: 더 큰 해상도 + TTA (예상 +0.02~0.03) - 완료!

**cv1에서 구현됨:**
- 해상도: 560x560 (14와 16 모두의 배수)
- TTA: 4-fold flip (HFlip x VFlip)

### 🔥 Priority 3: Optuna HPO (예상 +0.02~0.03) ⭐ NEW

**이제 CV가 정직해졌으니 HPO가 의미있음!**

```python
import optuna

def objective(trial):
    cfg = CFG()

    # ⭐ 작은 값부터 탐색! (DINOv2가 이미 좋은 feature 추출)
    cfg.hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
    cfg.num_layers = trial.suggest_int('num_layers', 1, 3)  # 1부터!
    cfg.dropout = trial.suggest_float('dropout', 0.1, 0.5)  # 높은 dropout도

    # lr, weight_decay
    cfg.lr = trial.suggest_float('lr', 1e-5, 1e-4, log=True)
    cfg.weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True)

    # 1-2 fold만 빠르게 검증
    cv_score = train_and_evaluate(cfg, folds=[0, 1])

    return cv_score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

#### 왜 작은 Head가 더 좋을 수 있는가?

| 요소 | 설명 |
|------|------|
| DINOv2 Large | 이미 1024 dim의 풍부한 feature 추출 |
| 데이터 크기 | 357개 이미지 (매우 작음!) |
| 결론 | 큰 Head = overfitting, 작은 Head = 일반화 ↑ |

#### Optuna 탐색 범위 (수정됨)

| 파라미터 | 이전 범위 | 수정된 범위 |
|----------|-----------|-------------|
| hidden_dim | 256, 512, 768 | **64, 128, 256, 512** |
| num_layers | 2, 3, 4 | **1, 2, 3** |
| dropout | 0.1~0.3 | **0.1~0.5** |

#### 예상 최적값 (가설)
```python
# 357개 이미지 + DINOv2 Large 조합
cfg.hidden_dim = 128  # 또는 256
cfg.num_layers = 1    # 또는 2
cfg.dropout = 0.3     # 높은 regularization
```

### 🔥 Priority 4: Weighted Loss (예상 +0.01~0.02)

```python
class WeightedR2Loss(nn.Module):
    """대회 평가 지표에 맞춘 Loss"""
    def __init__(self):
        super().__init__()
        # [Green, Dead, Clover, GDM, Total]
        self.weights = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5])

    def forward(self, pred, target):
        green, clover, dead = pred[:, 0:1], pred[:, 2:3], pred[:, 1:2]
        gdm_pred = green + clover
        total_pred = gdm_pred + dead

        full_pred = torch.cat([green, dead, clover, gdm_pred, total_pred], dim=1)

        # 가중 MSE (Dry_Total_g에 50% 가중치!)
        weights = self.weights.to(pred.device)
        mse = (full_pred - target) ** 2
        weighted_mse = (mse * weights).mean()

        return weighted_mse
```

### 🔥 Priority 5: Multi-Seed Ensemble (예상 +0.005~0.01)

```python
# 최적 파라미터로 다양한 seed 학습
seeds = [42, 123, 456]

all_preds = []
for seed in seeds:
    seed_everything(seed)
    model = train_model(best_cfg)  # Optuna 최적 파라미터
    all_preds.append(model.predict(test))

final_pred = np.mean(all_preds, axis=0)
```

### 🔥 Priority 6: 최적 앙상블 가중치 (예상 +0.01)

```python
from scipy.optimize import minimize

def optimize_ensemble_weights(oof_preds_list, oof_targets):
    """OOF 기반 최적 앙상블 가중치 찾기"""
    n_models = len(oof_preds_list)

    def objective(weights):
        weights = np.abs(weights) / np.abs(weights).sum()
        ensemble_pred = sum(w * p for w, p in zip(weights, oof_preds_list))
        return -competition_metric(oof_targets, ensemble_pred)

    x0 = np.ones(n_models) / n_models
    result = minimize(objective, x0, method='Nelder-Mead')

    optimal_weights = np.abs(result.x)
    optimal_weights = optimal_weights / optimal_weights.sum()

    return optimal_weights
```

---

## 📅 업데이트된 실행 계획

### Phase 1: 기준점 확립 (Day 1-2) ← 현재 진행중
```
1. ✅ CV 수정 (Sampling_Date 그룹핑) - 완료
2. ✅ 해상도 560x560 + TTA 구현 - 완료
3. ⏳ cv1 학습 완료 후 제출
4. CV-LB gap 확인 (목표: ≤0.02)
```

**현재 CV 결과 (진행중):**
- Fold 0: 0.7139
- Fold 1: 0.6474
- Fold 2: 0.6352
- 예상 평균: ~0.66-0.68

⚠️ **이전보다 CV가 낮은 이유**: CV가 정직해짐! (이전 CV는 data leakage로 거짓말)

### Phase 2: Optuna HPO (Day 3-4) ⭐ NEW
```
1. 작은 Head부터 탐색 (hidden_dim: 64~512, num_layers: 1~3)
2. 2-fold 빠른 검증으로 50+ trials
3. 최적 파라미터 확정
```

**Optuna 팁:**
```python
# 빠른 탐색을 위해 Fold 축소
cv_score = train_and_evaluate(cfg, folds=[0, 1])

# Pruning으로 시간 절약
pruner = optuna.pruners.MedianPruner()
study = optuna.create_study(direction='maximize', pruner=pruner)
```

### Phase 3: 최적 파라미터로 Full Training (Day 5-6)
```
1. Optuna best params로 5-fold 전체 학습
2. Multi-seed 앙상블 (best params × 3 seeds)
3. Weighted Loss 실험
4. 제출 및 LB 확인
```

### Phase 4: 앙상블 최적화 (Day 7-8)
```
1. OOF 기반 최적 가중치 찾기
2. 다양한 모델 조합 실험
3. Blending 또는 Stacking 시도
```

### Phase 5: 최종 제출 (Day 9)
```
1. 최적 조합 선택
2. 안전한 백업 제출
3. Final submission
```

---

## 📊 예상 개선 효과 (업데이트됨)

| 전략 | 예상 향상 | 난이도 | 상태 |
|------|----------|--------|------|
| CV 수정 (Sampling_Date) | +0.03~0.05 | 쉬움 | ✅ 완료 |
| 해상도 560 + TTA | +0.02~0.03 | 쉬움 | ✅ 완료 |
| **Optuna HPO** | +0.02~0.03 | 중간 | 🔜 다음 |
| Weighted Loss | +0.01~0.02 | 중간 | 대기 |
| Multi-seed | +0.005~0.01 | 쉬움 | 대기 |
| 앙상블 최적화 | +0.01 | 중간 | 대기 |

**총 예상 향상: +0.07~0.12 → 0.77~0.82 가능!**

---

## 💡 핵심 인사이트: 작은 Head가 좋은 이유

### DINOv2 Large 특성
```
Backbone output: 1024 dim
├── 사전학습으로 이미 풍부한 representation
├── Head는 "feature → target 매핑"만 하면 됨
└── 복잡한 Head = overfitting 위험 ↑
```

### 데이터 크기 고려
```
Train images: 357개 (매우 작음!)
├── 큰 Head = 파라미터 많음 = overfitting
├── 작은 Head = 파라미터 적음 = 일반화 ↑
└── 정직한 CV에서는 작은 모델이 유리
```

### 실제 사례
| Backbone | 데이터 크기 | 최적 Head |
|----------|------------|-----------|
| DINOv2-Large | 소규모 (357) | MLP 1-2 layers, 64-256 dim |
| DINOv2-Large | 대규모 (10K+) | MLP 2-3 layers, 256-512 dim |

---

## ⚠️ 주의사항

1. **CV-LB Correlation 확인**
   - CV 수정 후 Local CV와 LB의 상관관계 확인
   - CV ≈ 0.67이면 LB ≈ 0.65-0.68 예상

2. **Overfitting 주의**
   - 357개 이미지로 작은 데이터셋
   - 작은 Head + 높은 Dropout이 안전

3. **Private LB 대비**
   - Public 53% / Private 47% 분할
   - 과도한 LB probing 피하기

4. **Optuna 주의사항**
   - 너무 많은 trials는 CV overfitting 가능
   - 50-100 trials 정도가 적당

---

*Updated: 2026-01-19*
*Target: 0.70 → 0.79+*
*Key Updates: Optuna HPO 추가, 작은 Head 탐색 전략*
