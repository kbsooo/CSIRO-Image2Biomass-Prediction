# 🎯 CV3 Strategy: 0.70 → 0.72~0.73 달성 전략

## 📊 현재 상황

| 모델 | CV | Public LB | 특징 |
|------|-----|-----------|------|
| 이전 최고 (v27) | ~0.80 (거짓) | **0.70** | Data leakage CV |
| CV1 | 0.6366 | **0.68** | 정직한 CV, Sampling_Date 그룹핑 |
| CV2 (Freeze) | 0.5966 | - | 실패 |
| CV3 (Weighted) | 0.5253 | - | 실패 |
| **목표** | 0.68+ | **0.72~0.73** | |

---

## 🔥 CV3 핵심 변경사항

### 1. 이미지 전처리 (Discussion에서 +0.02 보고됨)

**문제점**: 이미지에 노이즈 요소들이 있음
- Orange timestamp (날짜/시간 텍스트)
- Bottom artifacts (cardboard, color charts)
- 모델이 "텍스트"나 "쓰레기"에 overfitting

**해결책**:
```python
import cv2
import numpy as np

def clean_image(img):
    """
    이미지 전처리: timestamp 제거 + bottom crop
    Discussion에서 LB +0.02 효과 보고됨
    """
    img = np.array(img)
    h, w = img.shape[:2]

    # 1. Bottom 10% crop (artifacts 제거)
    # cardboard, color charts 등이 하단에 자주 나타남
    img = img[0:int(h*0.90), :]

    # 2. Orange timestamp inpainting
    # 이미지에 있는 주황색 날짜/시간 텍스트 제거
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Orange color range (HSV)
    lower_orange = np.array([5, 150, 150])
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Dilate mask to cover text edges
    mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=2)

    # Inpaint if orange pixels found
    if np.sum(mask) > 0:
        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    return Image.fromarray(img)
```

### 2. WA State Dead=0 후처리 (Discussion 81 votes)

**발견**: Western Australia(WA) 32개 샘플 **전부** Dead=0
- 이건 100% 확실한 패턴
- 모델이 학습하든 안 하든, 후처리로 강제하면 손해 없음

**구현**:
```python
# Inference 시 후처리
def postprocess_predictions(preds, test_df):
    """
    State별 후처리
    - WA: Dead biomass = 0 (100% 확실)
    """
    preds = preds.copy()

    for idx, row in test_df.iterrows():
        if row['State'] == 'WA':
            # Dead = 0 강제
            preds[idx, 1] = 0.0  # Dry_Dead_g index

            # GDM과 Total도 재계산
            green = preds[idx, 0]   # Dry_Green_g
            clover = preds[idx, 2]  # Dry_Clover_g
            preds[idx, 3] = green + clover  # GDM_g
            preds[idx, 4] = green + clover  # Dry_Total_g (Dead=0이므로)

    return preds
```

### 3. 기존 CV1 설정 유지 (검증됨)

CV1이 현재 가장 좋은 성능:
```python
class CFG:
    # === Model Architecture (CV1과 동일) ===
    hidden_dim = 256
    num_layers = 2
    dropout = 0.3

    # === Backbone ===
    freeze_backbone = False  # Fine-tune이 더 좋았음

    # === Training ===
    lr = 2e-4
    weight_decay = 1e-4
    batch_size = 16
    epochs = 30
    patience = 7

    # === Resolution ===
    img_size = (560, 560)  # 14와 16의 공배수

    # === Loss ===
    use_weighted_loss = False  # MSE가 더 좋았음
```

---

## 📁 CV3 파일 구조

```
cv3_train.py          # 학습 코드 (전처리 추가)
cv3_infer.py          # 추론 코드 (후처리 추가)
```

---

## 🔧 CV3 Train 코드 변경사항

### Dataset 클래스 수정

```python
class BiomassDataset(Dataset):
    def __init__(self, df, data_path, transform=None,
                 height_mean=None, height_std=None,
                 ndvi_mean=None, ndvi_std=None,
                 return_idx=False,
                 use_clean_image=True):  # ⭐ 추가
        self.df = df.reset_index(drop=True)
        self.data_path = data_path
        self.transform = transform
        self.return_idx = return_idx
        self.use_clean_image = use_clean_image  # ⭐ 추가

        # ... (나머지 동일)

    def clean_image(self, img):
        """이미지 전처리: timestamp 제거 + bottom crop"""
        img = np.array(img)
        h, w = img.shape[:2]

        # 1. Bottom 10% crop
        img = img[0:int(h*0.90), :]

        # 2. Orange timestamp inpainting
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower_orange = np.array([5, 150, 150])
        upper_orange = np.array([25, 255, 255])
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=2)

        if np.sum(mask) > 0:
            img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

        return Image.fromarray(img)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.data_path / row['image_path']).convert('RGB')

        # ⭐ 전처리 적용
        if self.use_clean_image:
            img = self.clean_image(img)

        width, height = img.size
        mid = width // 2

        left_img = img.crop((0, 0, mid, height))
        right_img = img.crop((mid, 0, width, height))

        # ... (나머지 동일)
```

### Transforms 수정 (crop 후 resize)

```python
def get_train_transforms(cfg):
    return T.Compose([
        T.Resize(cfg.img_size),  # clean_image에서 crop했으므로 resize만
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=cfg.hue_jitter),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms(cfg):
    return T.Compose([
        T.Resize(cfg.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
```

---

## 🔧 CV3 Inference 코드 변경사항

### 후처리 함수

```python
def postprocess_predictions(preds, test_df):
    """
    State별 후처리
    WA: Dead = 0 (Discussion에서 발견된 100% 패턴)
    """
    preds = preds.copy()

    wa_count = 0
    for idx in range(len(test_df)):
        state = test_df.iloc[idx]['State']

        if state == 'WA':
            wa_count += 1
            # Dry_Dead_g = 0 강제
            old_dead = preds[idx, 1]
            preds[idx, 1] = 0.0

            # GDM과 Total 재계산
            green = preds[idx, 0]   # Dry_Green_g
            clover = preds[idx, 2]  # Dry_Clover_g
            dead = preds[idx, 1]    # Dry_Dead_g (now 0)

            preds[idx, 3] = green + clover         # GDM_g
            preds[idx, 4] = green + clover + dead  # Dry_Total_g

    print(f"✓ WA samples processed: {wa_count} (Dead forced to 0)")
    return preds
```

### TTA + 후처리 통합

```python
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

            pred, _ = model(l.to(device), r.to(device))
            preds.append(pred.cpu())

    return torch.stack(preds).mean(0)


def inference_cv3(models, test_loader, test_df, device):
    """CV3 추론: TTA + 후처리"""
    all_preds = []

    for left, right in tqdm(test_loader, desc="Inference"):
        batch_preds = []
        for model in models:
            model.eval()
            pred = predict_with_tta(model, left, right, device)
            batch_preds.append(pred)

        # 5-fold 앙상블
        ensemble_pred = torch.stack(batch_preds).mean(0)
        all_preds.append(ensemble_pred.numpy())

    preds = np.concatenate(all_preds)

    # ⭐ 후처리 적용
    preds = postprocess_predictions(preds, test_df)

    return preds
```

---

## 📊 예상 효과 분석

### 개별 효과

| 변경사항 | 예상 효과 | 근거 |
|----------|----------|------|
| 이미지 전처리 | +0.01~0.02 | Discussion에서 0.60→0.62 보고 |
| WA Dead=0 후처리 | +0.005~0.01 | 32개 샘플 100% 패턴 |
| CV1 기반 (Fine-tune + MSE) | baseline | 이미 LB 0.68 |

### 누적 효과 예상

```
CV1 baseline:        LB 0.68
+ 이미지 전처리:     LB 0.69~0.70
+ WA 후처리:         LB 0.70~0.71
```

---

## 🚀 추가 부스팅 전략

### CV3 완료 후 추가 실험

#### 1. Multi-seed 앙상블 (CV3 × 3 seeds)
```python
seeds = [42, 123, 456]
# 각 seed로 CV3 학습 → 3개 모델 앙상블
```
**예상: +0.01~0.02 → LB 0.71~0.72**

#### 2. CV3 + 이전 v27 앙상블
```python
# 새 모델과 이전 모델 앙상블
final_pred = 0.6 * cv3_pred + 0.4 * v27_pred
```
**예상: +0.01 → LB 0.72~0.73**

#### 3. OOF 기반 가중치 최적화
```python
from scipy.optimize import minimize

def optimize_weights(oof_list, targets):
    def objective(w):
        w = np.abs(w) / np.abs(w).sum()
        pred = sum(wi * oof for wi, oof in zip(w, oof_list))
        return -competition_metric(targets, pred)

    result = minimize(objective, np.ones(len(oof_list))/len(oof_list))
    return np.abs(result.x) / np.abs(result.x).sum()
```

---

## 📅 실행 계획

### Day 1: CV3 학습 + 검증
```
1. cv3_train.py 작성 (이미지 전처리 추가)
2. 5-fold 학습 실행 (~3-4시간)
3. CV 점수 확인 (목표: CV1과 비슷하거나 높게)
```

### Day 2: CV3 제출 + 앙상블
```
1. cv3_infer.py 작성 (TTA + WA 후처리)
2. CV3 단독 제출 → LB 확인 (목표: 0.70~0.71)
3. CV3 + v27 앙상블 제출 → LB 확인 (목표: 0.71~0.72)
```

### Day 3: Multi-seed 부스팅
```
1. CV3 seed 123으로 재학습
2. CV3 seed 456으로 재학습
3. 3-seed 앙상블 제출 → LB 확인 (목표: 0.72~0.73)
```

---

## ⚠️ 주의사항

### 1. 이미지 전처리 일관성
- **Train과 Test 모두 동일한 전처리 적용 필수**
- clean_image() 함수를 Dataset에서 호출

### 2. Bottom crop 비율 조정
```python
# 기본: 10% crop
img = img[0:int(h*0.90), :]

# 너무 많이 crop하면 중요한 정보 손실
# 너무 적게 crop하면 artifacts 남음
# 필요시 0.85~0.95 범위에서 실험
```

### 3. Orange mask 범위 조정
```python
# 기본 HSV 범위
lower_orange = np.array([5, 150, 150])
upper_orange = np.array([25, 255, 255])

# 이미지마다 다를 수 있으므로 시각적 확인 권장
```

### 4. CV-LB 상관관계 유지
- CV3의 CV 점수가 CV1보다 낮으면 전처리가 해로운 것
- CV 점수 확인 후 제출 여부 결정

---

## 📈 성공 지표

| 단계 | 목표 CV | 목표 LB | 달성 기준 |
|------|---------|---------|----------|
| CV3 학습 | ≥0.63 | - | CV1 수준 유지 |
| CV3 제출 | - | ≥0.70 | 이전 최고와 동등 |
| + v27 앙상블 | - | ≥0.71 | +0.01 개선 |
| + Multi-seed | - | **≥0.72** | 최종 목표 |

---

## 🔑 핵심 요약

```
CV3 = CV1 + 이미지 전처리 + WA 후처리

변경 최소화 원칙:
- 모델 구조: CV1과 동일 (검증됨)
- Loss: MSE (검증됨)
- 해상도: 560x560 (검증됨)
- 추가: 이미지 전처리만!
```

**가장 중요한 것**: CV1이 잘 동작하므로, 전처리만 추가하고 나머지는 건드리지 않기!

---

*Created: 2025-01-20*
*Target: LB 0.72~0.73*
*Base: CV1 (LB 0.68)*
