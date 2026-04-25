# 실험 결과 보고서

**실험 일시**: 2026-04-23 (타임스탬프: `20260423_043207`)  
**실행 스크립트**: `run_experiments.sh`  
**로그 디렉토리**: `logs/`  
**결과 CSV**: `results_all.csv`

---

## 공통 실험 설정

| 항목 | 값 |
|------|-----|
| Label | `family` (70 클래스) |
| Epochs | 30 |
| Batch size | 32 |
| Learning rate | 0.001 |
| Optimizer | Adam (weight_decay=1e-4) |
| CV 전략 | 4-Fold Stratified K-Fold |
| Seed | 42 |
| 데이터셋 | FGVC-Aircraft 원본 10,000장 |
| Test set | 2,000장 고립 (20%, stratified) |
| Train+Val pool | 8,000장 |

---

## 실험 구성

| # | 실험명 | 모델 | Attention | 데이터 |
|---|--------|------|-----------|--------|
| 1 | Vanilla_SimpleCNN | SimpleCNN (scratch) | none | original |
| 2 | Vanilla_ResNet34 | ResNet-34 (pretrained) | none | original |
| 3 | Tuning_ResNet34 | ResNet-34 (pretrained) | none | excluded 제거 + added 추가 |
| 4 | Tuning_ResNet34_CBAM | ResNet-34 (pretrained) | CBAM (layer3+layer4) | excluded 제거 + added 추가 |

---

## 결과 요약

### 핵심 지표

| # | 모델 | Fold1 | Fold2 | Fold3 | Fold4 | **Mean Val** | **Test Acc** | **CN Acc** |
|---|------|:-----:|:-----:|:-----:|:-----:|:------------:|:------------:|:----------:|
| 1 | SimpleCNN | 23.70% | 26.75% | 26.10% | 22.85% | 24.85% | 26.90% | 18.33% |
| 2 | ResNet-34 vanilla | 89.30% | 88.70% | 88.10% | 87.70% | 88.45% | 89.45% | 87.68% |
| **3** | **ResNet-34 tuning** | **88.25%** | **88.80%** | **88.35%** | **87.35%** | **88.19%** | **89.70%** ✅ | **88.19%** ✅ |
| 4 | ResNet-34+CBAM | 80.40% | 80.40% | 78.90% | 81.50% | 80.30% | 85.40% | 83.63% |

> **CN Acc** (Class-Normalized Accuracy): 클래스별 정확도의 평균. 클래스 불균형 영향을 제거한 지표.

---

## 실험별 상세 분석

### Exp 1 — Vanilla SimpleCNN

- **파라미터 수**: 2,522,950
- **학습 방식**: scratch (사전학습 없음), backbone freeze 없음
- **Fold당 소요 시간**: ~14분

**학습 추이 (Fold 1 기준)**

| Epoch | Train Loss/Acc | Val Loss/Acc |
|-------|---------------|--------------|
| 1 | 4.1667 / 6.6% | 3.9812 / 8.1% |
| 30 | 2.7248 / 21.1% | 2.6144 / 23.2% |

**특이사항**  
- 30 epoch 종료 시점에도 val acc가 22~26% 수준으로 수렴 미달
- Test acc(26.9%)와 CN acc(18.3%) 간 8.6%p 차이 → 빈도 높은 클래스에 편향된 예측 경향

---

### Exp 2 — Vanilla ResNet-34

- **사전학습**: ImageNet weights
- **학습 방식**: Epoch 1~5 FC layer만 학습 (backbone freeze), Epoch 6부터 backbone unfreeze (lr=1e-4)
- **Fold당 소요 시간**: ~14분

**학습 추이 (Fold 1 기준)**

| Epoch | Train Loss/Acc | Val Loss/Acc | 비고 |
|-------|---------------|--------------|------|
| 1 | 3.6098 / 14.8% | 3.0103 / 26.9% | FC warmup |
| 6 | 1.1094 / 67.4% | 0.9153 / 72.4% | backbone unfreeze 직후 |
| 30 | 0.0033 / 99.95% | 0.4359 / 89.1% | |

**특이사항**
- Epoch 5→6 전환 시 val acc가 ~26% → ~72%로 급등 (전이학습의 핵심 효과)
- Epoch 30 기준 train acc ~100%, val acc ~88~89% → 경미한 과적합

---

### Exp 3 — Tuning ResNet-34

- **데이터 변경**: excluded 이미지 제거 + added 이미지 180장 train에 추가
- **Train 크기**: 6,180장/fold (vanilla 대비 +180장)
- **그 외 설정**: Exp 2와 동일

**Exp 2 대비 변화**

| 지표 | Exp 2 | Exp 3 | 차이 |
|------|-------|-------|------|
| Mean Val Acc | 88.45% | 88.19% | -0.26%p |
| Ensemble Test Acc | 89.45% | **89.70%** | **+0.25%p** |
| CN Acc | 87.68% | **88.19%** | **+0.51%p** |

**특이사항**
- Mean Val은 소폭 하락했으나 Test / CN 지표는 모두 향상
- CN Acc 개선(+0.51%p)이 Test Acc 개선(+0.25%p)보다 큼 → excluded 제거로 학습 노이즈 감소 효과

---

### Exp 4 — Tuning ResNet-34 + CBAM

- **CBAM 삽입 위치**: layer3, layer4 (총 9개 모듈)
- **데이터**: Exp 3과 동일 (excluded 제거 + added 추가)

**학습 추이 (Fold 1 기준)**

| Epoch | Train Loss/Acc | Val Loss/Acc | 비고 |
|-------|---------------|--------------|------|
| 1 | 4.1104 / 7.8% | 4.0854 / 8.0% | FC warmup + CBAM 초기화 |
| 6 | 3.6683 / 10.6% | 3.4239 / 14.1% | backbone unfreeze 직후 |
| 30 | 0.1138 / 96.6% | 0.7616 / 80.3% | |

**Exp 3 대비 변화**

| 지표 | Exp 3 | Exp 4 | 차이 |
|------|-------|-------|------|
| Mean Val Acc | 88.19% | 80.30% | **-7.89%p** |
| Ensemble Test Acc | 89.70% | 85.40% | **-4.30%p** |
| CN Acc | 88.19% | 83.63% | **-4.56%p** |

**성능 하락 원인 분석**
1. **수렴 지연**: backbone unfreeze 직후(Epoch 6) val acc가 14.1%에 불과 (Exp 2/3의 ~72% 대비 현저히 낮음) — CBAM 모듈이 randomly initialized된 상태에서 학습 신호를 분산시킴
2. **과적합**: Epoch 30 기준 train acc 96~97%, val acc 79~80% 로 train-val 간 격차가 큼
3. **에폭 부족**: 9개 CBAM 모듈을 충분히 학습하기에 30 epoch은 부족한 것으로 판단

---

## 종합 비교

```
Test Accuracy
─────────────────────────────────────────────────────────────
 Exp 3  ResNet-34 tuning       ██████████████████████  89.70%  ← 최고
 Exp 2  ResNet-34 vanilla      █████████████████████   89.45%
 Exp 4  ResNet-34+CBAM tuning  ████████████████████    85.40%
 Exp 1  SimpleCNN              ██                      26.90%
─────────────────────────────────────────────────────────────
```

### 주요 인사이트

1. **전이학습 효과 압도적**: ResNet-34 pretrained(89.45%) vs SimpleCNN scratch(26.90%) → 62.6%p 차이
2. **데이터 튜닝 효과 소폭**: excluded 제거 + added 추가로 Test +0.25%p, CN +0.51%p 개선
3. **CBAM은 30 epoch에서 역효과**: 추가 파라미터 대비 학습 시간 부족으로 오히려 4.3%p 하락. 에폭 증가(50~100) 또는 CBAM 모듈 warm-up 전략 필요
4. **최고 성능 모델**: Exp 3 (ResNet-34 tuning) — Test **89.70%**, CN **88.19%**

---

## 생성된 파일 목록

| 파일 | 설명 |
|------|------|
| `results_all.csv` | 4개 실험 종합 결과 |
| `history_cnn_noattn_vanilla_family_*.csv` | Exp 1 fold별 epoch 기록 |
| `history_resnet34_noattn_vanilla_family_*.csv` | Exp 2 fold별 epoch 기록 |
| `history_resnet34_noattn_tuning_family_*.csv` | Exp 3 fold별 epoch 기록 |
| `history_resnet34_cbam_tuning_family_*.csv` | Exp 4 fold별 epoch 기록 |
| `curve_*.png` | 학습 곡선 (Loss / Val Acc) |
| `confusion_*.png` | Confusion Matrix (Ensemble) |
| `best_fold{1-4}_*.pth` | 각 fold 최고 val acc 체크포인트 |
| `logs/1_Vanilla_SimpleCNN_20260423_043207.log` | Exp 1 전체 로그 |
| `logs/2_Vanilla_ResNet34_20260423_043207.log` | Exp 2 전체 로그 |
| `logs/3_Tuning_ResNet34_20260423_043207.log` | Exp 3 전체 로그 |
| `logs/4_Tuning_ResNet34_CBAM_20260423_043207.log` | Exp 4 전체 로그 |
