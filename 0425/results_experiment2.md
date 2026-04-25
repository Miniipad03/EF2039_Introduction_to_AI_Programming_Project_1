# Experiment 2 결과 정리 (2026-04-25)

---

## 실험 개요

| 항목 | 설정 |
|------|------|
| Dataset | FGVC-Aircraft, label=family (70 classes) |
| 평가 방식 | 4-Fold CV + Weighted Soft Voting Ensemble |
| Test set | 전체의 20% 사전 격리 (seed=42) |
| 공통 설정 | batch_size=32, lr=0.001, optimizer=Adam, weight_decay=1e-4 |

---

## 전체 결과 요약

| # | 실험명 | Model | Attn | Data | Epochs | Test Acc | CN Acc | Params | Train Time |
|---|--------|-------|------|------|--------|:--------:|:------:|-------:|----------:|
| 1 | Baseline_SimpleCNN | CNN | - | vanilla | 100 | 55.00% | 50.06% | 2.52M | 4,595s |
| 2 | Baseline_ResNet34 | ResNet-34 | none | vanilla | 30 | **89.90%** | **88.15%** | 21.32M | 1,434s |
| 3 | DataAblation_ExclOnly | ResNet-34 | none | excl | 30 | 89.85% | 88.02% | 21.32M | 1,450s |
| 4 | DataAblation_AddedOnly | ResNet-34 | none | added | 30 | 89.45% | 87.68% | 21.32M | 1,473s |
| 5 | DataAblation_Tuning | ResNet-34 | none | tuning | 30 | **90.05%** | **88.33%** | 21.32M | 1,468s |
| 6 | Attention_Channel | ResNet-34 | channel | tuning | 30 | **90.10%** | **88.51%** | 21.47M | 1,469s |
| 7 | Attention_Spatial | ResNet-34 | spatial | tuning | 30 | 89.70% | 87.86% | 21.32M | 1,393s |
| 8 | Attention_CBAM | ResNet-34 | cbam | tuning | 30 | 89.55% | 87.96% | 21.47M | 1,530s |

> CN Acc: class-normalized accuracy (클래스별 샘플 수 불균형 보정)

---

## Section 1 — Baseline 비교

### SimpleCNN vs ResNet-34

| | SimpleCNN | ResNet-34 |
|---|:---------:|:---------:|
| Test Acc | 55.00% | 89.90% |
| CN Acc | 50.06% | 88.15% |
| Params | 2.52M | 21.32M |
| Train Time | 4,595s (~76분) | 1,434s (~24분) |
| Epochs | 100 | 30 |

- Test Acc 차이: **+34.9%p** (전이학습 효과)
- SimpleCNN은 100 epoch을 돌렸음에도 ResNet-34의 30 epoch에 크게 못 미침
- 파라미터 수는 SimpleCNN이 약 1/8 수준이나 성능 격차가 압도적

### Overfitting 분석

(final epoch 기준 4-fold 평균, history_*.csv 출처)

| 실험 | Final Train Acc | Final Val Acc | Gap |
|------|:--------------:|:-------------:|:---:|
| SimpleCNN | 52.22% | 51.15% | 1.07% |
| ResNet34_vanilla | 99.98% | 88.34% | 11.65% |
| ResNet34_excl | 99.98% | 88.66% | 11.32% |
| ResNet34_added | 99.98% | 88.65% | 11.33% |
| ResNet34_tuning | 99.98% | 88.44% | 11.54% |
| ResNet34_channel | 99.94% | 88.06% | 11.88% |
| ResNet34_spatial | 99.95% | 88.52% | 11.43% |
| ResNet34_cbam | 99.96% | 87.64% | **12.32%** |

- **SimpleCNN**: train/val 격차 거의 없음 → overfitting이 아닌 **underfitting**. 100 epoch에서도 수렴 중으로 판단, 모델 표현력 자체가 부족
- **ResNet-34 전 변종**: train ~99.98%, val ~88% → **overfitting** 경향. pretrained features로 train set 암기. dropout 추가 또는 epoch 축소 여지 있음
- **CBAM**이 gap 가장 큼 (12.32%): randomly initialized attention 모듈이 train set 과적합을 가중시키는 것으로 추정

---

## Section 2 — Data Ablation

ResNet-34 (attn=none) 고정, 데이터 설정만 변경.

| Data 설정 | Test Acc | CN Acc | 변화 (vs vanilla) |
|-----------|:--------:|:------:|-----------------:|
| vanilla (기준) | 89.90% | 88.15% | — |
| excl_only | 89.85% | 88.02% | -0.05% / -0.13% |
| added_only | 89.45% | 87.68% | -0.45% / -0.47% |
| **tuning (excl+added)** | **90.05%** | **88.33%** | **+0.15% / +0.18%** |

### 해석

- **excl_only**: 노이즈 이미지 제거만으로는 미미한 변화 (-0.05%). 제거된 샘플이 적거나 해당 이미지의 영향이 제한적
- **added_only**: 오히려 소폭 하락 (-0.45%). 추가 이미지만 넣고 노이즈를 그대로 두면 효과가 상쇄됨
- **tuning (둘 다 적용)**: 가장 높은 성능. 노이즈 제거 + 데이터 보강이 시너지를 냄

→ `pick_best_data.py`가 **tuning** 설정을 Section 3 기준으로 자동 선택

---

## Section 3 — Attention 비교

ResNet-34 + tuning data 고정, attention 방식만 변경 (layer3 + layer4에 삽입).
기준선: ResNet-34 + tuning + no-attn (90.05%)

| Attention | Test Acc | CN Acc | vs 기준선 | Params 증가 |
|-----------|:--------:|:------:|----------:|------------:|
| none (기준) | 90.05% | 88.33% | — | — |
| **channel** | **90.10%** | **88.51%** | **+0.05% / +0.18%** | +147K |
| spatial | 89.70% | 87.86% | -0.35% / -0.47% | +0.9K |
| cbam | 89.55% | 87.96% | -0.50% / -0.37% | +148K |

### 해석

- **Channel attention**: 유일하게 기준선 대비 개선. "어떤 채널이 중요한가"를 선택적으로 학습하는 방식이 fine-grained 분류에 유효
- **Spatial attention**: 소폭 하락. 7×7 spatial map에서의 위치 선택이 기종 구분에 결정적이지 않을 수 있음
- **CBAM**: 두 attention 결합이지만 가장 낮은 성능. 파라미터가 늘어나는 데 비해 30 epoch으로는 최적화가 부족한 것으로 판단 (이전 실험에서도 동일한 패턴 확인됨)
- 전반적으로 attention 효과가 미미한 이유: 30 epoch이라는 제한된 학습량에서 randomly initialized attention 모듈의 수렴이 불완전할 가능성

---

## 클래스별 분석 — 취약 클래스

전 실험에서 반복적으로 낮은 정확도를 보이는 클래스:

| Class | 특징 | 대표 취약 실험 |
|-------|------|---------------|
| **Boeing 717** | DC-9 계열 파생형, 외형 거의 동일 | 모든 실험 bottom 5 |
| **DC-9** | Boeing 717과 혼동 빈번 | 모든 실험 bottom 5 |
| **C-47** | 구형 프로펠러기, 유사 기종 다수 | 모든 실험 bottom 5 |
| **A310** | A300 파생형, 크기·형태 유사 | 모든 실험 bottom 5 |
| **A300** | A310과 혼동 | vanilla/spatial에서 빈번 |

→ **혼동 패턴**: 파생 관계에 있는 기종 간 오분류가 주된 실패 원인. family 레벨 분류임에도 시각적으로 거의 동일한 쌍이 존재함 (Boeing 717 ↔ DC-9, A300 ↔ A310)

---

## 최종 최고 성능

| 기준 | 실험 | 값 |
|------|------|----|
| Test Acc 최고 | ResNet-34 + channel + tuning | **90.10%** |
| CN Acc 최고 | ResNet-34 + channel + tuning | **88.51%** |
| 파라미터 대비 효율 | ResNet-34 + spatial + tuning | 89.70% / 21.32M |

---

## 보고서 작성 시 활용 포인트

- SimpleCNN underfitting → "모델 표현력 한계" 서술
- ResNet-34 overfitting → "전이학습 기반 모델의 한계, 추가 정규화 여지" 서술
- Data tuning 효과: excl+added 조합이 단독보다 우월 → "데이터 품질과 다양성의 시너지"
- Channel attention 미세 개선 → "fine-grained 분류에서 채널 선택성의 유효성"
- Boeing 717 / DC-9 혼동 → confusion matrix에서 해당 셀 강조하여 failure analysis 서술
