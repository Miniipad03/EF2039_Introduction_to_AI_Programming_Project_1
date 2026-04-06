# FGVC (Fine-Grained Visual Classification) 연구 흐름 정리

> 우리 팀 프로젝트 배경 이해 및 보고서 Citation 참고용
> 작성일: 2026-04-06

---

## 개요

Fine-Grained Visual Classification(FGVC)은 같은 상위 카테고리에 속하는 하위 클래스들 간의 **미세한 시각적 차이**를 구별하는 과제다.
예: Boeing 737-700 vs. 737-800처럼 외형이 매우 유사한 항공기 기종 구별.

우리 팀의 프로젝트는 FGVC-Aircraft 데이터셋을 사용하며, **1세대 딥러닝 방법론(BBox 크롭 + CNN)** 을 기반으로 한다.

---

## 세대별 발전 흐름

### 1단계 — 딥러닝 이전: 수작업 특징 추출 (Pre-Deep Learning Baseline)

수학적 공식으로 특징점(SIFT 등)을 추출한 뒤, Bag-of-Visual-Words(BoVW) 방식으로 통계 분류.

- **한계:** 날개 형태, 창문 배열 같은 미세한 기하학적 차이를 수식만으로 잡기 어려움
- **성능:** 약 48% (Accuracy @ family level)
- **핵심 논문:**
  - Maji et al., "Fine-Grained Visual Classification of Aircraft," arXiv:1306.5151, 2013.
    — FGVC-Aircraft 데이터셋 제안 논문. BoVW 기반 베이스라인 수립.

---

### 2단계 — 1세대 딥러닝: CNN + 구조적 정보(BBox) 활용

**우리 팀이 현재 채택한 방법론의 세대.**

전체 이미지를 그대로 넣으면 배경(하늘, 활주로) 노이즈가 학습을 방해하므로,
제공된 BBox로 객체만 크롭한 뒤 CNN에 입력하는 전략이 표준화되었다.

- **주요 전략:**
  - BBox 크롭으로 불필요한 배경 제거 → 분류 성능 향상
  - 항공기 전체를 보는 CNN + 특정 부위(엔진, 날개 끝 등)를 확대해서 보는 CNN 결합 (Part-based ensemble)
- **성능:** 약 80–85% (Accuracy @ variant level, ResNet 기준)
- **핵심 논문:**
  - Lin et al., "Bilinear CNN Models for Fine-Grained Visual Recognition," ICCV 2015.
    — 두 CNN의 출력을 외적(outer product)으로 결합. 미세 특징 포착력 대폭 향상. 정확도 단숨에 84% 달성.
  - Huang et al., "Part-Stacked CNN for Fine-Grained Visual Categorization," CVPR 2016.
    — BBox 기반 전체 객체 이미지 + 세부 부품 이미지 함께 입력. **우리 프로젝트 구조와 가장 유사한 논문.**

---

### 3단계 — 2세대 딥러닝: Attention 기반 자동 부위 탐색 (Weakly Supervised)

"사람이 BBox를 직접 제공하지 않아도 모델이 스스로 중요 부위를 찾을 수 있는가?"에서 출발.

- **주요 전략:**
  - 모델 내부에 부위 탐색 서브넷(Navigator)을 내장
  - 스스로 "꼬리 날개와 엔진을 봐야 해" 판단 → 중요 부위 BBox 자동 생성 → 크롭/확대 후 추가 학습
- **성능:** 약 90% 이상
- **핵심 논문:**
  - Yang et al., "Learning to Navigate for Fine-grained Classification (NTS-Net)," ECCV 2018.
  - Zheng et al., "Learning Multi-Attention Convolutional Neural Network for Fine-Grained Image Recognition (MA-CNN)," ICCV 2017.

---

### 4단계 — 3세대 딥러닝: Vision Transformer (ViT) 기반

CNN 구조를 넘어 이미지 패치 간 관계를 Self-Attention으로 포착하는 최신 트렌드.

- **주요 전략:**
  - 이미지를 패치로 분할 → Self-Attention으로 전역 문맥(global context)과 로컬 미세 특징 동시 포착
- **성능:** 약 95% 이상
- **핵심 논문:**
  - He et al., "TransFG: A Transformer Architecture for Fine-Grained Recognition," AAAI 2022.

---

## 우리 프로젝트와의 연결

| 항목 | 내용 |
|---|---|
| 데이터셋 | FGVC-Aircraft (Maji et al., 2013) |
| 분류 레벨 | Family (70 classes) — 베이스라인 목표 ≥ 70% |
| 전처리 | 20px 하단 배너 제거 → BBox 크롭 → 224×224 리사이즈 → ImageNet 정규화 |
| 모델 (Baseline) | Custom CNN (3–4 Conv layers, ReLU, MaxPool, Dropout, FC) |
| 모델 (Advanced) | ResNet18/34 Transfer Learning (ImageNet pretrained) |
| 학습 | Adam (lr=0.001) / SGD+momentum; CrossEntropyLoss |
| 평가 | Class-normalised average accuracy, Confusion matrix, Grad-CAM |

### 고도화 아이디어 (향후 실험 후보)

1. **Part-Stacked 앙상블:** BBox 이미지를 4분할하여 여러 크롭에서 예측값 합산 (Part-Stacked CNN 아이디어)
2. **Bilinear Pooling:** 두 CNN 피처맵의 외적으로 미세 특징 강화
3. **Weakly Supervised Attention:** 테스트 단계에서 BBox 없이 자동으로 중요 부위 탐색 (NTS-Net 스타일)

---

## 참고문헌 (Citation)

```
[1] S. Maji, E. Rahtu, J. Kannala, M. Blaschko, and A. Vedaldi,
    "Fine-Grained Visual Classification of Aircraft," arXiv:1306.5151, 2013.

[2] T.-Y. Lin, A. RoyChowdhury, and S. Maji,
    "Bilinear CNN Models for Fine-Grained Visual Recognition,"
    in Proc. ICCV, 2015.

[3] S. Huang, Z. Xu, D. Tao, and Y. Zhang,
    "Part-Stacked CNN for Fine-Grained Visual Categorization,"
    in Proc. CVPR, 2016.

[4] H. Zheng, J. Fu, T. Mei, and J. Luo,
    "Learning Multi-Attention Convolutional Neural Network for Fine-Grained Image Recognition,"
    in Proc. ICCV, 2017.

[5] Z. Yang, T. Luo, D. Wang, Z. Hu, J. Gao, and L. Wang,
    "Learning to Navigate for Fine-grained Classification,"
    in Proc. ECCV, 2018.

[6] J. He, J.-N. Chen, S. Liu, A. Kortylewski, C. Yang, Y. Bai, and C. Wang,
    "TransFG: A Transformer Architecture for Fine-Grained Recognition,"
    in Proc. AAAI, 2022.
```
