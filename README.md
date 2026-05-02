# 🛩️ FGVC-Aircraft Image Classification & Active Labeling Pipeline

<div align="center">
  <em>An Advanced Computer Vision Project featuring K-Fold Cross-Validation, Attention Mechanisms, and an Interactive Labeling Web Environment.</em>
</div>

---

## ⚡ Quick Start

```bash
# 1. Setup environment
source activate.sh

# 2. Download dataset
python download_dataset.py

# 3. Train best model
python train.py --model resnet34d --attn channel --tuning --epochs 30

# 4. (Optional) Launch labeling tool
cd labeling-tool && npm install && npm run dev

# 5. (Optional) Launch inference demo
cd inference-demo && npm install && npm run dev
```

---

## 🎯 1. Project Goal
The primary objective of this project is to build a highly robust and accurate image classification system for the **FGVC-Aircraft** dataset. Instead of relying on a standard model fit, this repository introduces a systematic, production-ready pipeline:
- **State-of-the-art architectures**: Utilizes `ResNet-18` and `ResNet-34` enhanced with Attention Mechanisms (`CBAM`, `Spatial`, `Channel`) to focus on fine-grained aircraft features.
- **Robust Generalization**: Implements **4-Fold Cross Validation** with a **Weighted Soft Voting Ensemble** to maximize test accuracy and reduce overfitting.
- **Active Data Engineering**: Features an interactive SvelteKit-based Labeling Tool allowing developers to proactively _add custom images_ or _exclude noisy/incorrect samples_ to continuously tune dataset quality.

## 📂 2. Dataset Information
This project utilizes the **FGVC-Aircraft 2013b** dataset, which consists of images spanning over 100 different aircraft structural variants.

- **Data Fetching:** Handled strictly via `download_dataset.py`.
- **Annotation Levels:** Supports hierarchical labeling (`variant`, `family`, `manufacturer`).

### Data Engineering (`dataset_utils.py`)
`dataset_utils.py` is the core of the data pipeline, providing the custom `FGVCAircraft` and `SampleDataset` classes that merge the original dataset with labeling tool outputs:
1. **Excluded Data Filtering**: Reads `excluded.json` (images flagged via the labeling tool) and skips them during dataset loading to prevent noisy samples from reaching the model.
2. **Custom Image Injection**: Loads newly added images from `added_images.json` and locks them into the training split only, preventing evaluation data leakage.
3. **6:2:2 Dynamic Stratified Re-Split**: Unlike the default 1:1:1 FGVC split, this recalculates split sizes dynamically when new images are added, maintaining a 6:2:2 train/val/test ratio while preserving per-class balance.
4. **Bounding Box Crop (`crop_bbox`)**: When enabled, crops each image to the aircraft bounding box using coordinates from `images_box.txt`, removing irrelevant background.
5. **Bottom Banner Removal (`RemoveBottomBanner`)**: FGVC images include a ~20px copyright banner at the bottom that can bias feature maps. This transform removes it from original images only, leaving user-added images untouched.

## 🛠️ 3. Environment and Dependencies
This pipeline is optimized for PyTorch 2.6.0+ and supports automatic CUDA detection.

### One-Click Setup
Use the provided bash script to automatically create a python virtual environment, detect your CUDA version, and install the exact correct binaries.
```bash
source activate.sh
```

### Manual Setup (requirements.txt)
If managing your own environment:
```bash
pip install -r requirements.txt
```
**Key Dependencies**:
- `torch>=2.6.0`, `torchvision>=0.21.0`
- `scikit-learn`, `pandas`, `seaborn`, `matplotlib`, `pillow`, `tqdm`

## 🎛️ 4. Data Tuning and Preprocessing via Labeling Tool
The included SvelteKit-based Labeling Tool acts as a direct **Train-Set Controller** rather than a passive viewing portal. Launching the web environment permits the rapid application of Active Learning mechanics:

- **Data Excluding**: Flagging an incorrect or corrupted image silently logs it into a JSON within `data/excluded`. The `dataset_utils.py` instantly adapts upon next execution, stripping it out of the pipeline.
- **Data Adding**: Users can upload explicitly harvested images, draw bounding boxes, and label them directly on the web application. These definitions propagate immediately to `data/added_images`.
- **Seamless Pipeline Integration**: You never touch code to scale or clean data. Toggling the `--use_excluded` / `--use_added` (or `--tuning`) arguments inside `train.py` dynamically forces your entire web workflow into the tensor processing batches.

### Launching the Labeling Tool
```bash
cd labeling-tool
npm install       # First time only
npm run dev       # Start the web server
```
Visit `http://localhost:5173` to open the labeling interface.

## 🚂 5. Training Instructions
The main entry point for training is `train.py`. It supports extensive hyperparameter tuning and model configuration.

### Basic Run (Vanilla ResNet-34)
```bash
python train.py --model resnet34 --epochs 30 --batch_size 32
```

### Advanced Run (Tuning & Attention)
```bash
python train.py --model resnet34 --attn cbam --tuning --epochs 30
```

### 🔑 Key Architecture Arguments
Below are detailed explanations for the critical structural options you can select via `--model` and `--attn`.

**Models (`--model`)**:
- `resnet18`: A lightweight 18-layer residual network providing extremely fast training iterations and establishing a solid classification baseline.
- `resnet34`: A deeper 34-layer residual network capable of extracting more complex, granular representations for fine-grained aircraft features.
- `resnet34d`: A custom-modified ResNet-34 natively incorporating structural Dropout before the fully connected layers, drastically mitigating overfitting in dense representations.

**Attention Modules (`--attn`)**:
- `channel`: Channel Attention module that weights feature maps independently, teaching the network *what* specific channel features are most important to the current prediction.
- `spatial`: Spatial Attention module operating structurally across the feature map axes, guiding the network mathematically on *where* to focus physically within the image.
- `cbam`: Convolutional Block Attention Module that sequentially applies both channel and spatial attentions for the most comprehensive, synergistic feature refinement.

### Comprehensive Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | — | Architecture: `resnet18`, `resnet34`, `resnet34d` (see above) |
| `--attn` | — | Attention module: `channel`, `spatial`, `cbam` (see above) |
| `--tuning` | `False` | Shortcut: enables both `--use_excluded` and `--use_added` |
| `--use_excluded` | `False` | Skip images flagged as noisy by the Labeling Tool |
| `--use_added` | `False` | Inject custom images from the Labeling Tool into training |
| `--folds` | `4` | Number of folds (K) for cross-validation |
| `--epochs` | `30` | Number of training epochs per fold |
| `--batch_size` | `32` | Samples per training batch |
| `--lr` | `0.001` | Peak learning rate (Cosine Annealing schedule) |
| `--optimizer` | `adamw` | Optimizer: `adam`, `adamw`, `sgd` |
| `--weight_decay` | `1e-4` | L2 regularization coefficient |
| `--label_smoothing` | `0.0` | Label smoothing factor (e.g. `0.1` reduces overconfidence) |
| `--label` | `family` | Label hierarchy: `variant`, `family`, `manufacturer` |
| `--crop_bbox` | `False` | Crop input images to bounding box before training |
| `--seed` | `42` | Random seed for reproducible fold splits |
| `--save_csv` | `results.csv` | Output file for experiment logs |
| `--outdir` | `.` | Output directory for plots, matrices, and metrics |
| `--added_repeat` | `1` | Oversampling multiplier for custom-added images |

## 📊 6. Evaluation Instructions
The evaluation phase is built directly into the end of the `train.py` script. After processing all `K` folds, the script performs a **Weighted Soft Voting Ensemble**, weighting each model by its validation accuracy, to predict against the isolated Test set.

All results are automatically flushed to your `--outdir` generating:
1. **`results.csv`**: A master log appending the performance of every experiment.
2. **`history_[tag].csv`**: Epoch-by-epoch loss/accuracy metrics.
3. **`curve_[tag].png`**: Beautiful matplotlib visualizations of your learning curves.
4. **`confusion_[tag].png`**: A normalized Seaborn heatmap of the test predictions.
5. **`classwise_[tag].csv`**: Per-class accuracy metrics to identify underperforming aircraft variants.

> **Pro-Tip**: Use the provided utility scripts to fetch your best weights/settings:
> ```bash
> python pick_best_model.py results.csv
> python pick_best_data.py results.csv
> ```

## 🖥️ 7. How to Run the Inference Demo
Instead of a standard local python GUI, this project features a modern **SvelteKit** web application `inference-demo`. This demo allows you to easily test your trained models in a user-friendly web interface.

### Key Features
- **Drag & Drop Image Upload**: Easily upload images of aircraft for inference.
- **Bounding Box Cropping**: Draw a bounding box directly on the canvas to focus inference on the aircraft area.
- **Model Selection & Ensemble**: Choose one or multiple `.pth` models trained by `train.py`. Selecting multiple models will automatically perform an ensemble prediction.
- **Visual Results**: Beautifully visualized top predictions with confidence score bars.

### Prerequisites

**1. Node.js (npm)**
Download and install Node.js from https://nodejs.org (LTS version recommended). npm is bundled with Node.js.

**2. Python Path Configuration**
The inference server spawns a Python subprocess to run `scripts/infer.py`. By default it uses the system `python` command, but if you're using a conda environment you must update the path in `inference-demo/src/routes/api/infer/+server.js`:

```js
// Line ~35 — replace with your actual Python path
const result = spawnSync(
    'python',   // e.g. 'C:\\Users\\yourname\\miniconda3\\envs\\YOUR_ENV\\python.exe'
    args,
    ...
```

To find your conda environment's Python path:
```bash
conda activate YOUR_ENV
python -c "import sys; print(sys.executable)"
```

### Launching the Web App
```bash
cd inference-demo
npm install         # Install dependencies (first time only)
npm run dev         # Start the Development Server
```
Visit `http://localhost:5173` (or the port specified in your console, e.g., `5174`) to interact with the demo.

---

## 📈 8. Experimental Results

### 8.1 Overview

All experiments share the following base configuration: **4-Fold Cross Validation**, AdamW optimizer, lr=0.001 (Cosine Annealing), weight_decay=5e-3, batch_size=32, BBox Crop, RandomRotation±15°, CutMix α=0.5, `family` label (70 classes).

| # | Model | Attention | Data Strategy | Mean Val Acc | **Test Acc** |
|---|-------|-----------|---------------|-------------|-------------|
| 1 | SimpleCNN | — | Vanilla | 31.39% | 36.15% |
| 2 | ResNet-34d | None | Vanilla | 90.61% | 91.75% |
| 3 | ResNet-34d | None | Excluded | 90.76% | 91.90% |
| 4 | ResNet-34d | None | Added | 90.69% | 92.20% |
| 5 | ResNet-34d | None | Tuning (Excl+Add) | 90.54% | 92.25% |
| 6 | ResNet-34d | **Channel** | Tuning | **91.00%** | **93.00%** |
| 7 | ResNet-34d | Spatial | Tuning | 90.55% | 92.45% |
| 8 | ResNet-34d | CBAM | Tuning | 90.64% | 91.50% |

> **Best Model: ResNet-34d + Channel Attention + Tuning → 93.00% Test Accuracy**

### 8.2 Baseline vs Best Model

| | SimpleCNN | ResNet-34d + Channel + Tuning |
|--|-----------|-------------------------------|
| Test Accuracy | 36.15% | **93.00%** |
| Parameters | 2.5M | 21.5M |
| Training Time | 3,305s | 2,003s |
| Epochs | 100 | 50 |
| Overfitting | Severe (train≈20%, val≈31%) | Mild (train≈82%, val≈91%) |

SimpleCNN failed to learn meaningful discriminative features across 70 visually similar aircraft families even after 100 epochs. ResNet-34d with ImageNet pretrained weights surpassed 90% within 50 epochs, demonstrating the critical role of transfer learning for fine-grained datasets.

### 8.3 Effect of Each Component

**Data Strategy:**
- Excluded only: +0.15%p
- Added only: +0.45%p
- Tuning (both): +0.50%p

**Attention Module (over Tuning baseline 92.25%):**
- Channel: **+0.75%p** ← Best
- Spatial: +0.20%p
- CBAM: -0.75%p

Channel Attention outperformed CBAM despite being a subset of it. After BBox cropping removes background, the spatial attention in CBAM likely suppresses useful aircraft regions — making pure channel-wise feature selection the more effective strategy.

---

## 🔬 9. Error Analysis

### 9.1 Lowest Performing Classes (Best Model)

| Class | Samples | Accuracy |
|-------|---------|----------|
| C-47 | 20 | 60.0% |
| DC-10 | 20 | 70.0% |
| DC-9 | 20 | 70.0% |
| A330 | 40 | 72.5% |
| A300 | 20 | 75.0% |

**Root Causes:**
- **C-47 vs DC-3**: C-47 is a military derivative of DC-3 — nearly identical fuselage and wing shape
- **DC-10 vs MD-11**: Both share the tri-engine layout (two underwing + one tail), and MD-11 is the direct evolution of DC-10
- **DC-9 vs MD-80/MD-90**: All rear twin-engine narrowbodies from the same design lineage
- **A330 vs A300**: Similar wide-body fuselage cross-section and wing planform

### 9.2 Perfect Accuracy Classes

Military jets (F-16, Tornado, Spitfire), business jets (Gulfstream, Global Express, Falcon 900), and utility aircraft (C-130, DR-400) all achieved **100% accuracy**, as their silhouettes are visually distinct from commercial airliners.

### 9.3 Visual Explanation (Grad-CAM)

**Grad-CAM (Gradient-weighted Class Activation Mapping)** was utilized to visually verify the model's decision-making process. The generated heatmaps confirmed that our attention-enhanced models successfully pinpoint and focus on the most discriminative aircraft structures—such as engine configurations, tail designs, and wing shapes—rather than relying on arbitrary background noise.

---

## 💬 10. Discussion & Conclusion

### Key Findings

1. **Transfer learning is essential** for fine-grained classification with limited data. The 56.85%p gap between SimpleCNN and ResNet-34d cannot be explained by model size alone — ImageNet pretraining provides reusable low/mid-level features critical for shape recognition.

2. **Channel Attention > CBAM** when input is spatially pre-cropped. The BBox crop already handles spatial noise; adding spatial attention on top introduces unnecessary suppression of informative aircraft regions.

3. **Active Labeling has compounding value.** While individual gains (Excluded: +0.15%p, Added: +0.45%p) appear modest, the system enables continuous dataset improvement without any code changes — a meaningful operational advantage.

4. **K-Fold Ensemble improves reliability.** Fold-to-fold val accuracy variance stayed within 1.5%p, confirming stable generalization rather than sensitivity to a particular data split.

### Future Work
- Targeted data collection for confused class pairs (C-47/DC-3, DC-10/MD-11)
- Higher input resolution (448×448) for finer detail recognition

---

## 📚 References

1. Maji et al. (2013). Fine-grained visual classification of aircraft. *arXiv:1306.5151*
2. He et al. (2016). Deep residual learning for image recognition. *CVPR 2016*
3. Woo et al. (2018). CBAM: Convolutional block attention module. *ECCV 2018*
4. Yun et al. (2019). CutMix: Training strategy that makes use of sample mixing. *ICCV 2019*
5. Loshchilov & Hutter (2019). Decoupled weight decay regularization. *ICLR 2019*
