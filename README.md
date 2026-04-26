# 🛩️ FGVC-Aircraft Image Classification & Active Labeling Pipeline

<div align="center">
  <em>An Advanced Computer Vision Project featuring K-Fold Cross-Validation, Attention Mechanisms, and an Interactive Labeling Web Environment.</em>
</div>

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
- **Data Engineering (`dataset_utils.py`)**:
  - Automatically crops out the ~20px bottom banners from images (which contain noisy copyright text).
  - Merges the default annotations with user-defined JSON configurations from the `labeling-tool`.
  - Performs a **6:2:2 Stratified Re-Split** (Train/Val/Test) dynamically ensuring classes remain balanced even when new data is injected.

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

## 🚂 4. Training Instructions
The main entry point for training is `train.py`. It supports extensive hyperparameter tuning and model configuration. 

### Basic Run (Vanilla ResNet-34)
```bash
python train.py --model resnet34 --epochs 30 --batch_size 32
```

### Advanced Run (Tuning & Attention)
To train a `ResNet-34` with a `CBAM` attention mechanism, while respecting your manually added/excluded datasets from the labeling tool:
```bash
python train.py --model resnet34 --attn cbam --tuning --epochs 30
```

### Key Training Arguments
- `--model`: Architecture choice (`cnn`, `resnet18`, `resnet34`, `resnet34d`).
- `--attn`: Attention module (`none`, `channel`, `spatial`, `cbam`).
- `--tuning`: Shortcut flag that enables `--use_excluded` & `--use_added`.
- `--folds`: Number of K-Fold splits (Default: `4`).
- `--label_smoothing`: Regularization parameter to prevent overconfidence (e.g., `0.1`).

## 📊 5. Evaluation Instructions
The evaluation phase is built directly into the end of the `train.py` script. After processing all `K` folds, the script performs a **Weighted Soft Voting Ensemble**, weighting each model by its validation accuracy, to predict against the isolated Test set.

All results are automatically flushed to your `--outdir` (default: root) generating:
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

## 🖥️ 6. How to Run the Inference Demo (Labeling Tool)
Instead of a standard local python GUI, this project features a modern **SvelteKit** web application `labeling-tool`. This serves as both an inference visualizer and an active data-centric AI refinement environment. 

Here you can view images, flag poor data (sends to `excluded/`), or inject new web-scraped data (sends to `added_images/`).

### Launching the Web App
Make sure you have Node.js (`npm`) installed.
```bash
cd labeling-tool
npm ci              # Install precise dependency tree
npm run prepare     # Sync SvelteKit routes
npm run dev         # Start the Development Server
```
Visit `http://localhost:5173` (or the port specified in your console) to interact with the demo. The JSON results will be automatically parsed by `train.py` on your next training run!
