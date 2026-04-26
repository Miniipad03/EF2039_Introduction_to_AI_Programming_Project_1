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

### Data Engineering (`dataset_utils.py` Detailed Analysis)
`dataset_utils.py` is the operational core of the data pipeline, housing the custom `FGVCAircraft` and `SampleDataset` classes specifically engineered to gracefully blend raw dataset inputs with injected custom intelligence:
1. **Purging Excluded Data**: Reads the `excluded.json` objects (flagged via the labeling-tool) and actively prohibits their initialization to prevent feeding the model noisy or erroneous samples.
2. **Train-Set Forced Injection**: Ingests your newly scraped intelligence via `added_images.json`. To prevent evaluation data leakage, it explicitly locks all custom additions dynamically into the Training split.
3. **6:2:2 Dynamic Stratified Re-Split**: Unlike the generic 1:1:1 FGVC methodology, this logic recalculates the overall scale (`N_total`) dynamically considering new data, forcing a 0.4 proportional chunk exactly back into validation/test splits while completely maintaining class stratifications.
4. **Targeted Bounding Box Extraction (`crop_bbox`)**: When enabled, mathematically crops precisely around the aircraft using the `images_box.txt` coords, destroying background noise variables.
5. **Smart Bottom Banner Auto-Removal (`RemoveBottomBanner`)**: FGVC default images ship with ~20px high copyright banners that destructively bias Neural Network Feature maps. This custom transformer securely lops off the bottom footprint of native images without accidentally tampering with user-injected datasets.

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

### Comprehensive Training Arguments
`train.py` exposes exactly 18 command-line parameters for precision neural network deployment:

- `--model`: Structural neural network architecture selection (`cnn`, `resnet18`, `resnet34`, `resnet34d`).
- `--attn`: Specifies an attention mechanism to be injected into deeper ResNet layers (`none`, `channel`, `spatial`, `cbam`).
- `--tuning`: Active Learning shortcut flag that triggers both `--use_excluded` and `--use_added` simultaneously.
- `--use_excluded`: Forces the dataset engine to ignore noisy images defined by the Labeling Tool.
- `--use_added`: Instructs the dataset engine to inject custom Web Labeling Tool images directly into the Train subset.
- `--folds`: The exact split integer (K) utilized for Cross-Validation (Default: `4`).
- `--epochs`: Limits the absolute number of backward propagation cycles over the full dataset (Default: `30`).
- `--batch_size`: Specifies the discrete volume of isolated samples fed forward at a time (Default: `32`).
- `--lr`: Declares the maximum optimizer learning rate, subject to dynamic Cosine Annealing (Default: `0.001`).
- `--optimizer`: Gradient Descent algorithm implementation selector (`adam`, `adamw`, `sgd`).
- `--weight_decay`: Constant L2 parameter penalty punishing infinitely expanding weights (Default: `1e-4`).
- `--label_smoothing`: Modifies strict Ground Truth `1.0` targets into softer probabilistic vectors to suppress overconfidence (Default: `0.0`).
- `--label`: Designates the chosen hierarchy tier for FGVC categorical predictions (`variant`, `family`, `manufacturer`).
- `--crop_bbox`: Clips away external image context, limiting the actual input tensors physically onto the object Bounding Box scale.
- `--seed`: Hardcoded initialization integer to perfectly replicate fold shuffling parameters globally (Default: `42`).
- `--save_csv`: Name of the aggregate logging file that continuously tracks distinct experiment iterations (Default: `results.csv`).
- `--outdir`: Export destination string for generated matrices, charts, and metrics (Default: `.`).
- `--added_repeat`: Dynamic multiplier (oversampling element) replicating injected custom web images to amplify logic footprints structurally (Default: `1`).

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

## 🖥️ 7. How to Run the Inference Demo (Labeling Tool)
Instead of a standard local python GUI, this project features a modern **SvelteKit** web application `labeling-tool`. This serves as both an inference visualizer and an active data-centric AI refinement environment. 

### Launching the Web App
Make sure you have Node.js (`npm`) installed.
```bash
cd labeling-tool
npm ci              # Install precise dependency tree
npm run prepare     # Sync SvelteKit routes
npm run dev         # Start the Development Server
```
Visit `http://localhost:5173` (or the port specified in your console) to interact with the demo. The JSON results will be automatically parsed by `train.py` on your next training run!
