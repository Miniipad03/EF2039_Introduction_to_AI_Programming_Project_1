"""
src/train_baseline.py — Baseline Custom CNN 학습
실행: python src/train_baseline.py  (프로젝트 루트에서)

결과:
  results/baseline_log.csv       — epoch별 train/val loss, val accuracy
  results/baseline_curve.png     — 학습 곡선
  results/baseline_best.pth      — best val accuracy 기준 모델 가중치
"""

import os
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 프로젝트 루트를 sys.path에 추가 (src/ 하위에서 실행 시 대비)
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import AircraftDataset, get_transforms

os.makedirs("results", exist_ok=True)

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
DATA_ROOT = Path("data/fgvc-aircraft-2013b/data")
IMAGE_DIR = DATA_ROOT / "images"

TRAIN_ANN = DATA_ROOT / "images_manufacturer_train.txt"
VAL_ANN   = DATA_ROOT / "images_manufacturer_val.txt"

LOG_CSV    = Path("results/baseline_log.csv")
CURVE_PNG  = Path("results/baseline_curve.png")
BEST_PTH   = Path("results/baseline_best.pth")

# ── 하이퍼파라미터 ────────────────────────────────────────────────────────────
BATCH_SIZE = 64
EPOCHS     = 30
LR         = 0.01
MOMENTUM   = 0.9
WEIGHT_DECAY = 1e-4
NUM_CLASSES  = 30


# ── 모델 정의 ─────────────────────────────────────────────────────────────────
class BaselineCNN(nn.Module):
    """
    Simple 3-block CNN for aircraft manufacturer classification.

    Input  : (B, 3, 224, 224)
    Output : (B, 30)
    """

    def __init__(self, num_classes=30):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: (3, 224, 224) → (32, 112, 112)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: (32, 112, 112) → (64, 56, 56)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: (64, 56, 56) → (128, 28, 28)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # AdaptiveAvgPool: (128, 28, 28) → (128, 4, 4)
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ── 학습 / 검증 함수 ─────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset & DataLoader
    train_ds = AircraftDataset(TRAIN_ANN, IMAGE_DIR, transform=get_transforms("train"))
    val_ds   = AircraftDataset(VAL_ANN,   IMAGE_DIR, transform=get_transforms("val"))

    # label2idx는 train set 기준으로 구성되므로 val도 동일 매핑 사용
    # (annotation 파일에 동일한 30개 클래스가 존재하므로 일치)
    num_classes = train_ds.num_classes()
    print(f"클래스 수: {num_classes} (train: {len(train_ds)}, val: {len(val_ds)})")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    # Model, Loss, Optimizer, Scheduler
    model = BaselineCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # CSV 헤더
    with open(LOG_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc"])

    # 학습 루프
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_epoch   = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # CSV 기록
        with open(LOG_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{val_acc:.4f}"])

        # Best model 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save(model.state_dict(), BEST_PTH)

        print(f"Epoch [{epoch:2d}/{EPOCHS}]  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.5f}"
              + (" ← best" if val_acc == best_val_acc else ""))

    # ── 학습 곡선 저장 ────────────────────────────────────────────────────────
    epochs = range(1, EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["val_acc"], color="green", label="Val Accuracy")
    ax2.axhline(best_val_acc, color="red", linestyle="--",
                label=f"Best: {best_val_acc:.4f} @ ep{best_epoch}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Baseline CNN — FGVC-Aircraft (Manufacturer, 30 classes)", fontsize=11)
    plt.tight_layout()
    plt.savefig(CURVE_PNG, dpi=150)
    plt.close()

    # ── 최종 결과 출력 ────────────────────────────────────────────────────────
    final_val_acc = history["val_acc"][-1]
    print("\n" + "=" * 55)
    print("  학습 완료")
    print("=" * 55)
    print(f"  최종 val accuracy (epoch {EPOCHS}): {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
    print(f"  best val accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%) @ epoch {best_epoch}")

    # Overfitting 간단 판단
    last5_train = sum(history["train_loss"][-5:]) / 5
    last5_val   = sum(history["val_loss"][-5:]) / 5
    gap = last5_val - last5_train
    if gap > 0.5:
        obs = "Overfitting 의심 (val_loss >> train_loss)"
    elif history["val_acc"][-1] < history["val_acc"][EPOCHS // 2]:
        obs = "Val accuracy 정체 or 하락 — learning rate 조정 또는 regularization 검토"
    else:
        obs = "안정적으로 수렴 중"
    print(f"  관찰: {obs}")
    print(f"\n  저장 파일:")
    print(f"    {LOG_CSV}")
    print(f"    {CURVE_PNG}")
    print(f"    {BEST_PTH}")
    print("=" * 55)


if __name__ == "__main__":
    main()
