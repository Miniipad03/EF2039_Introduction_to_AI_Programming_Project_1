import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix
import seaborn as sns

from dataset_utils import FGVCAircraft


# ── Custom transform: 하단 20px 배너 제거 ─────────────────────────────────────
class RemoveBottomBanner:
    """이미지 하단의 20px 사진 작가 크레딧 배너를 제거합니다."""
    def __init__(self, pixels: int = 20):
        self.pixels = pixels

    def __call__(self, img):
        w, h = img.size
        return img.crop((0, 0, w, h - self.pixels))


# ── 모델 정의 ─────────────────────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    """3-4개의 Conv 레이어로 구성된 기본 CNN."""
    def __init__(self, num_classes: int = 70):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: 28 -> 14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "cnn":
        return SimpleCNN(num_classes=num_classes)
    elif model_name in ("resnet18", "resnet34"):
        weights = (models.ResNet18_Weights.DEFAULT if model_name == "resnet18"
                   else models.ResNet34_Weights.DEFAULT)
        model = (models.resnet18(weights=weights) if model_name == "resnet18"
                 else models.resnet34(weights=weights))
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from cnn, resnet18, resnet34")


# ── Argument Parser ────────────────────────────────────────────────────────────
def get_args():
    parser = argparse.ArgumentParser(description="FGVC-Aircraft CNN Trainer")
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--lr",            type=float, default=0.001)
    parser.add_argument("--epochs",        type=int,   default=30)
    parser.add_argument("--optimizer",     type=str,   default="adam",
                        choices=["adam", "sgd"])
    parser.add_argument("--model",         type=str,   default="resnet18",
                        choices=["cnn", "resnet18", "resnet34"])
    parser.add_argument("--label",         type=str,   default="family",
                        choices=["variant", "family", "manufacturer"])
    parser.add_argument("--crop_bbox",     action="store_true",
                        help="BBox 영역만 crop하여 학습")
    parser.add_argument("--original_only", action="store_true",
                        help="라벨링 툴의 추가/제거 데이터를 무시하고 원본 데이터만 사용")
    parser.add_argument("--save_csv",      type=str,   default="results.csv")
    parser.add_argument("--weight_decay",  type=float, default=1e-4)
    return parser.parse_args()


# ── 학습 / 평가 루프 ──────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device, return_preds=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if return_preds:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    acc = correct / total

    if return_preds:
        return avg_loss, acc, all_preds, all_labels
    return avg_loss, acc


# ── 클래스 정규화 평균 정확도 (논문 기준 메트릭) ─────────────────────────────
def class_normalized_accuracy(preds, labels, num_classes):
    preds  = np.array(preds)
    labels = np.array(labels)
    per_class = np.zeros(num_classes)
    counts    = np.zeros(num_classes)
    for c in range(num_classes):
        mask = labels == c
        counts[c]    = mask.sum()
        per_class[c] = (preds[mask] == c).sum()
    valid = counts > 0
    return (per_class[valid] / counts[valid]).mean()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = get_args()

    print("===== Input Arguments =====")
    for k, v in vars(args).items():
        print(f"  {k:<14}: {v}")
    print("===========================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Windows에서 num_workers=0이면 메인 프로세스가 데이터 로딩을 처리해 GPU 병목 발생
    # 2로 설정 시 에러가 나면 0으로 낮출 것
    num_workers = 2 if os.name == 'nt' else 4

    # ── Transform 정의 ──────────────────────────────────────────────────────
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std  = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        RemoveBottomBanner(20),
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    eval_transform = transforms.Compose([
        RemoveBottomBanner(20),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    # ── 데이터셋 로드 ────────────────────────────────────────────────────────
    # --original_only: 라벨링 툴 수정 무시, 원본 데이터만 사용
    use_custom = not args.original_only

    # split='all'로 통합 class_to_idx 생성 → train/val/test 간 클래스 인덱스 일치 보장
    ref_dataset = FGVCAircraft(split='all', label=args.label,
                               use_excluded=use_custom, use_added=use_custom)
    shared_class_to_idx = ref_dataset.class_to_idx
    shared_classes      = ref_dataset.classes
    num_classes         = len(shared_classes)

    def make_dataset(split, transform):
        ds = FGVCAircraft(split=split, label=args.label,
                          transform=transform, crop_bbox=args.crop_bbox,
                          use_excluded=use_custom, use_added=use_custom)
        ds.class_to_idx = shared_class_to_idx
        ds.classes      = shared_classes
        return ds

    train_dataset = make_dataset('train', train_transform)
    val_dataset   = make_dataset('val',   eval_transform)
    test_dataset  = make_dataset('test',  eval_transform)

    print(f"Label level : {args.label}  |  Classes: {num_classes}  "
          f"| {'원본' if args.original_only else '커스텀'} 데이터 사용")
    train_dataset.summary()

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_dataset,  shuffle=False, **loader_kwargs)

    # ── ResNet warmup: fc만 먼저 학습 ────────────────────────────────────────
    model = build_model(args.model, num_classes).to(device)

    if args.model in ("resnet18", "resnet34"):
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
        warmup_epochs = 5
    else:
        warmup_epochs = 0

    # ── Optimizer: freeze 이후 생성 (trainable param만 포함) ─────────────────
    def make_optimizer(params):
        if args.optimizer == "adam":
            return optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        return optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    trainable = lambda: filter(lambda p: p.requires_grad, model.parameters())
    optimizer = make_optimizer(trainable())
    criterion = nn.CrossEntropyLoss()

    # Cosine scheduler: warmup 이후 남은 에포크 기준
    remaining_epochs = max(args.epochs - warmup_epochs, 1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs)

    # ── 학습 루프 ────────────────────────────────────────────────────────────
    list_train_loss, list_val_loss, list_val_acc = [], [], []
    best_val_acc = 0.0

    import time
    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        # Warmup 종료 시 backbone unfreeze + optimizer에 backbone 파라미터 추가
        if args.model in ("resnet18", "resnet34") and epoch == warmup_epochs + 1:
            backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]
            for p in backbone_params:
                p.requires_grad = True
            optimizer.add_param_group({
                'params': backbone_params,
                'lr': args.lr * 0.1,  # backbone은 낮은 lr로 fine-tune
                'weight_decay': args.weight_decay,
            })
            print(f"[Epoch {epoch}] Backbone unfreeze (backbone lr={args.lr * 0.1:.0e})")

        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        epoch_secs = time.time() - epoch_start

        # warmup 이후에만 scheduler step
        if epoch > warmup_epochs:
            scheduler.step()

        list_train_loss.append(train_loss)
        list_val_loss.append(val_loss)
        list_val_acc.append(val_acc)

        print(f"[Epoch {epoch:3d}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
              f"Time: {epoch_secs:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_{args.model}_{args.label}.pth")

    total_secs = time.time() - train_start
    print(f"\n총 학습 시간: {total_secs // 60:.0f}m {total_secs % 60:.1f}s")

    # ── 테스트 평가 (best model 기준) ────────────────────────────────────────
    model.load_state_dict(torch.load(f"best_{args.model}_{args.label}.pth",
                                     map_location=device, weights_only=True))
    test_loss, test_acc, preds, gt = evaluate(model, test_loader, criterion, device,
                                              return_preds=True)
    cn_acc = class_normalized_accuracy(preds, gt, num_classes)

    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | "
          f"Class-Norm Acc: {cn_acc:.4f}")

    # ── 에포크별 히스토리 CSV ────────────────────────────────────────────────
    data_tag = "original" if args.original_only else "custom"
    tag = f"{args.model}_{args.label}_{data_tag}_bs{args.batch_size}_lr{args.lr}_opt{args.optimizer}"

    history_df = pd.DataFrame({
        "epoch":      range(1, args.epochs + 1),
        "train_loss": list_train_loss,
        "val_loss":   list_val_loss,
        "val_acc":    list_val_acc,
    })
    history_df.to_csv(f"history_{tag}.csv", index=False)
    print(f"History saved → history_{tag}.csv")

    # ── 학습 곡선 플롯 ───────────────────────────────────────────────────────
    epochs_range = range(1, args.epochs + 1)
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs_range, list_train_loss, label="Train Loss")
    ax1.plot(epochs_range, list_val_loss,   label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss"); ax1.legend()

    ax2.plot(epochs_range, list_val_acc, label="Val Accuracy")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy"); ax2.legend()

    plt.tight_layout()
    plt.savefig(f"curve_{tag}.png")
    plt.close()
    print(f"Learning curve saved → curve_{tag}.png")

    # ── Confusion Matrix ─────────────────────────────────────────────────────
    cm      = confusion_matrix(gt, preds, labels=list(range(num_classes)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig_size = max(12, num_classes // 3)
    _, ax = plt.subplots(figsize=(fig_size, fig_size))

    # 클래스 수가 적으면 레이블 표시 (manufacturer ~30, family ~70)
    show_labels = num_classes <= 70
    tick_labels = shared_classes if show_labels else False
    sns.heatmap(cm_norm, ax=ax, cmap="Blues",
                xticklabels=tick_labels, yticklabels=tick_labels,
                vmin=0, vmax=1)
    if show_labels:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=6)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=6)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix ({args.label}, {data_tag})")
    plt.tight_layout()
    plt.savefig(f"confusion_{tag}.png", dpi=150)
    plt.close()
    print(f"Confusion matrix saved → confusion_{tag}.png")

    # ── 요약 결과 CSV 추가 기록 ──────────────────────────────────────────────
    row = {
        "model":         args.model,
        "label":         args.label,
        "data":          data_tag,
        "batch_size":    args.batch_size,
        "lr":            args.lr,
        "epochs":        args.epochs,
        "optimizer":     args.optimizer,
        "best_val_acc":  best_val_acc,   # best checkpoint 기준
        "test_acc":      test_acc,
        "cn_acc":        cn_acc,
    }
    df = pd.DataFrame([row])
    file_exists = os.path.isfile(args.save_csv)
    df.to_csv(args.save_csv, mode="a", header=not file_exists, index=False)
    print(f"Results appended → {args.save_csv}")


if __name__ == "__main__":
    main()
