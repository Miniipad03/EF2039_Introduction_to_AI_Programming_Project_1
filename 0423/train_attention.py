import argparse
import os
import sys
import pathlib
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix
import seaborn as sns


sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from dataset_utils import FGVCAircraft


# ══════════════════════════════════════════════════════════════════════════════
#  Attention Modules
# ══════════════════════════════════════════════════════════════════════════════

class ChannelAttention(nn.Module):
    """SE-style channel attention (CBAM channel branch)."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c = x.shape[:2]
        avg  = x.mean(dim=[2, 3])          # (B, C)
        mx   = x.amax(dim=[2, 3])          # (B, C)
        scale = self.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * scale.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """CBAM spatial branch."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg  = x.mean(dim=1, keepdim=True)   # (B, 1, H, W)
        mx   = x.amax(dim=1, keepdim=True)   # (B, 1, H, W)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """Channel → Spatial attention (Woo et al., 2018)."""
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(spatial_kernel)

    def forward(self, x):
        return self.spatial(self.channel(x))


# ── 실험 설정 핵심부: 여기 한 곳만 수정하면 새로운 attention 타입 추가 가능 ──────
ATTN_REGISTRY: dict[str, type] = {
    "none":    nn.Identity,
    "channel": ChannelAttention,
    "spatial": SpatialAttention,
    "cbam":    CBAM,
}

def make_attn(attn_type: str, channels: int) -> nn.Module:
    """attn_type 문자열 → attention 모듈 인스턴스."""
    if attn_type not in ATTN_REGISTRY:
        raise ValueError(f"Unknown attention type '{attn_type}'. "
                         f"Available: {list(ATTN_REGISTRY)}")
    cls = ATTN_REGISTRY[attn_type]
    if cls is nn.Identity:
        return nn.Identity()
    # SpatialAttention은 channels 인자 불필요
    if cls is SpatialAttention:
        return cls()
    return cls(channels)


# ══════════════════════════════════════════════════════════════════════════════
#  Attention Injection
# ══════════════════════════════════════════════════════════════════════════════

class BlockWithAttention(nn.Module):
    """ResNet BasicBlock을 감싸 출력에 attention을 적용한다.

    적용 위치: block(x) → attention → 출력
    (residual이 합쳐진 뒤 적용 — post-block attention)
    """
    def __init__(self, block: nn.Module, attn: nn.Module):
        super().__init__()
        self.block = block
        self.attn  = attn

    def forward(self, x):
        return self.attn(self.block(x))


def inject_attention(layer: nn.Sequential, attn_type: str) -> None:
    """layer 내 모든 BasicBlock을 BlockWithAttention으로 교체 (in-place)."""
    if attn_type == "none":
        return
    for i, block in enumerate(layer):
        channels = block.conv2.out_channels
        layer[i] = BlockWithAttention(block, make_attn(attn_type, channels))


# ── 삽입 위치 설정: shallow / deep 범위를 여기서만 바꾸면 됨 ──────────────────
#   shallow: 저수준 feature (작은 receptive field, 높은 해상도)
#   deep   : 고수준 feature (큰 receptive field, 낮은 해상도)
SHALLOW_LAYERS = ("layer1", "layer2")
DEEP_LAYERS    = ("layer3", "layer4")


# ══════════════════════════════════════════════════════════════════════════════
#  GradCAM
# ══════════════════════════════════════════════════════════════════════════════

class GradCAM:
    """마지막 conv 레이어에 hook을 걸어 class activation map을 생성한다."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model       = model
        self.activations: torch.Tensor | None = None
        self.gradients:   torch.Tensor | None = None
        self._hooks = [
            target_layer.register_forward_hook(
                lambda m, inp, out: setattr(self, "activations", out.detach())
            ),
            target_layer.register_full_backward_hook(
                lambda m, g_in, g_out: setattr(self, "gradients", g_out[0].detach())
            ),
        ]

    def compute(self, x: torch.Tensor, class_idx: torch.Tensor | None = None):
        """CAM 텐서 (B, 1, H, W) 와 예측 class index를 반환한다."""
        self.model.zero_grad()
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
        logits.gather(1, class_idx.view(-1, 1)).sum().backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)   # (B, C, 1, 1)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        return cam, class_idx                                       # (B, 1, h, w)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


def get_gradcam_layer(model: nn.Module, layer_name: str = "layer4") -> nn.Module:
    """지정 layer의 마지막 블록 안 conv2를 반환한다.
    BlockWithAttention으로 감싸진 경우 inner block을 자동으로 벗긴다.
    """
    layer      = getattr(model, layer_name)
    last_block = layer[-1]
    inner      = last_block.block if isinstance(last_block, BlockWithAttention) else last_block
    return inner.conv2


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _denorm(t: torch.Tensor) -> np.ndarray:
    """(3, H, W) normalized tensor → (H, W, 3) numpy [0, 1]."""
    img = (t.cpu() * _IMAGENET_STD + _IMAGENET_MEAN).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def save_gradcam_grid(
    model: nn.Module,
    loader: DataLoader,
    gradcam: GradCAM,
    class_names: list[str],
    n: int,
    save_path: str,
    device: torch.device,
) -> None:
    """테스트 이미지 n장에 대해 원본 / GradCAM overlay 그리드를 저장한다."""
    model.eval()
    imgs_buf, lbls_buf = [], []
    for imgs, lbls in loader:
        imgs_buf.append(imgs)
        lbls_buf.append(lbls)
        if sum(x.size(0) for x in imgs_buf) >= n:
            break

    images = torch.cat(imgs_buf)[:n]   # (N, 3, H, W)  — CPU, normalized
    labels = torch.cat(lbls_buf)[:n]

    fig, axes = plt.subplots(n, 2, figsize=(7, 3.2 * n))
    if n == 1:
        axes = axes[np.newaxis]

    for i in range(n):
        img_dev = images[i:i+1].to(device)
        lbl     = labels[i].item()

        cam, pred_idx = gradcam.compute(img_dev)
        pred = pred_idx[0].item()

        # CAM → image 해상도로 업샘플 후 [0, 1] 정규화
        cam_up = F.interpolate(cam, size=images.shape[-2:],
                               mode="bilinear", align_corners=False)
        cam_np = cam_up[0, 0].cpu().numpy()
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)

        img_np  = _denorm(images[i])
        heatmap = plt.cm.jet(cam_np)[..., :3]          # RGB jet colormap
        overlay = np.clip(0.5 * img_np + 0.5 * heatmap, 0, 1)

        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"GT: {class_names[lbl]}", fontsize=7)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(overlay)
        title_color = "green" if pred == lbl else "red"
        axes[i, 1].set_title(f"Pred: {class_names[pred]}", fontsize=7, color=title_color)
        axes[i, 1].axis("off")

    plt.suptitle("GradCAM  (green=correct, red=wrong)", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"GradCAM grid saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Simple CNN (scratch training — attention / warmup 없음)
# ══════════════════════════════════════════════════════════════════════════════

class SimpleCNN(nn.Module):
    """4-block CNN for FGVC-Aircraft (scratch training)."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,   32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32,  64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128,256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ══════════════════════════════════════════════════════════════════════════════
#  Model Builder
# ══════════════════════════════════════════════════════════════════════════════

def build_model(model_name: str, num_classes: int,
                shallow_attn: str, deep_attn: str) -> nn.Module:
    if model_name == "simplecnn":
        model = SimpleCNN(num_classes)
        print(f"  SimpleCNN (scratch)  |  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model

    weights = (models.ResNet18_Weights.DEFAULT if model_name == "resnet18"
               else models.ResNet34_Weights.DEFAULT)
    model   = (models.resnet18(weights=weights) if model_name == "resnet18"
               else models.resnet34(weights=weights))
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    for name in SHALLOW_LAYERS:
        inject_attention(getattr(model, name), shallow_attn)
    for name in DEEP_LAYERS:
        inject_attention(getattr(model, name), deep_attn)

    n_attn = sum(1 for m in model.modules() if isinstance(m, (ChannelAttention, SpatialAttention, CBAM)))
    print(f"  Attention modules injected: {n_attn}  "
          f"(shallow={shallow_attn}, deep={deep_attn})")
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  Custom Transform
# ══════════════════════════════════════════════════════════════════════════════

class RemoveBottomBanner:
    def __init__(self, pixels: int = 20):
        self.pixels = pixels

    def __call__(self, img):
        w, h = img.size
        return img.crop((0, 0, w, h - self.pixels))


# ══════════════════════════════════════════════════════════════════════════════
#  Argument Parser
# ══════════════════════════════════════════════════════════════════════════════

def get_args():
    parser = argparse.ArgumentParser(description="FGVC-Aircraft Attention Trainer")
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--lr",            type=float, default=0.001)
    parser.add_argument("--epochs",        type=int,   default=30)
    parser.add_argument("--optimizer",     type=str,   default="adam",
                        choices=["adam", "sgd"])
    parser.add_argument("--model",         type=str,   default="resnet18",
                        choices=["resnet18", "resnet34", "simplecnn"])
    parser.add_argument("--label",         type=str,   default="family",
                        choices=["variant", "family", "manufacturer"])
    parser.add_argument("--crop_bbox",     action="store_true")
    parser.add_argument("--original_only", action="store_true")
    parser.add_argument("--save_csv",      type=str,   default="results_attention.csv")
    parser.add_argument("--weight_decay",  type=float, default=1e-4)

    # ── 실험 핵심 인자 ────────────────────────────────────────────────────────
    attn_choices = list(ATTN_REGISTRY.keys())   # ['none', 'channel', 'spatial', 'cbam']
    parser.add_argument("--shallow_attn", type=str, default="none",
                        choices=attn_choices,
                        help=f"layer1+layer2에 삽입할 attention 종류 {attn_choices}")
    parser.add_argument("--deep_attn",    type=str, default="cbam",
                        choices=attn_choices,
                        help=f"layer3+layer4에 삽입할 attention 종류 {attn_choices}")

    # ── GradCAM ───────────────────────────────────────────────────────────────
    layer_choices = ["layer1", "layer2", "layer3", "layer4"]
    parser.add_argument("--gradcam",       action=argparse.BooleanOptionalAction, default=True,
                        help="테스트 후 GradCAM 시각화 저장 (--no-gradcam으로 비활성화)")
    parser.add_argument("--gradcam_n",     type=int, default=16,
                        help="시각화할 테스트 샘플 수 (default: 16)")
    parser.add_argument("--gradcam_layer", type=str, default="layer4",
                        choices=layer_choices,
                        help=f"GradCAM target layer {layer_choices} (default: layer4)")
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  Train / Eval
# ══════════════════════════════════════════════════════════════════════════════

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
        total   += labels.size(0)
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
            total   += labels.size(0)
            if return_preds:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / total
    acc = correct / total
    if return_preds:
        return avg_loss, acc, all_preds, all_labels
    return avg_loss, acc


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


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = get_args()

    print("===== Input Arguments =====")
    for k, v in vars(args).items():
        print(f"  {k:<14}: {v}")
    print("===========================")

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 2 if os.name == "nt" else 4
    print(f"Using device: {device}")

    # ── Transform ────────────────────────────────────────────────────────────
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

    # ── Dataset ──────────────────────────────────────────────────────────────
    use_custom = not args.original_only

    # dataset_utils가 6:2:2 re-split과 추가 이미지 train 고정을 처리한다.
    def make_dataset(split, transform):
        return FGVCAircraft(split=split, label=args.label,
                            transform=transform, crop_bbox=args.crop_bbox,
                            use_excluded=use_custom, use_added=use_custom)

    train_dataset = make_dataset("train", train_transform)
    val_dataset   = make_dataset("val",   eval_transform)
    test_dataset  = make_dataset("test",  eval_transform)

    # 클래스 정보는 어느 split이든 동일 (dataset_utils가 전체 기준으로 생성)
    shared_class_to_idx = train_dataset.class_to_idx
    shared_classes      = train_dataset.classes
    num_classes         = len(shared_classes)

    print(f"Label: {args.label}  |  Classes: {num_classes}  "
          f"| {'원본' if args.original_only else '커스텀'} 데이터")
    print(f"Split 6:2:2  |  Train: {len(train_dataset)}, "
          f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_dataset,  shuffle=False, **loader_kwargs)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args.model, num_classes,
                        args.shallow_attn, args.deep_attn).to(device)

    is_resnet = args.model in ("resnet18", "resnet34")

    # ResNet warmup: fc만 먼저 학습 (SimpleCNN은 scratch이므로 스킵)
    warmup_epochs = 0
    if is_resnet:
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
        warmup_epochs = 5

    def make_optimizer(params):
        if args.optimizer == "adam":
            return optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        return optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    trainable  = lambda: filter(lambda p: p.requires_grad, model.parameters())
    optimizer  = make_optimizer(trainable())
    criterion  = nn.CrossEntropyLoss()
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - warmup_epochs, 1))

    # ── 실험 태그 ─────────────────────────────────────────────────────────────
    data_tag = "original" if args.original_only else "custom"
    if is_resnet:
        tag = (f"{args.model}_s{args.shallow_attn}_d{args.deep_attn}_"
               f"{args.label}_{data_tag}_bs{args.batch_size}_lr{args.lr}_opt{args.optimizer}")
    else:
        tag = (f"{args.model}_{args.label}_{data_tag}"
               f"_bs{args.batch_size}_lr{args.lr}_opt{args.optimizer}")
    ckpt_path = f"best_{tag}.pth"

    # ── 학습 루프 ─────────────────────────────────────────────────────────────
    list_train_loss, list_val_loss, list_val_acc = [], [], []
    best_val_acc = 0.0
    train_start  = time.time()

    for epoch in range(1, args.epochs + 1):
        # Warmup 종료: backbone + attention 파라미터 unfreeze
        if epoch == warmup_epochs + 1:
            backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]
            for p in backbone_params:
                p.requires_grad = True
            optimizer.add_param_group({
                "params":       backbone_params,
                "lr":           args.lr * 0.1,
                "weight_decay": args.weight_decay,
            })
            print(f"[Epoch {epoch}] Backbone + attention unfreeze "
                  f"(backbone lr={args.lr * 0.1:.0e})")

        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        epoch_secs = time.time() - epoch_start

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
            torch.save(model.state_dict(), ckpt_path)

    total_secs = time.time() - train_start
    print(f"\n총 학습 시간: {total_secs // 60:.0f}m {total_secs % 60:.1f}s")

    # ── 테스트 ────────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    test_loss, test_acc, preds, gt = evaluate(
        model, test_loader, criterion, device, return_preds=True)
    cn_acc = class_normalized_accuracy(preds, gt, num_classes)

    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | "
          f"Class-Norm Acc: {cn_acc:.4f}")

    # ── History CSV ───────────────────────────────────────────────────────────
    history_df = pd.DataFrame({
        "epoch":      range(1, args.epochs + 1),
        "train_loss": list_train_loss,
        "val_loss":   list_val_loss,
        "val_acc":    list_val_acc,
    })
    history_df.to_csv(f"history_{tag}.csv", index=False)
    print(f"History saved → history_{tag}.csv")

    # ── 학습 곡선 ─────────────────────────────────────────────────────────────
    epochs_range = range(1, args.epochs + 1)
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs_range, list_train_loss, label="Train Loss")
    ax1.plot(epochs_range, list_val_loss,   label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title(f"Loss  [shallow={args.shallow_attn}, deep={args.deep_attn}]")
    ax1.legend()
    ax2.plot(epochs_range, list_val_acc, label="Val Accuracy")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy"); ax2.legend()
    plt.tight_layout()
    plt.savefig(f"curve_{tag}.png")
    plt.close()
    print(f"Learning curve saved → curve_{tag}.png")

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    cm      = confusion_matrix(gt, preds, labels=list(range(num_classes)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    fig_size   = max(12, num_classes // 3)
    show_labels = num_classes <= 70
    tick_labels = shared_classes if show_labels else False
    _, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(cm_norm, ax=ax, cmap="Blues",
                xticklabels=tick_labels, yticklabels=tick_labels, vmin=0, vmax=1)
    if show_labels:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=6)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=6)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix  shallow={args.shallow_attn}, deep={args.deep_attn}")
    plt.tight_layout()
    plt.savefig(f"confusion_{tag}.png", dpi=150)
    plt.close()
    print(f"Confusion matrix saved → confusion_{tag}.png")

    # ── GradCAM (ResNet 계열만 지원) ─────────────────────────────────────────
    if args.gradcam and is_resnet:
        target_layer = get_gradcam_layer(model, args.gradcam_layer)
        gradcam = GradCAM(model, target_layer)
        save_gradcam_grid(
            model, test_loader, gradcam,
            class_names=shared_classes,
            n=args.gradcam_n,
            save_path=f"gradcam_{args.gradcam_layer}_{tag}.png",
            device=device,
        )
        gradcam.remove_hooks()

    # ── 요약 결과 CSV (실험 간 누적) ──────────────────────────────────────────
    row = {
        "model":        args.model,
        "shallow_attn": args.shallow_attn,
        "deep_attn":    args.deep_attn,
        "label":        args.label,
        "data":         data_tag,
        "batch_size":   args.batch_size,
        "lr":           args.lr,
        "epochs":       args.epochs,
        "optimizer":    args.optimizer,
        "best_val_acc": best_val_acc,
        "test_acc":     test_acc,
        "cn_acc":       cn_acc,
    }
    df = pd.DataFrame([row])
    file_exists = os.path.isfile(args.save_csv)
    df.to_csv(args.save_csv, mode="a", header=not file_exists, index=False)
    print(f"Results appended → {args.save_csv}")


if __name__ == "__main__":
    main()

