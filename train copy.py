"""
train.py — FGVC-Aircraft K-Fold Trainer
4-Fold CV + Weighted Soft Voting Ensemble

실험 예시:
  Vanilla CNN:        python train.py --model cnn
  Vanilla ResNet-34:  python train.py --model resnet34
  Tuning  ResNet-34:  python train.py --model resnet34 --tuning
  Tuning  +CBAM:      python train.py --model resnet34 --attn cbam --tuning
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
import seaborn as sns

from dataset_utils import FGVCAircraft, IMAGES_DIR, USER_IMAGES, SampleDataset
from models import build_model, build_model_dropout


# ══════════════════════════════════════════════════════════════════════════════
#  Custom Transform
# ══════════════════════════════════════════════════════════════════════════════

class ResizeWithPad:
    """비율을 유지하며 size×size로 리사이즈 후 남는 영역을 0으로 패딩."""
    def __init__(self, size=224):
        self.size = size

    def __call__(self, img):
        w, h   = img.size
        ratio  = self.size / max(w, h)
        new_w  = int(w * ratio)
        new_h  = int(h * ratio)
        img    = transforms.functional.resize(img, (new_h, new_w))
        pad_w  = self.size - new_w
        pad_h  = self.size - new_h
        img    = transforms.functional.pad(
            img,
            (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2),
        )
        return img


# ══════════════════════════════════════════════════════════════════════════════
#  Argument Parser
# ══════════════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(description='FGVC-Aircraft K-Fold Trainer')
    p.add_argument('--model',        type=str,   default='resnet34',
                   choices=['cnn', 'resnet18', 'resnet34', 'resnet34d'])
    p.add_argument('--attn',         type=str,   default='none',
                   choices=['none', 'channel', 'spatial', 'cbam'],
                   help='Attention type : channel / spatial / cbam on layer3+layer4.')
    p.add_argument('--tuning',       action='store_true',
                   help='Shortcut: enable both --use_excluded and --use_added')
    p.add_argument('--use_excluded', action='store_true',
                   help='Remove excluded images from dataset')
    p.add_argument('--use_added',    action='store_true',
                   help='Add custom images to train set')
    p.add_argument('--folds',        type=int,   default=4)
    p.add_argument('--epochs',       type=int,   default=50)
    p.add_argument('--batch_size',   type=int,   default=32)
    p.add_argument('--lr',           type=float, default=0.001)
    p.add_argument('--optimizer',       type=str,   default='adamw', choices=['adam', 'adamw', 'sgd'])
    p.add_argument('--weight_decay',    type=float, default=5e-3)
    p.add_argument('--label_smoothing', type=float, default=0.0)
    p.add_argument('--label',        type=str,   default='family',
                   choices=['variant', 'family', 'manufacturer'])
    p.add_argument('--crop_bbox',    action='store_true')
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--save_csv',     type=str,   default='results.csv')
    p.add_argument('--outdir',       type=str,   default='.',
                   help='Output directory for all files')
    p.add_argument('--added_repeat', type=int,   default=1,
                   help='추가 이미지 반복 배수 (use_added=True일 때만 적용)')
    p.add_argument('--rotation',     type=int,   default=0,
                   help='RandomRotation 최대 각도 (0=비활성)')
    p.add_argument('--cutmix',       type=float, default=0.0,
                   help='CutMix alpha (0=비활성, 권장=1.0)')
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  Training Utilities
# ══════════════════════════════════════════════════════════════════════════════

def cutmix_batch(images, labels, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(images.size(0))
    H, W = images.size(2), images.size(3)
    cut_w = int(W * np.sqrt(1 - lam))
    cut_h = int(H * np.sqrt(1 - lam))
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, x2 = np.clip(cx - cut_w // 2, 0, W), np.clip(cx + cut_w // 2, 0, W)
    y1, y2 = np.clip(cy - cut_h // 2, 0, H), np.clip(cy + cut_h // 2, 0, H)
    images = images.clone()
    images[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return images, labels, labels[idx], lam


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, args=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        use_cutmix = args is not None and args.cutmix > 0 and np.random.random() < 0.5
        if use_cutmix:
            images, labels_a, labels_b, lam = cutmix_batch(images, labels, args.cutmix)
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            out  = model(images)
            loss = (lam * criterion(out, labels_a) + (1 - lam) * criterion(out, labels_b)
                    if use_cutmix else criterion(out, labels))
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device, return_preds=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    use_amp = device.type == 'cuda'
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast('cuda', enabled=use_amp):
                out  = model(images)
                loss = criterion(out, labels)
            total_loss += loss.item() * images.size(0)
            preds = out.argmax(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            if return_preds:
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / total
    acc      = correct / total
    return (avg_loss, acc, all_preds, all_labels) if return_preds else (avg_loss, acc)


def class_normalized_accuracy(preds, labels, num_classes):
    preds, labels = np.array(preds), np.array(labels)
    per_class = np.zeros(num_classes)
    counts    = np.zeros(num_classes)
    for c in range(num_classes):
        mask = labels == c
        counts[c]    = mask.sum()
        per_class[c] = (preds[mask] == c).sum()
    valid = counts > 0
    return (per_class[valid] / counts[valid]).mean()


# ══════════════════════════════════════════════════════════════════════════════
#  Weighted Soft Voting Ensemble
# ══════════════════════════════════════════════════════════════════════════════

def ensemble_predict(ckpt_paths, val_accs, test_loader, num_classes, args, device):
    """각 fold 모델의 softmax를 val_acc 가중 평균해 최종 예측."""
    weights = np.array(val_accs, dtype=float)
    weights /= weights.sum()

    ensemble_probs = None
    gt = None

    for ckpt, w in zip(ckpt_paths, weights):
        model = (build_model_dropout(num_classes, args.attn)
                 if args.model == 'resnet34d'
                 else build_model(args.model, num_classes, args.attn)).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.eval()

        probs_list, labels_list = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                    logits = model(images.to(device))
                probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
                labels_list.extend(labels.numpy())

        fold_probs = np.concatenate(probs_list, axis=0)
        if ensemble_probs is None:
            ensemble_probs = w * fold_probs
            gt = np.array(labels_list)
        else:
            ensemble_probs += w * fold_probs

    return ensemble_probs.argmax(axis=1), gt


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args   = get_args()
    if args.tuning:
        args.use_excluded = True
        args.use_added    = True
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("===== Input Arguments =====")
    for k, v in vars(args).items():
        print(f"  {k:<14}: {v}")
    print("===========================")

    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 2 if os.name == 'nt' else max(os.cpu_count() - 2, 1)
    scaler      = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    print(f"Device: {device}  |  AMP: {scaler is not None}  |  Workers: {num_workers}")

    # ── Transform ─────────────────────────────────────────────────────────────
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std  = (0.229, 0.224, 0.225)

    _aug = [
        ResizeWithPad(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    ]
    if args.rotation > 0:
        _aug.append(transforms.RandomRotation(args.rotation))
    _aug += [transforms.ToTensor(), transforms.Normalize(imagenet_mean, imagenet_std)]
    train_transform = transforms.Compose(_aug)
    eval_transform = transforms.Compose([
        ResizeWithPad(224),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    # ── 데이터 로드 ───────────────────────────────────────────────────────────
    # tuning=True : excluded 제거 후 로드, added 포함 (split 이후 train에만 배치)
    # tuning=False: 원본 FGVC 전체 (수정 없음)
    ref_ds = FGVCAircraft(
        split='all', label=args.label, transform=None,
        crop_bbox=args.crop_bbox,
        use_excluded=args.use_excluded,
        use_added=args.use_added,
        resplit=False,
    )

    if args.use_added:
        original_samples = [s for s in ref_ds.samples if s['path'].parent == IMAGES_DIR]
        added_samples    = [s for s in ref_ds.samples if s['path'].parent == USER_IMAGES]
    else:
        original_samples = ref_ds.samples
        added_samples    = []

    class_to_idx = ref_ds.class_to_idx
    classes      = ref_ds.classes
    num_classes  = len(classes)

    print(f"Label: {args.label}  |  Classes: {num_classes}")
    print(f"Mode: excl={args.use_excluded}, added={args.use_added}  |  "
          f"Original: {len(original_samples)}, Added(train-only): {len(added_samples)} ×{args.added_repeat}"
          f" = {len(added_samples) * args.added_repeat}")

    # ── Test Set 고립 ─────────────────────────────────────────────────────────
    all_idx    = list(range(len(original_samples)))
    all_labels = [class_to_idx[s['label']] for s in original_samples]

    train_val_idx, test_idx, y_tv, _ = train_test_split(
        all_idx, all_labels,
        test_size=0.2, random_state=args.seed, stratify=all_labels,
    )

    test_samples = [original_samples[i] for i in test_idx]
    test_ds      = SampleDataset(test_samples, class_to_idx, eval_transform, args.crop_bbox)
    loader_kw    = dict(batch_size=args.batch_size, num_workers=num_workers,
                        pin_memory=True, persistent_workers=(num_workers > 0))
    test_loader  = DataLoader(test_ds, shuffle=False, **loader_kw)

    print(f"Test set isolated: {len(test_idx)} samples (20%, seed={args.seed})")
    print(f"Train+Val pool   : {len(train_val_idx)} samples → {args.folds}-fold CV")

    # ── K-Fold CV ─────────────────────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    is_resnet     = args.model != 'cnn'
    warmup_epochs = 5 if is_resnet else 0

    if args.use_excluded and args.use_added:
        data_tag = 'tuning'
    elif args.use_excluded:
        data_tag = 'excl'
    elif args.use_added:
        data_tag = 'added'
    else:
        data_tag = 'vanilla'
    attn_tag = args.attn if args.attn != 'none' else 'noattn'
    rot_tag = f"_rot{args.rotation}" if args.rotation > 0 else ""
    cm_tag  = f"_cm{args.cutmix}"   if args.cutmix  > 0 else ""
    exp_tag  = (f"{args.model}_{attn_tag}_{data_tag}_{args.label}"
                f"_lr{args.lr}_opt{args.optimizer}_wd{args.weight_decay}{rot_tag}{cm_tag}")

    fold_val_accs    = []
    fold_ckpt_paths  = []
    all_histories    = []
    total_train_secs = 0
    num_params       = 0

    for fold, (tr_pos, vl_pos) in enumerate(skf.split(train_val_idx, y_tv)):
        print(f"\n{'='*60}")
        print(f"  Fold {fold+1}/{args.folds}")
        print(f"{'='*60}")

        actual_train = [train_val_idx[i] for i in tr_pos]
        actual_val   = [train_val_idx[i] for i in vl_pos]

        fold_train = [original_samples[i] for i in actual_train] + added_samples * args.added_repeat
        fold_val   = [original_samples[i] for i in actual_val]

        print(f"  Train: {len(fold_train)}, Val: {len(fold_val)}, Test: {len(test_samples)}")

        train_ds = SampleDataset(fold_train, class_to_idx, train_transform, args.crop_bbox)
        val_ds   = SampleDataset(fold_val,   class_to_idx, eval_transform,  args.crop_bbox)
        train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
        val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)

        model = (build_model_dropout(num_classes, args.attn)
                 if args.model == 'resnet34d'
                 else build_model(args.model, num_classes, args.attn)).to(device)
        if num_params == 0:
            num_params = sum(p.numel() for p in model.parameters())

        if is_resnet:
            for n, p in model.named_parameters():
                if 'fc' not in n:
                    p.requires_grad = False

        def make_optimizer(params):
            if args.optimizer == 'adam':
                return optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
            if args.optimizer == 'adamw':
                return optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
            return optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

        optimizer = make_optimizer(filter(lambda p: p.requires_grad, model.parameters()))
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs - warmup_epochs, 1))

        fold_ckpt    = str(outdir / f"best_fold{fold+1}_{exp_tag}.pth")
        best_val_acc = 0.0
        fold_start   = time.time()

        for epoch in range(1, args.epochs + 1):
            if is_resnet and epoch == warmup_epochs + 1:
                backbone_params = [p for n, p in model.named_parameters() if 'fc' not in n]
                for p in backbone_params:
                    p.requires_grad = True
                optimizer.add_param_group({
                    'params': backbone_params,
                    'lr': args.lr * 0.1,
                    'weight_decay': args.weight_decay,
                })
                # Scheduler를 재생성해 backbone param group까지 Cosine decay 적용
                remaining = max(args.epochs - warmup_epochs, 1)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=remaining)
                print(f"  [Epoch {epoch}] Backbone unfreeze (lr={args.lr*0.1:.0e})")

            epoch_start = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, args)
            val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
            if epoch > warmup_epochs:
                scheduler.step()

            all_histories.append({
                'fold': fold + 1, 'epoch': epoch,
                'train_loss': train_loss, 'train_acc': train_acc,
                'val_loss': val_loss, 'val_acc': val_acc,
            })

            print(f"  [F{fold+1} E{epoch:3d}/{args.epochs}] "
                  f"Train {train_loss:.4f}/{train_acc:.4f}  "
                  f"Val {val_loss:.4f}/{val_acc:.4f}  "
                  f"({time.time()-epoch_start:.1f}s)")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), fold_ckpt)

        fold_secs         = time.time() - fold_start
        total_train_secs += fold_secs
        print(f"  Fold {fold+1} done — best val acc: {best_val_acc:.4f}  "
              f"({fold_secs//60:.0f}m {fold_secs%60:.0f}s)")
        fold_val_accs.append(best_val_acc)
        fold_ckpt_paths.append(fold_ckpt)

    # ── Weighted Soft Voting Ensemble ─────────────────────────────────────────
    weights = np.array(fold_val_accs) / sum(fold_val_accs)
    print(f"\n{'='*60}")
    print(f"  Ensemble — Weighted Soft Voting")
    print(f"  Fold val accs: {[f'{a:.4f}' for a in fold_val_accs]}")
    print(f"  Weights      : {[f'{w:.4f}' for w in weights]}")
    print(f"{'='*60}")

    preds, gt = ensemble_predict(
        fold_ckpt_paths, fold_val_accs, test_loader, num_classes, args, device)

    test_acc = float((preds == gt).mean())
    cn_acc   = float(class_normalized_accuracy(preds, gt, num_classes))
    print(f"Ensemble Test Acc : {test_acc:.4f}")
    print(f"Class-Norm Acc    : {cn_acc:.4f}")

    # ── History CSV ───────────────────────────────────────────────────────────
    hist_path = outdir / f"history_{exp_tag}.csv"
    pd.DataFrame(all_histories).to_csv(hist_path, index=False)
    print(f"History → {hist_path}")

    # ── Learning Curve (mean ± std across folds) ─────────────────────────────
    hist_df = pd.DataFrame(all_histories)
    epochs  = sorted(hist_df['epoch'].unique())

    def fold_stats(col):
        mat = np.array([hist_df[hist_df['fold'] == f + 1][col].values
                        for f in range(args.folds)])
        return mat.mean(axis=0), mat.std(axis=0)

    tl_mean, tl_std = fold_stats('train_loss')
    vl_mean, vl_std = fold_stats('val_loss')
    ta_mean, ta_std = fold_stats('train_acc')
    va_mean, va_std = fold_stats('val_acc')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for f in range(args.folds):
        fh = hist_df[hist_df['fold'] == f + 1]
        ax1.plot(fh['epoch'], fh['train_loss'], color='gray', linewidth=0.7, alpha=0.2)
        ax1.plot(fh['epoch'], fh['val_loss'],   color='gray', linewidth=0.7, alpha=0.2)
        ax2.plot(fh['epoch'], fh['train_acc'],  color='gray', linewidth=0.7, alpha=0.2)
        ax2.plot(fh['epoch'], fh['val_acc'],    color='gray', linewidth=0.7, alpha=0.2)

    ax1.plot(epochs, tl_mean, '--', color='tab:blue',   linewidth=2, label='Train')
    ax1.fill_between(epochs, tl_mean - tl_std, tl_mean + tl_std, color='tab:blue',   alpha=0.15)
    ax1.plot(epochs, vl_mean, '-',  color='tab:orange', linewidth=2, label='Val')
    ax1.fill_between(epochs, vl_mean - vl_std, vl_mean + vl_std, color='tab:orange', alpha=0.15)

    ax2.plot(epochs, ta_mean, '--', color='tab:blue',   linewidth=2, label='Train')
    ax2.fill_between(epochs, ta_mean - ta_std, ta_mean + ta_std, color='tab:blue',   alpha=0.15)
    ax2.plot(epochs, va_mean, '-',  color='tab:orange', linewidth=2,
             label=f'Val (mean={float(np.mean(fold_val_accs)):.3f})')
    ax2.fill_between(epochs, va_mean - va_std, va_mean + va_std, color='tab:orange', alpha=0.15)

    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Loss (mean ± std)'); ax1.legend(fontsize=9)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy (mean ± std)'); ax2.legend(fontsize=9)
    plt.suptitle(f"{exp_tag}\nEnsemble Test={test_acc:.4f}  CN={cn_acc:.4f}", fontsize=9)
    plt.tight_layout()
    plt.savefig(outdir / f"curve_{exp_tag}.png", dpi=150)
    plt.close()
    print(f"Curve → {outdir / f'curve_{exp_tag}.png'}")

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    cm      = confusion_matrix(gt, preds, labels=list(range(num_classes)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    show_labels = num_classes <= 70
    tick_labels = classes if show_labels else False
    _, ax = plt.subplots(figsize=(max(12, num_classes // 3),) * 2)
    sns.heatmap(cm_norm, ax=ax, cmap='Blues',
                xticklabels=tick_labels, yticklabels=tick_labels, vmin=0, vmax=1)
    if show_labels:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=6)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=6)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix (Ensemble)\n{exp_tag}')
    plt.tight_layout()
    plt.savefig(outdir / f"confusion_{exp_tag}.png", dpi=150)
    plt.close()
    print(f"Confusion → {outdir / f'confusion_{exp_tag}.png'}")

    # ── Class-wise Accuracy CSV ───────────────────────────────────────────────
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    cw_rows = []
    for c in range(num_classes):
        mask = gt == c
        if mask.sum() == 0:
            continue
        cw_rows.append({
            'class':     idx_to_class[c],
            'n_samples': int(mask.sum()),
            'accuracy':  float((preds[mask] == c).sum() / mask.sum()),
        })
    cw_path = outdir / f"classwise_{exp_tag}.csv"
    pd.DataFrame(cw_rows).sort_values('accuracy').to_csv(cw_path, index=False)
    print(f"Class-wise → {cw_path}")

    # ── Predictions CSV ───────────────────────────────────────────────────────
    pred_rows = [{
        'image_id':   s['path'].stem,
        'true_label': idx_to_class[int(gt[i])],
        'pred_label': idx_to_class[int(preds[i])],
        'correct':    bool(preds[i] == gt[i]),
    } for i, s in enumerate(test_samples)]
    pred_path = outdir / f"predictions_{exp_tag}.csv"
    pd.DataFrame(pred_rows).to_csv(pred_path, index=False)
    print(f"Predictions → {pred_path}")

    # ── Summary CSV ───────────────────────────────────────────────────────────
    row = {
        'model': args.model, 'attn': args.attn, 'tuning': args.tuning,
        'use_excluded': args.use_excluded, 'use_added': args.use_added,
        'crop_bbox': args.crop_bbox, 'added_repeat': args.added_repeat,
        'label': args.label, 'folds': args.folds, 'epochs': args.epochs,
        'batch_size': args.batch_size, 'lr': args.lr, 'optimizer': args.optimizer,
        'weight_decay': args.weight_decay, 'label_smoothing': args.label_smoothing,
        'rotation': args.rotation, 'cutmix': args.cutmix,
        'num_params': num_params,
        'train_time_sec': round(total_train_secs),
        **{f'fold{i+1}_val_acc': fold_val_accs[i] for i in range(args.folds)},
        'mean_val_acc': float(np.mean(fold_val_accs)),
        'ensemble_test_acc': test_acc,
        'ensemble_cn_acc':   cn_acc,
    }
    csv_path   = outdir / args.save_csv
    file_exists = csv_path.exists()
    pd.DataFrame([row]).to_csv(csv_path, mode='a', header=not file_exists, index=False)
    print(f"Results → {csv_path}")


if __name__ == '__main__':
    main()

