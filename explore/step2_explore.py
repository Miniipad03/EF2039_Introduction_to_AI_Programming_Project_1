"""
Step 2 — Dataset Exploration 스크립트
실행: python explore/step2_explore.py  (프로젝트 루트에서)

출력:
  explore/outputs/class_distribution.png
  explore/outputs/sample_images.png
"""

import os
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

os.makedirs("explore/outputs", exist_ok=True)

DATA_ROOT = Path("data/fgvc-aircraft-2013b/data")
IMAGES_DIR = DATA_ROOT / "images"


def read_annotation(filepath):
    """annotation 파일 → [(image_id, label), ...]"""
    samples = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            samples.append((parts[0], parts[1]))
    return samples


# ── 1. 클래스별 샘플 수 분포 (manufacturer) ────────────────────────────────────
print("=" * 50)
print("1. 클래스별 샘플 수 분포 (manufacturer, train)")
print("=" * 50)

train_samples = read_annotation(DATA_ROOT / "images_manufacturer_train.txt")
label_counts = Counter(label for _, label in train_samples)
labels_sorted = sorted(label_counts.keys())
counts = [label_counts[l] for l in labels_sorted]

max_label = max(label_counts, key=label_counts.get)
min_label = min(label_counts, key=label_counts.get)
print(f"가장 많은 클래스: '{max_label}' ({label_counts[max_label]}개)")
print(f"가장 적은 클래스: '{min_label}' ({label_counts[min_label]}개)")
print(f"최대/최소 비율: {label_counts[max_label] / label_counts[min_label]:.1f}x")

mean_count = sum(counts) / len(counts)
std_count = (sum((c - mean_count) ** 2 for c in counts) / len(counts)) ** 0.5
print(f"평균: {mean_count:.1f}개, 표준편차: {std_count:.1f}")
if std_count / mean_count > 0.3:
    print("→ 클래스 불균형 있음 (CV > 0.3)")
else:
    print("→ 클래스 분포 비교적 균등")

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(range(len(labels_sorted)), counts, color="steelblue", edgecolor="white")
ax.set_xticks(range(len(labels_sorted)))
ax.set_xticklabels(labels_sorted, rotation=60, ha="right", fontsize=8)
ax.set_xlabel("Manufacturer")
ax.set_ylabel("Number of Training Images")
ax.set_title("Class Distribution (Manufacturer, Train Split)")
ax.axhline(mean_count, color="red", linestyle="--", linewidth=1.2, label=f"Mean ({mean_count:.0f})")
ax.legend()
plt.tight_layout()
plt.savefig("explore/outputs/class_distribution.png", dpi=150)
plt.close()
print("→ 저장: explore/outputs/class_distribution.png")


# ── 2. 클래스별 예시 이미지 시각화 ────────────────────────────────────────────
print("\n" + "=" * 50)
print("2. 클래스별 예시 이미지 시각화 (상위 6개 클래스, 각 3장)")
print("=" * 50)

top6 = [label for label, _ in label_counts.most_common(6)]
print(f"상위 6개 클래스: {top6}")

# 클래스별 image_id 리스트 구성
class_to_ids = {}
for img_id, label in train_samples:
    class_to_ids.setdefault(label, []).append(img_id)

fig, axes = plt.subplots(6, 3, figsize=(9, 18))
fig.suptitle("Sample Images by Manufacturer (Top 6 Classes, 3 each)", fontsize=13)

for row, cls in enumerate(top6):
    ids = class_to_ids[cls]
    sampled = random.sample(ids, min(3, len(ids)))
    for col, img_id in enumerate(sampled):
        img_path = IMAGES_DIR / f"{img_id}.jpg"
        ax = axes[row][col]
        try:
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
        ax.set_title(f"{cls}\n{img_id}", fontsize=7)
        ax.axis("off")

plt.tight_layout()
plt.savefig("explore/outputs/sample_images.png", dpi=120)
plt.close()
print("→ 저장: explore/outputs/sample_images.png")


# ── 3. 이미지 크기 분포 ────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("3. 이미지 크기 분포 (train set 전체)")
print("=" * 50)

train_ids = [img_id for img_id, _ in train_samples]
widths, heights = [], []

for img_id in train_ids:
    img_path = IMAGES_DIR / f"{img_id}.jpg"
    try:
        with Image.open(img_path) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)
    except Exception:
        pass

print(f"처리된 이미지 수: {len(widths)}")
print(f"Width  — min: {min(widths)}, max: {max(widths)}, mean: {sum(widths)/len(widths):.1f}")
print(f"Height — min: {min(heights)}, max: {max(heights)}, mean: {sum(heights)/len(heights):.1f}")

# 정사각형 비율 확인
square = sum(1 for w, h in zip(widths, heights) if w == h)
landscape = sum(1 for w, h in zip(widths, heights) if w > h)
portrait = sum(1 for w, h in zip(widths, heights) if w < h)
print(f"가로형(landscape): {landscape}, 세로형(portrait): {portrait}, 정사각형: {square}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(widths, bins=40, color="steelblue", edgecolor="white")
axes[0].set_title("Image Width Distribution (Train)")
axes[0].set_xlabel("Width (px)")
axes[0].set_ylabel("Count")

axes[1].hist(heights, bins=40, color="darkorange", edgecolor="white")
axes[1].set_title("Image Height Distribution (Train)")
axes[1].set_xlabel("Height (px)")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("explore/outputs/image_size_distribution.png", dpi=150)
plt.close()
print("→ 저장: explore/outputs/image_size_distribution.png")

print("\nDone.")
