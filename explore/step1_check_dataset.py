"""
Step 1 — Dataset 확인 스크립트
실행: python explore/step1_check_dataset.py  (프로젝트 루트에서)
"""

import os
from pathlib import Path

DATA_ROOT = Path("data/fgvc-aircraft-2013b/data")
IMAGES_DIR = DATA_ROOT / "images"


def count_lines(filepath):
    with open(filepath, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def print_section(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print('='*50)


# ── 전체 이미지 수 ─────────────────────────────────────────────────────────────
print_section("전체 이미지 수")
images = list(IMAGES_DIR.glob("*.jpg"))
print(f"총 이미지 수: {len(images)}")


# ── 클래스 계층별 클래스 수 ────────────────────────────────────────────────────
print_section("클래스 계층별 클래스 수")
class_files = {
    "variant": "variants.txt",
    "family": "families.txt",
    "manufacturer": "manufacturers.txt",
}
for level, filename in class_files.items():
    txt = DATA_ROOT / filename
    n = count_lines(txt)
    print(f"  {level:15s}: {n:4d} classes")


# ── split별 이미지 수 ──────────────────────────────────────────────────────────
print_section("Split별 이미지 수 (manufacturer 기준)")
for split in ("train", "val", "test"):
    ann = DATA_ROOT / f"images_manufacturer_{split}.txt"
    n = count_lines(ann)
    print(f"  {split:6s}: {n:5d} images")

print_section("Split별 이미지 수 (variant 기준)")
for split in ("train", "val", "test"):
    ann = DATA_ROOT / f"images_variant_{split}.txt"
    n = count_lines(ann)
    print(f"  {split:6s}: {n:5d} images")

print_section("Split별 이미지 수 (family 기준)")
for split in ("train", "val", "test"):
    ann = DATA_ROOT / f"images_family_{split}.txt"
    n = count_lines(ann)
    print(f"  {split:6s}: {n:5d} images")


# ── annotation 파일 포맷 예시 ─────────────────────────────────────────────────
print_section("Annotation 파일 포맷 예시 (images_manufacturer_train.txt, 첫 5줄)")
with open(DATA_ROOT / "images_manufacturer_train.txt", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        parts = line.strip().split(" ", 1)
        image_id, label = parts[0], parts[1]
        print(f"  image_id={image_id!r:12s}  label={label!r}")

print_section("Annotation 파일 포맷 예시 (images_variant_train.txt, 첫 5줄)")
with open(DATA_ROOT / "images_variant_train.txt", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        parts = line.strip().split(" ", 1)
        image_id, label = parts[0], parts[1]
        print(f"  image_id={image_id!r:12s}  label={label!r}")

print_section("Manufacturer 클래스 목록")
with open(DATA_ROOT / "manufacturers.txt", encoding="utf-8") as f:
    manufacturers = [line.strip() for line in f if line.strip()]
for i, m in enumerate(manufacturers):
    print(f"  {i+1:2d}. {m}")

print("\nDone.")
