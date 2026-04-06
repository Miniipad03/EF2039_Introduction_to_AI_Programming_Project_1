"""
FGVC-Aircraft 데이터셋 다운로드 및 확인 스크립트
Usage: python download_dataset.py --data_dir ./data
"""

import argparse
import os
import torchvision.datasets as datasets


def download_dataset(data_dir: str, annotation_level: str = "family"):
    print(f"Downloading FGVC-Aircraft to '{data_dir}' ...")

    # transform 없이 원본 그대로 로드 (전처리는 별도 스크립트에서)
    train_dataset = datasets.FGVCAircraft(
        root=data_dir,
        split="train",
        annotation_level=annotation_level,
        download=True,
    )
    val_dataset = datasets.FGVCAircraft(
        root=data_dir,
        split="val",
        annotation_level=annotation_level,
        download=True,
    )
    test_dataset = datasets.FGVCAircraft(
        root=data_dir,
        split="test",
        annotation_level=annotation_level,
        download=True,
    )

    return train_dataset, val_dataset, test_dataset


def print_summary(train_dataset, val_dataset, test_dataset):
    print("\n=== Dataset Summary ===")
    print(f"Annotation level : {train_dataset._annotation_level}")
    print(f"Classes          : {len(train_dataset.classes)}")
    print(f"Train samples    : {len(train_dataset)}")
    print(f"Val   samples    : {len(val_dataset)}")
    print(f"Test  samples    : {len(test_dataset)}")
    print(f"Total            : {len(train_dataset) + len(val_dataset) + len(test_dataset)}")
    print("\nFirst 5 classes:", train_dataset.classes[:5])

    # 샘플 이미지 shape 확인
    img, label = train_dataset[0]
    print(f"\nSample image shape : {img.shape}")
    print(f"Sample label       : {train_dataset.classes[label]}")


def main():
    parser = argparse.ArgumentParser(description="Download FGVC-Aircraft dataset")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory to download dataset into")
    parser.add_argument("--annotation_level", type=str, default="family",
                        choices=["variant", "family", "manufacturer"],
                        help="Classification hierarchy level")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    train_dataset, val_dataset, test_dataset = download_dataset(
        args.data_dir, args.annotation_level
    )
    print_summary(train_dataset, val_dataset, test_dataset)

    print("\nDownload complete.")


if __name__ == "__main__":
    main()
