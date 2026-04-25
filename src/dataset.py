"""
src/dataset.py — AircraftDataset (PyTorch Dataset)

사용 예시:
    from src.dataset import AircraftDataset, get_transforms

    train_ds = AircraftDataset(
        annotation_file="data/fgvc-aircraft-2013b/data/images_manufacturer_train.txt",
        image_dir="data/fgvc-aircraft-2013b/data/images",
        transform=get_transforms("train"),
    )
    val_ds = AircraftDataset(..., transform=get_transforms("val"))
"""

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class AircraftDataset(Dataset):
    """
    FGVC-Aircraft annotation 파일 기반 Dataset.

    Parameters
    ----------
    annotation_file : str | Path
        '{image_id} {label}' 형식의 annotation 파일 경로
    image_dir : str | Path
        이미지 디렉토리 경로 (*.jpg 파일들이 위치)
    transform : callable, optional
        torchvision transforms
    """

    def __init__(self, annotation_file, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform

        # annotation 파일 파싱 → [(image_id, label), ...]
        self.samples = []
        with open(annotation_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                image_id, label = parts[0], parts[1]
                self.samples.append((image_id, label))

        # label → index 변환 딕셔너리 (알파벳 정렬로 고정)
        all_labels = sorted(set(label for _, label in self.samples))
        self.label2idx = {label: idx for idx, label in enumerate(all_labels)}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}
        self.classes = all_labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_id, label = self.samples[idx]
        img_path = self.image_dir / f"{image_id}.jpg"

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, self.label2idx[label]

    def num_classes(self):
        return len(self.classes)


def get_transforms(mode: str) -> transforms.Compose:
    """
    mode: 'train' | 'val' | 'test'
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if mode == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])
