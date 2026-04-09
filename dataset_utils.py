"""
FGVC-Aircraft Dataset Utility
==============================
라벨링 툴의 excluded.json / added_images.json을 반영한 데이터셋 로더.

사용 예시:
    from dataset_utils import FGVCAircraft

    ds = FGVCAircraft(split='train', label='variant')
    print(len(ds))          # 유효 이미지 수 (제거된 이미지 제외)
    img, label = ds[0]      # PIL Image, 클래스 인덱스

    # PyTorch DataLoader와 함께 사용
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4)
"""

import json
import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent
DATA_ROOT      = PROJECT_ROOT / 'data' / 'fgvc-aircraft-2013b' / 'data'
IMAGES_DIR     = DATA_ROOT / 'images'
TOOL_DATA      = PROJECT_ROOT / 'labeling-tool' / 'data'
USER_IMAGES    = TOOL_DATA / 'user_images'
EXCLUDED_DIR   = TOOL_DATA / 'excluded'
ADDED_DIR      = TOOL_DATA / 'added_images'


def _read_annotation(filename):
    """'image_id label name' 형식 파일 → {image_id: label} dict"""
    result = {}
    with open(DATA_ROOT / filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            space = line.index(' ')
            result[line[:space]] = line[space + 1:]
    return result


def _read_bbox():
    """images_box.txt → {image_id: (xmin, ymin, xmax, ymax)}"""
    bboxes = {}
    with open(DATA_ROOT / 'images_box.txt', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                bboxes[parts[0]] = tuple(int(x) for x in parts[1:5])
    return bboxes


class FGVCAircraft(Dataset):
    """
    Parameters
    ----------
    split   : 'train' | 'val' | 'test' | 'all'
    label   : 'variant' | 'family' | 'manufacturer'
    transform : torchvision transform (optional)
    crop_bbox : True면 bbox 영역만 crop해서 반환
    """

    def __init__(self, split='train', label='variant', transform=None, crop_bbox=False,
                 use_excluded=True, use_added=True):
        """
        Parameters
        ----------
        use_excluded : bool
            False면 excluded.json 무시 (원본 데이터 그대로 사용)
        use_added    : bool
            False면 added_images.json 무시 (원본 데이터 그대로 사용)
        """
        assert split in ('train', 'val', 'test', 'all')
        assert label in ('variant', 'family', 'manufacturer')

        self.transform  = transform
        self.crop_bbox  = crop_bbox

        # 제거 목록 로드 (excluded/ 디렉토리, ID별 파일)
        excluded_set = set()
        if use_excluded and EXCLUDED_DIR.exists():
            for f in EXCLUDED_DIR.glob('*.json'):
                excluded_set.add(json.loads(f.read_text())['id'])

        # 원본 annotation 로드
        splits = ['train', 'val', 'test'] if split == 'all' else [split]
        samples = []
        bboxes  = _read_bbox()

        for sp in splits:
            labels = _read_annotation(f'images_{label}_{sp}.txt')
            for img_id, lbl in labels.items():
                if img_id in excluded_set:
                    continue
                img_path = IMAGES_DIR / f'{img_id}.jpg'
                if not img_path.exists():
                    continue
                samples.append({
                    'id':    img_id,
                    'path':  img_path,
                    'label': lbl,
                    'split': sp,
                    'bbox':  bboxes.get(img_id),
                })

        # 추가 이미지 로드 (added_images/ 디렉토리, ID별 파일)
        if use_added and ADDED_DIR.exists():
            added_data = [json.loads(f.read_text()) for f in ADDED_DIR.glob('*.json')]
            for item in added_data:
                if item['id'] in excluded_set:
                    continue
                if split != 'all' and item['split'] != split:
                    continue
                img_path = USER_IMAGES / f'{item["id"]}.jpg'
                if not img_path.exists():
                    continue
                lbl = item.get(label) or item.get('variant')
                bbox = item.get('bbox')
                samples.append({
                    'id':    item['id'],
                    'path':  img_path,
                    'label': lbl,
                    'split': item['split'],
                    'bbox':  (bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']) if bbox else None,
                })

        # 클래스 인덱스 생성 (정렬된 순서로 고정)
        classes = sorted(set(s['label'] for s in samples))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.classes      = classes
        self.samples      = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = Image.open(s['path']).convert('RGB')

        if self.crop_bbox and s['bbox']:
            img = img.crop(s['bbox'])

        if self.transform:
            img = self.transform(img)

        return img, self.class_to_idx[s['label']]

    # ── 편의 메서드 ──────────────────────────────────────────────────────────

    def summary(self):
        """split별 클래스별 샘플 수 출력"""
        from collections import Counter
        split_counts = Counter(s['split'] for s in self.samples)
        print(f"총 {len(self.samples)}개 이미지 | "
              f"train: {split_counts['train']}, "
              f"val: {split_counts['val']}, "
              f"test: {split_counts['test']}")
        print(f"클래스 수: {len(self.classes)}")


# ── 빠른 검증 ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    for split in ('train', 'val', 'test'):
        ds = FGVCAircraft(split=split, label='variant')
        ds.summary()
    print("\n클래스 예시 (처음 5개):", FGVCAircraft('train').classes[:5])
