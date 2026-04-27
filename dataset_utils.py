"""
FGVC-Aircraft Dataset Utility
==============================
라벨링 툴의 excluded.json / added_images.json을 반영한 데이터셋 로더.

split 정책 (resplit=True 기본값):
  1. 추가 이미지(added_images)를 먼저 train에 고정 배정
  2. 전체(원본 + 추가) 기준 6:2:2가 되도록 원본 FGVC 데이터만 stratified re-split
     - val, test는 원본 데이터에서만 구성
     - 원본 내 val+test 비율 = 0.4 × N_total / N_orig

사용 예시:
    from dataset_utils import FGVCAircraft

    train_ds = FGVCAircraft(split='train', label='family')
    val_ds   = FGVCAircraft(split='val',   label='family')
    test_ds  = FGVCAircraft(split='test',  label='family')

    from torch.utils.data import DataLoader
    loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
"""

import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT    = PROJECT_ROOT / 'data' / 'fgvc-aircraft-2013b' / 'data'
IMAGES_DIR   = DATA_ROOT / 'images'
TOOL_DATA    = PROJECT_ROOT / 'labeling-tool' / 'data'
USER_IMAGES  = TOOL_DATA / 'user_images'
EXCLUDED_DIR = TOOL_DATA / 'excluded'
ADDED_DIR    = TOOL_DATA / 'added_images'


def _read_annotation(filename):
    """'image_id label' 형식 파일 → {image_id: label} dict"""
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
    split        : 'train' | 'val' | 'test' | 'all'
    label        : 'variant' | 'family' | 'manufacturer'
    transform    : torchvision transform (optional)
    crop_bbox    : True면 bbox 영역만 crop해서 반환
    use_excluded : False면 excluded 목록 무시
    use_added    : False면 added_images 무시
    resplit      : True(기본)면 6:2:2 stratified re-split 수행
                   False면 원본 annotation 파일의 split 그대로 사용
    random_state : re-split 재현성 시드 (기본 42)
    """

    def __init__(self, split='train', label='variant', transform=None, crop_bbox=False,
                 use_excluded=True, use_added=True, resplit=True, random_state=42):
        assert split in ('train', 'val', 'test', 'all')
        assert label in ('variant', 'family', 'manufacturer')

        self.transform = transform
        self.crop_bbox = crop_bbox

        # ── 1. 제거 목록 로드 ────────────────────────────────────────────────
        excluded_set = set()
        if use_excluded and EXCLUDED_DIR.exists():
            for f in sorted(EXCLUDED_DIR.glob('*.json')):
                excluded_set.add(json.loads(f.read_text())['id'])

        # ── 2. 추가 이미지 먼저 로드 → train 고정 ────────────────────────────
        #    비율 계산에 N_added가 필요하므로 원본보다 먼저 처리한다.
        added_samples = []
        if use_added and ADDED_DIR.exists():
            for f in sorted(ADDED_DIR.glob('*.json')):
                item = json.loads(f.read_text())
                if item['id'] in excluded_set:
                    continue
                img_path = USER_IMAGES / f'{item["id"]}.jpg'
                if not img_path.exists():
                    continue
                lbl  = item.get(label) or item.get('variant')
                bbox = item.get('bbox')
                added_samples.append({
                    'id':    item['id'],
                    'path':  img_path,
                    'label': lbl,
                    'split': 'train',   # 항상 train 고정
                    'bbox':  (bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax'])
                             if bbox else None,
                })

        # ── 3. 원본 FGVC 샘플 전체 로드 (세 split 파일 모두) ─────────────────
        original_samples = []
        bboxes = _read_bbox()
        for sp in ('train', 'val', 'test'):
            ann = _read_annotation(f'images_{label}_{sp}.txt')
            for img_id, lbl in ann.items():
                if img_id in excluded_set:
                    continue
                img_path = IMAGES_DIR / f'{img_id}.jpg'
                if not img_path.exists():
                    continue
                original_samples.append({
                    'id':    img_id,
                    'path':  img_path,
                    'label': lbl,
                    'split': sp,    # 원본 태그 (resplit=False 시 그대로 사용)
                    'bbox':  bboxes.get(img_id),
                })

        # ── 4. 클래스 인덱스 (전체 샘플 기준으로 split과 무관하게 일관성 유지) ─
        all_for_classes = added_samples + original_samples
        classes = sorted(set(s['label'] for s in all_for_classes))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.classes      = classes

        # ── 5. 6:2:2 Stratified Re-split ──────────────────────────────────────
        #    추가 이미지는 이미 train 고정이므로 원본 샘플에만 적용.
        #    val, test는 원본에서만 구성되므로, 전체 기준 6:2:2를 맞추려면
        #    원본 내 val+test 비율 = 0.4 × N_total / N_orig 로 계산해야 한다.
        if resplit:
            from sklearn.model_selection import train_test_split

            n_orig  = len(original_samples)
            n_added = len(added_samples)
            n_total = n_orig + n_added

            # 원본 샘플 안에서 val+test가 차지해야 할 비율
            # n_added가 클 경우 1.0 초과 방지 (train_test_split 오류 예방)
            val_test_ratio = min((0.4 * n_total) / n_orig, 0.90)

            indices     = list(range(n_orig))
            lbl_indices = [self.class_to_idx[s['label']] for s in original_samples]

            # 1차: train / (val+test)
            train_idx, tmp_idx, _, y_tmp = train_test_split(
                indices, lbl_indices,
                test_size=val_test_ratio, random_state=random_state,
                stratify=lbl_indices)
            # 2차: val / test (50:50)
            val_idx, test_idx = train_test_split(
                tmp_idx, test_size=0.5, random_state=random_state, stratify=y_tmp)

            for i in train_idx: original_samples[i]['split'] = 'train'
            for i in val_idx:   original_samples[i]['split'] = 'val'
            for i in test_idx:  original_samples[i]['split'] = 'test'

        # ── 6. 요청 split으로 필터링 ─────────────────────────────────────────
        all_samples = added_samples + original_samples
        self.samples = all_samples if split == 'all' else [
            s for s in all_samples if s['split'] == split
        ]

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
        """split별 샘플 수 및 클래스 수 출력"""
        from collections import Counter
        cnt   = Counter(s['split'] for s in self.samples)
        total = len(self.samples)
        train_r = cnt['train'] / total if total else 0
        val_r   = cnt['val']   / total if total else 0
        test_r  = cnt['test']  / total if total else 0
        print(f"총 {total}개 이미지  |  "
              f"train: {cnt['train']} ({train_r:.1%}), "
              f"val: {cnt['val']} ({val_r:.1%}), "
              f"test: {cnt['test']} ({test_r:.1%})")
        print(f"클래스 수: {len(self.classes)}")


# ── K-Fold 학습용 Dataset 래퍼 ───────────────────────────────────────────────

class RemoveBottomBanner:
    """FGVC 원본 이미지 하단 배너 제거용 transform."""
    def __init__(self, pixels=20):
        self.pixels = pixels

    def __call__(self, img):
        w, h = img.size
        return img.crop((0, 0, w, h - self.pixels))


class SampleDataset(Dataset):
    """fold별 샘플 리스트를 감싸는 Dataset.
    원본 FGVC 이미지(IMAGES_DIR)에만 배너 제거를 적용하고,
    added 이미지(USER_IMAGES)는 그대로 사용한다.
    """
    _banner = RemoveBottomBanner(20)

    def __init__(self, samples, class_to_idx, transform=None, crop_bbox=False):
        self.samples      = samples
        self.class_to_idx = class_to_idx
        self.transform    = transform
        self.crop_bbox    = crop_bbox

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = Image.open(s['path']).convert('RGB')
        if self.crop_bbox and s['bbox']:
            img = img.crop(s['bbox'])
        if s['path'].parent == IMAGES_DIR:
            img = self._banner(img)
        if self.transform:
            img = self.transform(img)
        return img, self.class_to_idx[s['label']]


# ── 빠른 검증 ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=== 전체 (resplit=True, 6:2:2) ===")
    FGVCAircraft(split='all', label='variant').summary()

    print("\n=== split별 크기 ===")
    for sp in ('train', 'val', 'test'):
        ds = FGVCAircraft(split=sp, label='variant')
        print(f"  {sp}: {len(ds)}")

    print("\n=== resplit=False (원본 1:1:1) ===")
    FGVCAircraft(split='all', label='variant', resplit=False).summary()

    print("\n클래스 예시 (처음 5개):", FGVCAircraft(split='all', label='variant').classes[:5])
