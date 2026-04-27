#!/usr/bin/env python3
"""FGVC-Aircraft inference (single or ensemble) — outputs JSON to stdout."""

import argparse
import contextlib
import io
import json
import re
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

# PROJECT_ROOT = EF2039_Introduction_to_AI_Programming_Project_1/
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import build_model, build_model_dropout  # noqa: E402


# ── Eval Transform (train.py의 eval_transform과 동일) ─────────────────────────

class ResizeWithPad:
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
        return transforms.functional.pad(
            img,
            (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2),
        )


EVAL_TRANSFORM = transforms.Compose([
    ResizeWithPad(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


# ── Helper Functions ──────────────────────────────────────────────────────────

def parse_model_info(pth_path: str):
    """파일명에서 model, attn, label_type 파싱."""
    stem = Path(pth_path).stem
    m = re.match(r'best_fold\d+_(.+)', stem)
    if not m:
        return 'resnet34', 'none', 'family'

    tag   = m.group(1).split('_lr')[0]   # e.g. resnet34_cbam_tuning_family
    parts = tag.split('_')

    model    = parts[0] if len(parts) > 0 else 'resnet34'
    attn_tag = parts[1] if len(parts) > 1 else 'noattn'
    label    = parts[3] if len(parts) > 3 else 'family'
    attn     = 'none' if attn_tag == 'noattn' else attn_tag
    return model, attn, label


def get_num_classes(sd: dict) -> int:
    """state dict에서 마지막 FC 레이어 출력 크기(= num_classes) 추출."""
    for key in ('fc.weight', 'fc.1.weight', 'classifier.4.weight'):
        if key in sd:
            return sd[key].shape[0]
    raise ValueError(f"num_classes를 state dict에서 알 수 없습니다. keys: {list(sd.keys())[:10]}")


def get_classes(label_type: str):
    """데이터셋에서 클래스 이름 목록 로드. 실패하면 None 반환."""
    try:
        from dataset_utils import FGVCAircraft
        with contextlib.redirect_stdout(io.StringIO()):
            ds = FGVCAircraft(split='all', label=label_type, resplit=False)
        return ds.classes
    except Exception:
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, nargs='+', help='모델 경로 (.pth) 하나 이상')
    parser.add_argument('--image',      required=True,  help='입력 이미지 경로')
    parser.add_argument('--top_k',      type=int, default=5)
    parser.add_argument('--bbox',       type=int, nargs=4, metavar=('XMIN', 'YMIN', 'XMAX', 'YMAX'), default=None)
    args = parser.parse_args()

    img = Image.open(args.image).convert('RGB')
    if args.bbox:
        img = img.crop(args.bbox)  # (left, upper, right, lower)
    x   = EVAL_TRANSFORM(img).unsqueeze(0)

    all_probs   = []
    model_infos = []

    for pth_path in args.model_path:
        model_name, attn, label_type = parse_model_info(pth_path)

        sd          = torch.load(pth_path, map_location='cpu', weights_only=True)
        num_classes = get_num_classes(sd)

        with contextlib.redirect_stdout(io.StringIO()):
            if model_name == 'resnet34d':
                model = build_model_dropout(num_classes, attn)
            else:
                model = build_model(model_name, num_classes, attn)

        model.load_state_dict(sd)
        model.eval()

        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]

        all_probs.append(probs)
        model_infos.append({
            'model':       model_name,
            'attn':        attn,
            'label_type':  label_type,
            'num_classes': num_classes,
        })

    # 앙상블 호환성 검증
    label_types     = {m['label_type']  for m in model_infos}
    num_classes_set = {m['num_classes'] for m in model_infos}
    if len(label_types) > 1:
        raise ValueError(f"앙상블 모델들의 label_type이 다릅니다: {label_types}")
    if len(num_classes_set) > 1:
        raise ValueError(f"앙상블 모델들의 num_classes가 다릅니다: {num_classes_set}")

    label_type  = model_infos[0]['label_type']
    num_classes = model_infos[0]['num_classes']
    classes     = get_classes(label_type)

    avg_probs = torch.stack(all_probs).mean(dim=0)

    top_k = min(args.top_k, num_classes)
    top_probs, top_idx = avg_probs.topk(top_k)

    predictions = [
        {
            'rank':       i + 1,
            'label':      classes[idx] if classes else f'Class {idx}',
            'confidence': round(float(p), 4),
        }
        for i, (p, idx) in enumerate(zip(top_probs.tolist(), top_idx.tolist()))
    ]

    print(json.dumps({
        'predictions': predictions,
        'model_info':  {
            'models':      [m['model'] for m in model_infos],
            'attns':       [m['attn']  for m in model_infos],
            'label_type':  label_type,
            'num_classes': num_classes,
            'ensemble':    len(args.model_path) > 1,
            'n_models':    len(args.model_path),
        },
    }))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        print(json.dumps({'error': str(e), 'traceback': traceback.format_exc()}),
              file=sys.stderr)
        sys.exit(1)
