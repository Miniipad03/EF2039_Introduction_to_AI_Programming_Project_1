#!/usr/bin/env python3
"""smoke_test.py — 본 실험 전 빠른 오류 체크
import / 데이터 로딩 / 모델 forward pass / pick_best_data 검증
"""

import os, sys, traceback, subprocess, tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

ok = fail = 0

def check(name, fn):
    global ok, fail
    print(f"  {name:<38} ", end='', flush=True)
    try:
        fn()
        print("OK")
        ok += 1
    except Exception:
        print("FAIL")
        traceback.print_exc()
        fail += 1

print("=" * 55)
print("  Smoke Test")
print("=" * 55)

# ── 1. imports ────────────────────────────────────────────────────────────────
def test_imports():
    import torch
    from dataset_utils import FGVCAircraft, SampleDataset
    from models import build_model

check("imports", test_imports)

# ── 2. 데이터 로딩 (각 data config) ──────────────────────────────────────────
from dataset_utils import FGVCAircraft

DATA_CONFIGS = [
    ("data: vanilla",    dict(use_excluded=False, use_added=False)),
    ("data: excl_only",  dict(use_excluded=True,  use_added=False)),
    ("data: added_only", dict(use_excluded=False, use_added=True)),
    ("data: tuning",     dict(use_excluded=True,  use_added=True)),
]

num_classes = None
for name, kwargs in DATA_CONFIGS:
    def _test(kw=kwargs):
        global num_classes
        ds = FGVCAircraft(split='all', label='family', transform=None,
                          crop_bbox=False, resplit=False, **kw)
        assert len(ds) > 0, "데이터셋이 비어있음"
        if num_classes is None:
            num_classes = len(ds.classes)
    check(name, _test)

# ── 3. 모델 빌드 + forward pass ───────────────────────────────────────────────
import torch
from models import build_model

nc  = num_classes or 70
x   = torch.randn(2, 3, 224, 224)

MODEL_CONFIGS = [
    ("model: SimpleCNN",         'cnn',      'none'),
    ("model: ResNet34 no-attn",  'resnet34', 'none'),
    ("model: ResNet34 channel",  'resnet34', 'channel'),
    ("model: ResNet34 spatial",  'resnet34', 'spatial'),
    ("model: ResNet34 cbam",     'resnet34', 'cbam'),
]

for name, model_name, attn in MODEL_CONFIGS:
    def _test(mn=model_name, at=attn):
        m = build_model(mn, num_classes=nc, attn=at).eval()
        with torch.no_grad():
            out = m(x)
        assert out.shape == (2, nc), f"출력 shape 오류: {out.shape}"
    check(name, _test)

# ── 4. pick_best_data.py ──────────────────────────────────────────────────────
import pandas as pd

def test_pick_best():
    rows = [
        dict(model='resnet34', attn='none', use_excluded=False, use_added=False,
             ensemble_test_acc=0.70, tuning=False),
        dict(model='resnet34', attn='none', use_excluded=True,  use_added=False,
             ensemble_test_acc=0.72, tuning=False),
        dict(model='resnet34', attn='none', use_excluded=True,  use_added=True,
             ensemble_test_acc=0.75, tuning=True),
    ]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False,
                                     dir=PROJECT_ROOT) as f:
        pd.DataFrame(rows).to_csv(f, index=False)
        tmp = f.name
    try:
        r = subprocess.run(
            [sys.executable, 'pick_best_data.py', tmp,
             '--model', 'resnet34', '--attn', 'none'],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, f"종료 코드 {r.returncode}: {r.stderr.strip()}"
        assert '--use_excluded' in r.stdout and '--use_added' in r.stdout, \
            f"예상 외 출력: {r.stdout!r}"
    finally:
        os.unlink(tmp)

check("pick_best_data", test_pick_best)

# ── 결과 ──────────────────────────────────────────────────────────────────────
print()
print("=" * 55)
print(f"  결과: {ok} OK  /  {fail} FAIL")
print("=" * 55)
sys.exit(1 if fail else 0)
