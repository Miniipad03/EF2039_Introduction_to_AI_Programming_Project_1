#!/bin/bash
# run_experiments.sh — experiment_overfitting
# Baseline vs AdamW / Dropout / CutMix 각각 1:1 비교
# 공통 고정: crop_bbox, ResizeWithPad, HFlip+ColorJitter, rotation=15, epochs=50, folds=4, seed=42
# 실행: bash experiment_overfitting/run_experiments.sh  (프로젝트 루트에서)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

PYTHON="$PROJECT_ROOT/.venv/bin/python"
OUTDIR="$SCRIPT_DIR"
RESULTS_CSV="results_all.csv"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

rm -f "$OUTDIR/$RESULTS_CSV"

# 전체 공통 고정 조건
FIXED="--label family --batch_size 32 --lr 0.001
       --folds 4 --seed 42 --crop_bbox --attn none
       --rotation 15 --epochs 50
       --save_csv $RESULTS_CSV --outdir $OUTDIR"

run_exp() {
    local idx="$1"
    local total="$2"
    local name="$3"
    shift 3
    local logfile="$LOG_DIR/${idx}_${name}_${TIMESTAMP}.log"

    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  [$idx/$total] $name"
    echo "  Log → $logfile"
    echo "══════════════════════════════════════════════════════"

    cd "$PROJECT_ROOT"
    "$PYTHON" train.py "$@" 2>&1 | tee "$logfile"

    echo "  [$idx/$total] $name 완료"
}

echo "실험 시작: $TIMESTAMP"
echo "출력 디렉토리: $OUTDIR"

# ════════════════════════════════════════════════════════════
#  Exp 1 — Baseline
#  ResNet34, Adam, wd=1e-5
# ════════════════════════════════════════════════════════════

run_exp 1 4 "Baseline" \
    $FIXED \
    --model resnet34 \
    --optimizer adam --weight_decay 1e-5

# ════════════════════════════════════════════════════════════
#  Exp 2 — Baseline + AdamW + wd=5e-3
#  오버피팅 억제: decoupled weight decay
# ════════════════════════════════════════════════════════════

run_exp 2 4 "AdamW_wd5e-3" \
    $FIXED \
    --model resnet34 \
    --optimizer adamw --weight_decay 5e-3

# ════════════════════════════════════════════════════════════
#  Exp 3 — Baseline + Dropout (ResNet34d)
#  오버피팅 억제: FC 앞 Dropout(p=0.5)
# ════════════════════════════════════════════════════════════

run_exp 3 4 "ResNet34d_dropout" \
    $FIXED \
    --model resnet34d \
    --optimizer adam --weight_decay 1e-5

# ════════════════════════════════════════════════════════════
#  Exp 4 — Baseline + CutMix
#  오버피팅 억제: 데이터 다양성 증가
# ════════════════════════════════════════════════════════════

run_exp 4 4 "CutMix" \
    $FIXED \
    --model resnet34 \
    --optimizer adam --weight_decay 1e-5 \
    --cutmix 1.0

# ════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════"
echo "  모든 실험 완료 (4/4)"
echo "  결과 CSV → $OUTDIR/$RESULTS_CSV"
echo "  로그     → $LOG_DIR/"
echo "══════════════════════════════════════════════════════"
