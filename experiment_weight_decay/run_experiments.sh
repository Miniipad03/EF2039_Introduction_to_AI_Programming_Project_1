#!/bin/bash
# run_experiments.sh — experiment_weight_decay
# AdamW + ResNet34에서 weight_decay 탐색 (1e-3 / 1e-2 / 5e-2)
# 나머지 조건 고정: Adam→AdamW, no tuning, no crop_bbox, epochs=30
# 실행: bash experiment_weight_decay/run_experiments.sh  (프로젝트 루트에서)

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

COMMON="--label family --batch_size 32 --lr 0.001
        --optimizer adamw --folds 4 --seed 42
        --model resnet34 --attn none --epochs 30 --crop_bbox
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
#  Weight Decay 탐색 — AdamW + ResNet34 (기타 조건 고정)
# ════════════════════════════════════════════════════════════

run_exp 1 3 "wd_1e-3" \
    $COMMON \
    --weight_decay 1e-3

run_exp 2 3 "wd_1e-2" \
    $COMMON \
    --weight_decay 1e-2

run_exp 3 3 "wd_5e-2" \
    $COMMON \
    --weight_decay 5e-2

# ════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════"
echo "  모든 실험 완료 (3/3)"
echo "  결과 CSV → $OUTDIR/$RESULTS_CSV"
echo "  로그     → $LOG_DIR/"
echo "══════════════════════════════════════════════════════"
