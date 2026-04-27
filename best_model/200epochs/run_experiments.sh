#!/bin/bash
# run_experiments.sh — best_model/200epochs
# Best config: ResNet34d + Channel Attention + Tuning(Excl+Added)
# AdamW, wd=5e-3, CutMix=0.5, rotation=15, crop_bbox, 200 epochs
# 실행: bash best_model/200epochs/run_experiments.sh  (프로젝트 루트에서)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

PYTHON="$PROJECT_ROOT/.venv/bin/python"
OUTDIR="$SCRIPT_DIR"
RESULTS_CSV="results_all.csv"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="$LOG_DIR/best_model_200ep_${TIMESTAMP}.log"

echo "실험 시작: $TIMESTAMP"
echo "Log → $LOGFILE"

cd "$PROJECT_ROOT"
"$PYTHON" train.py \
    --model resnet34d \
    --attn channel \
    --use_excluded --use_added \
    --label family \
    --batch_size 32 --lr 0.001 \
    --optimizer adamw --weight_decay 5e-3 \
    --folds 4 --seed 42 \
    --crop_bbox \
    --epochs 200 \
    --cutmix 0.5 --rotation 15 \
    --save_csv $RESULTS_CSV --outdir $OUTDIR \
    2>&1 | tee "$LOGFILE"

echo ""
echo "완료 → $OUTDIR"
