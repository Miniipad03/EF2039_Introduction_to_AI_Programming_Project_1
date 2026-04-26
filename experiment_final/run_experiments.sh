#!/bin/bash
# run_experiments.sh — experiment_final
# Base setting: AdamW + wd=5e-3 + ResNet34d(dropout) + CutMix=0.5
# exp2/3와 동일한 구조: CNN baseline / Data ablation / Attention 비교
# 실행: bash experiment_final/run_experiments.sh  (프로젝트 루트에서)

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
        --optimizer adamw --weight_decay 5e-3 --folds 4 --seed 42
        --crop_bbox --attn none --epochs 50
        --cutmix 0.5 --rotation 15
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
#  Section 1 — Baseline 비교
# ════════════════════════════════════════════════════════════

run_exp 1 8 "Baseline_SimpleCNN" \
    $COMMON \
    --model cnn \
    --epochs 100

run_exp 2 8 "Baseline_ResNet34d" \
    $COMMON \
    --model resnet34d

# ════════════════════════════════════════════════════════════
#  Section 2 — Data Ablation (ResNet34d 고정)
# ════════════════════════════════════════════════════════════

run_exp 3 8 "DataAblation_ExclOnly" \
    $COMMON \
    --model resnet34d \
    --use_excluded

run_exp 4 8 "DataAblation_AddedOnly" \
    $COMMON \
    --model resnet34d \
    --use_added

run_exp 5 8 "DataAblation_Tuning" \
    $COMMON \
    --model resnet34d \
    --tuning

# ════════════════════════════════════════════════════════════
#  Section 3 — Attention 비교 (best data config 자동 선택)
# ════════════════════════════════════════════════════════════

echo ""
echo "── Section 2 결과에서 최적 데이터 설정 선택 중... ──"
cd "$PROJECT_ROOT"
BEST_DATA=$("$PYTHON" pick_best_data.py "$OUTDIR/$RESULTS_CSV" \
            --model resnet34d --attn none)
echo "  → 선택된 데이터: '${BEST_DATA}'"

run_exp 6 8 "Attention_Channel" \
    $COMMON \
    --model resnet34d \
    --attn channel \
    $BEST_DATA

run_exp 7 8 "Attention_Spatial" \
    $COMMON \
    --model resnet34d \
    --attn spatial \
    $BEST_DATA

run_exp 8 8 "Attention_CBAM" \
    $COMMON \
    --model resnet34d \
    --attn cbam \
    $BEST_DATA

# ════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════"
echo "  모든 실험 완료 (8/8)"
echo "  결과 CSV → $OUTDIR/$RESULTS_CSV"
echo "  로그     → $LOG_DIR/"
echo "══════════════════════════════════════════════════════"
