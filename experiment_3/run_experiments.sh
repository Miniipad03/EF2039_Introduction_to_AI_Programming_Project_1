#!/bin/bash
# run_experiments.sh — experiment_3
# Overfitting 억제 실험: AdamW + weight_decay=5e-4 + label_smoothing=0.1
# experiment_2와 동일한 8개 구조, optimizer/regularization만 변경
# 실행: bash experiment_3/run_experiments.sh  (프로젝트 루트에서)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

PYTHON="$PROJECT_ROOT/.venv/bin/python"
OUTDIR="$SCRIPT_DIR"
RESULTS_CSV="results_all.csv"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 이전 실행 결과 초기화 (재실행 시 중복 행 방지)
rm -f "$OUTDIR/$RESULTS_CSV"

COMMON="--label family --batch_size 32 --lr 0.001
        --optimizer adamw --weight_decay 5e-4 --label_smoothing 0.1 --folds 4 --seed 42
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
echo "결과 CSV: $OUTDIR/$RESULTS_CSV"

# ════════════════════════════════════════════════════════════
#  Section 1 — Baseline
# ════════════════════════════════════════════════════════════

run_exp 1 8 "Baseline_SimpleCNN" \
    $COMMON \
    --epochs 100 \
    --model cnn \
    --attn none

run_exp 2 8 "Baseline_ResNet34" \
    $COMMON \
    --epochs 30 \
    --model resnet34 \
    --attn none

# ════════════════════════════════════════════════════════════
#  Section 2 — Data Ablation (ResNet-34, epochs=30)
# ════════════════════════════════════════════════════════════

run_exp 3 8 "DataAblation_ExclOnly" \
    $COMMON \
    --epochs 30 \
    --model resnet34 \
    --attn none \
    --use_excluded

run_exp 4 8 "DataAblation_AddedOnly" \
    $COMMON \
    --epochs 30 \
    --model resnet34 \
    --attn none \
    --use_added

run_exp 5 8 "DataAblation_Tuning" \
    $COMMON \
    --epochs 30 \
    --model resnet34 \
    --attn none \
    --tuning

# ════════════════════════════════════════════════════════════
#  Section 3 — Attention Comparison (ResNet-34, epochs=30)
# ════════════════════════════════════════════════════════════

echo ""
echo "── Section 2 결과에서 최고 데이터 설정 선택 중... ──"
cd "$PROJECT_ROOT"
BEST_DATA=$("$PYTHON" pick_best_data.py "$OUTDIR/$RESULTS_CSV" --model resnet34 --attn none)
echo "  → 선택된 데이터 플래그: '${BEST_DATA:-없음 (vanilla)}'"

run_exp 6 8 "Attention_Channel" \
    $COMMON \
    --epochs 30 \
    --model resnet34 \
    --attn channel \
    $BEST_DATA

run_exp 7 8 "Attention_Spatial" \
    $COMMON \
    --epochs 30 \
    --model resnet34 \
    --attn spatial \
    $BEST_DATA

run_exp 8 8 "Attention_CBAM" \
    $COMMON \
    --epochs 30 \
    --model resnet34 \
    --attn cbam \
    $BEST_DATA

# ════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════"
echo "  모든 실험 완료 (8/8)"
echo "  결과 CSV → $OUTDIR/$RESULTS_CSV"
echo "  로그     → $LOG_DIR/"
echo "══════════════════════════════════════════════════════"
