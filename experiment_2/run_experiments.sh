#!/bin/bash
# run_experiments.sh — experiment_2
# Section 1: Baseline       — SimpleCNN (epochs=100) / ResNet-34 (epochs=30)
# Section 2: Data Ablation  — excl / added / tuning 개별 효과 비교
# Section 3: Attention      — Section 2 최고 데이터 설정으로 channel / spatial / cbam 비교
# 실행: bash experiment_2/run_experiments.sh  (프로젝트 루트에서)

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

# ── 기존 설정 (오버피팅 실험 전) ─────────────────────────────────────────────
COMMON="--label family --batch_size 32 --lr 0.001
        --optimizer adam --weight_decay 1e-4 --folds 4 --seed 42
        --save_csv $RESULTS_CSV --outdir $OUTDIR"

# ── 오버피팅 억제 실험 설정 (위 COMMON 주석 처리 후 아래 주석 해제) ────────────
# COMMON="--label family --batch_size 32 --lr 0.001
#         --optimizer adamw --weight_decay 5e-4 --label_smoothing 0.1 --folds 4 --seed 42
#         --save_csv $RESULTS_CSV --outdir $OUTDIR"

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
#  SimpleCNN (scratch, epochs=100) vs ResNet-34 (pretrained, epochs=30)
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
#  excluded 제거 / added 추가 효과를 개별 및 조합으로 비교
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
#  Section 2 결과에서 test_acc 최고 데이터 설정을 자동 선택
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
