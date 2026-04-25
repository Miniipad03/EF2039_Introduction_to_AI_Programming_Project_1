#!/bin/bash
# run_experiments.sh — experiment_4_notuning
# Section 1: ResNet34 (baseline) vs ResNet34d (dropout) — bbox only, no tuning (순수 모델 비교)
# Section 2: 승자 모델로 tuning + added_repeat 1/2/3/5/10 비교
# 실행: bash experiment_4_notuning/run_experiments.sh  (프로젝트 루트에서)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

PYTHON="$PROJECT_ROOT/.venv/bin/python"
OUTDIR="$SCRIPT_DIR"
RESULTS_CSV="results_all.csv"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 이전 실행 결과 초기화
rm -f "$OUTDIR/$RESULTS_CSV"

# Section 1 공통: tuning 없이 순수 모델 비교
COMMON_S1="--label family --batch_size 32 --lr 0.001
        --optimizer adam --weight_decay 1e-4 --folds 4 --seed 42
        --crop_bbox --attn none --epochs 50
        --save_csv $RESULTS_CSV --outdir $OUTDIR"

# Section 2 공통: tuning ON + added_repeat 비교
COMMON_S2="--label family --batch_size 32 --lr 0.001
        --optimizer adam --weight_decay 1e-4 --folds 4 --seed 42
        --tuning --crop_bbox --attn none --epochs 50
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
#  Section 1 — ResNet34 vs ResNet34d (bbox only, no tuning)
#  목적: 순수 모델 성능 비교 (데이터 변수 제거)
# ════════════════════════════════════════════════════════════

run_exp 1 7 "ResNet34_notuning" \
    $COMMON_S1 \
    --model resnet34

run_exp 2 7 "ResNet34d_notuning" \
    $COMMON_S1 \
    --model resnet34d

# ════════════════════════════════════════════════════════════
#  Section 2 — 승자 모델로 added_repeat 비교 (tuning ON)
#  목적: curated 이미지 반복 횟수에 따른 성능 변화
# ════════════════════════════════════════════════════════════

echo ""
echo "── Section 1 결과에서 최고 모델 선택 중... ──"
cd "$PROJECT_ROOT"
BEST_MODEL=$("$PYTHON" pick_best_model.py "$OUTDIR/$RESULTS_CSV" \
             --candidates resnet34 resnet34d --attn none)
echo "  → 선택된 모델: '${BEST_MODEL}'"

run_exp 3 7 "AddedRepeat_x1" \
    $COMMON_S2 \
    --model $BEST_MODEL \
    --added_repeat 1

run_exp 4 7 "AddedRepeat_x2" \
    $COMMON_S2 \
    --model $BEST_MODEL \
    --added_repeat 2

run_exp 5 7 "AddedRepeat_x3" \
    $COMMON_S2 \
    --model $BEST_MODEL \
    --added_repeat 3

run_exp 6 7 "AddedRepeat_x5" \
    $COMMON_S2 \
    --model $BEST_MODEL \
    --added_repeat 5

run_exp 7 7 "AddedRepeat_x10" \
    $COMMON_S2 \
    --model $BEST_MODEL \
    --added_repeat 10

# ════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════"
echo "  모든 실험 완료 (7/7)"
echo "  결과 CSV → $OUTDIR/$RESULTS_CSV"
echo "  로그     → $LOG_DIR/"
echo "══════════════════════════════════════════════════════"
