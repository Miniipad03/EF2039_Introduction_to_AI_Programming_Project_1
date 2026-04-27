#!/bin/bash
# run_experiments.sh — 4개 실험 순차 실행
# 실행: bash run_experiments.sh  (프로젝트 루트에서)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 프로젝트 venv 파이썬 (스크립트와 같은 위치에 .venv 존재)
PYTHON="$SCRIPT_DIR/.venv/bin/python"

# 출력 파일은 모두 프로젝트 루트에 저장
OUTDIR="$SCRIPT_DIR"
RESULTS_CSV="results_all.csv"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 공통 인자
COMMON="--label family --epochs 30 --batch_size 32 --lr 0.001
        --optimizer adam --weight_decay 1e-4 --folds 4 --seed 42
        --save_csv $RESULTS_CSV --outdir $OUTDIR"

run_exp() {
    local idx="$1"
    local name="$2"
    shift 2
    local logfile="$LOG_DIR/${idx}_${name}_${TIMESTAMP}.log"

    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  [$idx/4] $name"
    echo "  Log → $logfile"
    echo "══════════════════════════════════════════════════════"

    cd "$SCRIPT_DIR"
    "$PYTHON" train.py "$@" 2>&1 | tee "$logfile"

    echo "  [$idx/4] $name 완료"
}

echo "실험 시작: $TIMESTAMP"
echo "출력 디렉토리: $OUTDIR"
echo "결과 CSV: $OUTDIR/$RESULTS_CSV"

# ── 실험 4개 ────────────────────────────────────────────────────────────────

# 1. Vanilla — Simple CNN (scratch, original data, 4-fold)
run_exp 1 "Vanilla_SimpleCNN" \
    $COMMON \
    --model cnn \
    --attn none

# 2. Vanilla — ResNet-34 (pretrained, original data, 4-fold)
run_exp 2 "Vanilla_ResNet34" \
    $COMMON \
    --model resnet34 \
    --attn none

# 3. Tuning — ResNet-34 (excluded 제거 + added 추가, 4-fold)
run_exp 3 "Tuning_ResNet34" \
    $COMMON \
    --model resnet34 \
    --attn none \
    --tuning

# 4. Tuning — ResNet-34 + CBAM (layer3+layer4, 4-fold)
run_exp 4 "Tuning_ResNet34_CBAM" \
    $COMMON \
    --model resnet34 \
    --attn cbam \
    --tuning

# ── 완료 ────────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════"
echo "  모든 실험 완료"
echo "  결과 CSV → $OUTDIR/$RESULTS_CSV"
echo "  로그     → $LOG_DIR/"
echo "══════════════════════════════════════════════════════"
