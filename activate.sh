#!/bin/bash
# 사용법: source activate.sh

VENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.venv"

# 가상환경이 없으면 생성 + 패키지 설치
if [ ! -d "$VENV_DIR" ]; then
    echo "[EFAI] 가상환경 생성 중..."
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --upgrade pip -q

    # CUDA 버전 감지 → PyTorch 선택
    if command -v nvcc &>/dev/null; then
        CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
        CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
        CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
        if [ "$CUDA_MAJOR" -ge 12 ]; then
            TORCH_TAG="cu124"
        elif [ "$CUDA_MAJOR" -eq 11 ] && [ "$CUDA_MINOR" -ge 8 ]; then
            TORCH_TAG="cu118"
        else
            TORCH_TAG="cpu"
        fi
    else
        TORCH_TAG="cpu"
    fi

    echo "[EFAI] PyTorch 설치 중 (tag: $TORCH_TAG)..."
    if [ "$TORCH_TAG" = "cpu" ]; then
        "$VENV_DIR/bin/pip" install torch==2.6.0 torchvision==0.21.0 -q
    else
        "$VENV_DIR/bin/pip" install \
            torch==2.6.0+${TORCH_TAG} \
            torchvision==0.21.0+${TORCH_TAG} \
            --index-url https://download.pytorch.org/whl/${TORCH_TAG} -q
    fi

    echo "[EFAI] 나머지 패키지 설치 중..."
    "$VENV_DIR/bin/pip" install \
        "numpy>=1.24.2" "pillow>=10.0.0" "matplotlib>=3.7.1" \
        "tqdm>=4.65.0" "scikit-learn>=1.3.0" "seaborn>=0.12.0" \
        pandas -q

    echo "[EFAI] 설치 완료"
fi

# 활성화
source "$VENV_DIR/bin/activate"
echo "[EFAI] 가상환경 활성화 완료 ($(python --version))"
