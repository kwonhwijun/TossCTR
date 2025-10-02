#!/bin/bash
# TossCTR FESeq 훈련 스크립트 (논문 코드 스타일)

home="colab_feseq/model_zoo"

export PYTHONPATH=${PWD}
echo "PYTHONPATH: ${PYTHONPATH}"

# 가상환경 활성화
echo "=== Activating virtual environment ==="
source venv/bin/activate

# FESeq 디렉토리로 이동
cd $home/FESeq

# H5 파일 존재 확인
echo "=== Checking H5 files ==="
if [ ! -f "../../data/tossctr/train.h5" ]; then
    echo "❌ train.h5 file not found!"
    echo "Please wait for preprocessing to complete."
    exit 1
fi

if [ ! -f "../../data/tossctr/feature_map.json" ]; then
    echo "❌ feature_map.json file not found!"
    echo "Please wait for preprocessing to complete."
    exit 1
fi

echo "✅ H5 files are ready!"

# 현재 디렉토리 확인
echo "Current directory: $(pwd)"
echo "Available configs:"
ls -la config/

echo ""
echo "=== Starting FESeq Training for TossCTR ==="
echo "============================================"

# TossCTR FESeq 훈련 실행
echo "=== Training FESeq TossCTR H5 ===" 
python run_expid.py --gpu -1 --expid FESeq_tossctr_h5

echo ""
echo "=== Training Completed ==="
echo "=========================="

# 결과 파일 확인
echo "=== Checking Results ==="
echo "CSV result files:"
ls -la *.csv 2>/dev/null || echo "No CSV result files found."

echo ""
echo "Checkpoint files:"
ls -la checkpoints/*FESeq_tossctr_h5* 2>/dev/null || echo "No checkpoint files found."

echo ""
echo "=== All Done! ==="

