#!/bin/bash
# TossCTR FESeq 훈련 - GPU 버전

home="colab_feseq/model_zoo"

export PYTHONPATH=${PWD}
echo "PYTHONPATH: ${PYTHONPATH}"

# 가상환경 활성화
source venv/bin/activate

cd $home/FESeq

# GPU 사용 가능 여부 확인
echo "=== Checking GPU ==="
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "nvidia-smi not available, using CPU mode"
fi

echo ""
echo "=== Training FESeq TossCTR with GPU ===" 
python run_expid.py --gpu 0 --expid FESeq_tossctr_h5

echo ""
echo "=== Training Complete! ==="

