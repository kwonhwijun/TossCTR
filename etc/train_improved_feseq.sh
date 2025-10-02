#!/bin/bash
# 개선된 FESeq 모델 훈련 스크립트

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
echo "Current directory: $(pwd)"
required_files=("../../../data/tossctr/train.h5" "../../../data/tossctr/valid.h5" "../../../data/tossctr/test.h5" "../../../data/tossctr/feature_map.json")

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ $file not found!"
        echo "Please wait for preprocessing to complete."
        exit 1
    else
        size=$(du -h "$file" | cut -f1)
        echo "✅ $file ($size)"
    fi
done

echo ""
echo "=== Training Improved FESeq TossCTR ==="
echo "🔧 Configuration:"
echo "   - Epochs: 15"
echo "   - Learning Rate: 5e-4"
echo "   - Embedding Dim: 64"
echo "   - Batch Size: 256"
echo "   - Transformer Layers: 2"
echo "   - Attention Heads: 4"
echo "   - DNN Units: [512, 256, 128]"
echo "   - Early Stop Patience: 5"
echo ""

# 개선된 FESeq 훈련 실행
echo "🚀 Starting improved training..."
python run_expid.py --gpu -1 --expid FESeq_tossctr_h5_improved

echo ""
echo "=== Training Completed ==="

# 결과 파일 확인
echo "=== Checking Results ==="
echo "CSV result files:"
ls -la *.csv 2>/dev/null || echo "No CSV result files found."

echo ""
echo "Checkpoint files:"
ls -la checkpoints/*FESeq_tossctr_h5_improved* 2>/dev/null || echo "No checkpoint files found."

echo ""
echo "=== Improved Training Complete! ==="
