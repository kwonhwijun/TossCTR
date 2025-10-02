#!/bin/bash
# TossCTR FESeq 훈련 - 간단 버전

home="colab_feseq/model_zoo"

export PYTHONPATH=${PWD}
echo "PYTHONPATH: ${PYTHONPATH}"

# 가상환경 활성화
source venv/bin/activate

cd $home/FESeq

# H5 파일들이 모두 준비되었는지 확인
echo "=== Checking H5 Files ==="
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
echo "=== Training FESeq TossCTR ===" 
python run_expid.py --gpu -1 --expid FESeq_tossctr_h5

echo ""
echo "=== Training Complete! ==="

