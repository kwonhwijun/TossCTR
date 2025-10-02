#!/bin/bash
# FESeq 모델 훈련 실행 스크립트

echo "🚀 FESeq TossCTR 훈련 시작"
echo "================================"

# 가상환경 활성화
echo "📦 가상환경 활성화..."
source /home/hj/TossCTR/venv/bin/activate

# 필요한 패키지 설치 (누락된 경우)
echo "📦 필요한 패키지 확인..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# FESeq 디렉토리로 이동
echo "📁 FESeq 디렉토리로 이동..."
cd /home/hj/TossCTR/colab_feseq/model_zoo/FESeq

# H5 파일 존재 확인
echo "📋 H5 파일 확인..."
if [ ! -f "../../data/tossctr/train.h5" ]; then
    echo "❌ train.h5 파일이 없습니다!"
    echo "   전처리가 완료될 때까지 기다려주세요."
    exit 1
fi

if [ ! -f "../../data/tossctr/feature_map.json" ]; then
    echo "❌ feature_map.json 파일이 없습니다!"
    echo "   전처리가 완료될 때까지 기다려주세요."
    exit 1
fi

echo "✅ H5 파일들이 준비되었습니다!"

# FESeq 훈련 실행
echo "🎯 FESeq 모델 훈련 시작..."
echo "================================"

python run_expid.py \
    --config ./config/ \
    --expid FESeq_tossctr_h5 \
    --gpu -1

echo "================================"
echo "🎉 FESeq 훈련 완료!"

# 결과 파일 확인
echo "📊 결과 파일 확인..."
ls -la *.csv 2>/dev/null || echo "결과 CSV 파일을 찾을 수 없습니다."
ls -la checkpoints/*FESeq_tossctr_h5* 2>/dev/null || echo "체크포인트 파일을 찾을 수 없습니다."

