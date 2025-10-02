# TossCTR 실행 가이드

## 🎯 목표
FESeq 모델을 사용하여 CTR(Click-Through Rate)을 예측하고 `submission.csv` 파일을 생성합니다.

## 📋 전제 조건

1. **데이터 준비 완료**
   - `data/raw/train.parquet` (학습 데이터)
   - `data/raw/test.parquet` (테스트 데이터)

2. **패키지 설치**
   ```bash
   cd /Users/hj/projects/TossCTR
   source venv/bin/activate  # 가상환경 활성화
   pip install -r requirements_feseq.txt
   ```

## 🚀 실행 방법

### 방법 1: 전체 파이프라인 한 번에 실행 (권장)

```bash
cd /Users/hj/projects/TossCTR
python scripts/run_full_pipeline.py
```

이 스크립트는 다음을 자동으로 수행합니다:
1. Toy 데이터 생성 (빠른 테스트용)
2. 전체 데이터 준비 (train/val/test 분할)
3. FESeq 모델 학습
4. 테스트 데이터 예측 및 `submission.csv` 생성

**예상 소요 시간**: 데이터 크기와 하드웨어에 따라 다름 (CPU: 수 시간, GPU: 1-2시간)

### 방법 2: 단계별 실행

#### 1단계: Toy 데이터로 빠른 테스트 (선택사항)

```bash
# Toy 데이터 생성
python scripts/create_toy_data.py --n_samples 1000

# Toy 데이터로 학습 테스트
python scripts/train.py --config configs/ --expid FESeq_toy --gpu -1 --mode train
```

#### 2단계: 전체 데이터 준비

```bash
python scripts/prepare_full_data.py
```

#### 3단계: 전체 데이터로 모델 학습

```bash
python scripts/train.py \
    --config configs/ \
    --expid FESeq_full \
    --gpu -1 \
    --mode train
```

- `--gpu -1`: CPU 사용 (GPU 있으면 `--gpu 0`으로 변경)

#### 4단계: 예측 및 submission.csv 생성

```bash
python scripts/predict.py \
    --config configs/ \
    --expid FESeq_full \
    --test_data data/raw/test.parquet \
    --output data/output/submission.csv \
    --gpu -1
```

## 📂 출력 파일

- **모델 체크포인트**: `checkpoints/tossctr_full/FESeq_full.model`
- **로그 파일**: `checkpoints/tossctr_full/FESeq_full.log`
- **최종 submission**: `data/output/submission.csv`
- **결과 요약**: `results.csv`

## ⚙️ 설정 커스터마이징

### 하이퍼파라미터 조정

`configs/model_config.yaml` 파일을 수정:

```yaml
FESeq_full:
    batch_size: 2048          # 배치 크기
    embedding_dim: 32         # 임베딩 차원
    dnn_hidden_units: [512, 256, 128]  # DNN 레이어
    num_heads: 4              # Attention heads
    stacked_transformer_layers: 2  # Transformer 레이어 수
    learning_rate: 1.0e-3     # 학습률
    epochs: 5                 # 에폭 수
```

### 피처 선택

`configs/dataset_config.yaml` 파일을 수정하여 사용할 피처를 변경할 수 있습니다.

## 🐛 문제 해결

### 메모리 부족 에러

- `batch_size`를 줄이기 (예: 1024, 512)
- 더 작은 모델 사용 (`embedding_dim`, `dnn_hidden_units` 줄이기)

### Import 에러

```bash
# fuxictr 재설치
pip install -e .

# 또는 모든 패키지 재설치
pip install -r requirements_feseq.txt --force-reinstall
```

### GPU 메모리 에러

```bash
# CPU로 전환
python scripts/train.py --config configs/ --expid FESeq_full --gpu -1
```

## 📊 결과 확인

### Submission 파일 형식

```csv
id,clicked
0,0.123456
1,0.234567
...
```

### 평가 지표

학습 과정에서 다음 지표가 출력됩니다:
- **AUC**: Area Under ROC Curve
- **LogLoss**: Binary Cross Entropy Loss

## 💡 팁

1. **빠른 테스트**: 먼저 toy 데이터로 테스트하여 파이프라인이 정상 작동하는지 확인
2. **GPU 사용**: GPU가 있다면 반드시 사용 (`--gpu 0`)
3. **조기 종료**: `early_stop_patience=2`로 설정되어 있어 성능이 개선되지 않으면 자동으로 종료
4. **체크포인트**: 학습 중단 시 체크포인트에서 재개 가능

## 📞 문의

문제가 발생하면 로그 파일(`checkpoints/tossctr_full/FESeq_full.log`)을 확인하세요.

---

**마지막 업데이트**: 2025-10-02

