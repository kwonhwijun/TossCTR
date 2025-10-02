# TossCTR FESeq 프로젝트 브리핑 문서

## 📋 프로젝트 개요

**프로젝트명**: TossCTR - Click-Through Rate Prediction  
**모델**: FESeq (Feature Interaction via Self-Attention and Sequence Learning)  
**목표**: 광고 클릭률 예측을 위한 딥러닝 모델 학습 및 제출 파일 생성  
**작업 환경**: macOS (darwin 23.5.0), Python 3.13

---

## 🎯 최종 목표

**`data/raw/` 디렉토리의 학습 데이터로 FESeq 모델을 학습하여, 테스트 데이터에 대한 예측값을 생성하고 `submission.csv` 파일을 만들어 제출하기**

### 입출력 정의
- **입력 (학습)**: `data/raw/train.parquet` (10,704,179 행)
- **입력 (테스트)**: `data/raw/test.parquet`
- **출력**: `data/output/submission.csv` (sample_submission.csv 형식)

---

## 📂 현재 프로젝트 구조

```
TossCTR/                                  # 프로젝트 루트
├── data/
│   ├── raw/                              # 원본 데이터
│   │   ├── train.parquet                 # 학습 데이터 (10.7M 행, 119 컬럼)
│   │   ├── test.parquet                  # 테스트 데이터 (예측 대상)
│   │   └── sample_submission.csv         # 제출 형식 예시
│   ├── toy/                              # 빠른 실험용 샘플 데이터
│   │   └── train_toy.csv                 # 1,000개 샘플 (생성 완료)
│   └── output/                           # 결과 저장 폴더
│       └── submission.csv                # 최종 제출 파일 (생성 필요)
│
├── fuxictr/                              # FuxiCTR 라이브러리 (CTR 예측 프레임워크)
│   ├── __init__.py
│   ├── features.py                       # 피처 정의 및 처리
│   ├── metrics.py                        # 평가 지표
│   ├── utils.py                          # 유틸리티 함수
│   ├── preprocess/
│   │   ├── build_dataset.py              # 데이터셋 빌드
│   │   ├── feature_processor.py          # 피처 전처리
│   │   └── utils.py
│   └── pytorch/
│       ├── models/                       # 베이스 모델
│       ├── layers/                       # 레이어 모듈
│       ├── dataloaders/                  # 데이터 로더
│       └── torch_utils.py
│
├── models/
│   └── feseq/                            # FESeq 모델 구현체
│       ├── __init__.py
│       ├── FESeq.py                      # FESeq 메인 모델
│       ├── interaction_layer.py          # Feature Interaction 레이어
│       └── pooling_layer.py              # Pooling 레이어
│
├── configs/                              # 설정 파일
│   ├── dataset_config.yaml               # 데이터셋 정의 (피처 타입, 경로 등)
│   └── model_config.yaml                 # 모델 하이퍼파라미터
│
├── scripts/                              # 실행 스크립트
│   ├── create_toy_data.py                # Toy 데이터 생성 (완료)
│   └── train.py                          # 학습 스크립트 (작성 완료, 테스트 필요)
│
├── notebooks/                            # Jupyter 노트북 (분석/EDA)
│   ├── 01_EDA.ipynb
│   ├── 02_Baseline.ipynb
│   └── 03_Feature_EDA.ipynb
│
├── colab_feseq/                          # 외부 레포지토리 (참고용, 정리 예정)
│   └── [다양한 모델 및 설정 파일들]
│
├── models/FESeq/                         # 원본 FESeq 레포지토리 (참고용)
│   └── [원본 코드 및 문서]
│
├── requirements.txt                      # 기본 패키지
├── requirements_feseq.txt                # FESeq용 패키지 (생성 완료)
└── README.md
```

---

## 📊 데이터 상세 정보

### 데이터 구조 (train.parquet)
- **총 행 수**: 10,704,179
- **총 컬럼 수**: 119
- **클릭률**: 약 1.8%

### 피처 구성

#### 1. **Categorical Features (5개)**
```python
["gender", "age_group", "inventory_id", "day_of_week", "hour"]
```
- 사용자 및 시간 정보
- 범주형 변수 → LabelEncoding 필요

#### 2. **Numeric Features (76개)**
```python
# 27개 l_feat 시리즈
["l_feat_1", "l_feat_2", ..., "l_feat_27"]

# 10개 feat_e 시리즈  
["feat_e_1", "feat_e_2", ..., "feat_e_10"]

# 6개 feat_d 시리즈
["feat_d_1", "feat_d_2", ..., "feat_d_6"]

# 8개 feat_c 시리즈
["feat_c_1", "feat_c_2", ..., "feat_c_8"]

# 6개 feat_b 시리즈
["feat_b_1", "feat_b_2", ..., "feat_b_6"]

# 18개 feat_a 시리즈
["feat_a_1", "feat_a_2", ..., "feat_a_18"]

# 추가 history 피처들 (필요시)
["history_a_*", "history_b_*"]
```
- 수치형 변수 → MinMaxScaler 정규화 필요

#### 3. **Sequence Feature (1개)**
```python
"seq"  # 사용자 행동 시퀀스 (쉼표로 구분된 inventory_id 리스트)
```
- 예: `"269,57,463,212,193,74,318,77,317"`
- FESeq의 핵심 입력
- 최대 길이: 500
- 구분자: `,` (쉼표)

#### 4. **Target Variable (1개)**
```python
"clicked"  # 0 또는 1 (클릭 여부)
```

---

## 🔧 기술 스택

### 필수 라이브러리
```
torch>=1.10.0
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
pyyaml>=5.4.1
h5py>=3.6.0
pyarrow>=6.0.0
tqdm>=4.62.0
```

### 프레임워크
- **FuxiCTR**: CTR 예측 전용 프레임워크
  - Feature 자동 처리
  - H5 데이터 포맷 사용 (효율적)
  - CSV → H5 자동 변환 기능
  
- **PyTorch**: 딥러닝 프레임워크

---

## 📝 작업 이력 및 현재 상태

### ✅ 완료된 작업

1. **프로젝트 구조 정리**
   - 외부 레포지토리(`models/FESeq`, `colab_feseq`)에서 필요한 파일만 추출
   - TossCTR 루트에 깔끔한 구조로 재구성
   - `fuxictr/` 라이브러리 복사 완료
   - `models/feseq/` 모델 코드 복사 완료

2. **Toy 데이터 생성**
   - `scripts/create_toy_data.py` 작성
   - `data/toy/train_toy.csv` 생성 (1,000개 샘플)
   - 빠른 실험 환경 준비 완료

3. **설정 파일 작성**
   - `configs/dataset_config.yaml`: 데이터셋 및 피처 정의
   - `configs/model_config.yaml`: FESeq 하이퍼파라미터
   - `tossctr_toy` 데이터셋 설정 완료

4. **학습 스크립트 작성**
   - `scripts/train.py` 작성 완료
   - 경로 및 import 구조 수정
   - 명령행 인자 처리 구현

5. **의존성 설치**
   - `requirements_feseq.txt` 작성
   - `h5py`, `tqdm`, `pyyaml` 설치 완료

### ⏸️ 진행 중 / 필요한 작업

1. **학습 스크립트 검증**
   - `scripts/train.py` 실행 테스트 필요
   - Toy 데이터로 빠른 검증 필요

2. **전체 데이터 학습**
   - `data/raw/train.parquet` (10.7M 행) 전처리
   - FESeq 모델 학습 (GPU 필요)
   - 하이퍼파라미터 튜닝

3. **예측 및 제출 파일 생성**
   - `data/raw/test.parquet` 로드
   - 학습된 모델로 예측
   - `data/output/submission.csv` 생성
   - `sample_submission.csv` 형식 준수 확인

---

## 🚀 실행 방법

### 1. Toy 데이터로 빠른 테스트 (권장)
```bash
cd /Users/hj/projects/TossCTR

# 학습 실행
python scripts/train.py \
    --config configs/ \
    --expid FESeq_toy \
    --gpu 0 \
    --mode train
```

### 2. 전체 데이터 학습
```bash
# Step 1: 전체 데이터용 config 추가 필요 (configs/에 추가)
# Step 2: 학습 실행
python scripts/train.py \
    --config configs/ \
    --expid FESeq_full \
    --gpu 0 \
    --mode train
```

### 3. 예측 및 제출 파일 생성 (스크립트 작성 필요)
```bash
python scripts/predict.py \
    --model checkpoints/FESeq_full/model.ckpt \
    --test_data data/raw/test.parquet \
    --output data/output/submission.csv
```

---

## 🔍 주요 이슈 및 해결 방법

### 1. **메모리 문제**
**이슈**: `train.parquet`가 10.7M 행으로 매우 큼  
**해결**:
- ✅ Toy 데이터로 먼저 테스트 (1,000개)
- 청크 단위 처리 (`scripts/create_toy_data.py` 참고)
- H5 포맷 사용 (FuxiCTR 자동 변환)
- 배치 크기 조정

### 2. **경로 문제**
**이슈**: `run_expid.py`의 경로가 `../../data` 형태로 상대 경로 사용  
**해결**:
- ✅ `scripts/train.py`에서 절대 경로 기반으로 수정
- `project_root` 변수로 기준점 명확화
- `sys.path` 동적 조정

### 3. **모듈 Import 문제**
**이슈**: `fuxictr` 모듈을 찾지 못함  
**해결**:
- ✅ `fuxictr/` 폴더를 TossCTR 루트로 복사
- ✅ `sys.path.insert(0, project_root)` 추가

### 4. **Sequence 데이터 형식**
**이슈**: Parquet에는 `seq` (쉼표 구분), Config에는 `^` 구분자 설정  
**해결**:
- `configs/dataset_config.yaml`에서 `splitter: ","` 설정
- 전처리 시 자동 변환

---

## 📐 Config 파일 상세

### `configs/dataset_config.yaml`
```yaml
tossctr_toy:
    data_root: ../data                    # 데이터 루트 (상대 경로)
    data_format: csv                      # csv 또는 h5
    train_data: ../data/toy/train_toy.csv
    valid_data: ../data/toy/train_toy.csv # Toy 데이터는 동일 파일 사용
    test_data: ../data/toy/train_toy.csv
    min_categr_count: 1                   # 최소 카테고리 빈도
    feature_cols: [...]                   # 피처 정의
    label_col: {name: "clicked", dtype: float}
```

### `configs/model_config.yaml`
```yaml
FESeq_toy:
    model: FESeq
    dataset_id: tossctr_toy               # dataset_config.yaml의 키와 일치
    batch_size: 64                        # Toy용 작은 배치
    embedding_dim: 8                      # 임베딩 차원
    dnn_hidden_units: [128, 64]          # DNN 레이어
    num_heads: 2                          # Attention heads
    stacked_transformer_layers: 1         # Transformer 레이어 수
    epochs: 2                             # 에폭 수
    gpu: 0                                # GPU 번호
    # ... 기타 하이퍼파라미터
```

---

## 🎓 FESeq 모델 구조

### 핵심 아이디어
1. **Feature Embedding**: 모든 피처를 임베딩 벡터로 변환
2. **Self-Attention**: Feature 간 상호작용 학습
3. **Sequence Modeling**: 사용자 행동 시퀀스 모델링 (Transformer)
4. **Feature Interaction**: 명시적/암시적 피처 교차
5. **DNN**: 최종 예측을 위한 Deep Neural Network

### 입력 처리 흐름
```
Raw Data
  ↓
Feature Encoding (Categorical → LabelEncoding, Numeric → MinMaxScaling)
  ↓
Embedding Layer (모든 피처 → dense vectors)
  ↓
Self-Attention (Feature Interaction)
  ↓
Sequence Modeling (Transformer on seq field)
  ↓
Feature Aggregation (Pooling)
  ↓
DNN Layers
  ↓
Output (clicked probability)
```

---

## ⚠️ 주의사항

1. **GPU 메모리**
   - 전체 데이터 학습 시 최소 16GB GPU 권장
   - 배치 크기 조정 필요 시: `batch_size` 감소

2. **데이터 전처리**
   - CSV → H5 변환 시간 소요 (최초 1회)
   - `feature_map.json` 자동 생성 확인
   - `feature_processor.pkl` 저장 확인

3. **시퀀스 길이**
   - `max_len: 50` 설정됨
   - 긴 시퀀스는 자동 잘림
   - 짧은 시퀀스는 패딩 추가

4. **클릭률 불균형**
   - 클릭률 1.8% (매우 불균형)
   - 평가 지표: AUC 사용 (logloss보다 적합)
   - 필요시 클래스 가중치 조정 고려

---

## 📋 다음 단계 체크리스트

### 즉시 수행 (우선순위 높음)
- [ ] `scripts/train.py` Toy 데이터로 테스트 실행
- [ ] 학습 과정 로그 확인 및 디버깅
- [ ] 모델 체크포인트 저장 확인
- [ ] 평가 지표 (AUC, logloss) 확인

### 중기 작업
- [ ] 전체 데이터용 config 추가 (`FESeq_full`)
- [ ] 전체 데이터 학습 실행 (시간 소요 예상)
- [ ] 하이퍼파라미터 튜닝
  - `embedding_dim`: 16, 32, 64 실험
  - `dnn_hidden_units`: 다양한 조합 시도
  - `learning_rate`: 조정

### 최종 작업
- [ ] `scripts/predict.py` 작성
  - 테스트 데이터 로드
  - 학습된 모델로 예측
  - `submission.csv` 생성
- [ ] 제출 파일 형식 검증
- [ ] 최종 제출

---

## 🔗 참고 자료

### FESeq 논문
- FESeq는 Behavior Sequence Transformer (BST) 기반
- 논문: "Behavior Sequence Transformer for E-commerce Recommendation in Alibaba" (DLP-KDD 2021)
- PDF: `/Users/hj/projects/TossCTR/paper/applsci-14-02760.pdf`

### 코드 참조
- 원본 레포지토리: `models/FESeq/`
- Colab 작업물: `colab_feseq/`

---

## 💬 추가 질문 시 확인 사항

1. **현재 작업 디렉토리**: `/Users/hj/projects/TossCTR`
2. **Python 버전**: 3.13
3. **가상환경**: `venv/` 또는 시스템 Python
4. **GPU 사용 가능 여부**: 확인 필요
5. **데이터 위치 확인**: `ls -lh data/raw/` 실행 결과

---

## 🚨 긴급 이슈 발생 시

### 학습이 안 될 때
1. Toy 데이터로 먼저 테스트
2. 로그 확인: `verbose: 1` 설정
3. GPU 메모리 확인: `nvidia-smi` (CUDA 사용 시)
4. 배치 크기 감소

### Import 에러 발생 시
1. `sys.path` 확인
2. `fuxictr/` 폴더 존재 확인
3. `models/feseq/` 폴더 존재 확인
4. 패키지 재설치: `pip install -r requirements_feseq.txt`

### 메모리 부족 시
1. 더 작은 Toy 데이터 생성 (100개)
2. 배치 크기 감소 (32, 16)
3. 임베딩 차원 감소 (4)
4. Sequence 길이 감소 (max_len: 20)

---

**문서 작성일**: 2025년 10월 2일  
**마지막 업데이트**: 학습 스크립트 작성 완료, 패키지 설치 완료  
**다음 LLM에게**: 위 정보를 바탕으로 `scripts/train.py` 실행 후 발생하는 이슈를 해결하고, 최종적으로 submission.csv 생성까지 완료해주세요.

