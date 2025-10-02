# GitHub Push 가이드

## 🚀 빠른 실행

터미널에서 다음 명령을 순서대로 실행하세요:

```bash
cd /Users/hj/projects/TossCTR

# 1. .gitkeep 파일 생성 (디렉토리 구조 유지)
mkdir -p data/raw data/output data/processed data/toy checkpoints
touch data/raw/.gitkeep data/output/.gitkeep data/processed/.gitkeep data/toy/.gitkeep checkpoints/.gitkeep

# 2. 핵심 파일들 추가
git add .gitignore
git add configs/
git add scripts/train.py scripts/predict.py scripts/prepare_full_data.py scripts/run_pipeline.py scripts/create_toy_data.py scripts/test_imports.py
git add fuxictr/
git add requirements.txt requirements_feseq.txt
git add README.md BRIEFING.md
git add data/raw/.gitkeep data/output/.gitkeep data/processed/.gitkeep data/toy/.gitkeep checkpoints/.gitkeep

# 3. FESeq 모델 소스 코드 강제 추가
git add -f models/FESeq/FESeq.py
git add -f models/FESeq/interaction_layer.py
git add -f models/FESeq/pooling_layer.py
git add -f models/FESeq/model_zoo/FESeq/src/*.py

# 4. 상태 확인
git status

# 5. Commit & Push
git commit -m "Add FESeq training pipeline and execution scripts

- Add training script (train.py)
- Add prediction script (predict.py) 
- Add data preparation scripts
- Add FESeq model source code
- Add fuxictr library
- Update configs for TossCTR dataset
- Add requirements_feseq.txt"

git push origin main
```

## 📦 Push되는 주요 파일들

### 필수 파일 (약 5-10MB):
- `configs/` - 설정 파일
- `scripts/` - 실행 스크립트 (train.py, predict.py 등)
- `fuxictr/` - FuxiCTR 라이브러리
- `models/FESeq/` - FESeq 모델 소스코드만
- `requirements_feseq.txt` - Python 패키지 목록

### 제외되는 파일 (Git에 올라가지 않음):
- `data/` - 데이터 파일들 (용량 큼)
- `venv/` - 가상환경
- `checkpoints/` - 학습된 모델 체크포인트
- `*.parquet`, `*.csv` - 데이터 파일
- `__pycache__/` - Python 캐시

## 🖥️ 서버에서 실행하기

GitHub에서 clone 후:

```bash
# 1. Clone
git clone https://github.com/kwonhwijun/TossCTR.git
cd TossCTR

# 2. 가상환경 생성 및 패키지 설치
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows

pip install -r requirements_feseq.txt

# 3. 데이터 업로드
# data/raw/ 폴더에 train.parquet, test.parquet 업로드

# 4. 실행
python scripts/run_pipeline.py
```

## ✅ Push 확인

Push 후 GitHub에서 확인:
- https://github.com/kwonhwijun/TossCTR

다음 파일들이 있는지 확인:
- ✓ configs/model_config.yaml
- ✓ scripts/train.py
- ✓ scripts/predict.py
- ✓ fuxictr/utils.py
- ✓ models/FESeq/FESeq.py
- ✓ requirements_feseq.txt
