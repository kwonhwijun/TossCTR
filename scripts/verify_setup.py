#!/usr/bin/env python3
"""
설치 및 설정 검증 스크립트
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("TossCTR 설정 검증")
print("=" * 80)

# 1. 필수 파일 확인
print("\n1. 필수 파일 확인:")
required_files = [
    'data/raw/train.parquet',
    'data/raw/test.parquet',
    'data/raw/sample_submission.csv',
    'configs/dataset_config.yaml',
    'configs/model_config.yaml',
    'scripts/train.py',
    'scripts/predict.py',
    'fuxictr/utils.py',
    'models/FESeq/model_zoo/FESeq/src/FESeq.py',
]

all_exist = True
for file in required_files:
    file_path = project_root / file
    exists = file_path.exists()
    status = "✓" if exists else "✗"
    print(f"  {status} {file}")
    if not exists:
        all_exist = False

if not all_exist:
    print("\n❌ 일부 필수 파일이 없습니다!")
    sys.exit(1)

# 2. Python 패키지 확인
print("\n2. Python 패키지 확인:")
packages = [
    'torch',
    'numpy',
    'pandas',
    'sklearn',
    'yaml',
    'h5py',
    'pyarrow',
    'tqdm',
]

missing_packages = []
for pkg in packages:
    try:
        __import__(pkg if pkg != 'sklearn' else 'sklearn')
        print(f"  ✓ {pkg}")
    except ImportError:
        print(f"  ✗ {pkg}")
        missing_packages.append(pkg)

if missing_packages:
    print(f"\n❌ 다음 패키지가 설치되지 않았습니다: {', '.join(missing_packages)}")
    print("  설치 명령: pip install -r requirements_feseq.txt")
    sys.exit(1)

# 3. Import 테스트
print("\n3. Import 테스트:")
try:
    print("  Testing fuxictr...")
    from fuxictr.utils import load_config, set_logger
    from fuxictr.features import FeatureMap
    from fuxictr.pytorch.torch_utils import seed_everything
    from fuxictr.pytorch.dataloaders import H5DataLoader
    from fuxictr.preprocess import FeatureProcessor, build_dataset
    print("  ✓ fuxictr 모듈")
    
    print("  Testing FESeq...")
    sys.path.insert(0, str(project_root / 'models' / 'FESeq'))
    from model_zoo.FESeq.src.FESeq import FESeq
    print("  ✓ FESeq 모델")
    
except Exception as e:
    print(f"  ✗ Import 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. 데이터 확인
print("\n4. 데이터 확인:")
try:
    import pandas as pd
    
    train_path = project_root / 'data/raw/train.parquet'
    train_df = pd.read_parquet(train_path, nrows=5)
    print(f"  ✓ train.parquet 읽기 가능 ({len(train_df)} rows loaded as sample)")
    print(f"    Columns: {list(train_df.columns)[:10]}...")
    
    test_path = project_root / 'data/raw/test.parquet'
    test_df = pd.read_parquet(test_path, nrows=5)
    print(f"  ✓ test.parquet 읽기 가능 ({len(test_df)} rows loaded as sample)")
    
except Exception as e:
    print(f"  ✗ 데이터 읽기 실패: {e}")
    sys.exit(1)

# 5. 디렉토리 생성
print("\n5. 출력 디렉토리 생성:")
output_dirs = [
    'data/output',
    'data/processed',
    'data/toy',
    'checkpoints',
]

for dir_name in output_dirs:
    dir_path = project_root / dir_name
    dir_path.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ {dir_name}")

print("\n" + "=" * 80)
print("✅ 모든 검증 통과! 파이프라인을 실행할 수 있습니다.")
print("=" * 80)
print("\n다음 명령으로 실행:")
print("  python scripts/run_full_pipeline.py")
print("\n또는 단계별 실행:")
print("  1. python scripts/create_toy_data.py")
print("  2. python scripts/prepare_full_data.py")
print("  3. python scripts/train.py --config configs/ --expid FESeq_full")
print("  4. python scripts/predict.py --config configs/ --expid FESeq_full")
print()

