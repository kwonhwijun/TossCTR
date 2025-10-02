#!/usr/bin/env python3
"""
전체 데이터 준비 스크립트 (train/val/test 분할)
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# 프로젝트 루트
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def prepare_full_data():
    """
    train.parquet을 train/val로 분할하고, test.parquet을 처리
    """
    logging.info("=" * 60)
    logging.info("전체 데이터 준비 시작")
    logging.info("=" * 60)
    
    # 경로 설정
    raw_dir = project_root / 'data' / 'raw'
    processed_dir = project_root / 'data' / 'processed'
    processed_dir.mkdir(exist_ok=True)
    
    train_parquet = raw_dir / 'train.parquet'
    test_parquet = raw_dir / 'test.parquet'
    
    # 1. 학습 데이터 로드 및 분할
    logging.info(f"Loading training data from: {train_parquet}")
    train_df = pd.read_parquet(train_parquet)
    logging.info(f"  - Total rows: {len(train_df):,}")
    logging.info(f"  - Columns: {train_df.columns.tolist()}")
    
    # 클릭률 확인
    if 'clicked' in train_df.columns:
        click_rate = train_df['clicked'].mean()
        logging.info(f"  - Click rate: {click_rate:.4%}")
    
    # Train/Val 분할 (시간순 split - 마지막 10%를 validation으로)
    n_total = len(train_df)
    n_train = int(n_total * 0.9)
    
    logging.info(f"\nSplitting data:")
    logging.info(f"  - Train: {n_train:,} rows (90%)")
    logging.info(f"  - Val: {n_total - n_train:,} rows (10%)")
    
    train_split = train_df.iloc[:n_train]
    val_split = train_df.iloc[n_train:]
    
    # 2. 테스트 데이터 로드
    logging.info(f"\nLoading test data from: {test_parquet}")
    test_df = pd.read_parquet(test_parquet)
    logging.info(f"  - Total rows: {len(test_df):,}")
    
    # 테스트 데이터에 더미 clicked 컬럼 추가 (전처리를 위해)
    if 'clicked' not in test_df.columns:
        test_df['clicked'] = 0
        logging.info("  - Added dummy 'clicked' column for preprocessing")
    
    # 3. CSV로 저장
    train_csv = processed_dir / 'train_full.csv'
    val_csv = processed_dir / 'val_full.csv'
    test_csv = processed_dir / 'test_full.csv'
    
    logging.info(f"\nSaving CSV files:")
    logging.info(f"  - Train: {train_csv}")
    train_split.to_csv(train_csv, index=False)
    
    logging.info(f"  - Val: {val_csv}")
    val_split.to_csv(val_csv, index=False)
    
    logging.info(f"  - Test: {test_csv}")
    test_df.to_csv(test_csv, index=False)
    
    # 4. 메모리 및 디스크 사용량 확인
    logging.info(f"\nFile sizes:")
    for csv_file in [train_csv, val_csv, test_csv]:
        size_mb = csv_file.stat().st_size / (1024 * 1024)
        logging.info(f"  - {csv_file.name}: {size_mb:.1f} MB")
    
    logging.info("\n" + "=" * 60)
    logging.info("✅ 전체 데이터 준비 완료!")
    logging.info("=" * 60)
    logging.info("\n다음 단계: 학습 실행")
    logging.info("  python scripts/train.py --config configs/ --expid FESeq_full --gpu 0")


if __name__ == '__main__':
    prepare_full_data()


