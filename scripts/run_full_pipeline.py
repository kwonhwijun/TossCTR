#!/usr/bin/env python3
"""
전체 파이프라인 실행: Toy 테스트 → 전체 데이터 학습 → Submission 생성
"""

import os
import sys
from pathlib import Path
import logging

# 프로젝트 루트
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_step(step_name, func):
    """단계 실행"""
    logging.info("=" * 80)
    logging.info(f"STEP: {step_name}")
    logging.info("=" * 80)
    try:
        func()
        logging.info(f"✅ {step_name} 완료\n")
    except Exception as e:
        logging.error(f"❌ {step_name} 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def step1_create_toy_data():
    """1단계: Toy 데이터 생성"""
    from scripts.create_toy_data import create_toy_dataset
    
    create_toy_dataset(
        input_path='data/raw/train.parquet',
        output_path='data/toy/train_toy.csv',
        n_samples=1000,  # 빠른 테스트를 위해 1,000개만
        method='sequential'
    )


def step2_prepare_full_data():
    """2단계: 전체 데이터 준비"""
    import pandas as pd
    
    logging.info("전체 데이터 준비 시작...")
    
    raw_dir = project_root / 'data' / 'raw'
    processed_dir = project_root / 'data' / 'processed'
    processed_dir.mkdir(exist_ok=True)
    
    # Train 데이터 로드 및 분할
    logging.info("Loading training data...")
    train_df = pd.read_parquet(raw_dir / 'train.parquet')
    logging.info(f"  Total rows: {len(train_df):,}")
    
    # Train/Val 분할 (90/10)
    n_total = len(train_df)
    n_train = int(n_total * 0.9)
    
    train_split = train_df.iloc[:n_train]
    val_split = train_df.iloc[n_train:]
    
    logging.info(f"Splitting: Train={len(train_split):,}, Val={len(val_split):,}")
    
    # Test 데이터 로드
    test_df = pd.read_parquet(raw_dir / 'test.parquet')
    if 'clicked' not in test_df.columns:
        test_df['clicked'] = 0
    
    # CSV로 저장
    train_split.to_csv(processed_dir / 'train_full.csv', index=False)
    val_split.to_csv(processed_dir / 'val_full.csv', index=False)
    test_df.to_csv(processed_dir / 'test_full.csv', index=False)
    
    logging.info("전체 데이터 준비 완료")


def step3_train_full():
    """3단계: 전체 데이터로 모델 학습"""
    from scripts.train import train_feseq
    
    logging.info("FESeq 모델 학습 시작 (전체 데이터)...")
    
    train_feseq(
        config_dir='configs/',
        experiment_id='FESeq_full',
        gpu=-1,  # CPU 사용 (GPU 있으면 0으로 변경)
        mode='train'
    )


def step4_predict():
    """4단계: 테스트 데이터 예측 및 submission.csv 생성"""
    from scripts.predict import predict
    
    logging.info("테스트 데이터 예측 시작...")
    
    predict(
        config_dir='configs/',
        experiment_id='FESeq_full',
        test_parquet='data/raw/test.parquet',
        output_csv='data/output/submission.csv',
        gpu=-1
    )


def main():
    logging.info("🚀 TossCTR 전체 파이프라인 시작")
    logging.info(f"작업 디렉토리: {project_root}\n")
    
    # 1단계: Toy 데이터 생성
    run_step("1. Toy 데이터 생성", step1_create_toy_data)
    
    # 2단계: 전체 데이터 준비
    run_step("2. 전체 데이터 준비 (Train/Val/Test 분할)", step2_prepare_full_data)
    
    # 3단계: 전체 데이터로 모델 학습
    run_step("3. FESeq 모델 학습 (전체 데이터)", step3_train_full)
    
    # 4단계: 예측 및 submission 생성
    run_step("4. 테스트 데이터 예측 및 Submission 생성", step4_predict)
    
    logging.info("=" * 80)
    logging.info("🎉 전체 파이프라인 완료!")
    logging.info("=" * 80)
    logging.info(f"\n최종 결과물: {project_root}/data/output/submission.csv")
    logging.info("이제 submission.csv를 제출할 수 있습니다.\n")


if __name__ == '__main__':
    main()

