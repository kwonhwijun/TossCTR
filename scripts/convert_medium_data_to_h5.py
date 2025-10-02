#!/usr/bin/env python3
"""
중간 크기 데이터셋 생성 (빠른 훈련용)
"""

import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
import logging
from fuxictr.preprocess import FeatureProcessor, build_dataset
from fuxictr.utils import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_medium_dataset(
    train_parquet_path: str,
    output_dir: str,
    dataset_id: str = 'tossctr_medium',
    n_samples: int = 500000,  # 50만개
    valid_ratio: float = 0.1
):
    """
    중간 크기 데이터셋 생성 (빠른 변환)
    """
    
    logging.info("="*60)
    logging.info(f"Creating {dataset_id} dataset with {n_samples:,} samples")
    logging.info("="*60)
    
    # 출력 디렉토리
    output_path = os.path.join(output_dir, dataset_id)
    os.makedirs(output_path, exist_ok=True)
    
    # Config 로드
    params = load_config('configs/', 'FESeq_toy')  # toy 설정 기반
    params['data_root'] = output_dir
    params['dataset_id'] = dataset_id
    
    # 1. 데이터 샘플링 (메모리 효율적)
    logging.info(f"Sampling {n_samples:,} from parquet...")
    
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(train_parquet_path)
    
    # 랜덤 샘플링 대신 앞에서 N개 (빠름!)
    df_list = []
    collected = 0
    
    for batch in pf.iter_batches(batch_size=100000):
        batch_df = batch.to_pandas()
        remaining = n_samples - collected
        
        if remaining <= len(batch_df):
            df_list.append(batch_df.head(remaining))
            break
        else:
            df_list.append(batch_df)
            collected += len(batch_df)
            logging.info(f"Collected: {collected:,}/{n_samples:,}")
    
    df = pd.concat(df_list, ignore_index=True)
    logging.info(f"✅ Loaded {len(df):,} samples")
    
    # 2. Train/Valid 분할
    logging.info("Splitting train/valid...")
    from sklearn.model_selection import train_test_split
    
    train_df, valid_df = train_test_split(
        df, 
        test_size=valid_ratio, 
        random_state=42,
        shuffle=True
    )
    
    logging.info(f"Train: {len(train_df):,}")
    logging.info(f"Valid: {len(valid_df):,}")
    
    # 3. CSV 임시 저장 (build_dataset에서 처리)
    train_csv = os.path.join(output_path, 'train_temp.csv')
    valid_csv = os.path.join(output_path, 'valid_temp.csv')
    
    logging.info("Saving temporary CSV files...")
    train_df.to_csv(train_csv, index=False)
    valid_df.to_csv(valid_csv, index=False)
    
    del df, train_df, valid_df
    import gc
    gc.collect()
    
    # 4. Feature processor 초기화 및 H5 변환
    logging.info("Building feature_map and transforming to h5...")
    feature_encoder = FeatureProcessor(**params)
    
    # build_dataset이 자동으로 h5 생성
    params['train_data'] = train_csv
    params['valid_data'] = valid_csv
    params['test_data'] = valid_csv  # 더미
    
    train_h5, valid_h5, test_h5 = build_dataset(feature_encoder, **params)
    
    # 5. 임시 CSV 삭제
    logging.info("Cleaning up temporary files...")
    os.remove(train_csv)
    os.remove(valid_csv)
    
    logging.info("="*60)
    logging.info("🎉 Medium dataset creation completed!")
    logging.info("="*60)
    logging.info(f"Output: {output_path}")
    logging.info(f"  - train.h5: {train_h5}")
    logging.info(f"  - valid.h5: {valid_h5}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_parquet', type=str, 
                       default='data/train.parquet')
    parser.add_argument('--output_dir', type=str,
                       default='data/')
    parser.add_argument('--dataset_id', type=str,
                       default='tossctr_medium')
    parser.add_argument('--n_samples', type=int, default=500000,
                       help='Number of samples (default: 500K)')
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    
    args = parser.parse_args()
    
    create_medium_dataset(
        train_parquet_path=os.path.join(project_root, args.train_parquet),
        output_dir=os.path.join(project_root, args.output_dir),
        dataset_id=args.dataset_id,
        n_samples=args.n_samples,
        valid_ratio=args.valid_ratio
    )


if __name__ == '__main__':
    main()
