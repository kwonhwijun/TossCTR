#!/usr/bin/env python3
"""
원본 Parquet 데이터에서 toy 데이터셋 생성
빠른 실험을 위해 100,000개 샘플만 추출하여 CSV로 저장
"""

import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_toy_dataset(
    input_path: str,
    output_path: str,
    n_samples: int = 100000,
    random_state: int = 42,
    method: str = 'sequential'
):
    """
    Parquet 파일에서 샘플링하여 toy CSV 생성 (메모리 효율적)
    
    Args:
        input_path: 입력 parquet 파일 경로
        output_path: 출력 CSV 파일 경로
        n_samples: 추출할 샘플 수
        random_state: 랜덤 시드
        method: 'sequential' (앞에서부터) 또는 'reservoir' (랜덤 샘플링)
    """
    logging.info(f"Reading parquet file: {input_path}")
    
    # Parquet 파일 메타데이터 확인
    pf = pq.ParquetFile(input_path)
    total_rows = pf.metadata.num_rows
    logging.info(f"Total rows in original file: {total_rows:,}")
    
    if n_samples >= total_rows:
        logging.warning(f"Requested samples ({n_samples}) >= total rows ({total_rows}). Using all data.")
        n_samples = total_rows
    
    if method == 'sequential':
        # 방법 1: 앞에서부터 n_samples개만 읽기 (가장 빠름)
        logging.info(f"Using sequential method: reading first {n_samples:,} rows")
        
        # iter_batches로 필요한 만큼만 읽기
        rows_collected = 0
        dfs = []
        
        for batch in pf.iter_batches(batch_size=50000):
            batch_df = batch.to_pandas()
            remaining = n_samples - rows_collected
            
            if remaining <= len(batch_df):
                # 필요한 만큼만 가져오고 종료
                dfs.append(batch_df.head(remaining))
                break
            else:
                dfs.append(batch_df)
                rows_collected += len(batch_df)
                logging.info(f"Collected {rows_collected:,} / {n_samples:,} rows")
        
        df_toy = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        
    elif method == 'reservoir':
        # 방법 2: Reservoir Sampling (진짜 랜덤 샘플링, 조금 느림)
        logging.info(f"Using reservoir sampling method for true random sampling")
        import numpy as np
        np.random.seed(random_state)
        
        # 청크로 읽으면서 Reservoir Sampling
        chunk_size = 100000
        reservoir = None
        rows_seen = 0
        
        for batch in pf.iter_batches(batch_size=chunk_size):
            chunk_df = batch.to_pandas()
            chunk_size_actual = len(chunk_df)
            
            if reservoir is None:
                # 첫 청크: 처음 n_samples개를 reservoir로
                reservoir = chunk_df.head(n_samples).copy()
                rows_seen = chunk_size_actual
            else:
                # 이후 청크: reservoir sampling 알고리즘
                for idx in range(chunk_size_actual):
                    rows_seen += 1
                    # 확률적으로 기존 샘플을 교체
                    j = np.random.randint(0, rows_seen)
                    if j < n_samples:
                        reservoir.iloc[j] = chunk_df.iloc[idx]
            
            if rows_seen % 500000 == 0:
                logging.info(f"Processed {rows_seen:,} rows...")
        
        df_toy = reservoir.reset_index(drop=True)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    logging.info(f"Loaded dataframe shape: {df_toy.shape}")
    logging.info(f"Columns: {list(df_toy.columns)}")
    logging.info(f"Final sample size: {len(df_toy):,} rows")
    
    # 레이블 분포 확인
    if 'clicked' in df_toy.columns:
        click_rate = df_toy['clicked'].mean()
        logging.info(f"Click rate: {click_rate:.4f}")
        logging.info(f"Clicked: {df_toy['clicked'].sum():,}, Not clicked: {(~df_toy['clicked'].astype(bool)).sum():,}")
    
    # CSV로 저장
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_toy.to_csv(output_path, index=False)
    logging.info(f"✅ Saved toy dataset to: {output_path}")
    logging.info(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    return df_toy


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Create toy dataset from parquet')
    parser.add_argument('--input', type=str, default='data/raw/train.parquet',
                       help='Input parquet file path')
    parser.add_argument('--output', type=str, default='data/toy/train_toy.csv',
                       help='Output CSV file path')
    parser.add_argument('--n_samples', type=int, default=100000,
                       help='Number of samples to extract')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--method', type=str, default='sequential',
                       choices=['sequential', 'reservoir'],
                       help='Sampling method: sequential (fast) or reservoir (true random)')
    
    args = parser.parse_args()
    
    create_toy_dataset(
        input_path=args.input,
        output_path=args.output,
        n_samples=args.n_samples,
        random_state=args.seed,
        method=args.method
    )
    
    logging.info("🎉 Toy dataset creation completed!")


if __name__ == "__main__":
    main()

