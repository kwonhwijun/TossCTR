#!/usr/bin/env python3
"""
Test Parquet을 H5로 변환 (fuxictr 방식 사용)
"""

import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
import logging
import pickle
import gc
from fuxictr.preprocess.build_dataset import save_h5

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def convert_test_in_chunks(
    test_parquet_path: str,
    feature_processor_path: str,
    output_h5_path: str,
    chunk_size: int = 100000
):
    """
    Test parquet을 청크로 나눠서 h5로 변환
    """
    
    # Feature processor 로드
    logging.info(f"Loading feature processor from: {feature_processor_path}")
    with open(feature_processor_path, 'rb') as f:
        feature_encoder = pickle.load(f)
    
    logging.info(f"Reading test data: {test_parquet_path}")
    
    # 전체 데이터를 한 번에 읽되, 청크 단위로 처리
    import pyarrow.parquet as pq
    
    pf = pq.ParquetFile(test_parquet_path)
    total_rows = pf.metadata.num_rows
    logging.info(f"Total test samples: {total_rows:,}")
    
    # 모든 청크를 처리하여 리스트에 저장
    all_arrays = {}
    total_processed = 0
    
    for batch_idx, batch in enumerate(pf.iter_batches(batch_size=chunk_size)):
        batch_df = batch.to_pandas()
        current_size = len(batch_df)
        
        logging.info(f"Processing batch {batch_idx + 1}: {current_size:,} samples "
                    f"({total_processed:,}/{total_rows:,})")
        
        # 더미 레이블 추가 (test 데이터용)
        if 'clicked' not in batch_df.columns:
            batch_df['clicked'] = 0
        
        # 전처리 및 변환
        batch_df = feature_encoder.preprocess(batch_df)
        darray_dict = feature_encoder.transform(batch_df)
        
        # 첫 번째 배치: 구조 초기화
        if not all_arrays:
            for key, arr in darray_dict.items():
                all_arrays[key] = [arr]
        else:
            # 이후 배치: 데이터 추가
            for key, arr in darray_dict.items():
                all_arrays[key].append(arr)
        
        total_processed += current_size
        progress = (total_processed / total_rows) * 100
        logging.info(f"Progress: {progress:.1f}%")
        
        # 메모리 정리
        del batch_df, darray_dict, batch
        gc.collect()
    
    # 모든 청크를 합치기
    logging.info("Concatenating all chunks...")
    import numpy as np
    
    final_arrays = {}
    for key, arr_list in all_arrays.items():
        logging.info(f"Concatenating {key}...")
        final_arrays[key] = np.concatenate(arr_list, axis=0)
        del arr_list  # 메모리 절약
        gc.collect()
    
    # H5로 저장
    logging.info(f"Saving to h5: {output_h5_path}")
    os.makedirs(os.path.dirname(output_h5_path), exist_ok=True)
    save_h5(final_arrays, output_h5_path)
    
    logging.info(f"✅ Test H5 파일 생성 완료!")
    logging.info(f"Total samples: {total_processed:,}")
    logging.info(f"Output: {output_h5_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_parquet', type=str, 
                       default='data/raw/test.parquet')
    parser.add_argument('--feature_processor', type=str,
                       default='data/tossctr_toy/feature_processor.pkl')
    parser.add_argument('--output', type=str,
                       default='data/tossctr_toy/test_full.h5')
    parser.add_argument('--chunk_size', type=int, default=100000,
                       help='Chunk size (default: 100k)')
    
    args = parser.parse_args()
    
    convert_test_in_chunks(
        test_parquet_path=os.path.join(project_root, args.test_parquet),
        feature_processor_path=os.path.join(project_root, args.feature_processor),
        output_h5_path=os.path.join(project_root, args.output),
        chunk_size=args.chunk_size
    )


if __name__ == '__main__':
    main()