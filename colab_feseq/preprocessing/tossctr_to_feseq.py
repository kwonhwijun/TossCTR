#!/usr/bin/env python3
"""
TossCTR 데이터를 FESeq 형식으로 변환하는 스크립트
10,000개 샘플로 테스트용 데이터셋 생성
"""

import os
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

def load_tossctr_sample(parquet_path: str, n_samples: int = 10000) -> pd.DataFrame:
    """TossCTR parquet 파일에서 샘플 데이터 로드"""
    print(f"Loading {n_samples} samples from {parquet_path}")
    
    pf = pq.ParquetFile(parquet_path)
    batch = next(pf.iter_batches(batch_size=n_samples))
    df = batch.to_pandas()
    
    print(f"Loaded shape: {df.shape}")
    return df

def preprocess_sequence(seq_str: str, max_len: int = 50) -> str:
    """시퀀스 데이터 전처리"""
    if pd.isna(seq_str) or seq_str == '':
        return ''
    
    # 쉼표로 분리된 시퀀스를 ^ 구분자로 변경하고 길이 제한
    tokens = str(seq_str).split(',')[:max_len]
    return '^'.join(tokens)

def encode_categorical_features(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    """범주형 피처 인코딩"""
    df_encoded = df.copy()
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            # 결측치를 'unknown'으로 처리
            df_encoded[col] = df_encoded[col].fillna('unknown').astype(str)
            
            # LabelEncoder 적용
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            print(f"Encoded {col}: {len(le.classes_)} unique values")
    
    return df_encoded

def convert_tossctr_to_feseq(df: pd.DataFrame) -> pd.DataFrame:
    """TossCTR 형식을 FESeq 형식으로 변환"""
    
    # 1. 피처 분류
    categorical_cols = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']
    
    # 수치형 피처 (l_feat, feat_e, feat_d, feat_c, feat_b, feat_a 계열)
    numeric_cols = [col for col in df.columns 
                   if any(col.startswith(prefix) for prefix in ['l_feat_', 'feat_e_', 'feat_d_', 'feat_c_', 'feat_b_', 'feat_a_'])]
    
    # history 피처는 제외 (너무 많음)
    numeric_cols = [col for col in numeric_cols if not col.startswith('history_')]
    
    print(f"Categorical features: {len(categorical_cols)}")
    print(f"Numeric features: {len(numeric_cols)}")
    
    # 2. 범주형 피처 인코딩
    df_processed = encode_categorical_features(df, categorical_cols)
    
    # 3. 수치형 피처 결측치 처리
    for col in numeric_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna(0.0)
    
    # 4. 시퀀스 피처 처리
    df_processed['seq_list'] = df_processed['seq'].apply(lambda x: preprocess_sequence(x, max_len=50))
    
    # 5. 필요한 컬럼만 선택
    selected_cols = categorical_cols + numeric_cols + ['seq_list', 'clicked']
    
    # 존재하는 컬럼만 선택
    available_cols = [col for col in selected_cols if col in df_processed.columns]
    df_final = df_processed[available_cols].copy()
    
    print(f"Final shape: {df_final.shape}")
    print(f"Final columns: {list(df_final.columns)}")
    
    return df_final

def split_and_save_data(df: pd.DataFrame, output_dir: str):
    """데이터를 train/val/test로 분할하고 저장"""
    
    # 균형잡힌 분할을 위해 stratify 사용
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df['clicked']
    )
    
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['clicked']
    )
    
    print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
    print(f"Train clicked ratio: {train_df['clicked'].mean():.3f}")
    print(f"Val clicked ratio: {val_df['clicked'].mean():.3f}")
    print(f"Test clicked ratio: {test_df['clicked'].mean():.3f}")
    
    # 저장
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_csv(f"{output_dir}/train_data.csv", index=False)
    val_df.to_csv(f"{output_dir}/val_data.csv", index=False)
    test_df.to_csv(f"{output_dir}/test_data.csv", index=False)
    
    print(f"Saved datasets to {output_dir}")
    
    return train_df, val_df, test_df

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert TossCTR data to FESeq format')
    parser.add_argument('--train_path', type=str, help='Path to train.parquet file')
    parser.add_argument('--test_path', type=str, help='Path to test.parquet file (for inference)')
    parser.add_argument('--output_dir', type=str, help='Output directory for processed data')
    parser.add_argument('--output_path', type=str, help='Output file path (for single file)')
    parser.add_argument('--n_samples', type=int, default=10000, help='Number of samples to load')
    
    args = parser.parse_args()
    
    # 데이터 경로들을 시도해볼 순서대로 정의 (colab, 로컬)
    possible_train_paths = [
        "/content/drive/MyDrive/data/TossCTR/raw/train.parquet",  # Colab 구글 드라이브
        "/content/TossCTR/data/raw/train.parquet",                # Colab 로컬
        "/Users/hj/projects/TossCTR/data/raw/train.parquet"       # 로컬 환경
    ]
    
    possible_output_dirs = [
        "/content/TossCTR/colab_feseq/data/tossctr",             # Colab
        "/Users/hj/projects/TossCTR/models/FESeq/data/tossctr"   # 로컬
    ]
    
    # 입력 파일 경로 결정
    if args.train_path:
        parquet_path = args.train_path
    elif args.test_path:
        parquet_path = args.test_path
    else:
        # 가능한 경로들 중에서 존재하는 첫 번째 파일 사용
        parquet_path = None
        for path in possible_train_paths:
            if os.path.exists(path):
                parquet_path = path
                print(f"✅ Found data file: {path}")
                break
        
        if parquet_path is None:
            print("❌ No train.parquet file found in any of the expected locations:")
            for path in possible_train_paths:
                print(f"   - {path}")
            return
    
    # 출력 디렉토리 결정
    if args.output_dir:
        output_dir = args.output_dir
    elif args.output_path:
        output_dir = os.path.dirname(args.output_path)
    else:
        # 가능한 출력 디렉토리 중에서 사용 가능한 첫 번째 선택
        output_dir = None
        for dir_path in possible_output_dirs:
            try:
                os.makedirs(dir_path, exist_ok=True)
                output_dir = dir_path
                print(f"✅ Using output directory: {dir_path}")
                break
            except:
                continue
        
        if output_dir is None:
            output_dir = "/tmp/tossctr"  # 임시 디렉토리 사용
            os.makedirs(output_dir, exist_ok=True)
            print(f"⚠️  Using temporary directory: {output_dir}")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 데이터 로드
    df = load_tossctr_sample(parquet_path, n_samples=args.n_samples)
    
    # 2. FESeq 형식으로 변환
    df_feseq = convert_tossctr_to_feseq(df)
    
    # 3. 단일 파일 출력 (추론용) 또는 분할 저장
    if args.output_path:
        # 추론용 단일 파일 저장
        df_feseq.to_csv(args.output_path, index=False)
        print(f"✅ Saved inference data to: {args.output_path}")
    else:
        # 훈련용 분할 저장
        train_df, val_df, test_df = split_and_save_data(df_feseq, output_dir)
        
        # 4. 샘플 데이터 확인
        print("\n=== Sample data ===")
        print(train_df.head(3))
        print("\n=== Sequence samples ===")
        for i in range(3):
            seq = train_df.iloc[i]['seq_list']
            print(f"Row {i}: {seq[:50]}...")
    
    print("\n✅ Conversion completed successfully!")

if __name__ == "__main__":
    main()
