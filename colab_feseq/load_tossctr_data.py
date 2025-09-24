#!/usr/bin/env python3
"""
TossCTR 원본 데이터를 Colab에서 로드하고 전처리하는 스크립트
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
from pathlib import Path

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

def load_and_process_tossctr(data_path: str, n_samples: int = 10000) -> pd.DataFrame:
    """TossCTR 데이터 로드 및 전처리"""
    
    print(f"Loading {n_samples} samples from {data_path}")
    
    # 데이터 로드 (CSV 또는 Parquet)
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path).head(n_samples)
    else:
        df = pd.read_csv(data_path).head(n_samples)
    
    print(f"Loaded shape: {df.shape}")
    
    # 피처 분류
    categorical_cols = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']
    
    # 수치형 피처 선택 (history 제외)
    numeric_cols = [col for col in df.columns 
                   if any(col.startswith(prefix) for prefix in ['l_feat_', 'feat_e_', 'feat_d_', 'feat_c_', 'feat_b_', 'feat_a_'])
                   and not col.startswith('history_')]
    
    print(f"Categorical features: {len(categorical_cols)}")
    print(f"Numeric features: {len(numeric_cols)}")
    
    # 범주형 피처 인코딩
    df_processed = encode_categorical_features(df, categorical_cols)
    
    # 수치형 피처 결측치 처리
    for col in numeric_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna(0.0)
    
    # 시퀀스 피처 처리
    df_processed['seq_list'] = df_processed['seq'].apply(lambda x: preprocess_sequence(x, max_len=50))
    
    # 필요한 컬럼만 선택
    selected_cols = categorical_cols + numeric_cols + ['seq_list', 'clicked']
    available_cols = [col for col in selected_cols if col in df_processed.columns]
    df_final = df_processed[available_cols].copy()
    
    print(f"Final shape: {df_final.shape}")
    print(f"Final columns: {list(df_final.columns)}")
    
    return df_final

def save_processed_data(df: pd.DataFrame, output_dir: str = "data/tossctr"):
    """전처리된 데이터를 train/val/test로 분할하여 저장"""
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 균형잡힌 분할
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
    train_df.to_csv(f"{output_dir}/train_data.csv", index=False)
    val_df.to_csv(f"{output_dir}/val_data.csv", index=False)
    test_df.to_csv(f"{output_dir}/test_data.csv", index=False)
    
    print(f"✅ Saved datasets to {output_dir}")
    
    return train_df, val_df, test_df

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process TossCTR data for FESeq")
    parser.add_argument("--data_path", required=True, help="Path to TossCTR data file")
    parser.add_argument("--n_samples", type=int, default=10000, help="Number of samples to use")
    parser.add_argument("--output_dir", default="data/tossctr", help="Output directory")
    
    args = parser.parse_args()
    
    # 데이터 로드 및 전처리
    df = load_and_process_tossctr(args.data_path, args.n_samples)
    
    # 저장
    save_processed_data(df, args.output_dir)
    
    print("✅ Data preprocessing completed!")

if __name__ == "__main__":
    main()

