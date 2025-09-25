#!/usr/bin/env python3
"""
TossCTR Parquet 데이터를 H5 형식으로 직접 변환하는 스크립트
메모리 효율적인 대용량 데이터 처리 지원
"""

import os
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import h5py
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from typing import Dict, List, Tuple, Optional
import yaml
import json
import pickle
from collections import Counter
import gc

# fuxictr imports
import sys
sys.path.append('/Users/hj/projects/TossCTR/colab_feseq')
from fuxictr.preprocess.feature_processor import FeatureProcessor
from fuxictr.preprocess.build_dataset import save_h5

logging.basicConfig(level=logging.INFO)

class TossCTRParquetProcessor:
    """TossCTR Parquet 데이터를 H5로 직접 변환하는 프로세서"""
    
    def __init__(self, config_path: str, data_root: str = "data"):
        """
        Args:
            config_path: dataset config YAML 파일 경로
            data_root: 데이터 루트 디렉토리
        """
        self.config_path = config_path
        self.data_root = data_root
        self.config = self._load_config()
        self.dataset_config = self.config['tossctr_dataset']
        
        # 출력 디렉토리 설정
        self.output_dir = os.path.join(data_root, 'tossctr')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 피처 설정 로드
        self.feature_cols = self.dataset_config['feature_cols']
        self.label_col = self.dataset_config['label_col']
        self.min_categr_count = self.dataset_config.get('min_categr_count', 1)
        
        # 피처 분류
        self._classify_features()
        
        # 전처리기 딕셔너리
        self.encoders = {}
        self.scalers = {}
        
    def _load_config(self) -> Dict:
        """설정 파일 로드"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _classify_features(self):
        """피처를 타입별로 분류"""
        self.categorical_features = []
        self.numeric_features = []
        self.sequence_features = []
        
        for feat_group in self.feature_cols:
            feat_type = feat_group['type']
            feat_names = feat_group['name']
            
            if isinstance(feat_names, str):
                feat_names = [feat_names]
                
            if feat_type == 'categorical':
                self.categorical_features.extend(feat_names)
            elif feat_type == 'numeric':
                self.numeric_features.extend(feat_names)
            elif feat_type == 'sequence':
                self.sequence_features.extend(feat_names)
        
        logging.info(f"Categorical features: {len(self.categorical_features)}")
        logging.info(f"Numeric features: {len(self.numeric_features)}")
        logging.info(f"Sequence features: {len(self.sequence_features)}")
    
    def _ensure_sequence_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """시퀀스 컬럼 보정: parquet에 seq_list가 없고 seq만 있는 경우 생성"""
        # seq_list가 필요한데 없고, seq가 있으면 파생 생성
        for col in self.sequence_features:
            if col not in df.columns:
                # 흔히 원본에 'seq'로 존재함
                if 'seq' in df.columns:
                    logging.info(f"Deriving missing sequence column '{col}' from 'seq'")
                    df[col] = df['seq'].apply(lambda x: self.preprocess_sequence(x, max_len=50))
                else:
                    # 완전 없는 경우 빈 시퀀스 채움
                    logging.warning(f"Sequence column '{col}' not found. Filling with PAD-only sequences.")
                    df[col] = ''
        return df

    def load_parquet_chunks(self, parquet_path: str, chunk_size: int = 50000) -> pd.DataFrame:
        """Parquet 파일을 청크 단위로 로드 (존재하는 컬럼만 안전하게 선택)"""
        logging.info(f"Loading parquet file: {parquet_path}")
        
        pf = pq.ParquetFile(parquet_path)
        total_rows = pf.metadata.num_rows
        logging.info(f"Total rows: {total_rows:,}")
        
        # parquet에 실제 존재하는 컬럼으로 필터링
        schema_cols = set(pf.schema.names)
        required_cols = set(self.categorical_features + self.numeric_features + self.sequence_features + [self.label_col['name']])
        read_cols = [c for c in required_cols if c in schema_cols]
        # seq_list가 필요하지만 없음 + 'seq'가 있으면 함께 읽어서 파생
        if any((c not in schema_cols) for c in self.sequence_features) and ('seq' in schema_cols):
            if 'seq' not in read_cols:
                read_cols.append('seq')
        
        chunks = []
        for batch in pf.iter_batches(batch_size=chunk_size, columns=read_cols):
            chunk_df = batch.to_pandas()
            chunks.append(chunk_df)
        
        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        # 시퀀스 컬럼 보정
        if not df.empty:
            df = self._ensure_sequence_columns(df)
        logging.info(f"Loaded shape: {df.shape}")
        return df
    
    def preprocess_sequence(self, seq_str: str, max_len: int = 50) -> str:
        """시퀀스 데이터 전처리"""
        if pd.isna(seq_str) or seq_str == '':
            return ''
        
        # 쉼표로 분리된 시퀀스를 ^ 구분자로 변경하고 길이 제한
        tokens = str(seq_str).split(',')[:max_len]
        return '^'.join(tokens)
    
    def fit_categorical_features(self, df: pd.DataFrame):
        """범주형 피처 인코더 학습"""
        logging.info("Fitting categorical features...")
        
        for col in self.categorical_features:
            if col in df.columns:
                # 결측치 처리
                df[col] = df[col].fillna('unknown').astype(str)
                
                # 빈도 기반 필터링
                value_counts = df[col].value_counts()
                valid_values = value_counts[value_counts >= self.min_categr_count].index.tolist()
                
                # 저빈도 값들을 'rare'로 변경
                df[col] = df[col].apply(lambda x: x if x in valid_values else 'rare')
                
                # LabelEncoder 학습
                le = LabelEncoder()
                le.fit(df[col])
                self.encoders[col] = le
                
                logging.info(f"Encoded {col}: {len(le.classes_)} unique values")
    
    def fit_numeric_features(self, df: pd.DataFrame):
        """수치형 피처 스케일러 학습"""
        logging.info("Fitting numeric features...")
        
        for col in self.numeric_features:
            if col in df.columns:
                # 결측치를 0으로 처리
                df[col] = df[col].fillna(0.0).astype(float)
                
                # MinMaxScaler 학습
                scaler = MinMaxScaler()
                scaler.fit(df[[col]])
                self.scalers[col] = scaler
                
                logging.info(f"Fitted scaler for {col}")
    
    def fit_sequence_features(self, df: pd.DataFrame):
        """시퀀스 피처 전처리 및 어휘 구축"""
        logging.info("Fitting sequence features...")
        
        for col in self.sequence_features:
            if col in df.columns:
                # 시퀀스 전처리
                max_len = 50  # config에서 가져올 수 있음
                df[col] = df[col].apply(lambda x: self.preprocess_sequence(x, max_len))
                
                # 어휘 구축 (시퀀스의 모든 토큰 수집)
                all_tokens = []
                for seq in df[col]:
                    if seq:
                        tokens = seq.split('^')
                        all_tokens.extend(tokens)
                
                # 빈도 기반 어휘 구축
                token_counts = Counter(all_tokens)
                valid_tokens = [token for token, count in token_counts.items() 
                              if count >= self.min_categr_count]
                
                # 특수 토큰 추가
                vocab = ['__PAD__', '__OOV__'] + valid_tokens
                token_to_idx = {token: idx for idx, token in enumerate(vocab)}
                
                self.encoders[col] = {
                    'vocab': token_to_idx,
                    'vocab_size': len(vocab),
                    'max_len': max_len
                }
                
                logging.info(f"Built vocab for {col}: {len(vocab)} tokens")
    
    def transform_categorical_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """범주형 피처 변환"""
        arrays = {}
        
        for col in self.categorical_features:
            if col in df.columns and col in self.encoders:
                # 결측치 처리
                df[col] = df[col].fillna('unknown').astype(str)
                
                # 학습 시 보지 못한 값들을 'rare'로 처리
                le = self.encoders[col]
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'rare')
                
                # 변환
                try:
                    arrays[col] = le.transform(df[col]).astype(np.int32)
                except ValueError:
                    # 'rare' 값이 학습 시 없었다면 0으로 처리
                    df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                    arrays[col] = le.transform(df[col]).astype(np.int32)
        
        return arrays
    
    def transform_numeric_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """수치형 피처 변환"""
        arrays = {}
        
        for col in self.numeric_features:
            if col in df.columns and col in self.scalers:
                # 결측치 처리
                df[col] = df[col].fillna(0.0).astype(float)
                
                # 스케일링
                scaler = self.scalers[col]
                arrays[col] = scaler.transform(df[[col]]).astype(np.float32).flatten()
        
        return arrays
    
    def transform_sequence_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """시퀀스 피처 변환"""
        arrays = {}
        
        for col in self.sequence_features:
            if col in df.columns and col in self.encoders:
                encoder_info = self.encoders[col]
                vocab = encoder_info['vocab']
                max_len = encoder_info['max_len']
                
                # 시퀀스를 숫자 배열로 변환
                sequences = []
                for seq in df[col]:
                    if pd.isna(seq) or seq == '':
                        # 빈 시퀀스는 패딩으로 채움
                        sequences.append([vocab['__PAD__']] * max_len)
                    else:
                        tokens = seq.split('^')[:max_len]
                        # 토큰을 인덱스로 변환
                        token_ids = [vocab.get(token, vocab['__OOV__']) for token in tokens]
                        # 패딩 추가
                        if len(token_ids) < max_len:
                            token_ids.extend([vocab['__PAD__']] * (max_len - len(token_ids)))
                        sequences.append(token_ids)
                
                arrays[col] = np.array(sequences, dtype=np.int32)
        
        return arrays
    
    def transform_dataframe(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """데이터프레임을 H5 배열로 변환"""
        arrays = {}
        
        # 각 피처 타입별로 변환
        arrays.update(self.transform_categorical_features(df))
        arrays.update(self.transform_numeric_features(df))
        arrays.update(self.transform_sequence_features(df))
        
        # 레이블 변환
        label_col = self.label_col['name']
        if label_col in df.columns:
            arrays[label_col] = df[label_col].astype(np.float32).values
        
        return arrays
    
    def save_feature_map(self):
        """feature_map.json 파일 생성 (FuxiCTR 포맷 호환)"""
        features_list = []  # FuxiCTR는 [{name: spec}, ...] 포맷 사용
        num_fields = 0
        total_features = 0

        # 범주형 피처
        for col in self.categorical_features:
            if col in self.encoders:
                vocab_size = len(self.encoders[col].classes_)
                spec = {
                    "type": "categorical",
                    "vocab_size": vocab_size
                }
                features_list.append({col: spec})
                total_features += vocab_size
                num_fields += 1

        # 수치형 피처
        for col in self.numeric_features:
            if col in self.scalers:
                spec = {
                    "type": "numeric"
                }
                features_list.append({col: spec})
                total_features += 1
                num_fields += 1

        # 시퀀스 피처
        for col in self.sequence_features:
            if col in self.encoders:
                encoder_info = self.encoders[col]
                spec = {
                    "type": "sequence",
                    "dtype": "str",
                    "vocab_size": encoder_info['vocab_size'],
                    "max_len": encoder_info['max_len'],
                    # 토큰 임베딩은 inventory_id와 공유
                    "share_embedding": "inventory_id",
                    "feature_encoder": None
                }
                features_list.append({col: spec})
                total_features += encoder_info['vocab_size']
                num_fields += 1

        feature_map = {
            "dataset_id": "tossctr_dataset",
            "num_fields": num_fields,
            "total_features": total_features,
            "input_length": 0,  # FeatureMap.load에서 재계산됨
            "features": features_list,
            "labels": [self.label_col['name']]
        }

        feature_map_path = os.path.join(self.output_dir, "feature_map.json")
        with open(feature_map_path, 'w') as f:
            json.dump(feature_map, f, indent=2)
        logging.info(f"Saved feature_map.json: {feature_map['num_fields']} fields, {feature_map['total_features']} features")
    
    def save_processors(self):
        """전처리기들을 pickle로 저장"""
        processor_data = {
            'encoders': self.encoders,
            'scalers': self.scalers,
            'categorical_features': self.categorical_features,
            'numeric_features': self.numeric_features,
            'sequence_features': self.sequence_features
        }
        
        processor_path = os.path.join(self.output_dir, "feature_processor.pkl")
        with open(processor_path, 'wb') as f:
            pickle.dump(processor_data, f)
        
        logging.info(f"Saved processors to {processor_path}")
    
    def process_and_save_h5(self, df: pd.DataFrame, filename: str):
        """데이터프레임을 H5로 변환하여 저장"""
        logging.info(f"Processing and saving {filename}...")
        
        # 변환
        arrays = self.transform_dataframe(df)
        
        # H5 저장
        h5_path = os.path.join(self.output_dir, f"{filename}.h5")
        save_h5(arrays, h5_path)
        
        logging.info(f"Saved {filename}.h5 with {len(df)} samples")
    
    def split_data(self, df: pd.DataFrame, 
                  train_ratio: float = 0.7, 
                  val_ratio: float = 0.15, 
                  test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """데이터를 train/val/test로 분할"""
        logging.info("Splitting data...")
        
        # 클릭률 균형을 위해 stratify 사용
        train_df, temp_df = train_test_split(
            df, test_size=(val_ratio + test_ratio), 
            random_state=42, stratify=df[self.label_col['name']]
        )
        
        val_df, test_df = train_test_split(
            temp_df, test_size=test_ratio/(val_ratio + test_ratio), 
            random_state=42, stratify=temp_df[self.label_col['name']]
        )
        
        logging.info(f"Split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        logging.info(f"Click rates - Train: {train_df[self.label_col['name']].mean():.3f}, "
                    f"Val: {val_df[self.label_col['name']].mean():.3f}, "
                    f"Test: {test_df[self.label_col['name']].mean():.3f}")
        
        return train_df, val_df, test_df
    
    def process_full_pipeline(self, train_parquet_path: str, 
                            chunk_size: int = 50000,
                            n_samples: Optional[int] = None):
        """전체 파이프라인 실행"""
        logging.info("Starting full pipeline...")
        
        # 1. 데이터 로드
        df = self.load_parquet_chunks(train_parquet_path, chunk_size)
        
        # 샘플링 (필요한 경우)
        if n_samples and n_samples < len(df):
            df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
            logging.info(f"Sampled {n_samples} rows")
        
        # 2. 데이터 분할
        train_df, val_df, test_df = self.split_data(df)
        
        # 3. 전처리기 학습 (train 데이터로만)
        logging.info("Fitting preprocessors on training data...")
        self.fit_categorical_features(train_df.copy())
        self.fit_numeric_features(train_df.copy())
        self.fit_sequence_features(train_df.copy())
        
        # 4. 전처리기 및 feature_map 저장
        self.save_processors()
        self.save_feature_map()
        
        # 5. 각 데이터셋을 H5로 변환 및 저장
        self.process_and_save_h5(train_df, "train")
        self.process_and_save_h5(val_df, "valid")
        self.process_and_save_h5(test_df, "test")
        
        logging.info("✅ Full pipeline completed!")
        
        # 메모리 정리
        del df, train_df, val_df, test_df
        gc.collect()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert TossCTR parquet to H5 format')
    parser.add_argument('--train_path', type=str, required=True, 
                       help='Path to train.parquet file')
    parser.add_argument('--config_path', type=str, required=True,
                       help='Path to dataset config YAML file')
    parser.add_argument('--data_root', type=str, default='data',
                       help='Data root directory')
    parser.add_argument('--chunk_size', type=int, default=50000,
                       help='Chunk size for processing large files')
    parser.add_argument('--n_samples', type=int, default=None,
                       help='Number of samples to use (None for all)')
    
    args = parser.parse_args()
    
    # 프로세서 초기화
    processor = TossCTRParquetProcessor(
        config_path=args.config_path,
        data_root=args.data_root
    )
    
    # 전체 파이프라인 실행
    processor.process_full_pipeline(
        train_parquet_path=args.train_path,
        chunk_size=args.chunk_size,
        n_samples=args.n_samples
    )


if __name__ == "__main__":
    main()
