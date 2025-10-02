#!/usr/bin/env python3
"""
TossCTR Parquet 데이터를 H5 형식으로 직접 변환하는 스크립트
메모리 효율적인 대용량 데이터 처리 지원 (개선된 버전)
"""
import os
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import h5py
import logging
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from typing import Dict, List, Tuple, Optional, Generator
import yaml
import json
import pickle
from collections import Counter
import gc

# fuxictr imports
import sys
sys.path.append('/home/hj/TossCTR/colab_feseq')
from fuxictr.preprocess.build_dataset import save_h5

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TossCTRParquetProcessor:
    """TossCTR Parquet 데이터를 H5로 직접 변환하는 프로세서 (스트리밍 최적화)"""

    def __init__(self, config_path: str, data_root: str = "data"):
        self.config_path = config_path
        self.data_root = data_root
        self.config = self._load_config()
        self.dataset_config = self.config['tossctr_dataset']
        self.output_dir = os.path.join(data_root, 'tossctr')
        os.makedirs(self.output_dir, exist_ok=True)
        self.feature_cols = self.dataset_config['feature_cols']
        self.label_col = self.dataset_config['label_col']
        self.min_categr_count = self.dataset_config.get('min_categr_count', 1)
        self._classify_features()
        self.encoders = {}
        self.scalers = {}
        self.fitted = False

    def _load_config(self) -> Dict:
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _classify_features(self):
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

    def preprocess_sequence(self, seq_str: str, max_len: int = 50) -> str:
        if pd.isna(seq_str) or seq_str == '':
            return ''
        tokens = str(seq_str).split(',')[:max_len]
        return '^'.join(tokens)

    # --- 개선된 핵심 로직: 스트리밍 처리 ---

    def iter_parquet_chunks(self, parquet_path: str, chunk_size: int, columns: Optional[List[str]] = None) -> Generator[pd.DataFrame, None, None]:
        """Parquet 파일을 안전하게 청크 단위로 읽어오는 제너레이터"""
        pf = pq.ParquetFile(parquet_path)
        if columns is None:
            columns = pf.schema.names
        
        # Parquet 파일에 실제 존재하는 컬럼만 읽도록 필터링
        schema_cols = set(pf.schema.names)
        read_cols = [c for c in columns if c in schema_cols]
        # 'seq' 컬럼이 필요한 경우 추가
        if 'seq' in schema_cols and 'seq' not in read_cols and any('seq' in c for c in self.sequence_features):
             read_cols.append('seq')
        
        logging.info(f"Reading columns: {read_cols}")
        for batch in pf.iter_batches(batch_size=chunk_size, columns=read_cols):
            chunk_df = batch.to_pandas()
            # seq 컬럼으로부터 시퀀스 피처 파생
            for col in self.sequence_features:
                if col not in chunk_df.columns and 'seq' in chunk_df.columns:
                    chunk_df[col] = chunk_df['seq'].apply(lambda x: self.preprocess_sequence(x, max_len=500)) # preprocess_sequence는 이미 정의됨
            yield chunk_df

    def fit_processors_from_chunks(self, parquet_path: str, chunk_size: int):
        """(Pass 1) Parquet 청크를 순회하며 전처리기 학습"""
        logging.info("--- Starting Pass 1: Fitting Preprocessors ---")
        
        # 임시 저장소 초기화
        categr_vocabs = {col: set() for col in self.categorical_features}
        seq_vocabs = {col: Counter() for col in self.sequence_features}
        numeric_stats = {col: {'min': np.inf, 'max': -np.inf} for col in self.numeric_features}

        all_cols = self.categorical_features + self.numeric_features + self.sequence_features
        
        for i, chunk_df in enumerate(self.iter_parquet_chunks(parquet_path, chunk_size, all_cols)):
            logging.info(f"Fitting on chunk {i+1}...")
            # Categorical: 고유값 수집
            for col in self.categorical_features:
                if col in chunk_df.columns:
                    categr_vocabs[col].update(chunk_df[col].dropna().astype(str).unique())
            
            # Numeric: 최소/최대값 추적
            for col in self.numeric_features:
                if col in chunk_df.columns:
                    col_min, col_max = chunk_df[col].min(), chunk_df[col].max()
                    numeric_stats[col]['min'] = min(numeric_stats[col]['min'], col_min)
                    numeric_stats[col]['max'] = max(numeric_stats[col]['max'], col_max)

            # Sequence: 토큰 빈도 수집
            for col in self.sequence_features:
                if col in chunk_df.columns:
                    tokens = chunk_df[col].dropna().str.split('^').explode()
                    seq_vocabs[col].update(tokens[tokens != ''])
            
            gc.collect()

        # 학습된 정보로 최종 전처리기 생성
        logging.info("Finalizing preprocessors...")
        for col, vocab_set in categr_vocabs.items():
            le = LabelEncoder().fit(list(vocab_set) + ['unknown', 'rare'])
            self.encoders[col] = le
        
        for col, stats in numeric_stats.items():
            scaler = MinMaxScaler()
            scaler.fit(np.array([[stats['min']], [stats['max']]]))
            self.scalers[col] = scaler

        for col, token_counts in seq_vocabs.items():
            valid_tokens = [token for token, count in token_counts.items() if count >= self.min_categr_count]
            vocab = ['__PAD__', '__OOV__'] + valid_tokens
            self.encoders[col] = {
                'vocab': {token: idx for idx, token in enumerate(vocab)},
                'vocab_size': len(vocab),
                'max_len': 50
            }
        self.fitted = True
        logging.info("--- Pass 1 Finished ---")
    
    def transform_chunks_to_h5(self, parquet_path: str, chunk_size: int, split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)):
        """(Pass 2) Parquet 청크를 변환하고 H5 파일에 바로 저장"""
        if not self.fitted:
            raise RuntimeError("Processors have not been fitted. Run fit_processors_from_chunks first.")
        
        logging.info("--- Starting Pass 2: Transforming and Saving to H5 ---")
        
        # H5 파일 및 데이터셋 초기화
        h5_paths = {
            "train": os.path.join(self.output_dir, "train.h5"),
            "valid": os.path.join(self.output_dir, "valid.h5"),
            "test": os.path.join(self.output_dir, "test.h5"),
        }
        h5_files = {name: h5py.File(path, 'w') for name, path in h5_paths.items()}
        
        # 데이터 분할을 위한 인덱스 생성
        pf = pq.ParquetFile(parquet_path)
        total_rows = pf.num_row_groups * pf.metadata.row_group(0).num_rows # Approximate for simplicity
        indices = np.arange(total_rows)
        np.random.shuffle(indices)
        
        train_end = int(total_rows * split_ratios[0])
        valid_end = train_end + int(total_rows * split_ratios[1])
        
        split_map = np.empty(total_rows, dtype=object)
        split_map[:train_end] = "train"
        split_map[train_end:valid_end] = "valid"
        split_map[valid_end:] = "test"

        # 첫 번째 청크로 데이터셋 구조 결정 및 H5 파일 생성
        first_chunk = next(self.iter_parquet_chunks(parquet_path, chunk_size, list(self.encoders.keys()) + list(self.scalers.keys()) + [self.label_col['name']]))
        transformed_first_chunk = self.transform_dataframe(first_chunk)

        for name, h5_file in h5_files.items():
            for key, arr in transformed_first_chunk.items():
                shape = (0,) + arr.shape[1:]
                maxshape = (None,) + arr.shape[1:]
                h5_file.create_dataset(key, shape=shape, maxshape=maxshape, dtype=arr.dtype, chunks=True)

        # 전체 데이터를 순회하며 변환 및 저장
        current_idx = 0
        all_cols = list(self.encoders.keys()) + list(self.scalers.keys()) + [self.label_col['name']]
        
        for i, chunk_df in enumerate(self.iter_parquet_chunks(parquet_path, chunk_size, all_cols)):
            logging.info(f"Transforming chunk {i+1}...")
            
            # 현재 청크의 인덱스에 해당하는 split 가져오기
            chunk_end_idx = current_idx + len(chunk_df)
            chunk_splits = split_map[current_idx:chunk_end_idx]
            current_idx = chunk_end_idx
            
            # 데이터 변환
            arrays = self.transform_dataframe(chunk_df)

            # split에 따라 해당 H5 파일에 추가
            for split_name in ["train", "valid", "test"]:
                mask = (chunk_splits == split_name)
                if not np.any(mask):
                    continue

                for key, arr in arrays.items():
                    data_to_append = arr[mask]
                    dset = h5_files[split_name][key]
                    dset.resize(dset.shape[0] + len(data_to_append), axis=0)
                    dset[-len(data_to_append):] = data_to_append
            
            gc.collect()

        # 파일 닫기
        for h5_file in h5_files.values():
            h5_file.close()
        
        logging.info("--- Pass 2 Finished ---")

    def process_full_pipeline_stream(self, train_parquet_path: str, chunk_size: int = 50000):
        """메모리 효율적인 전체 파이프라인 실행"""
        logging.info("Starting memory-efficient streaming pipeline...")
        # 1. (Pass 1) 전처리기 학습
        self.fit_processors_from_chunks(train_parquet_path, chunk_size)
        
        # 2. 전처리기 및 feature_map 저장
        self.save_processors() # 기존 함수 재사용
        self.save_feature_map() # 기존 함수 재사용

        # 3. (Pass 2) 데이터 변환 및 H5 저장 (분할 포함)
        self.transform_chunks_to_h5(train_parquet_path, chunk_size)

        logging.info("✅ Memory-efficient streaming pipeline completed!")

    # --- 기존 함수들 (transform_*, save_* 등)은 여기에 그대로 유지 ---
    # (코드가 길어져 생략, 제공해주신 원본 코드의 함수들을 그대로 사용하면 됩니다)
    def transform_dataframe(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        arrays = {}
        arrays.update(self.transform_categorical_features(df))
        arrays.update(self.transform_numeric_features(df))
        arrays.update(self.transform_sequence_features(df))
        label_col = self.label_col['name']
        if label_col in df.columns:
            arrays[label_col] = df[label_col].astype(np.float32).values
        return arrays

    def transform_categorical_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        arrays = {}
        for col in self.categorical_features:
            if col in df.columns and col in self.encoders:
                df[col] = df[col].fillna('unknown').astype(str)
                le = self.encoders[col]
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'rare')
                try:
                    arrays[col] = le.transform(df[col]).astype(np.int32)
                except ValueError:
                    df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                    arrays[col] = le.transform(df[col]).astype(np.int32)
        return arrays

    def transform_numeric_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        arrays = {}
        for col in self.numeric_features:
            if col in df.columns and col in self.scalers:
                df[col] = df[col].fillna(0.0).astype(float)
                scaler = self.scalers[col]
                arrays[col] = scaler.transform(df[[col]]).astype(np.float32).flatten()
        return arrays

    def transform_sequence_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        arrays = {}
        for col in self.sequence_features:
            if col in df.columns and col in self.encoders:
                encoder_info = self.encoders[col]
                vocab = encoder_info['vocab']
                max_len = encoder_info['max_len']
                sequences = []
                for seq in df[col]:
                    if pd.isna(seq) or seq == '':
                        sequences.append([vocab['__PAD__']] * max_len)
                    else:
                        tokens = seq.split('^')[:max_len]
                        token_ids = [vocab.get(token, vocab['__OOV__']) for token in tokens]
                        if len(token_ids) < max_len:
                            token_ids.extend([vocab['__PAD__']] * (max_len - len(token_ids)))
                        sequences.append(token_ids)
                arrays[col] = np.array(sequences, dtype=np.int32)
        return arrays

    def save_feature_map(self):
        features_list = []
        num_fields = 0
        total_features = 0
        for col in self.categorical_features:
            if col in self.encoders:
                vocab_size = len(self.encoders[col].classes_)
                spec = {"type": "categorical", "vocab_size": vocab_size}
                features_list.append({col: spec})
                total_features += vocab_size
                num_fields += 1
        for col in self.numeric_features:
            if col in self.scalers:
                spec = {"type": "numeric"}
                features_list.append({col: spec})
                total_features += 1
                num_fields += 1
        for col in self.sequence_features:
            if col in self.encoders:
                encoder_info = self.encoders[col]
                spec = {
                    "type": "sequence", "dtype": "str",
                    "vocab_size": encoder_info['vocab_size'],
                    "max_len": encoder_info['max_len'],
                    "share_embedding": "inventory_id", "feature_encoder": None
                }
                features_list.append({col: spec})
                total_features += encoder_info['vocab_size']
                num_fields += 1
        feature_map = {
            "dataset_id": "tossctr_dataset", "num_fields": num_fields,
            "total_features": total_features, "input_length": 0,
            "features": features_list, "labels": [self.label_col['name']]
        }
        feature_map_path = os.path.join(self.output_dir, "feature_map.json")
        with open(feature_map_path, 'w') as f:
            json.dump(feature_map, f, indent=2)
        logging.info(f"Saved feature_map.json: {feature_map['num_fields']} fields, {feature_map['total_features']} features")

    def save_processors(self):
        processor_data = {
            'encoders': self.encoders, 'scalers': self.scalers,
            'categorical_features': self.categorical_features,
            'numeric_features': self.numeric_features,
            'sequence_features': self.sequence_features
        }
        processor_path = os.path.join(self.output_dir, "feature_processor.pkl")
        with open(processor_path, 'wb') as f:
            pickle.dump(processor_data, f)
        logging.info(f"Saved processors to {processor_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convert TossCTR parquet to H5 format (Memory-Efficient)')
    parser.add_argument('--train_path', type=str, required=True, help='Path to train.parquet file')
    parser.add_argument('--config_path', type=str, required=True, help='Path to dataset config YAML file')
    parser.add_argument('--data_root', type=str, default='data', help='Data root directory')
    parser.add_argument('--chunk_size', type=int, default=100000, help='Chunk size for processing large files')
    
    args = parser.parse_args()
    
    processor = TossCTRParquetProcessor(config_path=args.config_path, data_root=args.data_root)
    # 새로 만든 스트리밍 파이프라인 함수 호출
    processor.process_full_pipeline_stream(train_parquet_path=args.train_path, chunk_size=args.chunk_size)

if __name__ == "__main__":
    main()