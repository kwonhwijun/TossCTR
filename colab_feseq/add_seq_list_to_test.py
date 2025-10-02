#!/usr/bin/env python3
"""
기존 test.h5 파일에 seq_list 데이터를 추가하는 스크립트
"""

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import os

def preprocess_sequence(seq_str, max_len=50):
    """시퀀스 데이터 전처리"""
    if pd.isna(seq_str) or seq_str == '':
        return ''
    
    # 쉼표로 분리된 시퀀스를 ^ 구분자로 변경하고 길이 제한
    tokens = str(seq_str).split(',')[:max_len]
    return '^'.join(tokens)

def load_seq_from_parquet(parquet_path, max_len=50):
    """parquet 파일에서 seq 컬럼을 로드하고 seq_list로 변환"""
    print(f"Loading seq column from {parquet_path}")
    
    # seq 컬럼만 로드
    df = pd.read_parquet(parquet_path, columns=['seq'])
    
    # seq -> seq_list 변환
    seq_list = df['seq'].apply(lambda x: preprocess_sequence(x, max_len))
    
    print(f"Loaded {len(seq_list)} sequences")
    return seq_list

def encode_sequences(seq_list, vocab_size=50, max_len=50):
    """시퀀스를 정수 배열로 인코딩"""
    print("Encoding sequences...")
    
    # 모든 토큰 수집
    all_tokens = set()
    for seq in seq_list:
        if seq:
            tokens = seq.split('^')
            all_tokens.update(tokens)
    
    # 어휘 생성 (상위 vocab_size-2개 토큰 사용, PAD와 OOV 제외)
    token_list = list(all_tokens)[:vocab_size-2]
    vocab = ['__PAD__', '__OOV__'] + token_list
    token_to_idx = {token: idx for idx, token in enumerate(vocab)}
    
    print(f"Built vocabulary with {len(vocab)} tokens")
    
    # 시퀀스 인코딩
    encoded_sequences = []
    for seq in seq_list:
        if pd.isna(seq) or seq == '':
            # 빈 시퀀스는 패딩으로 채움
            encoded_sequences.append([token_to_idx['__PAD__']] * max_len)
        else:
            tokens = seq.split('^')[:max_len]
            # 토큰을 인덱스로 변환
            token_ids = [token_to_idx.get(token, token_to_idx['__OOV__']) for token in tokens]
            # 패딩 추가
            if len(token_ids) < max_len:
                token_ids.extend([token_to_idx['__PAD__']] * (max_len - len(token_ids)))
            encoded_sequences.append(token_ids)
    
    return np.array(encoded_sequences, dtype=np.int32)

def add_seq_list_to_h5(h5_path, seq_array):
    """기존 H5 파일에 seq_list 데이터 추가"""
    print(f"Adding seq_list to {h5_path}")
    
    with h5py.File(h5_path, 'a') as f:  # 'a' 모드로 열어서 추가
        # seq_list가 이미 있으면 삭제
        if 'seq_list' in f:
            del f['seq_list']
        
        # seq_list 데이터 추가
        f.create_dataset('seq_list', data=seq_array, compression='gzip')
        print(f"Added seq_list dataset with shape {seq_array.shape}")
        
        # 기존 키들 확인
        print("H5 file contents:")
        for key in f.keys():
            print(f"  {key}: {f[key].shape}")

def main():
    # 경로 설정
    base_dir = "/Users/hj/projects/TossCTR"
    parquet_path = os.path.join(base_dir, "data/raw/test.parquet")
    h5_path = os.path.join(base_dir, "colab_feseq/data/tossctr/test.h5")
    
    # 파일 존재 확인
    if not os.path.exists(parquet_path):
        print(f"❌ Parquet file not found: {parquet_path}")
        return
    
    if not os.path.exists(h5_path):
        print(f"❌ H5 file not found: {h5_path}")
        return
    
    try:
        # 1. parquet에서 seq 데이터 로드
        seq_list = load_seq_from_parquet(parquet_path)
        
        # 2. 시퀀스 인코딩
        seq_array = encode_sequences(seq_list)
        
        # 3. H5 파일에 추가
        add_seq_list_to_h5(h5_path, seq_array)
        
        print("✅ Successfully added seq_list to test.h5")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

