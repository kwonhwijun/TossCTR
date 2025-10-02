#!/usr/bin/env python3
"""
빠른 테스트를 위한 작은 데이터셋 생성 스크립트
"""
import h5py
import numpy as np
import pandas as pd
import os

def create_tiny_h5_datasets():
    """기존 H5 파일에서 작은 샘플을 추출하여 테스트용 파일 생성"""
    
    base_dir = "colab_feseq/data/tossctr"
    
    # 원본 파일들
    files = ['train.h5', 'valid.h5', 'test.h5']
    
    for filename in files:
        src_path = os.path.join(base_dir, filename)
        tiny_filename = filename.replace('.h5', '_tiny.h5')
        dst_path = os.path.join(base_dir, tiny_filename)
        
        if not os.path.exists(src_path):
            print(f"⚠️  원본 파일이 없습니다: {src_path}")
            continue
            
        print(f"📝 {filename} -> {tiny_filename} 생성 중...")
        
        try:
            # 원본 파일 열기
            with h5py.File(src_path, 'r') as src_f:
                # 데이터 크기 확인
                if len(src_f.keys()) == 0:
                    print(f"⚠️  {filename}이 비어있습니다.")
                    continue
                
                # 첫 번째 키의 데이터 길이 확인 (모든 필드가 같은 길이여야 함)
                first_key = list(src_f.keys())[0]
                total_samples = len(src_f[first_key])
                
                # 최대 1000개 샘플만 추출 (더 빠른 테스트)
                sample_size = min(1000, total_samples)
                print(f"  📊 전체 {total_samples}개 중 {sample_size}개 샘플 추출")
                
                # 작은 파일 생성
                with h5py.File(dst_path, 'w') as dst_f:
                    for key in src_f.keys():
                        data = src_f[key][:sample_size]  # 처음 sample_size개만 복사
                        dst_f.create_dataset(key, data=data, compression='gzip')
                        
                print(f"✅ {tiny_filename} 생성 완료!")
                
        except Exception as e:
            print(f"❌ {filename} 처리 중 오류: {e}")

def check_h5_files():
    """H5 파일들의 상태 확인"""
    base_dir = "colab_feseq/data/tossctr"
    files = ['train.h5', 'valid.h5', 'test.h5']
    
    print("📊 H5 파일 상태 확인:")
    for filename in files:
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            try:
                with h5py.File(filepath, 'r') as f:
                    if len(f.keys()) > 0:
                        first_key = list(f.keys())[0]
                        sample_count = len(f[first_key])
                        print(f"  ✅ {filename}: {sample_count:,}개 샘플, {len(f.keys())}개 필드")
                    else:
                        print(f"  ⚠️  {filename}: 빈 파일")
            except Exception as e:
                print(f"  ❌ {filename}: 읽기 오류 - {e}")
        else:
            print(f"  ❌ {filename}: 파일 없음")

if __name__ == "__main__":
    print("🚀 빠른 테스트용 작은 데이터셋 생성기")
    print("=" * 50)
    
    # H5 파일 상태 확인
    check_h5_files()
    
    print("\n📝 작은 테스트 데이터셋 생성 중...")
    create_tiny_h5_datasets()
    
    print("\n✅ 완료! 이제 다음 명령어로 빠른 테스트를 실행하세요:")
    print("cd colab_feseq/model_zoo/FESeq")
    print("python run_expid.py --config ./config --expid FESeq_tossctr_quick_test --gpu -1")

