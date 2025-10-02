#!/usr/bin/env python3
"""
초고속 테스트를 위한 극소 데이터셋 생성 스크립트
"""
import h5py
import numpy as np
import os

def create_micro_h5_datasets():
    """극소 데이터셋 생성 - 단 10개 샘플만"""
    
    base_dir = "colab_feseq/data/tossctr"
    
    # 원본 파일들
    files = ['train.h5', 'valid.h5', 'test.h5']
    
    for filename in files:
        src_path = os.path.join(base_dir, filename)
        micro_filename = filename.replace('.h5', '_micro.h5')
        dst_path = os.path.join(base_dir, micro_filename)
        
        if not os.path.exists(src_path):
            print(f"⚠️  원본 파일이 없습니다: {src_path}")
            continue
            
        print(f"📝 {filename} -> {micro_filename} 생성 중...")
        
        try:
            # 원본 파일 열기
            with h5py.File(src_path, 'r') as src_f:
                if len(src_f.keys()) == 0:
                    print(f"⚠️  {filename}이 비어있습니다.")
                    continue
                
                # 단 10개 샘플만 추출
                sample_size = 10
                print(f"  📊 극소 데이터셋: {sample_size}개 샘플만 추출")
                
                # 극소 파일 생성
                with h5py.File(dst_path, 'w') as dst_f:
                    for key in src_f.keys():
                        data = src_f[key][:sample_size]  # 처음 10개만 복사
                        dst_f.create_dataset(key, data=data, compression='gzip')
                        
                print(f"✅ {micro_filename} 생성 완료!")
                
        except Exception as e:
            print(f"❌ {filename} 처리 중 오류: {e}")

if __name__ == "__main__":
    print("⚡ 초고속 테스트용 극소 데이터셋 생성기")
    print("=" * 50)
    
    create_micro_h5_datasets()
    
    print("\n✅ 완료! 이제 초고속 테스트를 실행하세요!")

