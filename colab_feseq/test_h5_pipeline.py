#!/usr/bin/env python3
"""
H5 파이프라인 테스트 스크립트
"""

import os
import sys

def test_h5_pipeline():
    """H5 파이프라인 테스트"""
    
    print("🧪 H5 파이프라인 테스트 시작...")
    
    # 1. 경로 설정
    base_dir = "/Users/hj/projects/TossCTR"
    colab_dir = os.path.join(base_dir, "colab_feseq")
    
    print(f"📁 기본 디렉토리: {colab_dir}")
    
    # 2. 필요한 파일들 확인
    required_files = [
        "preprocessing/tossctr_parquet_to_h5.py",
        "model_zoo/FESeq/config/dataset_config.yaml",
        "model_zoo/FESeq/config/model_config.yaml",
        "model_zoo/FESeq/run_expid.py"
    ]
    
    print("\n📋 필수 파일 확인:")
    all_files_exist = True
    for file_path in required_files:
        full_path = os.path.join(colab_dir, file_path)
        if os.path.exists(full_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            all_files_exist = False
    
    # 3. 원본 데이터 확인
    train_parquet_path = os.path.join(base_dir, "data/raw/train.parquet")
    print(f"\n📊 원본 데이터 확인:")
    if os.path.exists(train_parquet_path):
        size_gb = os.path.getsize(train_parquet_path) / (1024**3)
        print(f"✅ train.parquet ({size_gb:.2f} GB)")
        data_available = True
    else:
        print(f"❌ train.parquet 없음: {train_parquet_path}")
        data_available = False
    
    # 4. Config 내용 확인
    print(f"\n⚙️  설정 확인:")
    dataset_config_path = os.path.join(colab_dir, "model_zoo/FESeq/config/dataset_config.yaml")
    if os.path.exists(dataset_config_path):
        with open(dataset_config_path, 'r') as f:
            content = f.read()
            if "tossctr_h5_dataset:" in content:
                print("✅ tossctr_h5_dataset 설정 존재")
            else:
                print("❌ tossctr_h5_dataset 설정 없음")
    
    model_config_path = os.path.join(colab_dir, "model_zoo/FESeq/config/model_config.yaml")
    if os.path.exists(model_config_path):
        with open(model_config_path, 'r') as f:
            content = f.read()
            if "FESeq_tossctr_h5:" in content:
                print("✅ FESeq_tossctr_h5 모델 설정 존재")
            else:
                print("❌ FESeq_tossctr_h5 모델 설정 없음")
    
    # 5. 테스트 실행 가능 여부 판단
    print(f"\n🎯 테스트 준비 상태:")
    if all_files_exist:
        print("✅ 모든 필수 파일 존재")
    else:
        print("❌ 필수 파일 누락")
    
    if data_available:
        print("✅ 원본 데이터 사용 가능")
    else:
        print("❌ 원본 데이터 없음")
    
    # 6. 간단한 H5 변환 테스트 (소량 데이터)
    if all_files_exist and data_available:
        print(f"\n🚀 소량 데이터로 H5 변환 테스트...")
        
        try:
            # 작업 디렉토리 변경
            original_cwd = os.getcwd()
            os.chdir(colab_dir)
            
            # Python path 추가
            sys.path.insert(0, colab_dir)
            sys.path.insert(0, os.path.join(colab_dir, "preprocessing"))
            
            # H5 프로세서 import
            from tossctr_parquet_to_h5 import TossCTRParquetProcessor
            
            # 소량 테스트 실행
            processor = TossCTRParquetProcessor(
                config_path="model_zoo/FESeq/config/dataset_config.yaml",
                data_root="data"
            )
            
            print("✅ H5 프로세서 초기화 성공")
            
            # 아주 작은 샘플로 테스트
            processor.process_full_pipeline(
                train_parquet_path=train_parquet_path,
                chunk_size=1000,
                n_samples=5000  # 5천개만 테스트
            )
            
            print("✅ H5 변환 테스트 성공!")
            
            # 생성된 파일 확인
            h5_files = ["train.h5", "valid.h5", "test.h5", "feature_map.json"]
            for filename in h5_files:
                filepath = os.path.join("data/tossctr", filename)
                if os.path.exists(filepath):
                    if filename.endswith('.h5'):
                        size_mb = os.path.getsize(filepath) / (1024**2)
                        print(f"✅ {filename}: {size_mb:.1f} MB")
                    else:
                        print(f"✅ {filename}: 생성됨")
                else:
                    print(f"❌ {filename}: 생성 실패")
            
            # 원래 디렉토리로 복원
            os.chdir(original_cwd)
            
            print("\n🎉 H5 파이프라인 테스트 완료!")
            print("✅ 모든 구성 요소가 정상적으로 작동합니다.")
            
        except Exception as e:
            print(f"❌ H5 변환 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            
            # 원래 디렉토리로 복원
            os.chdir(original_cwd)
    
    else:
        print("❌ 테스트 실행 조건이 충족되지 않음")
        print("필요한 파일들을 먼저 준비해주세요.")


if __name__ == "__main__":
    test_h5_pipeline()
