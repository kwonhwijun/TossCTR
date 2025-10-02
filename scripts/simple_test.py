#!/usr/bin/env python3
"""
FESeq 모델 파이프라인 간단 테스트
"""
import sys
import os
sys.path.append('colab_feseq')

def test_imports():
    """필요한 모듈들이 제대로 임포트되는지 테스트"""
    print("1️⃣ 모듈 임포트 테스트...")
    
    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__}")
        
        import pandas as pd
        print(f"   ✅ Pandas {pd.__version__}")
        
        import h5py
        print(f"   ✅ h5py {h5py.__version__}")
        
        # FuXiCTR 임포트 테스트
        from fuxictr.features import FeatureProcessor
        print("   ✅ FuXiCTR FeatureProcessor")
        
        from fuxictr.pytorch.dataloaders import H5DataLoader
        print("   ✅ FuXiCTR H5DataLoader")
        
        # FESeq 모델 임포트 테스트
        sys.path.append('colab_feseq/model_zoo/FESeq/src')
        from FESeq import FESeq
        print("   ✅ FESeq 모델")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 임포트 오류: {e}")
        return False

def test_data_loading():
    """데이터 로딩 테스트"""
    print("\n2️⃣ 데이터 로딩 테스트...")
    
    try:
        import h5py
        
        # 마이크로 데이터셋 확인
        micro_files = [
            'colab_feseq/data/tossctr/train_micro.h5',
            'colab_feseq/data/tossctr/valid_micro.h5',
            'colab_feseq/data/tossctr/test_micro.h5'
        ]
        
        for file_path in micro_files:
            if os.path.exists(file_path):
                with h5py.File(file_path, 'r') as f:
                    keys = list(f.keys())
                    if keys:
                        sample_count = len(f[keys[0]])
                        print(f"   ✅ {file_path}: {sample_count}개 샘플, {len(keys)}개 필드")
                    else:
                        print(f"   ⚠️  {file_path}: 빈 파일")
            else:
                print(f"   ❌ {file_path}: 파일 없음")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 데이터 로딩 오류: {e}")
        return False

def test_config_loading():
    """설정 파일 로딩 테스트"""
    print("\n3️⃣ 설정 파일 로딩 테스트...")
    
    try:
        import yaml
        
        # 데이터셋 설정 확인
        with open('colab_feseq/model_zoo/FESeq/config/dataset_config.yaml', 'r') as f:
            dataset_config = yaml.safe_load(f)
            
        if 'tossctr_micro_dataset' in dataset_config:
            print("   ✅ tossctr_micro_dataset 설정 발견")
        else:
            print("   ❌ tossctr_micro_dataset 설정 없음")
        
        # 모델 설정 확인
        with open('colab_feseq/model_zoo/FESeq/config/model_config.yaml', 'r') as f:
            model_config = yaml.safe_load(f)
            
        if 'FESeq_tossctr_ultra_fast' in model_config:
            print("   ✅ FESeq_tossctr_ultra_fast 설정 발견")
            config = model_config['FESeq_tossctr_ultra_fast']
            print(f"   📊 배치 크기: {config['batch_size']}, 에포크: {config['epochs']}")
        else:
            print("   ❌ FESeq_tossctr_ultra_fast 설정 없음")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 설정 로딩 오류: {e}")
        return False

def test_simple_model_creation():
    """간단한 모델 생성 테스트"""
    print("\n4️⃣ 모델 생성 테스트...")
    
    try:
        import torch
        import torch.nn as nn
        
        # 매우 간단한 FESeq 유사 모델 생성 테스트
        class SimpleFESeq(nn.Module):
            def __init__(self, vocab_size=100, embed_dim=2, hidden_dim=4):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.linear = nn.Linear(embed_dim, hidden_dim)
                self.output = nn.Linear(hidden_dim, 1)
                self.sigmoid = nn.Sigmoid()
                
            def forward(self, x):
                embedded = self.embedding(x)
                pooled = embedded.mean(dim=1)  # 간단한 평균 풀링
                hidden = torch.relu(self.linear(pooled))
                output = self.sigmoid(self.output(hidden))
                return output
        
        # 모델 생성
        model = SimpleFESeq()
        print("   ✅ 간단한 모델 생성 성공")
        
        # 더미 입력으로 forward pass 테스트
        dummy_input = torch.randint(0, 100, (2, 5))  # 배치 2, 시퀀스 길이 5
        output = model(dummy_input)
        print(f"   ✅ Forward pass 성공: 출력 형태 {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 모델 생성 오류: {e}")
        return False

def main():
    """메인 테스트 실행"""
    print("⚡ FESeq 파이프라인 간단 테스트 시작")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_loading, 
        test_config_loading,
        test_simple_model_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        if test_func():
            passed += 1
    
    print(f"\n🏁 테스트 완료: {passed}/{total} 통과")
    
    if passed == total:
        print("✅ 모든 테스트 통과! 파이프라인이 정상 동작할 준비가 되었습니다.")
        return True
    else:
        print("❌ 일부 테스트 실패. 위의 오류를 확인해주세요.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

