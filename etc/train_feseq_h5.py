#!/usr/bin/env python3
"""
FESeq 모델로 TossCTR H5 데이터 훈련하는 스크립트
"""

import os
import sys
import time
import logging
from pathlib import Path

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).parent
FESEQ_PATH = PROJECT_ROOT / "colab_feseq" / "model_zoo" / "FESeq"

def setup_environment():
    """환경 설정"""
    # 작업 디렉토리를 FESeq 모델 디렉토리로 변경
    os.chdir(FESEQ_PATH)
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print(f"🚀 FESeq 훈련 시작")
    print(f"📁 작업 디렉토리: {os.getcwd()}")
    print(f"🐍 Python 경로: {sys.executable}")

def check_h5_files():
    """H5 파일들이 존재하는지 확인"""
    h5_files = [
        "../../data/tossctr/train.h5",
        "../../data/tossctr/valid.h5", 
        "../../data/tossctr/test.h5",
        "../../data/tossctr/feature_map.json"
    ]
    
    print("\n📋 H5 파일 확인:")
    all_exist = True
    
    for file_path in h5_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024**2)  # MB
            print(f"✅ {file_path} ({file_size:.1f} MB)")
        else:
            print(f"❌ {file_path} - 파일이 없습니다!")
            all_exist = False
    
    if not all_exist:
        print("\n⚠️  H5 파일들이 완전히 생성되지 않았습니다.")
        print("   전처리 스크립트가 완료될 때까지 기다려주세요.")
        return False
    
    return True

def run_feseq_training(gpu_id=-1, experiment_id="FESeq_tossctr_h5"):
    """FESeq 모델 훈련 실행"""
    print(f"\n🎯 FESeq 모델 훈련 시작")
    print(f"🔧 실험 ID: {experiment_id}")
    print(f"🖥️  GPU ID: {gpu_id}")
    
    # 훈련 명령어 구성
    cmd = f"python run_expid.py --config ./config/ --expid {experiment_id} --gpu {gpu_id}"
    
    print(f"\n💻 실행 명령어:")
    print(f"   {cmd}")
    
    # 시작 시간 기록
    start_time = time.time()
    
    # 훈련 실행
    print(f"\n🏃 훈련 실행 중...")
    print("=" * 60)
    
    exit_code = os.system(cmd)
    
    # 완료 시간 계산
    end_time = time.time()
    duration = end_time - start_time
    
    print("=" * 60)
    
    if exit_code == 0:
        print(f"✅ 훈련 완료! 소요 시간: {duration/60:.1f}분")
        return True
    else:
        print(f"❌ 훈련 실패! 종료 코드: {exit_code}")
        return False

def check_results():
    """훈련 결과 확인"""
    print(f"\n📊 훈련 결과 확인:")
    
    # 체크포인트 디렉토리 확인
    checkpoint_dir = Path("./checkpoints")
    if checkpoint_dir.exists():
        checkpoint_files = list(checkpoint_dir.glob("*FESeq_tossctr_h5*"))
        if checkpoint_files:
            print(f"✅ 체크포인트 파일 생성됨:")
            for file in checkpoint_files:
                print(f"   📄 {file}")
        else:
            print(f"⚠️  체크포인트 파일을 찾을 수 없습니다.")
    
    # 결과 CSV 파일 확인
    result_files = list(Path(".").glob("*FESeq_tossctr_h5*.csv"))
    if result_files:
        print(f"✅ 결과 파일 생성됨:")
        for file in result_files:
            print(f"   📄 {file}")
    else:
        print(f"⚠️  결과 파일을 찾을 수 없습니다.")

def main():
    """메인 함수"""
    print("🚀 FESeq TossCTR 훈련 스크립트")
    print("=" * 50)
    
    # 1. 환경 설정
    setup_environment()
    
    # 2. H5 파일 확인
    if not check_h5_files():
        print("\n❌ H5 파일이 준비되지 않았습니다. 종료합니다.")
        return False
    
    # 3. 사용자 확인
    print(f"\n❓ FESeq 모델 훈련을 시작하시겠습니까?")
    print(f"   - 데이터셋: tossctr_h5_dataset")
    print(f"   - 모델: FESeq")
    print(f"   - 예상 시간: 30분-2시간")
    
    response = input("계속 진행하시겠습니까? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("훈련이 취소되었습니다.")
        return False
    
    # 4. GPU 설정
    gpu_input = input("GPU ID를 입력하세요 (CPU는 -1, 기본값: -1): ").strip()
    try:
        gpu_id = int(gpu_input) if gpu_input else -1
    except ValueError:
        gpu_id = -1
    
    # 5. FESeq 훈련 실행
    success = run_feseq_training(gpu_id=gpu_id)
    
    # 6. 결과 확인
    if success:
        check_results()
        print(f"\n🎉 FESeq 훈련이 성공적으로 완료되었습니다!")
    else:
        print(f"\n💥 FESeq 훈련 중 오류가 발생했습니다.")
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n\n⏹️  사용자가 훈련을 중단했습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 예상치 못한 오류: {e}")
        sys.exit(1)

