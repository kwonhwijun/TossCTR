#!/usr/bin/env python3
"""
FESeq 모델로 테스트 데이터 추론 및 submission 파일 생성
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# FESeq 경로 설정
FESEQ_PATH = Path("/home/hj/TossCTR/colab_feseq/model_zoo/FESeq")
sys.path.append("/home/hj/TossCTR/colab_feseq")

def setup_environment():
    """환경 설정"""
    os.chdir(FESEQ_PATH)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print(f"🚀 FESeq 추론 시작")
    print(f"📁 작업 디렉토리: {os.getcwd()}")

def run_feseq_test():
    """FESeq 모델로 테스트 데이터 추론"""
    print(f"\n🎯 FESeq 모델 테스트 추론 시작")
    
    # run_expid.py의 test 모드 실행
    cmd = f"PYTHONPATH=/home/hj/TossCTR/colab_feseq python run_expid.py --config ./config/ --expid FESeq_tossctr_h5 --gpu -1 --mode test"
    
    print(f"\n💻 실행 명령어:")
    print(f"   {cmd}")
    
    print(f"\n🏃 테스트 추론 실행 중...")
    print("=" * 60)
    
    exit_code = os.system(cmd)
    
    print("=" * 60)
    
    if exit_code == 0:
        print(f"✅ 테스트 추론 완료!")
        return True
    else:
        print(f"❌ 테스트 추론 실패! 종료 코드: {exit_code}")
        return False

def check_prediction_files():
    """예측 결과 파일 확인"""
    print(f"\n📊 예측 결과 파일 확인:")
    
    # 가능한 예측 파일 위치들
    possible_paths = [
        "predictions/",
        "checkpoints/",
        "./",
        "results/"
    ]
    
    prediction_files = []
    for path in possible_paths:
        if os.path.exists(path):
            files = list(Path(path).glob("*prediction*")) + list(Path(path).glob("*test*"))
            prediction_files.extend(files)
    
    if prediction_files:
        print(f"✅ 예측 파일 발견:")
        for file in prediction_files:
            print(f"   📄 {file}")
        return prediction_files
    else:
        print(f"⚠️  예측 파일을 찾을 수 없습니다.")
        return []

def create_submission_from_h5():
    """H5 테스트 데이터와 모델을 직접 사용하여 submission 생성"""
    print(f"\n🔧 직접 추론으로 submission 생성...")
    
    try:
        # FuxiCTR 모듈 import
        from fuxictr.utils import load_config
        from fuxictr.features import FeatureMapAbsTime
        from fuxictr.pytorch.dataloaders import H5DataLoader
        import src as model_zoo
        import torch
        
        print("✅ 모듈 로드 완료")
        
        # 설정 로드
        params = load_config("./config/", "FESeq_tossctr_h5")
        params['gpu'] = -1
        
        # Feature map 로드
        data_dir = os.path.join(params['data_root'], params['dataset_id'])
        feature_map_json = os.path.join(data_dir, "feature_map.json")
        feature_map = FeatureMapAbsTime(params['dataset_id'], data_dir)
        feature_map.load(feature_map_json, params)
        
        print("✅ Feature map 로드 완료")
        
        # 모델 로드
        model_class = getattr(model_zoo, params['model'])
        model = model_class(feature_map, params=params, **params)
        
        # 체크포인트 로드
        checkpoint_path = f"./checkpoints/{params['dataset_id']}/FESeq_tossctr_h5.model"
        if os.path.exists(checkpoint_path):
            model.load_weights(checkpoint_path)
            print(f"✅ 모델 체크포인트 로드: {checkpoint_path}")
        else:
            print(f"❌ 체크포인트를 찾을 수 없습니다: {checkpoint_path}")
            return False
        
        # 테스트 데이터 로더
        test_gen = H5DataLoader(feature_map, stage='test', **params).make_iterator()
        
        if test_gen:
            print("✅ 테스트 데이터 로더 생성 완료")
            
            # 예측 실행
            print("🔮 예측 실행 중...")
            predictions = model.predict(test_gen)
            
            print(f"✅ 예측 완료! 예측 결과 shape: {predictions.shape}")
            
            # sample_submission.csv 로드
            sample_submission = pd.read_csv("/home/hj/TossCTR/data/sample_submission.csv")
            print(f"📋 Sample submission 로드: {len(sample_submission)} rows")
            
            # 예측 결과가 확률인 경우 클래스로 변환 (threshold 0.5)
            if predictions.max() <= 1.0 and predictions.min() >= 0.0:
                # 확률 값 그대로 사용 (competition에서 확률을 요구할 수도 있음)
                final_predictions = predictions.flatten()
            else:
                # 이진 분류로 변환
                final_predictions = (predictions > 0.5).astype(int).flatten()
            
            # submission 파일 생성
            if len(final_predictions) == len(sample_submission):
                sample_submission['clicked'] = final_predictions
            else:
                print(f"⚠️  예측 결과 수({len(final_predictions)})와 submission 행 수({len(sample_submission)})가 다릅니다.")
                # 길이를 맞춤
                min_len = min(len(final_predictions), len(sample_submission))
                sample_submission = sample_submission.iloc[:min_len].copy()
                sample_submission['clicked'] = final_predictions[:min_len]
            
            # 결과 저장
            submission_path = "/home/hj/TossCTR/submission_feseq.csv"
            sample_submission.to_csv(submission_path, index=False)
            
            print(f"✅ Submission 파일 생성 완료: {submission_path}")
            print(f"📊 예측 분포:")
            print(f"   - 클릭 예측 (1): {(sample_submission['clicked'] > 0.5).sum():,}")
            print(f"   - 비클릭 예측 (0): {(sample_submission['clicked'] <= 0.5).sum():,}")
            print(f"   - 평균 클릭률: {sample_submission['clicked'].mean():.4f}")
            
            return True
        else:
            print("❌ 테스트 데이터 로더 생성 실패")
            return False
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 함수"""
    print("🚀 FESeq 테스트 추론 및 Submission 생성")
    print("=" * 50)
    
    # 1. 환경 설정
    setup_environment()
    
    # 2. 직접 추론으로 submission 생성
    success = create_submission_from_h5()
    
    if success:
        print(f"\n🎉 Submission 파일이 성공적으로 생성되었습니다!")
        print(f"📄 파일 위치: /home/hj/TossCTR/submission_feseq.csv")
    else:
        print(f"\n💥 Submission 생성 중 오류가 발생했습니다.")
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n\n⏹️  사용자가 추론을 중단했습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 예상치 못한 오류: {e}")
        sys.exit(1)
