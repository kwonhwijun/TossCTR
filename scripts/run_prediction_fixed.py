#!/usr/bin/env python3
"""
FESeq 모델로 예측 생성 및 submission 파일 생성 (Fixed version)
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# 경로 설정
os.chdir('/home/hj/TossCTR/colab_feseq/model_zoo/FESeq')
sys.path.append('/home/hj/TossCTR/colab_feseq')

# FuxiCTR imports
from fuxictr.utils import load_config, set_logger
from fuxictr.features import FeatureMapAbsTime  
from fuxictr.pytorch.dataloaders import H5DataLoader
from fuxictr.pytorch.torch_utils import seed_everything

# FESeq 직접 import
from src.FESeq import FESeq

def main():
    """예측 생성 및 submission 파일 생성"""
    # 설정 로드
    experiment_id = "FESeq_tossctr_h5"
    params = load_config("./config/", experiment_id)
    params['gpu'] = -1
    
    set_logger(params)
    logging.info(f"Generating predictions for {experiment_id}")
    seed_everything(seed=params['seed'])
    
    # Feature map 로드
    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    feature_map = FeatureMapAbsTime(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)
    
    # 모델 초기화 (직접 FESeq 클래스 사용)
    model = FESeq(feature_map, params=params, **params)
    
    # 체크포인트 로드
    checkpoint_path = f"./checkpoints/{params['dataset_id']}/{experiment_id}.model"
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)
        logging.info(f"Model loaded from {checkpoint_path}")
    else:
        logging.error(f"Checkpoint not found: {checkpoint_path}")
        return False
    
    # 테스트 데이터 로드
    test_gen = H5DataLoader(feature_map, stage='test', **params).make_iterator()
    
    if test_gen:
        logging.info("Generating predictions...")
        
        # 예측 생성
        predictions = model.predict(test_gen)
        logging.info(f"Predictions generated: shape {predictions.shape}")
        
        # 확률값을 클릭률로 변환 (0~1 사이 값)
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            # 다중 클래스인 경우 첫 번째 클래스 확률 사용
            click_probabilities = predictions[:, 1]  # 클릭 확률
        else:
            # 이진 분류인 경우
            click_probabilities = predictions.flatten()
        
        # Sample submission 로드
        sample_submission = pd.read_csv("/home/hj/TossCTR/data/sample_submission.csv")
        logging.info(f"Sample submission loaded: {len(sample_submission)} rows")
        
        # 예측 결과와 sample submission 길이 맞추기
        if len(click_probabilities) != len(sample_submission):
            logging.warning(f"Prediction length ({len(click_probabilities)}) != submission length ({len(sample_submission)})")
            min_len = min(len(click_probabilities), len(sample_submission))
            click_probabilities = click_probabilities[:min_len]
            sample_submission = sample_submission.iloc[:min_len].copy()
        
        # Submission 파일 생성
        sample_submission['clicked'] = click_probabilities
        
        # 결과 저장
        submission_path = "/home/hj/TossCTR/submission_feseq.csv"
        sample_submission.to_csv(submission_path, index=False)
        
        # 통계 출력
        logging.info(f"Submission saved to: {submission_path}")
        logging.info(f"Prediction statistics:")
        logging.info(f"  Mean click probability: {click_probabilities.mean():.4f}")
        logging.info(f"  Min: {click_probabilities.min():.4f}")
        logging.info(f"  Max: {click_probabilities.max():.4f}")
        logging.info(f"  Std: {click_probabilities.std():.4f}")
        
        # 첫 10개 예측 샘플 출력
        print("\n📊 첫 10개 예측 샘플:")
        print(sample_submission.head(10))
        
        return True
    
    else:
        logging.error("Failed to create test data loader")
        return False

if __name__ == "__main__":
    print("🚀 FESeq 예측 생성 시작 (Fixed)")
    print("=" * 50)
    
    try:
        success = main()
        if success:
            print("\n🎉 예측 생성 완료!")
        else:
            print("\n❌ 예측 생성 실패!")
            
    except Exception as e:
        print(f"\n💥 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
