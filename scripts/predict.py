import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMapAbsTime
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.pytorch.dataloaders.h5_dataloader import DataLoader
from fuxictr.preprocess import FeatureProcessor, build_dataset

# FESeq 모델 import
sys.path.insert(0, os.path.join(project_root, 'models'))
from model_zoo.FESeq.src.FESeq import FESeq

def predict_and_submit(config_dir, experiment_id, test_h5_path, output_path, gpu : int = 0) :
    logging.info(f"예측 시작 {experiment_id}")

    params = load_config(config_dir, experiment_id)
    params['gpu'] = gpu
    set_logger(params)
    seed_everything(seed=params['seed'])

    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feautre_map_json = os.path.join(data_dir, "feature_map.json")

    feature_map = FeatureMapAbsTime(params['dataset_id'], data_dir)
    feature_map.load(feautre_map_json, params)
    logging.info(f"Feature map 로드 완료")

    model = FESeq(feature_map, params, **params)

    checkpoint_path = os.path.join(project_root, 'checkpoints', params['dataset_id'], 
                                   f"{experiment_id}.model")

    logging.info(f"Loading model from: {checkpoint_path}")
    model.load_weights(checkpoint_path)

    # 테스트 데이터 로드
    logging.info(f"Loading test data from: {test_h5_path}")
    test_dataset = DataLoader(feature_map, test_h5_path, 
                              batch_size=params['batch_size'], 
                              shuffle=False)

    # 예측 수행
    logging.info("Generating predictions...")
    predictions = model.predict(test_dataset)

    # 확률값 추출
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        click_probabilities = predictions[:, 1]  # 클릭 확률
    else:
        click_probabilities = predictions.flatten()
    
    logging.info(f"Predictions shape: {predictions.shape}")
    logging.info(f"Generated {len(click_probabilities)} predictions")

    sample_submission_path = os.path.join(project_root, "data", "sample_submission.csv")

    if os.path.exists(sample_submission_path):
        sample_submission = pd.read_csv(sample_submission_path)
        logging.info(f"Sample submission loaded: {len(sample_submission)} rows")
        
        # 길이 맞추기
        if len(click_probabilities) != len(sample_submission):
            logging.warning(f"Length mismatch: predictions({len(click_probabilities)}) "
                          f"vs submission({len(sample_submission)})")
            min_len = min(len(click_probabilities), len(sample_submission))
            click_probabilities = click_probabilities[:min_len]
            sample_submission = sample_submission.iloc[:min_len].copy()
        
        # 예측값 할당
        sample_submission['clicked'] = click_probabilities
        
        # 저장
        output_path = os.path.join(project_root, output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sample_submission.to_csv(output_path, index=False)
        
        # 통계 출력
        logging.info(f"✅ Submission saved to: {output_path}")
        logging.info(f"📊 Prediction statistics:")
        logging.info(f"   Mean: {click_probabilities.mean():.6f}")
        logging.info(f"   Min: {click_probabilities.min():.6f}")
        logging.info(f"   Max: {click_probabilities.max():.6f}")
        logging.info(f"   Std: {click_probabilities.std():.6f}")
        
        print("\n" + "="*60)
        print("🎉 Submission 파일 생성 완료!")
        print("="*60)
        print(f"📄 파일 경로: {output_path}")
        print(f"\n📊 예측 통계:")
        print(f"   평균 클릭 확률: {click_probabilities.mean():.6f}")
        print(f"   최소값: {click_probabilities.min():.6f}")
        print(f"   최대값: {click_probabilities.max():.6f}")
        print(f"   표준편차: {click_probabilities.std():.6f}")
        
        # 첫 10개 샘플 출력
        print(f"\n📋 첫 10개 예측 샘플:")
        print(sample_submission.head(10).to_string(index=False))
        print("="*60)
        
    else:
        logging.error(f"Sample submission file not found: {sample_submission_path}")
        raise FileNotFoundError(f"Sample submission not found: {sample_submission_path}")


def main():
    parser = argparse.ArgumentParser(description='Predict and generate submission')
    parser.add_argument('--config', type=str, default='./configs/',
                       help='Config directory path')
    parser.add_argument('--expid', type=str, default='FESeq_toy',
                       help='Experiment ID')
    parser.add_argument('--test_h5', type=str, 
                       default='data/tossctr_toy/test.h5',
                       help='Test H5 file path (relative to project root)')
    parser.add_argument('--output', type=str, 
                       default='data/output/submission.csv',
                       help='Output submission file path')
    parser.add_argument('--gpu', type=int, default=-1,
                       help='GPU index (-1 for CPU)')
    
    args = parser.parse_args()
    
    predict_and_submit(
        config_dir=args.config,
        experiment_id=args.expid,
        test_h5_path=os.path.join(project_root, args.test_h5),
        output_path=args.output,
        gpu=args.gpu
    )


if __name__ == '__main__':
    main()