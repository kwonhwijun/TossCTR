#!/usr/bin/env python3
"""
FESeq 모델 학습 스크립트
"""

import os
import sys

# TossCTR 루트를 파이썬 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import logging
import argparse
from datetime import datetime
from pathlib import Path
import gc

from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.pytorch.dataloaders import H5DataLoader
from fuxictr.preprocess import FeatureProcessor, build_dataset

# FESeq 모델 import - 경로 수정
sys.path.insert(0, os.path.join(project_root, 'models', 'FESeq'))
from model_zoo.FESeq.src.FESeq import FESeq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train_feseq(config_dir: str, experiment_id: str, gpu: int = -1, mode: str = 'train'):
    """
    FESeq 모델 학습
    
    Args:
        config_dir: 설정 파일 디렉토리
        experiment_id: 실험 ID
        gpu: GPU 번호 (-1이면 CPU)
        mode: 'train' 또는 'test'
    """
    logging.info(f"🚀 Starting experiment: {experiment_id}")
    logging.info(f"Config directory: {config_dir}")
    
    # 설정 로드
    params = load_config(config_dir, experiment_id)
    params['gpu'] = gpu
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params['seed'])
    
    # 데이터 디렉토리 설정
    data_dir = os.path.join(project_root, params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    
    # CSV 데이터 전처리
    if params["data_format"] == "csv":
        logging.info("Building feature_map and transforming to h5 data...")
        feature_encoder = FeatureProcessor(**params)
        params["train_data"], params["valid_data"], params["test_data"] = \
            build_dataset(feature_encoder, **params)
    
    # Feature map 로드
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)
    logging.info("Feature specs: " + print_to_json(feature_map.features))
    
    # 모델 초기화 - params 전달 방식 수정
    model = FESeq(feature_map, params)
    model.count_parameters()
    
    # 학습 모드
    if mode == "train":
        logging.info("=" * 50)
        logging.info("Training FESeq model...")
        logging.info("=" * 50)
        
        train_gen, valid_gen = H5DataLoader(feature_map, stage='train', **params).make_iterator()
        model.fit(train_gen, validation_data=valid_gen, **params)
        
        logging.info('****** Validation evaluation ******')
        valid_result = model.evaluate(valid_gen)
        del train_gen, valid_gen
        gc.collect()
    
    # 테스트 평가
    logging.info('******** Test evaluation ********')
    model.load_weights(model.checkpoint)
    
    test_gen = H5DataLoader(feature_map, stage='test', **params).make_iterator()
    test_result = {}
    if test_gen:
        test_result = model.evaluate(test_gen)
    
    # 결과 저장
    result_filename = os.path.join(project_root, 'results.csv')
    with open(result_filename, 'a+') as fw:
        fw.write(' {},[command] {},[exp_id] {},[dataset_id] {},[test] {}\n'.format(
            datetime.now().strftime('%Y%m%d-%H%M%S'),
            ' '.join(sys.argv),
            experiment_id,
            params['dataset_id'],
            print_to_list(test_result)
        ))
    
    logging.info(f"✅ Results saved to: {result_filename}")
    logging.info("🎉 Training completed!")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train FESeq model')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'test'],
                       help='Train or test mode')
    parser.add_argument('--config', type=str, default='./configs/',
                       help='Config directory path')
    parser.add_argument('--expid', type=str, default='FESeq_toy',
                       help='Experiment ID')
    parser.add_argument('--gpu', type=int, default=-1,
                       help='GPU index (-1 for CPU)')
    
    args = parser.parse_args()
    
    train_feseq(
        config_dir=args.config,
        experiment_id=args.expid,
        gpu=args.gpu,
        mode=args.mode
    )


if __name__ == '__main__':
    main()
