#!/usr/bin/env python3
"""
전체 파이프라인 실행 스크립트
1. 전체 데이터 준비
2. 모델 학습
3. 예측 및 submission 생성
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

project_root = Path(__file__).parent.parent
os.chdir(project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_command(cmd, description):
    """명령어 실행"""
    logging.info("=" * 60)
    logging.info(f"Step: {description}")
    logging.info(f"Command: {' '.join(cmd)}")
    logging.info("=" * 60)
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        logging.error(f"❌ Failed: {description}")
        sys.exit(1)
    
    logging.info(f"✅ Completed: {description}\n")


def main():
    logging.info("🚀 TossCTR 전체 파이프라인 시작")
    logging.info(f"Working directory: {project_root}\n")
    
    # Step 1: 전체 데이터 준비
    run_command(
        ['python', 'scripts/prepare_full_data.py'],
        "전체 데이터 준비 (train/val/test 분할)"
    )
    
    # Step 2: 모델 학습
    run_command(
        ['python', 'scripts/train.py', 
         '--config', 'configs/', 
         '--expid', 'FESeq_full', 
         '--gpu', '-1',  # CPU 사용 (GPU 있으면 0으로 변경)
         '--mode', 'train'],
        "FESeq 모델 학습"
    )
    
    # Step 3: 예측 및 submission 생성
    run_command(
        ['python', 'scripts/predict.py',
         '--config', 'configs/',
         '--expid', 'FESeq_full',
         '--test_data', 'data/raw/test.parquet',
         '--output', 'data/output/submission.csv',
         '--gpu', '-1'],  # CPU 사용 (GPU 있으면 0으로 변경)
        "테스트 데이터 예측 및 submission.csv 생성"
    )
    
    logging.info("=" * 60)
    logging.info("🎉 전체 파이프라인 완료!")
    logging.info("=" * 60)
    logging.info(f"최종 결과물: {project_root}/data/output/submission.csv")


if __name__ == '__main__':
    main()


