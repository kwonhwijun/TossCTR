#!/usr/bin/env python3
"""
Quick Git Push - 핵심 파일만 빠르게 push
"""

import os
import subprocess
from pathlib import Path

def run(cmd):
    print(f"\n$ {cmd}")
    os.system(cmd)

os.chdir('/Users/hj/projects/TossCTR')

print("=" * 80)
print("🚀 TossCTR GitHub Push")
print("=" * 80)

# 1. 디렉토리 구조 생성
print("\n📁 디렉토리 구조 생성...")
run("mkdir -p data/raw data/output data/processed data/toy checkpoints")
run("touch data/raw/.gitkeep data/output/.gitkeep data/processed/.gitkeep data/toy/.gitkeep checkpoints/.gitkeep")

# 2. 핵심 파일 추가
print("\n📦 핵심 파일 추가...")
run("git add .gitignore")
run("git add configs/")
run("git add scripts/train.py scripts/predict.py scripts/prepare_full_data.py scripts/run_pipeline.py scripts/create_toy_data.py scripts/test_imports.py")
run("git add fuxictr/")
run("git add requirements.txt requirements_feseq.txt")
run("git add README.md BRIEFING.md")
run("git add data/*/.gitkeep")

# 3. FESeq 모델 (강제 추가)
print("\n🤖 FESeq 모델 소스 코드 추가...")
run("git add -f models/FESeq/FESeq.py")
run("git add -f models/FESeq/interaction_layer.py")
run("git add -f models/FESeq/pooling_layer.py")
run("git add -f 'models/FESeq/model_zoo/FESeq/src/*.py'")

# 4. 상태 확인
print("\n📋 Git Status:")
run("git status")

# 5. Commit & Push
print("\n" + "=" * 80)
response = input("\n✅ 위 파일들을 commit & push 하시겠습니까? (y/N): ")

if response.lower() == 'y':
    commit_msg = """Add FESeq training pipeline and execution scripts

- Add training script (train.py) with FESeq model integration
- Add prediction script (predict.py) for submission generation
- Add data preparation scripts (prepare_full_data.py, run_pipeline.py)
- Add FESeq model source code from models/FESeq/
- Add fuxictr library for CTR prediction
- Update configs for TossCTR dataset (dataset_config.yaml, model_config.yaml)
- Add requirements_feseq.txt for dependency management
- Add utility scripts (create_toy_data.py, test_imports.py)"""
    
    run(f'git commit -m "{commit_msg}"')
    run("git push origin main")
    
    print("\n" + "=" * 80)
    print("✅ Push 완료!")
    print("=" * 80)
    print("\n서버에서 실행하기:")
    print("  git clone https://github.com/kwonhwijun/TossCTR.git")
    print("  cd TossCTR")
    print("  pip install -r requirements_feseq.txt")
    print("  python scripts/run_pipeline.py")
else:
    print("\n❌ 취소되었습니다.")

