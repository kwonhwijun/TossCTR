#!/usr/bin/env python3
"""
Git Push Helper Script - 핵심 파일만 선별하여 push
"""

import os
import subprocess
from pathlib import Path

project_root = Path(__file__).parent.parent
os.chdir(project_root)


def run_cmd(cmd):
    """명령어 실행"""
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode


def main():
    print("=" * 80)
    print("TossCTR Git Push Helper")
    print("=" * 80)
    
    # 1. Git status 확인
    print("\n📋 현재 Git Status:")
    print("-" * 80)
    run_cmd("git status")
    
    # 2. 핵심 파일들 추가
    print("\n📦 핵심 파일들 추가 중...")
    print("-" * 80)
    
    files_to_add = [
        # 설정 파일
        "configs/dataset_config.yaml",
        "configs/model_config.yaml",
        
        # 실행 스크립트
        "scripts/train.py",
        "scripts/predict.py",
        "scripts/prepare_full_data.py",
        "scripts/run_pipeline.py",
        "scripts/create_toy_data.py",
        "scripts/test_imports.py",
        
        # 라이브러리
        "fuxictr/",
        
        # FESeq 모델 (소스만)
        "models/FESeq/FESeq.py",
        "models/FESeq/interaction_layer.py",
        "models/FESeq/pooling_layer.py",
        "models/FESeq/model_zoo/FESeq/src/",
        
        # Requirements
        "requirements.txt",
        "requirements_feseq.txt",
        
        # 문서
        "README.md",
        "BRIEFING.md",
        
        # .gitignore
        ".gitignore",
    ]
    
    for file in files_to_add:
        file_path = project_root / file
        if file_path.exists():
            print(f"  ✓ {file}")
            run_cmd(f"git add {file}")
        else:
            print(f"  ⚠ {file} (파일 없음, 스킵)")
    
    # 3. .gitkeep 파일 추가 (디렉토리 구조 유지)
    print("\n📁 디렉토리 구조 유지...")
    print("-" * 80)
    
    dirs = ['data/raw', 'data/output', 'data/processed', 'data/toy', 'checkpoints']
    for dir_name in dirs:
        dir_path = project_root / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        gitkeep = dir_path / '.gitkeep'
        gitkeep.touch()
        run_cmd(f"git add {gitkeep}")
        print(f"  ✓ {dir_name}/.gitkeep")
    
    # 4. 추가된 파일 확인
    print("\n📝 추가된 파일 확인:")
    print("-" * 80)
    run_cmd("git status")
    
    # 5. Commit & Push
    print("\n🚀 Commit & Push:")
    print("-" * 80)
    
    commit_msg = input("Commit message (기본: 'Add FESeq training pipeline and scripts'): ").strip()
    if not commit_msg:
        commit_msg = "Add FESeq training pipeline and scripts"
    
    print(f"\nCommit message: {commit_msg}")
    confirm = input("Push하시겠습니까? (y/N): ").strip().lower()
    
    if confirm == 'y':
        run_cmd(f'git commit -m "{commit_msg}"')
        run_cmd("git push origin main")
        print("\n✅ Push 완료!")
    else:
        print("\n❌ 취소되었습니다.")
        print("   수동으로 push하려면:")
        print(f'   git commit -m "{commit_msg}"')
        print("   git push origin main")


if __name__ == '__main__':
    main()

