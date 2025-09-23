#!/usr/bin/env python3
"""
Colab에서 FESeq 실행을 위한 메인 스크립트
"""

import os
import sys
import subprocess

def setup_environment():
    """환경 설정"""
    # PYTHONPATH 설정
    current_dir = os.getcwd()
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # 환경 변수 설정
    os.environ['PYTHONPATH'] = current_dir
    print(f"✅ PYTHONPATH set to: {current_dir}")

def install_dependencies():
    """필요한 패키지 설치"""
    try:
        print("📦 Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        raise

def run_feseq_experiment(expid="FESeq_tossctr", gpu=0):
    """FESeq 실험 실행"""
    setup_environment()
    
    # FESeq 디렉토리로 이동
    feseq_dir = os.path.join(os.getcwd(), "model_zoo", "FESeq")
    os.chdir(feseq_dir)
    print(f"📁 Changed directory to: {feseq_dir}")
    
    # 실험 실행
    cmd = [
        sys.executable, "run_expid.py", 
        "--expid", expid,
        "--gpu", str(gpu)
    ]
    
    print(f"🚀 Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("✅ FESeq experiment completed successfully!")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ FESeq experiment failed: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run FESeq experiment")
    parser.add_argument("--expid", default="FESeq_tossctr", help="Experiment ID")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID (-1 for CPU)")
    parser.add_argument("--install-only", action="store_true", help="Only install dependencies")
    
    args = parser.parse_args()
    
    if args.install_only:
        install_dependencies()
    else:
        install_dependencies()
        run_feseq_experiment(args.expid, args.gpu)
