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
        # fuxictr 패키지가 이미 설치되어 있는지 확인
        try:
            import fuxictr
            # 현재 디렉토리의 setup.py 버전과 비교
            import pkg_resources
            installed_version = pkg_resources.get_distribution('fuxictr').version
            print(f"✅ fuxictr {installed_version} already installed.")
            return
        except (ImportError, pkg_resources.DistributionNotFound):
            pass
        
        print("📦 Installing dependencies silently...")
        # 조용한 설치 (출력 숨김)
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", ".", 
            "-q", "--quiet", "--no-deps", "--disable-pip-version-check"
        ], check=True, capture_output=True, text=True)
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        raise

def run_feseq_experiment(expid="FESeq_tossctr", gpu=0):
    """FESeq 실험 실행"""
    setup_environment()
    
    # 원래 디렉토리 저장
    original_dir = os.getcwd()
    
    # FESeq 디렉토리로 이동
    feseq_dir = os.path.join(original_dir, "model_zoo", "FESeq")
    os.chdir(feseq_dir)
    print(f"📁 Changed directory to: {feseq_dir}")
    
    # PYTHONPATH에 원래 디렉토리도 추가
    if original_dir not in sys.path:
        sys.path.insert(0, original_dir)
    os.environ['PYTHONPATH'] = f"{original_dir}:{feseq_dir}"
    print(f"✅ Updated PYTHONPATH: {os.environ['PYTHONPATH']}")
    
    # 설정 파일 확인
    config_files = ["config/dataset_config.yaml", "config/model_config.yaml"]
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✅ Found config: {config_file}")
        else:
            print(f"❌ Missing config: {config_file}")
    
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
    finally:
        # 원래 디렉토리로 복귀
        os.chdir(original_dir)

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

