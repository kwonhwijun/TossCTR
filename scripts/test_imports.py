#!/usr/bin/env python3
"""
Import 테스트 스크립트
"""

import os
import sys

# TossCTR 루트를 파이썬 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print(f"Project root: {project_root}")
print(f"sys.path: {sys.path[:3]}")

try:
    print("\n1. Testing fuxictr imports...")
    from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
    print("   ✓ fuxictr.utils")
    
    from fuxictr.features import FeatureMap
    print("   ✓ fuxictr.features")
    
    from fuxictr.pytorch.torch_utils import seed_everything
    print("   ✓ fuxictr.pytorch.torch_utils")
    
    from fuxictr.pytorch.dataloaders import H5DataLoader
    print("   ✓ fuxictr.pytorch.dataloaders")
    
    from fuxictr.preprocess import FeatureProcessor, build_dataset
    print("   ✓ fuxictr.preprocess")
    
    print("\n2. Testing FESeq model import...")
    sys.path.insert(0, os.path.join(project_root, 'models', 'FESeq'))
    from model_zoo.FESeq.src.FESeq import FESeq
    print("   ✓ FESeq model")
    
    print("\n✅ All imports successful!")
    
except Exception as e:
    print(f"\n❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
