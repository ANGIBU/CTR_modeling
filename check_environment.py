# check_environment.py

import sys
import platform

def check_environment():
    print("=" * 70)
    print("Environment Check")
    print("=" * 70)
    
    print(f"\nPython version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    
    try:
        import torch
        print(f"\nPyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            print(f"GPU name: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            
            print("\nGPU Test:")
            x = torch.rand(5, 3).cuda()
            print(f"  Tensor on GPU: {x.is_cuda}")
            print(f"  Current device: {torch.cuda.current_device()}")
        else:
            print("\n!!! WARNING: CUDA not available !!!")
            print("PyTorch installed version is CPU-only")
            print("\nTo install CUDA version:")
            print("  1. pip uninstall torch torchvision torchaudio -y")
            print("  2. pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
    except ImportError:
        print("\nPyTorch not installed")
    
    try:
        import lightgbm as lgb
        print(f"\nLightGBM version: {lgb.__version__}")
    except ImportError:
        print("\nLightGBM not installed")
    
    try:
        import xgboost as xgb
        print(f"XGBoost version: {xgb.__version__}")
    except ImportError:
        print("\nXGBoost not installed")
    
    try:
        import pandas as pd
        print(f"Pandas version: {pd.__version__}")
    except ImportError:
        print("\nPandas not installed")
    
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
    except ImportError:
        print("\nNumPy not installed")
    
    print("\n" + "=" * 70)
    print("Data Files Check")
    print("=" * 70)
    
    from pathlib import Path
    
    data_dir = Path("data")
    train_path = data_dir / "train.parquet"
    test_path = data_dir / "test.parquet"
    
    if train_path.exists():
        size_mb = train_path.stat().st_size / (1024**2)
        print(f"Train data: {size_mb:.1f}MB")
    else:
        print("Train data: NOT FOUND")
    
    if test_path.exists():
        size_mb = test_path.stat().st_size / (1024**2)
        print(f"Test data: {size_mb:.1f}MB")
    else:
        print("Test data: NOT FOUND")
    
    print("\n" + "=" * 70)
    print("Memory Check")
    print("=" * 70)
    
    try:
        import psutil
        vm = psutil.virtual_memory()
        print(f"Total RAM: {vm.total / (1024**3):.1f}GB")
        print(f"Available RAM: {vm.available / (1024**3):.1f}GB")
        print(f"Used RAM: {vm.used / (1024**3):.1f}GB ({vm.percent:.1f}%)")
    except ImportError:
        print("psutil not installed")
    
    print("\n" + "=" * 70)
    
    if torch.cuda.is_available():
        print("STATUS: Ready for GPU training")
    else:
        print("STATUS: GPU not available - reinstall PyTorch with CUDA support")
    
    print("=" * 70)

if __name__ == "__main__":
    check_environment()