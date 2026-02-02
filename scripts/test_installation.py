#!/usr/bin/env python3
"""
Test installation of all dependencies
Run this first to verify Week 0 setup
"""

import sys

def test_imports():
    """Test that all required packages are installed"""
    
    print("Testing package imports...\n")
    
    packages = {
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'matplotlib': 'Matplotlib',
        'mujoco': 'MuJoCo',
        'gymnasium': 'Gymnasium',
        'torch': 'PyTorch',
        'stable_baselines3': 'Stable-Baselines3',
        'cvxpy': 'CVXPY',
        'tensorboard': 'TensorBoard',
        'wandb': 'Weights & Biases',
        'yaml': 'PyYAML',
        'tqdm': 'tqdm',
        'pandas': 'Pandas'
    }
    
    failed = []
    
    for package, name in packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {name:<25} {version}")
        except ImportError as e:
            print(f"✗ {name:<25} NOT INSTALLED")
            failed.append(name)
    
    print()
    
    if failed:
        print(f"❌ {len(failed)} packages failed to import:")
        for pkg in failed:
            print(f"   - {pkg}")
        print("\nPlease install missing packages:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    else:
        print("✅ All packages installed successfully!")
        
        # Test CUDA availability
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available (CPU training will be slower)")
        
        return True

if __name__ == "__main__":
    print("="*60)
    print(" DA-VIL Installation Test")
    print("="*60 + "\n")
    
    test_imports()
    
    print("\n" + "="*60)
    print("Ready to proceed with Week 0!")
    print("="*60 + "\n")
