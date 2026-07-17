#!/usr/bin/env python
"""
Automatic MMHCL Training Script
This script verifies package versions and automatically trains the MMHCL model.
"""

import sys
import os
import subprocess
import importlib

# Required package versions
REQUIRED_VERSIONS = {
    'python': '3.12.12',
    'torch': '2.2.0+cu118',
    'numpy': '1.26.4',
    'scipy': '1.16.3',
    'sklearn': '1.6.1'
}

def check_package_version(package_name, required_version, actual_version):
    """Check if package version matches required version."""
    # Handle version strings with +cu118 suffix
    if '+' in required_version:
        required_base = required_version.split('+')[0]
        actual_base = actual_version.split('+')[0] if '+' in actual_version else actual_version
        return actual_base == required_base
    return actual_version == required_version

def verify_environment():
    """Verify that all required packages are installed with correct versions."""
    print("=" * 70)
    print("MMHCL Automatic Training - Environment Verification")
    print("=" * 70)
    
    # Check Python version
    python_version = sys.version.split()[0]
    print(f"Python version: {python_version}")
    if not check_package_version('python', REQUIRED_VERSIONS['python'], python_version):
        print(f"[WARNING]  WARNING: Python version mismatch!")
        print(f"   Required: {REQUIRED_VERSIONS['python']}, Found: {python_version}")
        return False
    print("[OK] Python version matches\n")
    
    # Check PyTorch
    try:
        import torch
        torch_version = torch.__version__
        print(f"PyTorch version: {torch_version}")
        if not check_package_version('torch', REQUIRED_VERSIONS['torch'], torch_version):
            print(f"[WARNING]  WARNING: PyTorch version mismatch!")
            print(f"   Required: {REQUIRED_VERSIONS['torch']}, Found: {torch_version}")
            return False
        print("[OK] PyTorch version matches")
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        if not cuda_available:
            print("[WARNING]  WARNING: CUDA is not available! Training will be slow on CPU.")
        else:
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
        print()
    except ImportError:
        print("[ERROR] PyTorch is not installed!")
        return False
    
    # Check NumPy
    try:
        import numpy as np
        numpy_version = np.__version__
        print(f"NumPy version: {numpy_version}")
        if not check_package_version('numpy', REQUIRED_VERSIONS['numpy'], numpy_version):
            print(f"[WARNING]  WARNING: NumPy version mismatch!")
            print(f"   Required: {REQUIRED_VERSIONS['numpy']}, Found: {numpy_version}")
            if numpy_version.startswith('2.'):
                print("   NumPy 2.x detected! This may cause compatibility issues.")
                print("   Please install: pip install 'numpy<2.0' --force-reinstall")
            return False
        print("[OK] NumPy version matches\n")
    except ImportError:
        print("[ERROR] NumPy is not installed!")
        return False
    
    # Check SciPy
    try:
        import scipy
        scipy_version = scipy.__version__
        print(f"SciPy version: {scipy_version}")
        if not check_package_version('scipy', REQUIRED_VERSIONS['scipy'], scipy_version):
            print(f"[WARNING]  WARNING: SciPy version mismatch!")
            print(f"   Required: {REQUIRED_VERSIONS['scipy']}, Found: {scipy_version}")
            return False
        print("[OK] SciPy version matches\n")
    except ImportError:
        print("[ERROR] SciPy is not installed!")
        return False
    
    # Check scikit-learn
    try:
        import sklearn
        sklearn_version = sklearn.__version__
        print(f"scikit-learn version: {sklearn_version}")
        if not check_package_version('sklearn', REQUIRED_VERSIONS['sklearn'], sklearn_version):
            print(f"[WARNING]  WARNING: scikit-learn version mismatch!")
            print(f"   Required: {REQUIRED_VERSIONS['sklearn']}, Found: {sklearn_version}")
            return False
        print("[OK] scikit-learn version matches\n")
    except ImportError:
        print("[ERROR] scikit-learn is not installed!")
        return False
    
    print("=" * 70)
    print("[OK] All package versions verified successfully!")
    print("=" * 70)
    return True

def check_dataset_availability(dataset_name):
    """Check if dataset files are available."""
    print(f"\nChecking dataset availability for '{dataset_name}'...")
    
    # The code uses args.data_path + args.dataset where data_path defaults to '../data/'
    # So for 'Clothing', it expects '../data/Clothing' (relative to codes directory)
    # But we're running from MMHCL directory, so we check multiple possible locations
    
    possible_paths = [
        f"data/{dataset_name}",  # Direct under data/
        f"data/{dataset_name.lower()}",  # Lowercase
        f"data/data/{dataset_name.lower()}",  # Nested structure
        f"../data/{dataset_name}",  # Relative from codes
        f"../data/{dataset_name.lower()}",  # Lowercase relative
    ]
    
    if dataset_name.lower() == 'tiktok':
        possible_paths.insert(0, "data/Tiktok")
        possible_paths.insert(1, "../data/Tiktok")
    
    dataset_path = None
    for path in possible_paths:
        # Check if 5-core directory exists
        core_path = os.path.join(path, '5-core')
        if os.path.exists(core_path):
            dataset_path = path
            break
    
    if dataset_path is None:
        print(f"[ERROR] Dataset directory not found in any expected location:")
        for path in possible_paths:
            print(f"   - {path}")
        print(f"\n[WARNING]  Please ensure the dataset is properly placed.")
        print(f"   Expected structure: data/{dataset_name}/5-core/{{train,val,test}}.json")
        print(f"   And: data/{dataset_name}/{{image_feat,text_feat}}.npy")
        return False
    
    print(f"[OK] Found dataset at: {dataset_path}")
    
    # Check for required files
    required_files = {
        'train.json': os.path.join(dataset_path, '5-core', 'train.json'),
        'val.json': os.path.join(dataset_path, '5-core', 'val.json'),
        'test.json': os.path.join(dataset_path, '5-core', 'test.json'),
        'image_feat.npy': os.path.join(dataset_path, 'image_feat.npy'),
        'text_feat.npy': os.path.join(dataset_path, 'text_feat.npy')
    }
    
    if dataset_name.lower() == 'tiktok':
        required_files['audio_feat.npy'] = os.path.join(dataset_path, 'audio_feat.npy')
    
    all_exist = True
    for name, path in required_files.items():
        if os.path.exists(path):
            print(f"[OK] {name}: Found")
        else:
            print(f"[ERROR] {name}: NOT FOUND at {path}")
            all_exist = False
    
    if not all_exist:
        print(f"\n[WARNING]  Some dataset files are missing!")
        print(f"   Please ensure all required files are present.")
        return False
    
    print(f"[OK] Dataset '{dataset_name}' is available!")
    return True

def run_training(dataset='Clothing', gpu_id=0, **kwargs):
    """Run the MMHCL training."""
    print("\n" + "=" * 70)
    print("Starting MMHCL Training")
    print("=" * 70)
    
    # Change to codes directory
    codes_dir = os.path.join(os.path.dirname(__file__), 'codes')
    if not os.path.exists(codes_dir):
        print(f"[ERROR] Error: codes directory not found at {codes_dir}")
        return False
    
    original_dir = os.getcwd()
    os.chdir(codes_dir)
    
    try:
        # Build training command
        cmd = [
            sys.executable, 'main.py',
            '--dataset', dataset,
            '--gpu_id', str(gpu_id),
            '--epoch', str(kwargs.get('epoch', 250)),
            '--verbose', str(kwargs.get('verbose', 5)),
            '--batch_size', str(kwargs.get('batch_size', 1024)),
            '--lr', str(kwargs.get('lr', 0.0001)),
            '--regs', str(kwargs.get('regs', 1e-3)),
            '--embed_size', str(kwargs.get('embed_size', 64)),
            '--topk', str(kwargs.get('topk', 5)),
            '--core', str(kwargs.get('core', 5)),
            '--User_layers', str(kwargs.get('User_layers', 3)),
            '--Item_layers', str(kwargs.get('Item_layers', 2)),
            '--user_loss_ratio', str(kwargs.get('user_loss_ratio', 0.03)),
            '--item_loss_ratio', str(kwargs.get('item_loss_ratio', 0.07)),
            '--temperature', str(kwargs.get('temperature', 0.6)),
            '--early_stopping_patience', str(kwargs.get('early_stopping_patience', 30)),
            '--early_stopping_min_epochs', str(kwargs.get('early_stopping_min_epochs', 75)),
            '--early_stopping_min_delta', str(kwargs.get('early_stopping_min_delta', 0.0001)),
            '--early_stopping_monitor', str(kwargs.get('early_stopping_monitor', 'val_recall@20')),
            '--early_stopping_mode', str(kwargs.get('early_stopping_mode', 'max')),
            '--early_stopping_restore_best', str(kwargs.get('early_stopping_restore_best', 1)),
            '--adaptive_patience', str(kwargs.get('adaptive_patience', 1)),
            '--use_reduce_lr', str(kwargs.get('use_reduce_lr', 1)),
            '--reduce_lr_factor', str(kwargs.get('reduce_lr_factor', 0.5)),
            '--reduce_lr_patience', str(kwargs.get('reduce_lr_patience', 3)),
            '--reduce_lr_min', str(kwargs.get('reduce_lr_min', 1e-6))
        ]
        
        print(f"\nTraining command:")
        print(" ".join(cmd))
        print("\n" + "=" * 70)
        
        # Run training
        result = subprocess.run(cmd, check=False)
        
        if result.returncode == 0:
            print("\n" + "=" * 70)
            print("[OK] Training completed successfully!")
            print("=" * 70)
            return True
        else:
            print("\n" + "=" * 70)
            print(f"[ERROR] Training failed with exit code {result.returncode}")
            print("=" * 70)
            return False
            
    except Exception as e:
        print(f"\n[ERROR] Error during training: {e}")
        return False
    finally:
        os.chdir(original_dir)

def main():
    """Main function to run automatic training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Automatic MMHCL Training')
    parser.add_argument('--dataset', type=str, default='Clothing',
                        choices=['Clothing', 'Sports', 'Tiktok'],
                        help='Dataset to train on')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--skip_verification', action='store_true',
                        help='Skip environment verification')
    parser.add_argument('--skip_dataset_check', action='store_true',
                        help='Skip dataset availability check')
    
    # Training hyperparameters
    parser.add_argument('--epoch', type=int, default=250,
                        help='Number of training epochs')
    parser.add_argument('--verbose', type=int, default=5,
                        help='Evaluation interval')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--regs', type=float, default=1e-3,
                        help='Regularization')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size')
    parser.add_argument('--topk', type=int, default=5,
                        help='K-NN sparsification')
    parser.add_argument('--core', type=int, default=5,
                        help='Core filtering')
    parser.add_argument('--User_layers', type=int, default=3,
                        help='User hypergraph layers')
    parser.add_argument('--Item_layers', type=int, default=2,
                        help='Item hypergraph layers')
    parser.add_argument('--user_loss_ratio', type=float, default=0.03,
                        help='User contrastive loss weight')
    parser.add_argument('--item_loss_ratio', type=float, default=0.07,
                        help='Item contrastive loss weight')
    parser.add_argument('--temperature', type=float, default=0.6,
                        help='InfoNCE temperature')
    parser.add_argument('--early_stopping_patience', type=int, default=30,
                        help='Early stopping patience (default: 30, model needs 200+ epochs)')
    parser.add_argument('--early_stopping_min_epochs', type=int, default=75,
                        help='Minimum epochs before early stopping can trigger (default: 75)')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.0001,
                        help='Minimum improvement to count as progress')
    parser.add_argument('--early_stopping_monitor', type=str, default='val_recall@20',
                        help='Metric to monitor for early stopping (val_recall@20 or val_ndcg@20)')
    parser.add_argument('--early_stopping_mode', type=str, default='max',
                        help='Early stopping mode: max or min')
    parser.add_argument('--early_stopping_restore_best', type=int, default=1,
                        help='Restore best model weights on early stop (1=yes, 0=no)')
    parser.add_argument('--adaptive_patience', type=int, default=0,
                        help='Use adaptive patience based on dataset size (1=yes, 0=no). Default: disabled')
    parser.add_argument('--use_reduce_lr', type=int, default=1,
                        help='Use ReduceLROnPlateau scheduler (1=yes, 0=no)')
    parser.add_argument('--reduce_lr_factor', type=float, default=0.5,
                        help='Factor to reduce LR by when plateau detected')
    parser.add_argument('--reduce_lr_patience', type=int, default=3,
                        help='Patience for ReduceLROnPlateau')
    parser.add_argument('--reduce_lr_min', type=float, default=1e-6,
                        help='Minimum learning rate for ReduceLROnPlateau')
    
    args = parser.parse_args()
    
    # Verify environment
    if not args.skip_verification:
        if not verify_environment():
            print("\n[ERROR] Environment verification failed!")
            print("   Please install the correct package versions or use --skip_verification to proceed anyway.")
            return 1
    else:
        print("[WARNING]  Skipping environment verification...")
    
    # Check dataset availability
    if not args.skip_dataset_check:
        if not check_dataset_availability(args.dataset):
            print("\n[ERROR] Dataset check failed!")
            print("   Please ensure the dataset is available or use --skip_dataset_check to proceed anyway.")
            return 1
    
    # Run training
    training_kwargs = {
        'epoch': args.epoch,
        'verbose': args.verbose,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'regs': args.regs,
        'embed_size': args.embed_size,
        'topk': args.topk,
        'core': args.core,
        'User_layers': args.User_layers,
        'Item_layers': args.Item_layers,
        'user_loss_ratio': args.user_loss_ratio,
        'item_loss_ratio': args.item_loss_ratio,
        'temperature': args.temperature,
        'early_stopping_patience': args.early_stopping_patience,
        'early_stopping_min_epochs': args.early_stopping_min_epochs,
        'early_stopping_min_delta': args.early_stopping_min_delta,
        'early_stopping_monitor': args.early_stopping_monitor,
        'early_stopping_mode': args.early_stopping_mode,
        'early_stopping_restore_best': args.early_stopping_restore_best,
        'adaptive_patience': args.adaptive_patience,
        'use_reduce_lr': args.use_reduce_lr,
        'reduce_lr_factor': args.reduce_lr_factor,
        'reduce_lr_patience': args.reduce_lr_patience,
        'reduce_lr_min': args.reduce_lr_min
    }
    
    success = run_training(args.dataset, args.gpu_id, **training_kwargs)
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())

