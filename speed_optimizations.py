"""
MMHCL Training Speed Optimizations
===================================
This module provides speed optimization utilities for MMHCL training.
Automatically applies optimizations for RTX 5000 Ada / CUDA 11.8 / PyTorch 2.0.1
"""

import os
import torch
import warnings

def apply_speed_optimizations(verbose=True):
    """
    Apply all available speed optimizations for MMHCL training.
    
    Optimizations applied:
    1. cuDNN autotuning for convolutions
    2. TensorFloat32 precision for faster matrix operations on Ampere+ GPUs
    3. Optimal CUDA memory allocation
    4. Multiprocessing spawn method for Windows
    5. Environment variables for performance
    
    Returns:
        dict: Summary of applied optimizations
    """
    optimizations = {}
    
    # 1. Enable cuDNN benchmark mode (finds fastest algorithms)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    optimizations['cudnn_benchmark'] = True
    
    # 2. Enable TensorFloat32 for RTX 30xx/40xx/Ada GPUs (Ampere+)
    # This provides ~2x speedup for matrix multiplications with minimal precision loss
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability[0] >= 8:  # Ampere (8.0) and Ada (8.9)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            optimizations['tf32_enabled'] = True
            if verbose:
                print(f"✓ TensorFloat32 enabled (GPU compute capability {capability[0]}.{capability[1]})")
        else:
            optimizations['tf32_enabled'] = False
            
    # 3. Optimize CUDA memory allocation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    optimizations['cuda_memory_optimized'] = True
    
    # 4. Set optimal number of threads for CPU operations
    num_threads = min(os.cpu_count(), 8)  # Cap at 8 to avoid overhead
    torch.set_num_threads(num_threads)
    optimizations['num_threads'] = num_threads
    
    # 5. Enable Flash Attention if available (PyTorch 2.0+)
    try:
        # Check if scaled_dot_product_attention is available
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            optimizations['flash_attention_available'] = True
            if verbose:
                print("✓ Flash Attention available via scaled_dot_product_attention")
    except:
        optimizations['flash_attention_available'] = False
    
    # 6. Suppress unnecessary warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='torch')
    
    # 7. Set multiprocessing start method for Windows
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    if verbose:
        print("\n" + "="*60)
        print("MMHCL Speed Optimizations Applied")
        print("="*60)
        print(f"✓ cuDNN benchmark mode: Enabled")
        print(f"✓ cuDNN enabled: True")
        print(f"✓ CPU threads: {num_threads}")
        print(f"✓ CUDA memory optimization: Enabled")
        if torch.cuda.is_available():
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print("="*60 + "\n")
    
    return optimizations


def optimize_sparse_operations():
    """
    Tips for optimizing sparse operations in MMHCL:
    - torch_sparse is installed and provides optimized sparse-dense matrix multiplication
    - torch_scatter provides optimized scatter operations for graph neural networks
    """
    try:
        import torch_sparse
        import torch_scatter
        return True
    except ImportError:
        return False


def get_optimal_batch_size(n_items, gpu_memory_gb=16):
    """
    Calculate optimal batch size based on GPU memory and dataset size.
    
    Args:
        n_items: Number of items in dataset
        gpu_memory_gb: GPU memory in GB (default: 16 for RTX 5000 Ada)
    
    Returns:
        int: Recommended batch size
    """
    # Heuristic based on MMHCL memory patterns
    if gpu_memory_gb >= 24:
        base_batch = 4096
    elif gpu_memory_gb >= 16:
        base_batch = 2048
    elif gpu_memory_gb >= 8:
        base_batch = 1024
    else:
        base_batch = 512
    
    # Adjust based on dataset size
    if n_items > 50000:
        return min(base_batch, 2048)
    elif n_items > 20000:
        return min(base_batch, 4096)
    else:
        return base_batch


def print_gpu_memory_usage():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {allocated:.2f}GB allocated / {reserved:.2f}GB reserved / {total:.1f}GB total")


def clear_gpu_cache():
    """Clear GPU cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class SpeedBenchmark:
    """Context manager to benchmark code execution time."""
    
    def __init__(self, name="Operation"):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        import time
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        import time
        elapsed = time.perf_counter() - self.start_time
        print(f"⏱ {self.name}: {elapsed:.4f}s")


# Auto-apply optimizations when module is imported
if __name__ != "__main__":
    # Only apply when imported, not when run directly
    pass


if __name__ == "__main__":
    # Test optimizations
    print("Testing MMHCL Speed Optimizations...")
    apply_speed_optimizations(verbose=True)
    
    print("\nSparse Operations Support:")
    if optimize_sparse_operations():
        print("✓ torch_sparse and torch_scatter are available")
    else:
        print("✗ Missing sparse operation libraries")
    
    print("\nGPU Memory Status:")
    print_gpu_memory_usage()
    
    print("\nRecommended batch sizes:")
    for n_items in [10000, 25000, 50000]:
        bs = get_optimal_batch_size(n_items, gpu_memory_gb=16)
        print(f"  {n_items:,} items → batch_size={bs}")
