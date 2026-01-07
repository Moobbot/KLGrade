#!/usr/bin/env python3
"""
GPU Test Script - Verify GPU availability and functionality
"""

import torch
import sys


def test_gpu():
    print("=" * 60)
    print("GPU Test Report")
    print("=" * 60)

    # 1. Check CUDA availability
    print(f"\n1. CUDA Available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("   ❌ CUDA is not available!")
        print("   Please check your PyTorch installation and NVIDIA drivers.")
        return False

    # 2. CUDA version
    print(f"2. CUDA Version: {torch.version.cuda}")

    # 3. GPU count
    gpu_count = torch.cuda.device_count()
    print(f"3. GPU Count: {gpu_count}")

    # 4. GPU details
    print(f"\n4. GPU Details:")
    for i in range(gpu_count):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"      - Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"      - Multi-Processors: {props.multi_processor_count}")
        print(f"      - Compute Capability: {props.major}.{props.minor}")

    # 5. Memory info
    print(f"\n5. Current GPU Memory:")
    for i in range(gpu_count):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"   GPU {i}:")
        print(f"      - Allocated: {allocated:.2f} GB")
        print(f"      - Reserved: {reserved:.2f} GB")

    # 6. Simple computation test
    print(f"\n6. Running GPU computation test...")
    try:
        # Create tensors on GPU
        x = torch.randn(1000, 1000, device="cuda")
        y = torch.randn(1000, 1000, device="cuda")

        # Matrix multiplication
        z = torch.matmul(x, y)

        # Ensure computation is done
        torch.cuda.synchronize()

        print("   ✅ GPU computation successful!")
        print(f"   Result shape: {z.shape}")

    except Exception as e:
        print(f"   ❌ GPU computation failed: {e}")
        return False

    # 7. cuDNN info
    print(f"\n7. cuDNN:")
    print(f"   Enabled: {torch.backends.cudnn.enabled}")
    print(f"   Version: {torch.backends.cudnn.version()}")

    print("\n" + "=" * 60)
    print("✅ All GPU tests passed!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = test_gpu()
    sys.exit(0 if success else 1)
