#!/usr/bin/env python3
"""
GPU Utilities - Shared GPU checking and info functions

Used by test_gpu.py and test_yolo_gpu.py to avoid code duplication.
"""

import torch
import sys


def check_gpu_available():
    """
    Check if CUDA/GPU is available.

    Returns:
        bool: True if CUDA is available, False otherwise
    """
    return torch.cuda.is_available()


def get_gpu_info():
    """
    Get detailed GPU information.

    Returns:
        dict: GPU information including:
            - available: bool
            - cuda_version: str
            - device_count: int
            - devices: list of device info dicts
    """
    if not torch.cuda.is_available():
        return {
            "available": False,
            "cuda_version": None,
            "device_count": 0,
            "devices": [],
        }

    devices = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        devices.append(
            {
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_gb": props.total_memory / 1024**3,
                "multi_processors": props.multi_processor_count,
                "compute_capability": f"{props.major}.{props.minor}",
            }
        )

    return {
        "available": True,
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "devices": devices,
    }


def get_gpu_memory_info(device=0):
    """
    Get current GPU memory usage.

    Args:
        device: GPU device index (default: 0)

    Returns:
        dict: Memory info with allocated and reserved in GB
    """
    if not torch.cuda.is_available():
        return {"allocated_gb": 0, "reserved_gb": 0}

    return {
        "allocated_gb": torch.cuda.memory_allocated(device) / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved(device) / 1024**3,
    }


def print_gpu_report():
    """
    Print a formatted GPU report to console.

    Returns:
        bool: True if GPU available and working, False otherwise
    """
    print("=" * 60)
    print("GPU Report")
    print("=" * 60)

    if not check_gpu_available():
        print("\n❌ CUDA is not available!")
        print("   Please check your PyTorch installation and NVIDIA drivers.")
        return False

    info = get_gpu_info()

    print(f"\n✅ CUDA Available: Yes")
    print(f"   CUDA Version: {info['cuda_version']}")
    print(f"   GPU Count: {info['device_count']}")

    print(f"\nGPU Details:")
    for device in info["devices"]:
        print(f"   GPU {device['id']}: {device['name']}")
        print(f"      - Total Memory: {device['total_memory_gb']:.2f} GB")
        print(f"      - Multi-Processors: {device['multi_processors']}")
        print(f"      - Compute Capability: {device['compute_capability']}")

    print(f"\nCurrent GPU Memory:")
    for i in range(info["device_count"]):
        mem = get_gpu_memory_info(i)
        print(f"   GPU {i}:")
        print(f"      - Allocated: {mem['allocated_gb']:.2f} GB")
        print(f"      - Reserved: {mem['reserved_gb']:.2f} GB")

    # cuDNN info
    print(f"\ncuDNN:")
    print(f"   Enabled: {torch.backends.cudnn.enabled}")
    print(f"   Version: {torch.backends.cudnn.version()}")

    print("\n" + "=" * 60)
    return True


def test_gpu_computation():
    """
    Run a simple GPU computation test.

    Returns:
        bool: True if computation successful, False otherwise
    """
    if not check_gpu_available():
        return False

    try:
        # Create tensors on GPU
        x = torch.randn(1000, 1000, device="cuda")
        y = torch.randn(1000, 1000, device="cuda")

        # Matrix multiplication
        z = torch.matmul(x, y)

        # Ensure computation is done
        torch.cuda.synchronize()

        print("\n✅ GPU computation test successful!")
        print(f"   Result shape: {z.shape}")
        return True

    except Exception as e:
        print(f"\n❌ GPU computation test failed: {e}")
        return False


if __name__ == "__main__":
    # Quick test when run directly
    success = print_gpu_report()
    if success:
        test_gpu_computation()

    sys.exit(0 if success else 1)
