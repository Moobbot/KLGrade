#!/usr/bin/env python3
"""
YOLO GPU Test - Quick 1-epoch test to verify YOLO can use GPU
"""

from ultralytics import YOLO
import torch


def test_yolo_gpu():
    print("=" * 60)
    print("YOLO GPU Test (1 epoch)")
    print("=" * 60)

    # Check GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available! Cannot run GPU test.")
        return False

    print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    print(
        f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

    # Load model
    print("\nLoading YOLO model...")
    model = YOLO("yolo11n.pt")

    # Quick training test
    print("\nRunning 1-epoch GPU test...")
    try:
        results = model.train(
            data="configs/yolo_5_class_baseline.yaml",
            epochs=1,
            batch=8,
            imgsz=640,
            device=0,  # GPU 0
            project="runs/gpu_test",
            name="quick_test",
            exist_ok=True,
            verbose=False,
            plots=False,
        )

        print("\n✅ YOLO GPU test successful!")
        print(f"   Training completed on GPU")
        print(f"   Results saved to: runs/gpu_test/quick_test")

        return True

    except Exception as e:
        print(f"\n❌ YOLO GPU test failed: {e}")
        return False


if __name__ == "__main__":
    import sys

    success = test_yolo_gpu()
    sys.exit(0 if success else 1)
