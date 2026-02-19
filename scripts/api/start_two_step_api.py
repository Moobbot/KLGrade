import argparse
import os
import sys
import uvicorn
from pathlib import Path

# Add project root to path
# Assuming script is in scripts/api/ (2 levels down from root)
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from api_two_step_yolo.inference.server import app


def main():
    parser = argparse.ArgumentParser(description="Start Two-Step YOLO API Server")

    # Model paths
    parser.add_argument(
        "--knee-model",
        type=str,
        default="runs/detect/knee_yolo11n_20260217_134003/weights/best.pt",
        help="Path to knee detection model weights",
    )
    parser.add_argument(
        "--lesion-model",
        type=str,
        default="runs/detect/lesion_8class_balanced/weights/best.pt",
        help="Path to lesion detection model weights",
    )

    # Server config
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind server to"
    )
    parser.add_argument("--port", type=int, default=9000, help="Port to bind server to")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run inference on (e.g., cuda:0 or cpu)",
    )

    args = parser.parse_args()

    # Set environment variables for the server module to pick up
    os.environ["KNEE_MODEL"] = args.knee_model
    os.environ["LESION_MODEL"] = args.lesion_model
    os.environ["DEVICE"] = args.device

    # Print configuration
    print(f"Starting Two-Step YOLO API Server")
    print(f"  - Host: {args.host}")
    print(f"  - Port: {args.port}")
    print(f"  - Knee Model: {args.knee_model}")
    print(f"  - Lesion Model: {args.lesion_model}")
    print(f"  - Device: {args.device}")

    # Run server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
