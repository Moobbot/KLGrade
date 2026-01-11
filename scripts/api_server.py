#!/usr/bin/env python3
"""
FastAPI Inference Server Entry Point (Knee Pipeline)

Usage:
    python scripts/api_server.py \
        --knee-model path/to/knee_det.pt \
        --grade-model path/to/kl_grade.pt \
        --port 8000
"""
import argparse
import sys
import os

# Ensure project root is in path
sys.path.append(os.getcwd())

from src.api.server import start_api_server

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KLGrade Pipeline API Server")
    parser.add_argument("--knee-model", type=str, default="runs/detect/my_knee_run_resplit/weights/best.pt", help="Path to Knee Detection model (.pt)")
    parser.add_argument("--grade-model", type=str, default="runs/detect_2026_01_09/E007_10class_conservative4/weights/best.pt", help="Path to KL Grading model (.pt)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    
    args = parser.parse_args()
    
    # Check paths
    if not os.path.exists(args.knee_model):
        print(f"Error: Knee model not found at {args.knee_model}")
        sys.exit(1)
    if not os.path.exists(args.grade_model):
        print(f"Error: Grade model not found at {args.grade_model}")
        sys.exit(1)
    
    start_api_server(args.host, args.port, args.knee_model, args.grade_model)
