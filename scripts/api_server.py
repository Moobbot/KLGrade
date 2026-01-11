#!/usr/bin/env python3
"""
FastAPI Inference Server Entry Point

Usage:
    python scripts/api_server.py --model path/to/model.pt --port 8000
"""
import argparse
import sys
import os

# Ensure project root is in path
sys.path.append(os.getcwd())

from src.api.server import start_api_server

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO API Server")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model (.pt)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    
    args = parser.parse_args()
    
    start_api_server(args.host, args.port, args.model)
