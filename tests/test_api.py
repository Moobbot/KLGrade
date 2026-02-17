#!/usr/bin/env python3
"""
Quick API test script
"""
import requests
import sys
from pathlib import Path

# Test image path
test_image = "datasets/processed_balanced/knees_cropped/resize_only/images/1.2.392.200036.9107.307.24972.20230209.145514.1046622_L.png"

if not Path(test_image).exists():
    print(f"Test image not found: {test_image}")
    # Try to find any image
    images_dir = Path("datasets/processed_balanced/knees_cropped/resize_only/images")
    if images_dir.exists():
        images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        if images:
            test_image = str(images[0])
            print(f"Using alternative image: {test_image}")
        else:
            print("No images found!")
            sys.exit(1)
    else:
        print(f"Images directory not found: {images_dir}")
        sys.exit(1)

# Test API
url = "http://localhost:9090/predict/"
print(f"\nTesting API at {url}")
print(f"Image: {test_image}\n")

try:
    with open(test_image, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files, timeout=30)

    if response.status_code == 200:
        result = response.json()
        print("✅ API Test Successful!")
        print(f"\nResults:")
        print(f"  Filename: {result.get('filename')}")
        print(f"  KL Grade: {result.get('kl_grade')}")
        print(f"  Knees detected: {result.get('knees_count')}")
        print(f"  Lesions detected: {result.get('lesions_count')}")

        if result.get("knees"):
            print(f"\n  Knee boxes:")
            for knee in result["knees"]:
                print(
                    f"    - Knee {knee['knee_id']}: confidence={knee['confidence']:.3f}"
                )

        if result.get("lesions"):
            print(f"\n  Lesions:")
            for lesion in result["lesions"]:
                print(
                    f"    - {lesion['class_name']}: confidence={lesion['confidence']:.3f}, knee_id={lesion['knee_id']}"
                )
    else:
        print(f"❌ API Error: {response.status_code}")
        print(response.text)

except requests.exceptions.ConnectionError:
    print("❌ Cannot connect to API server. Is it running?")
    print("Start server with: python api_two_step_yolo/inference/server.py")
except Exception as e:
    print(f"❌ Error: {e}")
