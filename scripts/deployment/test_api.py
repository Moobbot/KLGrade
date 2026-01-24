"""
Test Script for KLGrade KIOCMIL-CADA API

Tests all API endpoints with sample images.
"""

import requests
import json
import base64
import cv2
import numpy as np
from pathlib import Path
import time


class APITester:
    """Test client for KLGrade API."""

    def __init__(self, base_url: str = "http://localhost:8001"):
        """
        Initialize API tester.

        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url
        self.session = requests.Session()

    def test_health(self):
        """Test health check endpoint."""
        print("\n" + "=" * 60)
        print("Testing /health endpoint")
        print("=" * 60)

        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()

            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Status: {data['status']}")
            print(f"   Pipeline loaded: {data['pipeline_loaded']}")
            print(f"   Model type: {data['model_type']}")

            return True
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False

    def test_model_info(self):
        """Test model info endpoint."""
        print("\n" + "=" * 60)
        print("Testing /model_info endpoint")
        print("=" * 60)

        try:
            response = self.session.get(f"{self.base_url}/model_info")
            response.raise_for_status()

            data = response.json()
            print(f"✅ Model info retrieved")
            print(f"   Model type: {data['model_type']}")
            print(f"   Number of classes: {data['num_classes']}")
            print(f"   Context size: {data['ctx_size']}")
            print(f"   Patch size: {data['patch_size']}")
            print(f"   Device: {data['device']}")
            print(f"   Classes: {', '.join(data['class_names'][:5])}...")

            return True
        except Exception as e:
            print(f"❌ Model info failed: {e}")
            return False

    def test_predict(
        self, image_path: str, knee_conf: float = 0.5, lesion_conf: float = 0.5
    ):
        """
        Test prediction endpoint.

        Args:
            image_path: Path to test image
            knee_conf: Knee detection confidence threshold
            lesion_conf: Lesion detection confidence threshold
        """
        print("\n" + "=" * 60)
        print(f"Testing /predict endpoint with {image_path}")
        print("=" * 60)

        try:
            # Read image
            with open(image_path, "rb") as f:
                files = {"file": (Path(image_path).name, f, "image/jpeg")}
                data = {"knee_conf": knee_conf, "lesion_conf": lesion_conf}

                start_time = time.time()
                response = self.session.post(
                    f"{self.base_url}/predict", files=files, data=data
                )
                elapsed_time = (time.time() - start_time) * 1000

                response.raise_for_status()

            result = response.json()

            print(f"✅ Prediction successful")
            print(f"   Client processing time: {elapsed_time:.2f}ms")
            print(f"   Server processing time: {result['processing_time_ms']:.2f}ms")
            print(f"   Filename: {result['filename']}")
            print(f"   Knees detected: {result['num_knees_detected']}")

            for i, pred in enumerate(result["predictions"]):
                print(f"\n   Knee {i+1}:")
                print(
                    f"      BBox: ({pred['knee_bbox']['x1']}, {pred['knee_bbox']['y1']}) -> "
                    f"({pred['knee_bbox']['x2']}, {pred['knee_bbox']['y2']})"
                )
                print(
                    f"      Class: {pred['predicted_class']} (ID: {pred['predicted_class_id']})"
                )
                print(f"      Confidence: {pred['confidence']:.4f}")
                print(f"      JS lesions: {pred['num_js_lesions']}")
                print(f"      OST lesions: {pred['num_ost_lesions']}")

                # Show top 3 class probabilities
                probs = sorted(
                    pred["class_probabilities"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:3]
                print(f"      Top 3 probabilities:")
                for cls, prob in probs:
                    print(f"         {cls}: {prob:.4f}")

            return result
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            if hasattr(e, "response") and e.response is not None:
                print(f"   Response: {e.response.text}")
            return None

    def test_predict_visual(
        self,
        image_path: str,
        output_dir: str = "test_outputs",
        knee_conf: float = 0.5,
        lesion_conf: float = 0.5,
        include_gradcam: bool = True,
    ):
        """
        Test visual prediction endpoint.

        Args:
            image_path: Path to test image
            output_dir: Directory to save output images
            knee_conf: Knee detection confidence threshold
            lesion_conf: Lesion detection confidence threshold
            include_gradcam: Whether to include GradCAM
        """
        print("\n" + "=" * 60)
        print(f"Testing /predict_visual endpoint with {image_path}")
        print("=" * 60)

        try:
            # Read image
            with open(image_path, "rb") as f:
                files = {"file": (Path(image_path).name, f, "image/jpeg")}
                data = {
                    "knee_conf": knee_conf,
                    "lesion_conf": lesion_conf,
                    "include_gradcam": include_gradcam,
                }

                start_time = time.time()
                response = self.session.post(
                    f"{self.base_url}/predict_visual", files=files, data=data
                )
                elapsed_time = (time.time() - start_time) * 1000

                response.raise_for_status()

            result = response.json()

            print(f"✅ Visual prediction successful")
            print(f"   Client processing time: {elapsed_time:.2f}ms")
            print(f"   Server processing time: {result['processing_time_ms']:.2f}ms")
            print(f"   Knees detected: {result['num_knees_detected']}")

            # Save annotated image
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            if result.get("annotated_image_base64"):
                img_data = base64.b64decode(result["annotated_image_base64"])
                img_array = np.frombuffer(img_data, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                annotated_path = output_path / f"{Path(image_path).stem}_annotated.jpg"
                cv2.imwrite(str(annotated_path), img)
                print(f"   Saved annotated image to: {annotated_path}")

            # Save GradCAM image
            if result.get("gradcam_image_base64"):
                img_data = base64.b64decode(result["gradcam_image_base64"])
                img_array = np.frombuffer(img_data, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                gradcam_path = output_path / f"{Path(image_path).stem}_gradcam.jpg"
                cv2.imwrite(str(gradcam_path), img)
                print(f"   Saved GradCAM image to: {gradcam_path}")

            # Print predictions
            for i, pred in enumerate(result["predictions"]):
                print(
                    f"\n   Knee {i+1}: {pred['predicted_class']} ({pred['confidence']:.4f})"
                )

            return result
        except Exception as e:
            print(f"❌ Visual prediction failed: {e}")
            if hasattr(e, "response") and e.response is not None:
                print(f"   Response: {e.response.text}")
            return None

    def run_all_tests(self, test_image_path: str):
        """
        Run all API tests.

        Args:
            test_image_path: Path to test image
        """
        print("\n" + "=" * 60)
        print("KLGrade API Test Suite")
        print("=" * 60)
        print(f"API URL: {self.base_url}")
        print(f"Test image: {test_image_path}")

        results = {
            "health": False,
            "model_info": False,
            "predict": False,
            "predict_visual": False,
        }

        # Test health
        results["health"] = self.test_health()

        # Test model info
        results["model_info"] = self.test_model_info()

        # Test predict
        if Path(test_image_path).exists():
            results["predict"] = self.test_predict(test_image_path) is not None
            results["predict_visual"] = (
                self.test_predict_visual(test_image_path) is not None
            )
        else:
            print(f"\n⚠️  Test image not found: {test_image_path}")

        # Summary
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        for test, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{test:20s}: {status}")

        total = len(results)
        passed = sum(results.values())
        print(f"\nTotal: {passed}/{total} tests passed")

        return results


def main():
    """Main test function."""
    import argparse

    parser = argparse.ArgumentParser(description="Test KLGrade API")
    parser.add_argument(
        "--url", type=str, default="http://localhost:8001", help="API base URL"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to test image (optional, will use sample if not provided)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_outputs",
        help="Directory to save output images",
    )

    args = parser.parse_args()

    # Create tester
    tester = APITester(args.url)

    # Find test image
    if args.image:
        test_image = args.image
    else:
        # Try to find a sample image
        possible_paths = [
            "data/test/sample.jpg",
            "data/images/sample.jpg",
            "test_image.jpg",
        ]
        test_image = None
        for path in possible_paths:
            if Path(path).exists():
                test_image = path
                break

        if test_image is None:
            print("⚠️  No test image specified and no sample image found.")
            print("   Running only health and model_info tests.")
            tester.test_health()
            tester.test_model_info()
            return

    # Run all tests
    tester.run_all_tests(test_image)


if __name__ == "__main__":
    main()
