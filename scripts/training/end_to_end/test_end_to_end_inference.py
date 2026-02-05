"""
Quick inference test for end-to-end model.
Tests the trained model on a few sample images.
"""

import torch
import sys
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.kiocmil_with_detection import KiocmilWithDetection


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")

    # Initialize model
    model = KiocmilWithDetection(
        backbone_name="yolo11s",
        num_classes=10,
        pretrained_kiocmil=None,
        freeze_kiocmil=False,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"✅ Model loaded successfully")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Val Loss: {checkpoint.get('metrics', {}).get('val_loss', 'N/A'):.4f}")
    print(f"   Val Acc: {checkpoint.get('metrics', {}).get('val_accuracy', 'N/A'):.4f}")

    return model


def load_image(image_path: str, size: int = 640):
    """Load and preprocess image."""
    img = Image.open(image_path).convert("RGB")

    transform = T.Compose(
        [
            T.Resize((size, size)),
            T.ToTensor(),
        ]
    )

    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img_tensor


def test_inference(model, image_paths, device="cuda", image_size=640):
    """Run inference on sample images."""
    print(f"\n{'='*60}")
    print("Running inference on sample images")
    print(f"{'='*60}\n")

    with torch.no_grad():
        for img_path in image_paths:
            print(f"Processing: {img_path}")

            # Load image
            img = load_image(img_path, size=image_size).to(device)
            print(f"  Image shape: {img.shape}")

            # Run inference
            try:
                outputs = model(img)

                # Print results
                print(f"  ✅ Inference successful!")
                print(f"     Knee boxes shape: {outputs['knee_boxes'].shape}")
                print(f"     Knee confs shape: {outputs['knee_confs'].shape}")
                print(f"     Lesion boxes shape: {outputs['lesion_boxes'].shape}")
                print(f"     Lesion confs shape: {outputs['lesion_confs'].shape}")
                print(f"     Logits 10-class shape: {outputs['logits_10'].shape}")

                # Get prediction
                pred_class = torch.argmax(outputs["logits_10"], dim=1).item()
                confidence = torch.softmax(outputs["logits_10"], dim=1)[
                    0, pred_class
                ].item()
                print(
                    f"     Predicted class: {pred_class} (confidence: {confidence:.4f})"
                )

            except Exception as e:
                print(f"  ❌ Error: {e}")

            print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test end-to-end model inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/end_to_end/test_5epochs/best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="datasets/dataset_v0/images",
        help="Directory containing test images",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of sample images to test",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Input image size (default: 640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)",
    )

    args = parser.parse_args()

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load model
    model = load_model(args.checkpoint, device)

    # Get sample images
    image_dir = Path(args.image_dir)
    image_paths = sorted(list(image_dir.glob("*.jpg")))[: args.num_samples]

    if not image_paths:
        # Try PNG if no JPG found
        image_paths = sorted(list(image_dir.glob("*.png")))[: args.num_samples]

    if not image_paths:
        print(f"❌ No images found in {image_dir}")
        return

    print(f"\nFound {len(image_paths)} sample images")

    # Run inference
    test_inference(model, image_paths, device, args.image_size)

    print(f"\n{'='*60}")
    print("Inference test completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
