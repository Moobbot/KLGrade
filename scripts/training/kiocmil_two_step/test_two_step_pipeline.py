"""
Test Two-Step Pipeline: Knee Detection → KIOCMIL Classification

This script demonstrates the two-step approach:
1. Use YOLO to detect knees
2. Use KIOCMIL-CADA to classify KL grade
"""

import torch
from ultralytics import YOLO
from pathlib import Path
import sys
from PIL import Image
import numpy as np

# Add project root to path BEFORE importing src modules
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

from src.models.cada import KiocmilModelCADA


def load_knee_detector(checkpoint_path):
    """Load YOLO knee detection model."""
    print(f"Loading knee detector from {checkpoint_path}...")
    model = YOLO(checkpoint_path)
    print("✅ Knee detector loaded")
    return model


def load_kiocmil_classifier(checkpoint_path, num_classes=10, device="cuda"):
    """Load KIOCMIL-CADA classification model."""
    print(f"Loading KIOCMIL classifier from {checkpoint_path}...")

    model = KiocmilModelCADA(
        backbone_name="yolo11l",
        num_classes=num_classes,
        feature_dim=256,
        num_deformable_points=4,
        num_context_scales=3,
        use_positional_encoding=True,
        dropout=0.1,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # KIOCMIL checkpoints are direct state_dicts
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", "N/A")
    else:
        # Direct state_dict
        model.load_state_dict(checkpoint)
        epoch = "N/A"

    model = model.to(device)
    model.eval()

    print(f"✅ KIOCMIL classifier loaded (Epoch {epoch})")
    return model


def detect_knees(yolo_model, image_path, conf_threshold=0.5):
    """Detect knees in image using YOLO."""
    results = yolo_model(image_path, conf=conf_threshold, verbose=False)

    knees = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().item()
            knees.append(
                {"bbox": [int(x1), int(y1), int(x2), int(y2)], "confidence": conf}
            )

    return knees


def classify_knee(kiocmil_model, image_path, knee_bbox, device="cuda"):
    """Classify KL grade for a detected knee."""
    # Load image
    img = Image.open(image_path).convert("RGB")

    # Crop knee region
    x1, y1, x2, y2 = knee_bbox
    knee_img = img.crop((x1, y1, x2, y2))

    # Resize to expected size
    ctx_img = knee_img.resize((384, 384))

    # Convert to tensor
    ctx_tensor = torch.from_numpy(np.array(ctx_img)).permute(2, 0, 1).float() / 255.0

    # Prepare batch_data in KIOCMIL-CADA format
    # For now, use empty patches for JS and OST (in production, use actual lesion detection)
    batch_data = [
        {
            "knees": [
                {
                    "ctx": ctx_tensor,
                    "ctx_bbox": torch.tensor(
                        [0.0, 0.0, 1.0, 1.0], device=device
                    ),  # Normalized bbox
                    "js": torch.zeros(
                        0, 3, 224, 224, device=device
                    ),  # Empty JS patches
                    "js_bboxes": torch.zeros(0, 4, device=device),
                    "ost": torch.zeros(
                        0, 3, 224, 224, device=device
                    ),  # Empty OST patches
                    "ost_bboxes": torch.zeros(0, 4, device=device),
                }
            ]
        }
    ]

    # Inference
    with torch.no_grad():
        outputs = kiocmil_model(batch_data)

    # Get prediction from 10-class logits
    logits = outputs["logits_10"]
    pred_class = torch.argmax(logits, dim=1).item()
    confidence = torch.softmax(logits, dim=1)[0, pred_class].item()

    return pred_class, confidence


def visualize_results(image_path, knees, predictions, output_path):
    """Draw bounding boxes and predictions on image."""
    from PIL import ImageDraw, ImageFont

    # Load image
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Try to load a font, fallback to default if not available
    try:
        # Try common fonts
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if not Path(font_path).exists():
            font_path = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"

        font = ImageFont.truetype(font_path, 40)
        font_small = ImageFont.truetype(font_path, 20)
    except:
        font = ImageFont.load_default()
        font_small = font

    # Draw each knee
    for i, (knee, pred) in enumerate(zip(knees, predictions), 1):
        x1, y1, x2, y2 = knee["bbox"]

        # Draw bounding box (green)
        draw.rectangle([x1, y1, x2, y2], outline="green", width=5)

        # Draw knee number
        draw.text((x1 + 10, y1 + 10), f"Knee {i}", fill="green", font=font)

        # Draw detection confidence
        # det_text = f"Det: {knee['confidence']:.1%}"
        # draw.text((x1 + 10, y1 + 60), det_text, fill="green", font=font_small)

        # Draw KL grade prediction
        kl_text = f"{pred['label']}"
        draw.text((x1 + 10, y1 + 50), kl_text, fill="yellow", font=font)

        # Draw classification confidence
        cls_text = f"Conf: {pred['confidence']:.1%}"
        draw.text((x1 + 10, y1 + 95), cls_text, fill="yellow", font=font_small)

    # Save
    img.save(output_path)
    print(f"\n✅ Visualization saved to: {output_path}")


def test_two_step_pipeline(
    knee_detector_path,
    kiocmil_classifier_path,
    image_path,
    num_classes=10,
    device="cuda",
):
    """Run full two-step pipeline on an image."""
    print(f"\n{'='*60}")
    print(f"Testing Two-Step Pipeline")
    print(f"{'='*60}\n")

    # Load models
    knee_detector = load_knee_detector(knee_detector_path)
    kiocmil_classifier = load_kiocmil_classifier(
        kiocmil_classifier_path, num_classes, device
    )

    # Step 1: Detect knees
    print(f"\nStep 1: Detecting knees in {Path(image_path).name}...")
    knees = detect_knees(knee_detector, image_path)
    print(f"✅ Found {len(knees)} knee(s)")

    if len(knees) == 0:
        print("❌ No knees detected!")
        return

    # Step 2: Classify each knee
    print(f"\nStep 2: Classifying KL grade for each knee...")

    class_names = [
        "KL0-JS",
        "KL1-JS",
        "KL2-JS",
        "KL3-JS",
        "KL4-JS",
        "KL0-OST",
        "KL1-OST",
        "KL2-OST",
        "KL3-OST",
        "KL4-OST",
    ]

    predictions = []
    for i, knee in enumerate(knees, 1):
        print(f"\n  Knee {i}:")
        print(f"    BBox: {knee['bbox']}")
        print(f"    Detection Conf: {knee['confidence']:.2%}")

        pred_class, confidence = classify_knee(
            kiocmil_classifier, image_path, knee["bbox"], device
        )

        pred_label = (
            class_names[pred_class]
            if pred_class < len(class_names)
            else f"Class {pred_class}"
        )
        print(f"    KL Grade: {pred_label}")
        print(f"    Classification Conf: {confidence:.2%}")

        predictions.append({"label": pred_label, "confidence": confidence})

    # Visualize results
    output_vis_path = Path("runs/two_step_pipeline_vis.jpg")
    visualize_results(image_path, knees, predictions, output_vis_path)

    print(f"\n{'='*60}")
    print(f"Pipeline test completed!")
    print(f"{'='*60}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test Two-Step Pipeline")
    parser.add_argument(
        "--knee-detector",
        type=str,
        default="runs/my_knee_run_resplit/weights/best.pt",
        help="Path to YOLO knee detector checkpoint",
    )
    parser.add_argument(
        "--kiocmil-classifier",
        type=str,
        default="runs/kiocmil_cada/cada_10class_balanced/best_acc_model.pt",
        help="Path to KIOCMIL classifier checkpoint",
    )
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    parser.add_argument(
        "--num-classes", type=int, default=10, help="Number of classes (5 or 10)"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Check files exist
    if not Path(args.knee_detector).exists():
        print(f"❌ Knee detector not found: {args.knee_detector}")
        sys.exit(1)

    if not Path(args.kiocmil_classifier).exists():
        print(f"❌ KIOCMIL classifier not found: {args.kiocmil_classifier}")
        sys.exit(1)

    if not Path(args.image).exists():
        print(f"❌ Image not found: {args.image}")
        sys.exit(1)

    # Run pipeline
    test_two_step_pipeline(
        args.knee_detector,
        args.kiocmil_classifier,
        args.image,
        args.num_classes,
        args.device,
    )


if __name__ == "__main__":
    main()
