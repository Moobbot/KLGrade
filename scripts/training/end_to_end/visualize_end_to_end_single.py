"""
Visualize single image prediction (memory-efficient version).
Processes one image at a time to avoid OOM.
"""

import torch
import sys
from pathlib import Path
from PIL import Image, ImageDraw
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.kiocmil_with_detection import KiocmilWithDetection


def visualize_single_image(
    checkpoint_path, image_path, output_dir, image_size=384, device="cuda"
):
    """Load model, process one image, save result, and clean up."""

    # Load model
    model = KiocmilWithDetection(
        backbone_name="yolo11s",
        num_classes=10,
        pretrained_kiocmil=None,
        freeze_kiocmil=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load and preprocess image
    img_pil = Image.open(image_path).convert("RGB")
    img_resized = img_pil.resize((image_size, image_size))
    img_tensor = T.ToTensor()(img_resized).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(img_tensor)

    # Get predictions
    knee_boxes = outputs["knee_boxes"][0].cpu()
    knee_confs = outputs["knee_confs"][0, :, 0].cpu()
    lesion_boxes = outputs["lesion_boxes"][0].cpu()
    lesion_confs = outputs["lesion_confs"][0].cpu()
    logits = outputs["logits_10"][0].cpu()

    pred_class = torch.argmax(logits).item()
    confidence = torch.softmax(logits, dim=0)[pred_class].item()

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
    pred_label = (
        class_names[pred_class]
        if pred_class < len(class_names)
        else f"Class {pred_class}"
    )

    # Draw visualization
    img_vis = img_pil.copy()
    draw = ImageDraw.Draw(img_vis)
    w, h = img_vis.size

    def draw_boxes(boxes, confs, color, label, threshold=0.3, max_boxes=5):
        valid = (confs > threshold).nonzero(as_tuple=True)[0]
        if len(valid) == 0:
            return 0
        top = valid[torch.argsort(confs[valid], descending=True)][:max_boxes]
        for idx in top:
            cx, cy, bw, bh = boxes[idx]
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1, y1 - 15), f"{label} {confs[idx]:.2f}", fill=color)
        return len(top)

    n_knees = draw_boxes(knee_boxes, knee_confs, "green", "Knee")
    n_js = draw_boxes(lesion_boxes, lesion_confs[:, 0], "blue", "JS")
    n_ost = draw_boxes(lesion_boxes, lesion_confs[:, 1], "red", "OST")

    # Draw prediction
    draw.rectangle([0, 0, w, 40], fill="black")
    draw.text((10, 10), f"Prediction: {pred_label} ({confidence:.1%})", fill="white")

    # Save
    output_path = Path(output_dir) / f"vis_{Path(image_path).name}"
    img_vis.save(output_path)

    print(
        f"  ✅ {pred_label} ({confidence:.1%}) - {n_knees} knees, {n_js} JS, {n_ost} OST"
    )
    print(f"  Saved: {output_path.name}")

    # Cleanup
    del model, outputs, img_tensor
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    try:
        visualize_single_image(
            args.checkpoint, args.image, args.output_dir, args.image_size, args.device
        )
    except Exception as e:
        print(f"  ❌ Error: {e}")
        sys.exit(1)
