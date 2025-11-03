"""
Grad-CAM visualization for classifier models (e.g., ResNet).

Usage:
  python pipeline/gradcam.py --model models/cls_resnet50.pt --image <path_to_crop.jpg> --backbone resnet50

Saves grad-cam heatmap overlay next to the input image.
"""
import os
import sys
import argparse
import cv2
import numpy as np
import torch
from torch import nn

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipeline.model_cls import build_model
from config import CLASSES, IMG_SIZE


def preprocess_image(path, size=IMG_SIZE):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (size, size))
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - 0.5) / 0.25  # match normalization roughly
    return img, tensor.unsqueeze(0)  # returns BGR original for overlay, and normalized tensor [1,3,H,W]


def gradcam(model: nn.Module, x: torch.Tensor, target_layer_name: str, target_class: int = None):
    model.eval()
    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        activations['value'] = out.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0].detach()

    # resolve target layer; allow referring inside wrappers via 'backbone.' prefix transparently
    nm = dict([*model.named_modules()])
    target_layer = nm.get(target_layer_name, None)
    if target_layer is None and hasattr(model, 'backbone'):
        alt_name = f"backbone.{target_layer_name}"
        target_layer = nm.get(alt_name, None)
    if target_layer is None:
        raise ValueError(f"Layer {target_layer_name} not found. Use e.g. layer4.2 for ResNet50.")

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_backward_hook(backward_hook)

    logits = model(x)
    if target_class is None:
        target_class = int(torch.argmax(logits, dim=1))
    score = logits[0, target_class]

    model.zero_grad()
    score.backward(retain_graph=True)

    A = activations['value']  # [B, C, H, W]
    dA = gradients['value']   # [B, C, H, W]
    weights = torch.mean(dA, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
    cam = torch.sum(weights * A, dim=1, keepdim=False)  # [B, H, W]
    cam = torch.relu(cam)
    cam = cam[0].cpu().numpy()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    h1.remove()
    h2.remove()
    return cam, int(target_class), logits.detach().softmax(dim=1).cpu().numpy()[0]


def overlay_heatmap(img_bgr, cam, alpha=0.5):
    H, W = img_bgr.shape[:2]
    cam_resized = cv2.resize(cam, (W, H))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)
    return overlay


def parse_args():
    p = argparse.ArgumentParser(description="Grad-CAM for classifier")
    p.add_argument("--image", required=True, help="Path to a cropped classification image")
    p.add_argument("--model", required=True, help="Path to model weights (.pt)")
    p.add_argument("--backbone", default="resnet50", choices=["resnet50", "resnet101", "efficientnet_b0"])
    p.add_argument("--layer", default="layer4.2", help="Target layer name (e.g., layer4.2 for ResNet50)")
    p.add_argument("--out", default="gradcam_output.jpg")
    p.add_argument("--size", type=int, default=None, help="Resize for forward pass (defaults to config.IMG_SIZE)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.backbone, pretrained=False).to(device)
    sd = torch.load(args.model, map_location=device)
    model.load_state_dict(sd)

    size = args.size or IMG_SIZE
    img_bgr, x = preprocess_image(args.image, size=size)
    x = x.to(device)
    cam, cid, probs = gradcam(model, x, args.layer)
    print("Pred:", cid, list(CLASSES.values())[cid], "Prob:", float(probs[cid]))
    overlay = overlay_heatmap(img_bgr, cam)
    cv2.imwrite(args.out, overlay)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
