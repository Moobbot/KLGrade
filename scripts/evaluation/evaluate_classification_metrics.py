#!/usr/bin/env python3
"""
Script đánh giá classification metrics cho YOLO detection model.
Dùng model.val() (YOLO built-in) để lấy per-class precision/recall và confusion matrix,
sau đó tính Accuracy, F1, AUC từ confusion matrix.

Cách dùng:
    conda activate klgrade
    cd e:/CaoHoc/thesis/code/web-knee/KLGrade

    python scripts/evaluation/evaluate_classification_metrics.py \
        --model runs/detect/balanced_knees_cropped_8_class_80_10_10/weights/best.pt \
        --data datasets/splits/balanced_knees_cropped_8_class_80_10_10/dataset.yaml \
        --split test \
        --output runs/evaluate/8class_80_10_10_classification
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def evaluate(
    model_path: str,
    data_path: str,
    split: str,
    output_dir: str,
    conf: float = 0.001,
    iou: float = 0.6,
    device: str = "0",
    imgsz: int = 640,
):
    try:
        from ultralytics import YOLO
        from sklearn.metrics import f1_score
    except ImportError as e:
        print(f"Thiếu thư viện: {e}")
        print("Cài đặt: pip install scikit-learn ultralytics")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Đọc class names từ dataset.yaml ──────────────────────────────
    data_path = Path(data_path)
    with open(data_path, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    nc = data_cfg.get("nc", 8)
    names_raw = data_cfg.get("names", {})
    if isinstance(names_raw, dict):
        class_names = [names_raw[i] for i in range(nc)]
    else:
        class_names = list(names_raw)

    # ── Tạo dataset.yaml tạm với Windows path ─────────────────────────
    split_dir = data_path.parent.resolve()
    cwd = Path.cwd().resolve()

    # Chuyển các đường dẫn Linux → Windows absolute
    def fix_path(p_str):
        if not p_str:
            return p_str
        p = Path(p_str)
        parts = p.parts
        # Tìm "KLGrade" và lấy phần SAU nó
        for i, part in enumerate(parts):
            if part == "KLGrade" and i + 1 < len(parts):
                rel = Path(*parts[i + 1 :])  # ví dụ: datasets/splits/.../train.txt
                return str(cwd / rel)
        # Không tìm thấy KLGrade → tìm "datasets"
        for i, part in enumerate(parts):
            if part == "datasets":
                rel = Path(*parts[i:])
                return str(cwd / rel)
        # Fallback: nếu là relative thì join với split_dir
        if not p.is_absolute():
            return str(split_dir / p_str)
        return p_str

    train_txt = fix_path(data_cfg.get("train", ""))
    val_txt = fix_path(data_cfg.get("val", ""))
    test_txt = fix_path(data_cfg.get("test", ""))

    # Viết yaml tạm
    tmp_yaml = output_dir / "dataset_tmp.yaml"
    tmp_cfg = {
        "path": str(cwd),
        "train": train_txt,
        "val": val_txt,
        "test": test_txt,
        "nc": nc,
        "names": {i: n for i, n in enumerate(class_names)},
    }
    with open(tmp_yaml, "w", encoding="utf-8") as f:
        yaml.dump(tmp_cfg, f, allow_unicode=True)
    print(f"Dataset yaml tạm: {tmp_yaml}")
    print(f"  train: {train_txt}")
    print(f"  val  : {val_txt}")
    print(f"  test : {test_txt}")

    print("=" * 65)
    print(f"Model  : {model_path}")
    print(f"Dataset: {data_path.name} | nc={nc} | split={split}")
    print(f"Classes: {class_names}")
    print("=" * 65)

    # ── Chạy model.val() ─────────────────────────────────────────────
    model = YOLO(model_path)
    print(f"\nChạy val() trên tập {split}...")

    val_results = model.val(
        data=str(tmp_yaml),
        split=split,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        batch=8,
        project=str(output_dir),
        name="yolo_val",
        exist_ok=True,
        plots=True,
        verbose=False,
    )

    # ── Trích xuất detection metrics ─────────────────────────────────
    box = val_results.box
    map50 = box.map50
    map5095 = box.map
    prec_all = box.mp  # mean precision
    rec_all = box.mr  # mean recall

    # Per-class P, R (shape: [nc])
    per_cls_p = np.array(box.p) if hasattr(box, "p") else np.zeros(nc)
    per_cls_r = np.array(box.r) if hasattr(box, "r") else np.zeros(nc)
    per_cls_ap50 = np.array(box.ap50) if hasattr(box, "ap50") else np.zeros(nc)

    # Per-class F1
    per_cls_f1 = 2 * per_cls_p * per_cls_r / (per_cls_p + per_cls_r + 1e-9)

    # ── Đọc confusion matrix từ val_results ─────────────────────────
    # YOLO lưu confusion_matrix trong results.confusion_matrix
    cm_obj = val_results.confusion_matrix
    cm = np.array(cm_obj.matrix, dtype=int) if cm_obj is not None else None

    # Tính Accuracy từ confusion matrix (bỏ hàng/cột "background")
    accuracy = None
    f1_macro = None
    f1_weighted = None

    if cm is not None:
        # YOLO confusion matrix có kích thước (nc+1) x (nc+1) (bao gồm background)
        # Bỏ cột/hàng cuối (background)
        cm_cls = cm[:nc, :nc]

        total_correct = np.trace(cm_cls)
        total_pred = cm_cls.sum()
        accuracy = float(total_correct) / float(total_pred) if total_pred > 0 else 0.0

        # F1 từ confusion matrix
        tp = np.diag(cm_cls).astype(float)
        fp = cm_cls.sum(axis=0) - tp
        fn = cm_cls.sum(axis=1) - tp

        p_cls = tp / (tp + fp + 1e-9)
        r_cls = tp / (tp + fn + 1e-9)
        f1_cls = 2 * p_cls * r_cls / (p_cls + r_cls + 1e-9)

        support = cm_cls.sum(axis=1).astype(float)
        f1_macro = float(np.mean(f1_cls))
        f1_weighted = float(np.sum(f1_cls * support) / (support.sum() + 1e-9))

    # AUC ước tính từ per-class AP50
    auc_macro = float(np.mean(per_cls_ap50)) if len(per_cls_ap50) == nc else None

    # ── In kết quả ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("KẾT QUẢ DETECTION METRICS (YOLO val)")
    print("=" * 65)
    print(f"  mAP@50       : {map50:.4f}")
    print(f"  mAP@50-95    : {map5095:.4f}")
    print(f"  Precision    : {prec_all:.4f}")
    print(f"  Recall       : {rec_all:.4f}")

    print("\n" + "=" * 65)
    print("KẾT QUẢ CLASSIFICATION METRICS")
    print("=" * 65)
    if accuracy is not None:
        print(f"  Accuracy     : {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  F1 Macro     : {f1_macro:.4f}")
        print(f"  F1 Weighted  : {f1_weighted:.4f}")
    print(f"  AUC (mean AP@50): {auc_macro:.4f}" if auc_macro else "  AUC: N/A")

    print(f"\n{'Lớp':<20} {'P':>7} {'R':>7} {'F1':>7} {'AP50':>7}")
    print("-" * 50)
    for i, name in enumerate(class_names):
        p = per_cls_p[i] if i < len(per_cls_p) else 0
        r = per_cls_r[i] if i < len(per_cls_r) else 0
        f1 = per_cls_f1[i] if i < len(per_cls_f1) else 0
        ap = per_cls_ap50[i] if i < len(per_cls_ap50) else 0
        print(f"  {name:<18} {p:>7.3f} {r:>7.3f} {f1:>7.3f} {ap:>7.3f}")

    if cm is not None:
        print(f"\nConfusion Matrix (hàng=GT, cột=Pred, nc={nc}):")
        header = f"{'':>6}" + "".join(f"{n[:5]:>7}" for n in class_names)
        print(header)
        for i, row in enumerate(cm_cls):
            row_str = f"{class_names[i][:5]:>6}" + "".join(f"{v:>7}" for v in row)
            print(row_str)

    # ── Lưu kết quả JSON ────────────────────────────────────────────
    output = {
        "model": str(model_path),
        "dataset": str(data_path),
        "split": split,
        "detection_metrics": {
            "mAP50": float(map50),
            "mAP50_95": float(map5095),
            "precision": float(prec_all),
            "recall": float(rec_all),
        },
        "classification_metrics": {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "auc_mean_ap50": float(auc_macro) if auc_macro else None,
        },
        "per_class": {
            class_names[i]: {
                "precision": float(per_cls_p[i]) if i < len(per_cls_p) else 0,
                "recall": float(per_cls_r[i]) if i < len(per_cls_r) else 0,
                "f1": float(per_cls_f1[i]) if i < len(per_cls_f1) else 0,
                "ap50": float(per_cls_ap50[i]) if i < len(per_cls_ap50) else 0,
            }
            for i in range(nc)
        },
        "confusion_matrix": cm_cls.tolist() if cm is not None else None,
        "class_names": class_names,
    }

    out_file = output_dir / "classification_metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # ── Vẽ confusion matrix ─────────────────────────────────────────
    if cm is not None:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, axes = plt.subplots(1, 2, figsize=(22, 9))

            sns.heatmap(
                cm_cls,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                ax=axes[0],
            )
            acc_str = f"{accuracy*100:.1f}%" if accuracy else "N/A"
            axes[0].set_title(
                f"Confusion Matrix (counts)\nAccuracy = {acc_str}", fontsize=12
            )
            axes[0].set_xlabel("Predicted"), axes[0].set_ylabel("Ground Truth")
            plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right")

            cm_norm = cm_cls.astype(float) / (cm_cls.sum(axis=1, keepdims=True) + 1e-9)
            sns.heatmap(
                cm_norm,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                ax=axes[1],
            )
            f1_str = f"{f1_macro:.3f}" if f1_macro else "N/A"
            axes[1].set_title(
                f"Confusion Matrix (normalized)\nF1-macro = {f1_str}", fontsize=12
            )
            axes[1].set_xlabel("Predicted"), axes[1].set_ylabel("Ground Truth")
            plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")

            plt.tight_layout()
            cm_file = output_dir / "confusion_matrix_classification.png"
            plt.savefig(str(cm_file), dpi=150, bbox_inches="tight")
            plt.close()
            print(f"\nConfusion matrix plot: {cm_file}")
        except ImportError:
            print("Cài thêm: pip install matplotlib seaborn")

    print(f"\nKết quả lưu tại: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Đánh giá classification metrics cho YOLO (dùng model.val)"
    )
    parser.add_argument("--model", required=True, help="Đường dẫn weights .pt")
    parser.add_argument("--data", required=True, help="Đường dẫn dataset.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--output", default="runs/evaluate/classification", help="Thư mục lưu kết quả"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold (dùng thấp cho val)",
    )
    parser.add_argument("--iou", type=float, default=0.6, help="IoU NMS threshold")
    parser.add_argument("--device", type=str, default="0", help="Device: 0=GPU, cpu")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        data_path=args.data,
        split=args.split,
        output_dir=args.output,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        imgsz=args.imgsz,
    )
