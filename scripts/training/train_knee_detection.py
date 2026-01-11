
import sys
import os
import argparse
from pathlib import Path
import shutil
import wandb
import random
import json
from ultralytics import YOLO

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def setup_symlinks(base_dir: Path, source_images: Path, source_labels: Path):
    """
    Creates symlinks for images and labels in the base_dir to emulate YOLO structure.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    
    img_sym = base_dir / "images"
    lbl_sym = base_dir / "labels"
    
    # Remove existing if they are symlinks or empty dirs
    if img_sym.is_symlink() or img_sym.exists():
        if img_sym.is_symlink(): img_sym.unlink()
        elif img_sym.is_dir(): shutil.rmtree(img_sym)
    
    if lbl_sym.is_symlink() or lbl_sym.exists():
        if lbl_sym.is_symlink(): lbl_sym.unlink()
        elif lbl_sym.is_dir(): shutil.rmtree(lbl_sym)

    # Create symlinks
    os.symlink(source_images.absolute(), img_sym)
    os.symlink(source_labels.absolute(), lbl_sym)
    print(f"✅ Created symlinks in {base_dir}")

def create_new_splits(base_dir: Path, source_images: Path):
    """
    Generates new 70/15/15 splits physically from the source images.
    """
    print("\n🔄 Generating new splits (70% Train, 15% Val, 15% Test)...")
    
    # Get all .jpg images
    all_images = list(source_images.glob("*.jpg"))
    if not all_images:
        print("❌ No images found for splitting.")
        return {}
        
    random.shuffle(all_images)
    
    n_total = len(all_images)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    # Remaining for test
    
    train_imgs = all_images[:n_train]
    val_imgs = all_images[n_train:n_train+n_val]
    test_imgs = all_images[n_train+n_val:]
    
    split_out_dir = base_dir / "splits"
    split_out_dir.mkdir(parents=True, exist_ok=True)
    
    splits = {}
    
    for name, imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
        # Convert to paths relative to the symlinked structure for YOLO (absolute is safest)
        # We need them to point to {base_dir}/images/{filename}
        paths = [(base_dir / "images" / img.name).absolute() for img in imgs]
        
        out_file = split_out_dir / f"{name}.txt"
        with open(out_file, "w") as f:
            f.write("\n".join([str(p) for p in paths]))
            
        splits[name] = str(out_file.absolute())
        print(f"   - {name}: {len(imgs)} images")
        
    return splits

def prepare_splits(base_dir: Path, original_split_dir: Path):
    """
    Reads original splits (dataset_v0) and adapts paths.
    """
    print("\nUse existing splits from dataset_v0...")
    split_out_dir = base_dir / "splits"
    split_out_dir.mkdir(parents=True, exist_ok=True)
    
    created_splits = {}
    
    for split_name in ["train.txt", "val.txt", "test.txt"]:
        orig_file = original_split_dir / split_name
        if not orig_file.exists():
            continue
            
        with open(orig_file, "r") as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            line = line.strip()
            if not line: continue
            filename = Path(line).name
            new_path = (base_dir / "images" / filename).absolute()
            new_lines.append(str(new_path))
            
        out_file = split_out_dir / split_name
        with open(out_file, "w") as f:
            f.write("\n".join(new_lines))
            
        created_splits[split_name.replace(".txt", "")] = str(out_file.absolute())
        print(f"   - Adapted {split_name}: {len(new_lines)} images")
        
    return created_splits

def train_knee_yolo(
    epochs: int = 100,
    batch_size: int = 32,
    img_size: int = 640,
    model_name: str = "yolo11n.pt",
    device: str = "0",
    project: str = "runs/detect",
    name: str = "knee_detection",
    resplit: bool = False
):
    print("=" * 60)
    print("YOLO11 Training - Knee Detection (One Class)")
    print("=" * 60)
    
    # Paths
    root_dir = Path.cwd()
    data_v0 = root_dir / "dataset/dataset_v0"
    proc_dir = root_dir / "processed/knee_detection"
    orig_split_dir = root_dir / "processed/splits/dataset_v0"
    
    source_images = data_v0 / "images"
    source_labels = data_v0 / "labels-knee"
    
    if not source_images.exists():
        print(f"❌ Error: Image dir not found: {source_images}")
        return
        
    # 1. Setup Symlinks
    setup_symlinks(proc_dir, source_images, source_labels)
    
    # 2. Prepare Splits
    if resplit:
        splits = create_new_splits(proc_dir, source_images)
    else:
        splits = prepare_splits(proc_dir, orig_split_dir)
        
    if "train" not in splits:
        print("❌ Error: Valid train split not found.")
        return

    # 3. Create Dataset YAML
    yaml_path = proc_dir / "dataset.yaml"
    yaml_data = {
        "path": str(proc_dir.absolute()),
        "train": splits.get("train"),
        "val": splits.get("val"),
        "test": splits.get("test", None),
        "names": { 0: "knee" }
    }
    
    import yaml
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    
    # 4. WandB
    wandb_project = "KLGrade-Knee-Detection"
    wandb_name = f"{name}_resplit" if resplit else name
    
    wandb.init(
        project=wandb_project,
        name=wandb_name,
        config={
            "model": model_name,
            "epochs": epochs,
            "batch": batch_size,
            "img_size": img_size,
            "resplit": resplit,
            "augment": True
        }
    )
    
    # 5. Train
    print(f"\n📦 Loading YOLO: {model_name}")
    model = YOLO(model_name)
    
    print("\n🚀 Starting training with AUGMENTATION...")
    try:
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            project=project,
            name=wandb_name,
            patience=50,
            save=True,
            save_period=10,
            plots=True,
            exist_ok=True,
            # Augmentations
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            shear=2.0,
            perspective=0.0005,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.15,
        )
        print("\n✅ Training completed!")
        
        # 6. Validate & Log Metrics
        print("\n📊 Validating...")
        metrics = model.val()
        
        # Extract meaningful metrics
        evaluation = {
            "map50": round(metrics.box.map50, 4),
            "map50_95": round(metrics.box.map, 4),
            "precision": round(metrics.box.mp, 4),
            "recall": round(metrics.box.mr, 4)
        }
        
        print("\n📈 Final Metrics:")
        print(json.dumps(evaluation, indent=2))
        
        # Save to file
        res_file = Path(project) / wandb_name / "evaluation_metrics.json"
        with open(res_file, "w") as f:
            json.dump(evaluation, f, indent=4)
        print(f"✅ Saved metrics to {res_file}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--name", type=str, default="knee_detect")
    parser.add_argument("--resplit", action="store_true", help="Generate new 70/15/15 splits")
    parser.add_argument("--test", action="store_true", help="Run quick test (1 epoch)")
    
    args = parser.parse_args()
    
    if args.test:
        train_knee_yolo(epochs=1, batch_size=4, name="test_run", resplit=args.resplit)
    else:
        train_knee_yolo(
            epochs=args.epochs,
            batch_size=args.batch,
            model_name=args.model,
            device=args.device,
            name=args.name,
            resplit=args.resplit
        )
