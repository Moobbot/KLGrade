from pathlib import Path

base_dir = Path("E:/CaoHoc/thesis/KLGrade")
processed_dir = base_dir / "processed"

# keys = directory name in splits/, value = path to images
datasets = {
    "knee_10_class": processed_dir / "knee_10_class/images",
    "knee_4_class": processed_dir / "knee_4_class/images",
    "knee_8_class": processed_dir / "knee_8_class/images",
    "knee_5_class": processed_dir / "knee/dataset_yolo/images",
}

for split_name, img_dir in datasets.items():
    split_dir = base_dir / "splits" / split_name
    if not split_dir.exists():
        print(f"Skipping {split_name}, dir not found: {split_dir}")
        continue

    print(f"Processing {split_name} (Image Dir: {img_dir})...")
    # Verify image dir exists
    if not img_dir.exists():
        print(f"  WARNING: Image dir {img_dir} NOT found. Skipping.")
        continue

    for split_type in ["train", "val", "test"]:
        txt_file = split_dir / f"{split_type}.txt"
        if not txt_file.exists():
            continue

        lines = txt_file.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            continue

        # Check if already paths AND verify they point to correct location
        first_line = lines[0]
        if ":" in first_line or first_line.startswith("/"):
            # It has a path. Check if it matches expected root?
            # If it points to old location, we might want to update it.
            # But for now, if it's absolute, assume it's valid.
            # Actually, 5-class might be empty/relative.
            # Let's simple Check if strict filename (no slashes)
            if "\\" not in first_line and "/" not in first_line:
                pass  # Needs fixing
            else:
                print(f"  {split_type}.txt already has paths. Checking if valid...")
                # Optional: rewrite anyway to ensure consistency?
                # Let's rewrite anyway.
                pass

        # Regenerate paths
        new_lines = []
        for line in lines:
            # Extract just filename if line is already a path
            fname = Path(line).name

            full_path = img_dir / fname
            # Force forward slashes for cross-platform/YOLO compatibility
            new_lines.append(str(full_path.absolute()).replace("\\", "/"))

        # Write back
        txt_file.write_text("\n".join(new_lines), encoding="utf-8")
        print(f"  Updated {split_type}.txt with {len(new_lines)} paths.")
