#!/usr/bin/env python3
"""
Fix split files to use RELATIVE paths instead of absolute paths
This makes them work on both Windows and Linux
"""

from pathlib import Path


def fix_splits_to_relative():
    """Convert all split files to use relative paths."""

    base_dir = Path(".")
    splits_dir = base_dir / "splits"

    # Mapping: split name -> relative path to images
    datasets = {
        "knee_5_class": "processed/knee_5_class/images",
        "knee_10_class": "processed/knee_10_class/images",
        "knee_4_class": "processed/knee_4_class/images",
        "knee_8_class": "processed/knee_8_class/images",
        # Balanced datasets
        "balanced_knees_cropped": "datasets/balanced/knees_cropped/images",
        "balanced_knees_cropped_4_class": "datasets/balanced/knees_cropped_4_class/images",
        "balanced_knees_cropped_8_class": "datasets/balanced/knees_cropped_8_class/images",
        "balanced_knees_cropped_10_class": "datasets/balanced/knees_cropped_10_class/images",
        "balanced_full_xray": "datasets/balanced/full_xray/images",
        "balanced_full_xray_4_class": "datasets/balanced/full_xray_4_class/images",
        "balanced_full_xray_8_class": "datasets/balanced/full_xray_8_class/images",
        "balanced_full_xray_10_class": "datasets/balanced/full_xray_10_class/images",
    }

    print("🔧 Fixing split files to use relative paths...")
    print("=" * 60)

    for split_name, img_dir_rel in datasets.items():
        split_dir = splits_dir / split_name

        if not split_dir.exists():
            print(f"\n⏭️  Skipping {split_name} (not found)")
            continue

        print(f"\n📁 Processing {split_name}...")

        for split_type in ["train", "val", "test"]:
            txt_file = split_dir / f"{split_type}.txt"

            if not txt_file.exists():
                continue

            # Read existing lines
            lines = txt_file.read_text(encoding="utf-8").strip().splitlines()

            if not lines:
                continue

            # Convert to relative paths
            new_lines = []
            img_dir_abs = base_dir / img_dir_rel

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # 1. Check if line is already a valid relative path
                # e.g. "processed/knee_8_class/images/file.jpg"
                if (base_dir / line).exists():
                    new_lines.append(line)
                    continue

                # 2. Check if line is a valid filename in the image dir
                # e.g. "file.jpg" (and img_dir/file.jpg exists)
                name = Path(line).name
                if (img_dir_abs / name).exists():
                    rel_path = f"{img_dir_rel}/{name}"
                    new_lines.append(rel_path)
                    continue

                # 3. Try fixing extension issues (e.g. file.jpg.jpg -> file.jpg)
                # This handles the double extension issue we created
                stem_clean = name
                while stem_clean.endswith(".jpg") or stem_clean.endswith(".png"):
                    stem_clean = Path(stem_clean).stem

                # Try finding file with standard extensions
                found_ext = None
                for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                    if (img_dir_abs / f"{stem_clean}{ext}").exists():
                        found_ext = ext
                        break

                if found_ext:
                    fname = f"{stem_clean}{found_ext}"
                    rel_path = f"{img_dir_rel}/{fname}"
                    new_lines.append(rel_path)
                else:
                    # Final fallback: just assume it was correct mostly and warn
                    # If it was "foo.jpg.jpg", and we couldn't find "foo.jpg",
                    # maybe we should check if "foo.jpg" exists directly?
                    # We did that in step 2.

                    print(f"   ⚠️  Warning: Image not found for {line} in {img_dir_abs}")
                    # Keep original to avoid data loss, or keep try to make valid path
                    # Let's try to construct a valid path from stem_clean + .jpg as a guess
                    rel_path = f"{img_dir_rel}/{stem_clean}.jpg"
                    if not (base_dir / rel_path).exists():
                        # If we still can't find it, just keep the original line but made relative if possible
                        # or just skip? treating it as missing is safer for training.
                        # But let's append what we think it should be.
                        new_lines.append(
                            rel_path
                        )  # Add it anyway so we can see the path in split file
                    else:
                        new_lines.append(rel_path)

            # Write back
            txt_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            print(f"   ✅ {split_type}.txt: {len(new_lines)} paths (now relative)")

    print("\n" + "=" * 60)
    print("✅ All split files now use relative paths!")
    print("\nExample:")
    print("  processed/knee/images/1234.jpg")
    print("  (works on both Windows and Linux)")


if __name__ == "__main__":
    fix_splits_to_relative()
