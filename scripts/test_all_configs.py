import subprocess
import sys
from pathlib import Path

# List of configs to test
configs = [
    "configs/yolo_4_class_baseline.yaml",
    "configs/yolo_4_class_conservative.yaml",
    "configs/yolo_5_class_baseline.yaml",
    "configs/yolo_5_class_conservative.yaml",
    "configs/yolo_8_class_baseline.yaml",
    "configs/yolo_8_class_conservative.yaml",
    "configs/yolo_10_class_baseline.yaml",
    "configs/yolo_10_class_conservative.yaml",
    "configs/test_2epochs.yaml",
    "configs/test_minimal.yaml",
]

log_file = Path("docs/CONFIG_TEST_LOG.md")
log_content = ["# Configuration Test Results (2 Epochs)\n"]

for config in configs:
    config_path = Path(config)
    if not config_path.exists():
        print(f"Config not found: {config}")
        log_content.append(f"- **{config}**: ❌ File not found")
        continue

    print(f"Testing {config}...")

    # Run YOLO training for 2 epochs
    cmd = [
        ".venv/Scripts/yolo",
        "detect",
        "train",
        f"data={config}",
        "epochs=2",
        "batch=8",
        "imgsz=640",
        "project=runs/test_configs",
        f"name={config_path.stem}",
        "exist_ok=True",
    ]

    try:
        # Capture output
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            # check for "0 backgrounds" or similar success indicators in stdout/stderr
            if "0 backgrounds" in result.stdout or "0 backgrounds" in result.stderr:
                status = "✅ Success (0 backgrounds)"
            elif (
                "No labels found" in result.stdout or "No labels found" in result.stderr
            ):
                status = "❌ Failed (No labels)"
            else:
                # Fallback check
                status = "✅ Success (Command finished)"

            print(f"  Result: {status}")
            log_content.append(f"- **{config}**: {status}")
        else:
            print(f"  Result: ❌ Failed (Return code {result.returncode})")
            log_content.append(
                f"- **{config}**: ❌ Failed (Error)\n  <details><summary>Log</summary>\n\n```\n{result.stderr[-500:]}\n```\n</details>"
            )

    except Exception as e:
        print(f"  Result: ❌ Exception {e}")
        log_content.append(f"- **{config}**: ❌ Exception: {e}")

# Write log
log_file.write_text("\n".join(log_content), encoding="utf-8")
print(f"Test complete. Log saved to {log_file}")
