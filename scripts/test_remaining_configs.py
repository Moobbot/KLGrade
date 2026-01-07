import subprocess
import sys
from pathlib import Path

# List of configs to test (skipping 4 and 8 which passed)
configs = [
    "configs/yolo_5_class_baseline.yaml",
    "configs/yolo_5_class_conservative.yaml",
    "configs/yolo_10_class_baseline.yaml",
    "configs/yolo_10_class_conservative.yaml",
]

log_file = Path("docs/CONFIG_TEST_LOG_REMAINING.md")
log_content = ["# Configuration Test Results (Remaining)\n"]

for config in configs:
    config_path = Path(config)
    print(f"Testing {config}...")

    # Run YOLO training for 1 epoch (faster debug)
    cmd = [
        ".venv/Scripts/yolo",
        "detect",
        "train",
        f"data={config}",
        "epochs=1",
        "batch=8",
        "imgsz=640",
        "project=runs/test_configs",
        f"name={config_path.stem}",
        "exist_ok=True",
    ]

    try:
        # Capture output using utf-8 and errors=ignore to prevents crashes
        result = subprocess.run(
            cmd, capture_output=True, encoding="utf-8", errors="ignore"
        )

        status = "UNKNOWN"
        if result.returncode == 0:
            if "0 backgrounds" in result.stdout or "0 backgrounds" in result.stderr:
                status = "✅ Success (0 backgrounds)"
            elif (
                "No labels found" in result.stdout or "No labels found" in result.stderr
            ):
                status = "❌ Failed (No labels)"
            else:
                status = "✅ Success (Command finished)"
        else:
            status = f"❌ Failed (Return code {result.returncode})"

        print(f"  Result: {status}")
        log_content.append(
            f"## {config}\nStatus: {status}\n\n### Output Snippet:\n```\n{result.stderr[-1000:]}\n{result.stdout[-1000:]}\n```\n"
        )

    except Exception as e:
        print(f"  Result: ❌ Exception {e}")
        log_content.append(f"## {config}\n❌ Exception: {e}\n")

# Write log
log_file.write_text("\n".join(log_content), encoding="utf-8")
print(f"Test complete. Log saved to {log_file}")
