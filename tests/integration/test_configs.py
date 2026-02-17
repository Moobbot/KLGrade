#!/usr/bin/env python3
"""
Unified Config Testing Script

Test YOLO configurations with customizable parameters.
Consolidates test_all_configs.py and test_remaining_configs.py.

Usage:
    # Test all configs (2 epochs)
    python scripts/testing/test_configs.py

    # Test specific configs
    python scripts/testing/test_configs.py --configs yolo_5_class_baseline.yaml yolo_10_class_baseline.yaml

    # Test with 1 epoch (faster)
    python scripts/testing/test_configs.py --epochs 1

    # Custom log file
    python scripts/testing/test_configs.py --log-file my_test.md
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Default config sets
DEFAULT_CONFIGS = [
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

FIVE_TEN_CLASS_CONFIGS = [
    "configs/yolo_5_class_baseline.yaml",
    "configs/yolo_5_class_conservative.yaml",
    "configs/yolo_10_class_baseline.yaml",
    "configs/yolo_10_class_conservative.yaml",
]


def test_configs(
    configs_list,
    epochs=2,
    batch=8,
    imgsz=640,
    log_file="docs/CONFIG_TEST_LOG.md",
    yolo_cmd=".venv/Scripts/yolo",
):
    """
    Test YOLO configurations.

    Args:
        configs_list: List of config file paths
        epochs: Number of training epochs
        batch: Batch size
        imgsz: Image size
        log_file: Output log file path
        yolo_cmd: YOLO command path
    """
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    log_content = [
        f"# Configuration Test Results ({epochs} Epoch{'s' if epochs != 1 else ''})\n"
    ]

    print("=" * 60)
    print(f"Testing {len(configs_list)} configurations")
    print(f"Epochs: {epochs}, Batch: {batch}, ImgSz: {imgsz}")
    print("=" * 60)

    for config in configs_list:
        config_path = Path(config)

        if not config_path.exists():
            print(f"\n❌ Config not found: {config}")
            log_content.append(f"- **{config}**: ❌ File not found")
            continue

        print(
            f"\n[{configs_list.index(config)+1}/{len(configs_list)}] Testing {config}..."
        )

        # Build command
        cmd = [
            yolo_cmd,
            "detect",
            "train",
            f"data={config}",
            f"epochs={epochs}",
            f"batch={batch}",
            f"imgsz={imgsz}",
            "project=runs/test_configs",
            f"name={config_path.stem}",
            "exist_ok=True",
        ]

        try:
            # Run training
            result = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore"
            )

            # Determine status
            if result.returncode == 0:
                if "0 backgrounds" in result.stdout or "0 backgrounds" in result.stderr:
                    status = "✅ Success (0 backgrounds)"
                elif (
                    "No labels found" in result.stdout
                    or "No labels found" in result.stderr
                ):
                    status = "❌ Failed (No labels)"
                else:
                    status = "✅ Success (Command finished)"

                print(f"  Result: {status}")
                log_content.append(f"- **{config}**: {status}")
            else:
                print(f"  Result: ❌ Failed (Return code {result.returncode})")
                # Truncate error log to last 500 chars
                error_snippet = (
                    result.stderr[-500:] if result.stderr else "No error output"
                )
                log_content.append(
                    f"- **{config}**: ❌ Failed (Error)\n"
                    f"  <details><summary>Log</summary>\n\n```\n{error_snippet}\n```\n</details>"
                )

        except Exception as e:
            print(f"  Result: ❌ Exception {e}")
            log_content.append(f"- **{config}**: ❌ Exception: {e}")

    # Write log
    log_file.write_text("\n".join(log_content), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"✅ Test complete. Log saved to {log_file}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Test YOLO configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--configs", nargs="+", help="Config files to test (default: all configs)"
    )

    parser.add_argument(
        "--preset",
        choices=["all", "5-10-class"],
        default="all",
        help="Use preset config list (all or 5-10-class only)",
    )

    parser.add_argument(
        "--epochs", type=int, default=2, help="Number of training epochs (default: 2)"
    )

    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")

    parser.add_argument(
        "--imgsz", type=int, default=640, help="Image size (default: 640)"
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default="docs/CONFIG_TEST_LOG.md",
        help="Output log file (default: docs/CONFIG_TEST_LOG.md)",
    )

    parser.add_argument(
        "--yolo-cmd",
        type=str,
        default="yolo",  # Use 'yolo' directly (works on Linux/Mac with conda/pip install)
        help="YOLO command path (default: yolo)",
    )

    args = parser.parse_args()

    # Determine config list
    if args.configs:
        configs_list = args.configs
    elif args.preset == "5-10-class":
        configs_list = FIVE_TEN_CLASS_CONFIGS
    else:
        configs_list = DEFAULT_CONFIGS

    test_configs(
        configs_list=configs_list,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        log_file=args.log_file,
        yolo_cmd=args.yolo_cmd,
    )


if __name__ == "__main__":
    main()
