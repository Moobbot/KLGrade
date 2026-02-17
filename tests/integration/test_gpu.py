#!/usr/bin/env python3
"""
GPU Test Script - Verify GPU availability and functionality

Uses gpu_utils for shared GPU checking logic.
"""

import sys
from pathlib import Path

# Add scripts/testing to path for gpu_utils import
sys.path.insert(0, str(Path(__file__).parent))

from gpu_utils import print_gpu_report, test_gpu_computation


def test_gpu():
    """Run complete GPU test."""
    # Print detailed GPU report
    success = print_gpu_report()

    if not success:
        return False

    # Run computation test
    success = test_gpu_computation()

    if success:
        print("\n" + "=" * 60)
        print("✅ All GPU tests passed!")
        print("=" * 60)

    return success


if __name__ == "__main__":
    success = test_gpu()
    sys.exit(0 if success else 1)
