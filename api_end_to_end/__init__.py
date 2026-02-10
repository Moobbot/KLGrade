"""
End-to-End KIOCMIL Inference API Package

A standalone package for KL grade classification using End-to-End KIOCMIL models.
Supports 4-class, 5-class, 8-class, and 10-class classification.
"""

__version__ = "1.0.0"
__author__ = "KLGrade Team"

from .inference import EndToEndInference
from .output_formatters import (
    format_json,
    format_csv,
    format_visualization,
    save_summary,
)

__all__ = [
    "EndToEndInference",
    "format_json",
    "format_csv",
    "format_visualization",
    "save_summary",
]
