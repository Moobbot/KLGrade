"""
KLGrade Source Module

Main package for KLGrade object detection.
"""

# Make config easily accessible
from . import config
from . import datasets
from . import utils

__all__ = ["config", "datasets", "utils"]
