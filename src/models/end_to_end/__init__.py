"""End-to-end KIOCMIL models.

Models that perform both detection and classification.
"""

from .kiocmil_end_to_end import KiocmilEndToEnd
from .kiocmil_with_detection import KiocmilWithDetection

__all__ = [
    "KiocmilEndToEnd",
    "KiocmilWithDetection",
]
