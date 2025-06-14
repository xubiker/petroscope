"""
HRNetV2+OCR Module Initialization

This module initializes the HRNetV2+OCR segmentation model and makes it
available to the petroscope library.
"""

from petroscope.segmentation.models.hrnet.model import HRNetWithOCR
from petroscope.segmentation.models.hrnet.nn import HRNetOCR, HRNetV2

__all__ = ["HRNetWithOCR", "HRNetOCR", "HRNetV2"]
