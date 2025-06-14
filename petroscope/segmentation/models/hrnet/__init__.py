"""
HRNetV2 + OCR segmentation model package.

This package provides the HRNetV2 with Object-Contextual Representations (OCR)
implementation that exactly matches the original paper implementation.
"""

from .nn import HRNetWithOCR

__all__ = ["HRNetWithOCR"]
