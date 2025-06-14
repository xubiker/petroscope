"""
HRNetV2 segmentation model implementation.

This module provides the HRNetV2 wrapper class that inherits from
PatchSegmentationModel and integrates with the petroscope training pipeline.
"""

from typing import Any

from petroscope.segmentation.models.base import PatchSegmentationModel
from petroscope.utils.lazy_imports import torch  # noqa


class HRNetWithOCR(PatchSegmentationModel):
    """
    HRNetV2 with Object-Contextual Representations for segmentation.

    This model combines the high-resolution features from HRNet with
    a simplified attention mechanism for improved segmentation performance.

    Supported configurations:
    - Width: 18, 32, 48 (controls the number of channels in each stream)
    - OCR mid channels: Number of channels in the attention module
    - Dropout: Dropout rate for regularization
    - Use aux head: Whether to use auxiliary head during training
    """

    MODEL_REGISTRY: dict[str, str] = {
        # Model registry will be populated as models are trained
    }

    def __init__(
        self,
        n_classes: int,
        device: str,
        backbone: str = "hrnetv2_w32",
        pretrained: bool = True,
        ocr_mid_channels: int = 512,
        dropout: float = 0.1,
        use_aux_head: bool = True,
    ) -> None:
        """
        Initialize the HRNetV2 model.

        Args:
            n_classes: Number of segmentation classes
            device: Device to run the model on ('cuda', 'cpu', etc.)
            backbone: Backbone variant. Options:
                - "hrnetv2_w18": HRNetV2 with width 18
                - "hrnetv2_w32": HRNetV2 with width 32
                - "hrnetv2_w48": HRNetV2 with width 48
            pretrained: Whether to use pretrained weights (not implemented yet)
            ocr_mid_channels: Number of channels in OCR attention module
            dropout: Dropout rate for regularization
            use_aux_head: Whether to use auxiliary head during training
        """
        super().__init__(n_classes, device)

        from petroscope.segmentation.models.hrnet.nn import HRNetWithOCR

        # Parse backbone to get width
        if backbone == "hrnetv2_w18":
            width = 18
        elif backbone == "hrnetv2_w32":
            width = 32
        elif backbone == "hrnetv2_w48":
            width = 48
        else:
            raise ValueError(
                f"Unsupported backbone: {backbone}. "
                f"Supported: hrnetv2_w18, hrnetv2_w32, hrnetv2_w48"
            )

        self.backbone = backbone
        self.pretrained = pretrained
        self.ocr_mid_channels = ocr_mid_channels
        self.dropout = dropout
        self.use_aux_head = use_aux_head
        self.width = width

        # Create the model
        self.model = HRNetWithOCR(
            num_classes=n_classes,
            width=width,
            in_channels=3,
            ocr_mid_channels=ocr_mid_channels,
            dropout=dropout,
            use_aux_head=use_aux_head,
        ).to(self.device)

    def _get_checkpoint_data(self) -> dict[str, Any]:
        """Return model-specific data for checkpoint saving."""
        return {
            "n_classes": self.n_classes,
            "backbone": self.backbone,
            "pretrained": self.pretrained,
            "ocr_mid_channels": self.ocr_mid_channels,
            "dropout": self.dropout,
            "use_aux_head": self.use_aux_head,
        }

    @classmethod
    def _create_from_checkpoint(
        cls, checkpoint: dict, device: str
    ) -> "HRNetWithOCR":
        """Create an HRNetV2 model from checkpoint data."""
        # Extract architecture hyperparameters from checkpoint
        n_classes = checkpoint["n_classes"]
        backbone = checkpoint["backbone"]
        pretrained = checkpoint["pretrained"]
        ocr_mid_channels = checkpoint["ocr_mid_channels"]
        dropout = checkpoint["dropout"]
        use_aux_head = checkpoint["use_aux_head"]

        return cls(
            n_classes=n_classes,
            device=device,
            backbone=backbone,
            pretrained=pretrained,
            ocr_mid_channels=ocr_mid_channels,
            dropout=dropout,
            use_aux_head=use_aux_head,
        )
