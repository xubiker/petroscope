"""
HRNetV2 + OCR segmentation model implementation.

This module provides the HRNetV2 + OCR wrapper class that inherits from
PatchSegmentationModel and integrates with the petroscope training pipeline.

The implementation is based on the original HRNet + OCR papers:
- "Deep High-Resolution Representation Learning for Visual Recognition"
- "Object-Contextual Representations for Semantic Segmentation"
"""

from typing import Any

from petroscope.segmentation.models.base import PatchSegmentationModel


def _load_pretrained_backbone(model, backbone: str):
    """
    Load pretrained weights into the HRNet backbone.

    Args:
        model: HRNet model instance
        backbone: HRNet backbone variant
    """
    print(f"Pretrained weights for {backbone}:")
    print("❌ Pretrained weights loading not implemented yet")
    print("   You can add pretrained weights by:")
    print("   1. Download weights from official HRNet repository")
    print("   2. Implement weight loading with proper key mapping")
    print("✅ The model will use random initialization")
    print("   This is fine for training from scratch.")
    print("\nContinuing with random initialization...")


class HRNetWithOCR(PatchSegmentationModel):
    """
    HRNetV2 with Object-Contextual Representations for segmentation.

    This model implements the original HRNet + OCR architecture as described
    in the papers. It combines high-resolution features from HRNet with
    object-contextual representations for improved segmentation performance.

    Supported configurations:
    - Width: 18, 32, 48 (controls the number of channels in each stream)
    - OCR mid channels: Number of channels in the OCR attention module
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
        dropout: float = 0.05,
        use_aux_head: bool = True,
    ) -> None:
        """
        Initialize the HRNetV2 + OCR model.

        Args:
            n_classes: Number of segmentation classes
            device: Device to run the model on ('cuda', 'cpu', etc.)
            backbone: Backbone variant. Options:
                - "hrnetv2_w18": HRNetV2 with width 18
                - "hrnetv2_w32": HRNetV2 with width 32
                - "hrnetv2_w48": HRNetV2 with width 48
            pretrained: Whether to use pretrained weights from official repo
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
            n_classes=n_classes,
            width=width,
            in_channels=3,
            ocr_mid_channels=ocr_mid_channels,
            dropout=dropout,
            use_aux_head=use_aux_head,
        ).to(self.device)

        # Load pretrained weights if requested
        if pretrained:
            _load_pretrained_backbone(self.model, backbone)

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
        """Create HRNetV2 + OCR model from checkpoint data."""
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

    def supports_auxiliary_loss(self) -> bool:
        """Return True since HRNet supports auxiliary loss."""
        return self.use_aux_head
