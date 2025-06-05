"""
UPerNet segmentation model implementation.

This module provides the UPerNet wrapper class that inherits from
PatchSegmentationModel. It supports different backbone architectures.
"""

from typing import Any

from petroscope.segmentation.models.base import PatchSegmentationModel
from petroscope.utils.lazy_imports import torch  # noqa


class UPerNet(PatchSegmentationModel):
    """
    UPerNet segmentation model implementation.

    UPerNet is a unified perceptual parsing network that combines
    feature pyramid network (FPN) and pyramid pooling module (PPM)
    for multi-scale feature fusion.

    Supported backbone architectures:
    - ResNet: resnet18, resnet34, resnet50, resnet101, resnet152
    - ConvNeXt: convnext_tiny, convnext_small, convnext_base, convnext_large

    The model handles different input image sizes automatically through
    adaptive pooling and interpolation.
    """

    MODEL_REGISTRY: dict[str, str] = {
        # Model registry will be populated as models are trained and published
    }

    def __init__(
        self,
        n_classes: int,
        backbone: str,
        device: str,
        use_fpn: bool = True,
        pretrained: bool = True,
        dilated: bool = True,
    ) -> None:
        """
        Initialize the UPerNet model.

        Args:
            n_classes: Number of segmentation classes
            backbone: Backbone network. Supported options:
                - ResNet: "resnet18", "resnet34", "resnet50", "resnet101",
                  "resnet152"
                - ConvNeXt: "convnext_tiny", "convnext_small", "convnext_base",
                  "convnext_large"
            device: Device to run the model on ('cuda', 'cpu', etc.)
            use_fpn: Whether to use Feature Pyramid Network
            pretrained: Whether to use pretrained weights for the backbone
            dilated: Whether to use dilated convolutions in later layers
        """
        super().__init__(n_classes, device)

        from petroscope.segmentation.models.upernet.nn import UPerNet

        self.backbone = backbone
        self.use_fpn = use_fpn
        self.pretrained = pretrained
        self.dilated = dilated

        # Create the model with appropriate configuration
        self.model = UPerNet(
            n_classes=n_classes,
            backbone=backbone,
            use_fpn=use_fpn,
            pretrained=pretrained,
            dilated=dilated,
        ).to(self.device)

    def _get_checkpoint_data(self) -> dict[str, Any]:
        """Return model-specific data for checkpoint saving."""
        return {
            "n_classes": self.n_classes,
            "backbone": self.backbone,
            "use_fpn": self.use_fpn,
            "pretrained": self.pretrained,
            "dilated": self.dilated,
        }

    @classmethod
    def _create_from_checkpoint(
        cls, checkpoint: dict, device: str
    ) -> "UPerNet":
        """Create a UPerNet model from checkpoint data."""
        # Extract architecture hyperparameters from checkpoint
        n_classes = checkpoint["n_classes"]
        backbone = checkpoint["backbone"]
        use_fpn = checkpoint["use_fpn"]
        pretrained = checkpoint["pretrained"]
        dilated = checkpoint["dilated"]

        return cls(
            n_classes=n_classes,
            backbone=backbone,
            device=device,
            use_fpn=use_fpn,
            pretrained=pretrained,
            dilated=dilated,
        )
