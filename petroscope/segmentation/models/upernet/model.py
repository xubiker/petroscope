from pathlib import Path
from typing import TYPE_CHECKING, Dict, Any

import numpy as np

from petroscope.segmentation.models.base import PatchSegmentationModel
from petroscope.utils import logger
from petroscope.utils.lazy_imports import torch  # noqa
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
)


class UPerNet(PatchSegmentationModel):
    """
    UPerNet segmentation model implementation.

    UPerNet is a unified perceptual parsing network that combines
    feature pyramid network (FPN) and pyramid pooling module (PPM)
    for multi-scale feature fusion.
    """

    MODEL_REGISTRY: Dict[str, str] = {
        # Model registry will be populated as models are trained and published
    }

    def __init__(
        self, n_classes: int, backbone: str, device: str, use_fpn: bool = True
    ) -> None:
        """
        Initialize the UPerNet model.

        Args:
            n_classes: Number of segmentation classes
            backbone: Backbone network (e.g., resnet50)
            device: Device to run the model on ('cuda', 'cpu', etc.)
            use_fpn: Whether to use Feature Pyramid Network
        """
        super().__init__(n_classes, device)

        from petroscope.segmentation.models.upernet.nn import UPerNet

        self.backbone = backbone
        self.use_fpn = use_fpn
        # Create the model with appropriate weights
        self.model = UPerNet(
            n_classes=n_classes, backbone=backbone, use_fpn=use_fpn
        ).to(self.device)

    def _get_checkpoint_data(self) -> Dict[str, Any]:
        """Return model-specific data for checkpoint saving."""
        return {
            "n_classes": self.n_classes,
            "backbone": self.backbone,
            "use_fpn": self.use_fpn,
        }

    @classmethod
    def _create_from_checkpoint(
        cls, checkpoint: dict, device: str
    ) -> "UPerNet":
        """Create a UPerNet model from checkpoint data."""
        # Extract architecture hyperparameters from checkpoint
        n_classes = checkpoint["n_classes"]
        backbone = checkpoint["backbone"]
        use_fpn = checkpoint.get(
            "use_fpn", True
        )  # Default to True for backward compatibility

        return cls(
            n_classes=n_classes,
            backbone=backbone,
            device=device,
            use_fpn=use_fpn,
        )
