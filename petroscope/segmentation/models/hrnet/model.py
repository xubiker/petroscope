"""
HRNetV2+OCR Model Wrapper

This module provides the wrapper around the HRNetV2+OCR neural network architecture,
implementing the required GeoSegmModel interface for the petroscope library.
"""

from typing import Any


from petroscope.segmentation.models.base import PatchSegmentationModel
from petroscope.segmentation.models.hrnet.nn import HRNetOCR


class HRNetWithOCR(PatchSegmentationModel):
    """
    HRNetV2 with Object-Contextual Representations for semantic segmentation.

    This model combines HRNetV2 backbone with OCR module for better segmentation,
    particularly for handling class imbalance in mineral segmentation tasks.

    Args:
        n_classes: Number of segmentation classes
        device: Device to run the model on ('cuda', 'cpu', etc.)
        backbone: HRNetV2 backbone variant ('hrnetv2_w18', 'hrnetv2_w32', 'hrnetv2_w48')
        pretrained: Whether to use pretrained backbone weights
        ocr_mid_channels: Number of channels in OCR module
        dropout: Dropout rate for the model
        use_aux_head: Whether to use auxiliary segmentation head during training
    """

    # Registry of pretrained models (will be populated as models are trained)
    MODEL_REGISTRY = {}

    def __init__(
        self,
        n_classes: int,
        device: str,
        backbone: str = "hrnetv2_w32",
        pretrained: bool = True,
        ocr_mid_channels: int = 512,
        dropout: float = 0.1,
        use_aux_head: bool = True,
    ):
        super().__init__(n_classes, device)

        self.backbone_name = backbone
        self.pretrained = pretrained
        self.ocr_mid_channels = ocr_mid_channels
        self.dropout = dropout
        self.use_aux_head = use_aux_head

        self.model = HRNetOCR(
            n_classes=n_classes,
            backbone=backbone,
            pretrained=pretrained,
            ocr_mid_channels=ocr_mid_channels,
            dropout=dropout,
            use_aux_head=use_aux_head,
        )

        self.model.to(device)

    @classmethod
    def _create_from_checkpoint(
        cls, checkpoint: dict[str, Any], device: str
    ) -> "HRNetWithOCR":
        """
        Create a model instance from checkpoint data.

        Args:
            checkpoint: The loaded checkpoint dictionary
            device: Device to create the model on

        Returns:
            An initialized model instance
        """
        config = checkpoint["config"]

        return cls(
            n_classes=config["n_classes"],
            device=device,
            backbone=config.get("backbone", "hrnetv2_w32"),
            pretrained=False,  # Don't load pretrained weights since we're loading from checkpoint
            ocr_mid_channels=config.get("ocr_mid_channels", 512),
            dropout=config.get("dropout", 0.1),
            use_aux_head=config.get("use_aux_head", True),
        )

    def _get_checkpoint_data(self) -> dict[str, Any]:
        """
        Get model-specific data for checkpoint saving.

        Returns:
            Dictionary containing model-specific parameters
        """
        return {
            "config": {
                "n_classes": self.n_classes,
                "backbone": self.backbone_name,
                "ocr_mid_channels": self.ocr_mid_channels,
                "dropout": self.dropout,
                "use_aux_head": self.use_aux_head,
            }
        }

    @classmethod
    def best(cls) -> "HRNetWithOCR":
        """
        Get the best performing pretrained model.

        Returns:
            Best performing model instance
        """
        # Not implemented yet - will be populated as models are trained
        raise NotImplementedError(
            "No pretrained best model available yet. Please train the model first."
        )

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": "HRNetV2+OCR",
            "backbone": self.backbone_name,
            "n_classes": self.n_classes,
            "ocr_mid_channels": self.ocr_mid_channels,
            "use_aux_head": self.use_aux_head,
            "parameters": self.n_params_str,
        }
