from pathlib import Path
from typing import Any


from petroscope.segmentation.models.base import PatchSegmentationModel
from petroscope.utils import logger
from petroscope.utils.lazy_imports import torch  # noqa


class ResUNet(PatchSegmentationModel):
    """
    ResUNet segmentation model implementation.

    This implementation supports both the original ResUNet architecture
    and enhanced versions with pretrained ResNet backbones and dilated convolutions.
    """

    MODEL_REGISTRY: dict[str, str] = {
        "s1s2_resnet34_x05": (
            "http://www.xubiker.online/petroscope/segmentation_weights"
            "/resunet_resnet34/S1v2_S2v2_x05.pth"
        ),
    }

    def __init__(
        self,
        n_classes: int,
        layers: int = 4,
        filters: int = 64,
        device: str = "cuda",
        backbone: str = None,
        dilated: bool = False,
        pretrained: bool = True,
    ) -> None:
        """
        Initialize the ResUNet model.

        Args:
            n_classes: Number of segmentation classes
            layers: Number of layers in the network (only used if no backbone)
            filters: Number of starting filters (only used if no backbone)
            device: Device to run the model on ('cuda', 'cpu', etc.)
            backbone: Optional backbone network (e.g., 'resnet18', 'resnet34', etc.)
            dilated: Whether to use dilated convolutions
            pretrained: Whether to use pretrained backbone weights
        """
        super().__init__(n_classes, device)

        from petroscope.segmentation.models.resunet.nn import ResUNet

        self.layers = layers
        self.filters = filters
        self.backbone = backbone
        self.dilated = dilated
        self.pretrained = pretrained

        self.model = ResUNet(
            n_classes=n_classes,
            n_layers=layers,
            start_filters=filters,
            backbone=backbone,
            dilated=dilated,
            pretrained=pretrained,
        ).to(self.device)

    def _get_checkpoint_data(self) -> dict[str, Any]:
        """Return model-specific data for checkpoint saving."""
        return {
            "n_classes": self.n_classes,
            "layers": self.layers,
            "filters": self.filters,
            "backbone": self.backbone,
            "dilated": self.dilated,
            "pretrained": self.pretrained,
        }

    @classmethod
    def _create_from_checkpoint(
        cls, checkpoint: dict, device: str
    ) -> "ResUNet":
        """Create a ResUNet model from checkpoint data."""
        return cls(
            n_classes=checkpoint["n_classes"],
            layers=checkpoint.get("layers", 4),
            filters=checkpoint.get("filters", 64),
            device=device,
            backbone=checkpoint.get("backbone", None),
            dilated=checkpoint.get("dilated", False),
            pretrained=checkpoint.get("pretrained", True),
        )

    def _load_state_dict(self, checkpoint: dict) -> None:
        """Load model weights from checkpoint."""
        self.model.load_state_dict(checkpoint["model_state"], strict=False)
        logger.info("Loaded model weights.")
