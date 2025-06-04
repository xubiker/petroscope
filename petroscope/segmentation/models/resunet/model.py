from pathlib import Path
from typing import Dict, Any


from petroscope.segmentation.models.base import PatchSegmentationModel
from petroscope.utils import logger
from petroscope.utils.lazy_imports import torch  # noqa


class ResUNet(PatchSegmentationModel):
    MODEL_REGISTRY: Dict[str, str] = {
        "s1_x05": "http://www.xubiker.online/petroscope/segmentation_weights/resunet_s1_x05.pth",
        "s1_x05_calib": "http://www.xubiker.online/petroscope/segmentation_weights/resunet_s1_x05_calib.pth",
        "s2_x05": "http://www.xubiker.online/petroscope/segmentation_weights/resunet_s2_x05.pth",
        "s2_x05_calib": "http://www.xubiker.online/petroscope/segmentation_weights/resunet_s2_x05_calib.pth",
        # extra weights
        "__s1_x05_e5": "http://www.xubiker.online/petroscope/segmentation_weights/resunet_s1_x05_e5.pth",
        "__s1_x05_e10": "http://www.xubiker.online/petroscope/segmentation_weights/resunet_s1_x05_e10.pth",
        "__s1_x05_calib_e5": "http://www.xubiker.online/petroscope/segmentation_weights/resunet_s1_x05_calib_e5.pth",
        "__s1_x05_calib_e10": "http://www.xubiker.online/petroscope/segmentation_weights/resunet_s1_x05_calib_e10.pth",
        "__s2_x05_e5": "http://www.xubiker.online/petroscope/segmentation_weights/resunet_s2_x05_e5.pth",
        "__s2_x05_e10": "http://www.xubiker.online/petroscope/segmentation_weights/resunet_s2_x05_e10.pth",
        "__s2_x05_calib_e5": "http://www.xubiker.online/petroscope/segmentation_weights/resunet_s2_x05_calib_e5.pth",
        "__s2_x05_calib_e10": "http://www.xubiker.online/petroscope/segmentation_weights/resunet_s2_x05_calib_e10.pth",
    }

    def __init__(
        self, n_classes: int, layers: int, filters: int, device: str
    ) -> None:
        """
        Initialize the ResUNet model.

        Args:
            n_classes: Number of segmentation classes
            layers: Number of layers in the network
            filters: Number of starting filters
            device: Device to run the model on ('cuda', 'cpu', etc.)
        """
        super().__init__(n_classes, device)

        from petroscope.segmentation.models.resunet.nn import ResUNet

        self.layers = layers
        self.filters = filters
        self.model = ResUNet(
            n_classes=n_classes, n_layers=layers, start_filters=filters
        ).to(self.device)

    def _get_checkpoint_data(self) -> Dict[str, Any]:
        """Return model-specific data for checkpoint saving."""
        return {
            "n_classes": self.n_classes,
            "layers": self.layers,
            "filters": self.filters,
        }

    @classmethod
    def _create_from_checkpoint(
        cls, checkpoint: dict, device: str
    ) -> "ResUNet":
        """Create a ResUNet model from checkpoint data."""
        return cls(
            n_classes=checkpoint["n_classes"],
            layers=checkpoint["layers"],
            filters=checkpoint["filters"],
            device=device,
        )
