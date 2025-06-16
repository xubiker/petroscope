"""
HRNet segmentation model wrapper.

This module provides the HRNet wrapper class that inherits from
PatchSegmentationModel and integrates with the petroscope training pipeline.

The implementation is based on the original HRNet + OCR papers:
- "Deep High-Resolution Representation Learning for Visual Recognition"
- "Object-Contextual Representations for Semantic Segmentation"
"""

from typing import Any
import hashlib

from petroscope.segmentation.models.base import PatchSegmentationModel
from petroscope.utils import logger
from petroscope.utils.lazy_imports import torch  # noqa


def _generate_cache_filename(url: str) -> str:
    """Generate a shorter filename for the cache to avoid filesystem issues."""
    # Create a hash of the URL to generate a unique, short filename
    url_hash = hashlib.md5(url.encode()).hexdigest()[:16]
    return f"hrnetv2_imagenet_{url_hash}.pth"


def _load_pretrained_weights(model, backbone: str):
    """
    Load pretrained ImageNet weights into the HRNet backbone.

    Args:
        model: HRNet model instance
        backbone: HRNet backbone variant (e.g., "hrnetv2_w18")
    """
    if not backbone.startswith("hrnetv2_w"):
        logger.warning(
            f"Unknown backbone {backbone}, skipping pretrained weights"
        )
        return

    # Official HRNet ImageNet pretrained model URLs
    model_urls = {
        "hrnetv2_w18": (
            "https://opr0mq.dm.files.1drv.com/y4mIoWpP2n-LUohHHANpC0jrOixm1FZgO"
            "2OsUtP2DwIozH5RsoYVyv_De5wDgR6XuQmirMV3C0AljLeB-zQXevfLlnQpcNe"
            "JlT9Q8LwNYDwh3TsECkMTWXCUn3vDGJWpCxQcQWKONr5VQWO1hLEKPeJbbSZ6"
            "tgbWwJHgHF7592HY7ilmGe39o5BhHz7P9QqMYLBts6V7QGoaKrr0PL3wvvR4w"
        ),
        "hrnetv2_w32": (
            "https://opr74a.dm.files.1drv.com/y4mKOuRSNGQQlp6wm_a9bF-UEQwp6a10x"
            "FCLhm4bqjDu6aSNW9yhDRM7qyx0vK0WTh42gEaniUVm3h7pg0H-W0yJff5qQt"
            "oAX7Zze4vOsqjoIthp-FW3nlfMD0-gcJi8IiVrMWqVOw2N3MbCud6uQQrTaE"
            "AvAdNjtjMpym1JghN-F060rSQKmgtq5R-wJe185IyW4-_c5_ItbhYpCyLxdqdEQ"
        ),
        "hrnetv2_w48": (
            "https://optgaw.dm.files.1drv.com/y4mWNpya38VArcDInoPaL7GfPMgcop92G"
            "6YRkabO1QTSWkCbo7djk8BFZ6LK_KHHIYE8wqeSAChU58NVFOZEvqFaoz392O"
            "gcyBrq_f8XGkusQep_oQsuQ7DPQCUrdLwyze_NlsyDGWot0L9agkQ-M_SfNr"
            "10ETlCF5R7BdKDZdupmcMXZc-IE3Ysw1bVHdOH4l-XEbEKFAi6ivPUbeqlYkRMQ"
        ),
    }

    if backbone not in model_urls:
        logger.warning(f"No pretrained weights available for {backbone}")
        return

    url = model_urls[backbone]

    try:
        from torch.hub import load_state_dict_from_url

        print(
            f"🔄 Downloading ImageNet pretrained weights for "
            f"{backbone.upper()}..."
        )

        # Use custom filename to avoid filesystem issues with long URLs
        cache_filename = _generate_cache_filename(url)

        # Download weights with custom filename
        state_dict = load_state_dict_from_url(
            url, file_name=cache_filename, progress=True
        )

        # Filter out non-backbone weights (segmentation heads, etc.)
        model_dict = model.state_dict()
        pretrained_dict = {}
        skipped_keys = []

        for k, v in state_dict.items():
            # Skip classification heads and other non-backbone weights
            if any(skip in k for skip in ["classifier", "fc", "head"]):
                skipped_keys.append(k)
                continue

            # Try different key mappings for HRNet structure
            # The pretrained weights are for the backbone, but our model
            # has model.backbone. prefix
            possible_keys = [
                k,  # Direct match
                f"model.backbone.{k}",  # Add model.backbone prefix
                f"backbone.{k}",  # Add backbone prefix
            ]

            matched = False
            for mapped_key in possible_keys:
                if (
                    mapped_key in model_dict
                    and model_dict[mapped_key].shape == v.shape
                ):
                    pretrained_dict[mapped_key] = v
                    matched = True
                    break

            if not matched:
                skipped_keys.append(k)

        # Load the filtered weights
        model.load_state_dict(pretrained_dict, strict=False)

        loaded_keys = len(pretrained_dict)
        total_keys = len(state_dict)

        print(
            f"✅ Successfully loaded {loaded_keys}/{total_keys} "
            f"backbone weights"
        )
        if skipped_keys:
            print(
                "   Skipped segmentation-specific layers "
                "(cls_head, aux_head, OCR)"
            )

    except Exception as e:
        logger.warning(
            f"Failed to load pretrained weights for " f"{backbone}: {e}"
        )
        print("⚠️  Continuing with random initialization")


class HRNet(PatchSegmentationModel):
    """
    HRNet with Object-Contextual Representations for segmentation.

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
        "s1s2_w18_x05": (
            "http://www.xubiker.online/petroscope/segmentation_weights"
            "/hrnet_w18/S1v2_S2v2_x05.pth"
        ),
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
        Initialize the HRNet model.

        Args:
            n_classes: Number of segmentation classes
            device: Device to run the model on ('cuda', 'cpu', etc.)
            backbone: Backbone variant. Options:
                - "hrnetv2_w18": HRNet with width 18
                - "hrnetv2_w32": HRNet with width 32
                - "hrnetv2_w48": HRNet with width 48
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

        # Store parameters
        self.backbone = backbone
        self.pretrained = pretrained
        self.ocr_mid_channels = ocr_mid_channels
        self.dropout = dropout
        self.use_aux_head = use_aux_head
        self.width = width

        # Create the neural network
        self.model = HRNetWithOCR(
            n_classes=n_classes,
            width=width,
            in_channels=3,
            ocr_mid_channels=ocr_mid_channels,
            dropout=dropout,
            use_aux_head=use_aux_head,
            backbone=backbone,
        ).to(self.device)

        # Load pretrained weights if requested
        if pretrained:
            _load_pretrained_weights(self.model, backbone)

    def supports_auxiliary_loss(self) -> bool:
        """Return True since HRNet supports auxiliary loss."""
        return self.use_aux_head

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
    def _create_from_checkpoint(cls, checkpoint: dict, device: str) -> "HRNet":
        """Create HRNet model from checkpoint data."""
        return cls(
            n_classes=checkpoint["n_classes"],
            device=device,
            backbone=checkpoint["backbone"],
            pretrained=checkpoint.get("pretrained", False),
            ocr_mid_channels=checkpoint.get("ocr_mid_channels", 512),
            dropout=checkpoint.get("dropout", 0.05),
            use_aux_head=checkpoint.get("use_aux_head", True),
        )
