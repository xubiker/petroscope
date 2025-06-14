"""
UPerNet Neural Network Implementation

UPerNet: Unified Perceptual Parsing Network

This module implements the UPerNet segmentation architecture including:
- Support for multiple backbone architectures (ResNet, ConvNeXt)
- Feature Pyramid Network (FPN) for multi-scale feature fusion
- Pyramid Pooling Module (PPM) for capturing global context
- Unified architecture that combines the advantages of FPN and PPM

References:
    - Paper: "Unified Perceptual Parsing for Scene Understanding"
      https://arxiv.org/abs/1807.10221
"""

from typing import (
    TYPE_CHECKING,
    List,
    Dict,
    Tuple,
    Any,
)

# import torch-sensitive modules (satisfies Pylance and Flake8)
if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models
    from torchvision.models import (
        ResNet18_Weights,
        ResNet34_Weights,
        ResNet50_Weights,
        ResNet101_Weights,
        ResNet152_Weights,
        ConvNeXt_Tiny_Weights,
        ConvNeXt_Small_Weights,
        ConvNeXt_Base_Weights,
        ConvNeXt_Large_Weights,
    )

from petroscope.utils.lazy_imports import torch, nn, F, models  # noqa
from torchvision.models import (
    # ResNet pretrained weights
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    # ConvNeXt pretrained weights
    ConvNeXt_Tiny_Weights,
    ConvNeXt_Small_Weights,
    ConvNeXt_Base_Weights,
    ConvNeXt_Large_Weights,
)


class PyramidPoolingModule(nn.Module):
    """
    Pyramid Pooling Module from PSPNet.

    This module performs adaptive pooling at multiple scales to capture
    global context information and fuses them with the original features.

    Args:
        in_channels: Input feature channels
        pool_sizes: Tuple of pooling scales (e.g., (1, 2, 3, 6))
    """

    def __init__(self, in_channels: int, pool_sizes: Tuple[int, ...]):
        super().__init__()
        self.pool_sizes = pool_sizes

        # Reduce channels for efficiency
        self.reduced_channels = in_channels // len(pool_sizes)

        # More efficient implementation with shared reduction conv
        self.reduction_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, self.reduced_channels, kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(self.reduced_channels),
            nn.ReLU(inplace=True),
        )

        # Only use adaptive pooling in the features to minimize parameters
        self.pools = nn.ModuleList(
            [nn.AdaptiveAvgPool2d(output_size=s) for s in pool_sizes]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for PyramidPoolingModule.

        Args:
            x: Input feature tensor of shape (B, C, H, W)

        Returns:
            Concatenated feature tensor with original and pooled features
        """
        h, w = x.shape[2], x.shape[3]

        # Memory-efficient implementation
        # 1. Start with original features
        output = [x]

        # 2. Apply channel reduction once before pooling (more efficient)
        x_reduced = self.reduction_conv(x)

        # 3. Pool at each scale and upsample
        for pool in self.pools:
            # Pool, then upsample (more efficient than pooling, processing, then upsampling)
            pooled = pool(x_reduced)
            upsampled = F.interpolate(
                pooled, size=(h, w), mode="bilinear", align_corners=True
            )
            output.append(upsampled)

        # Concatenate all features
        return torch.cat(output, dim=1)


class FPN(nn.Module):
    """
    Feature Pyramid Network for multi-scale feature fusion.

    This module builds a feature pyramid from features at different scales
    by applying both top-down and lateral connections.

    Args:
        in_channels_list: List of input channels for each feature level
        out_channels: Output channels for all feature levels
    """

    def __init__(self, in_channels_list: List[int], out_channels: int):
        super().__init__()

        # Lateral convolutions (reduce channels for each input feature map)
        self.lateral_convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
                for in_channels in in_channels_list
            ]
        )

        # Output convolutions for each level after feature fusion
        self.fpn_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in range(len(in_channels_list))
            ]
        )

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass for FPN.

        Args:
            inputs: List of feature tensors at different scales

        Returns:
            List of output feature tensors at different scales
        """
        # Memory-optimized implementation:

        # 1. Apply 1x1 convs to get lateral features with same channel dimensions
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(inputs[i]))

        # 2. Optimized top-down pathway with in-place operations where possible
        prev_features = laterals[-1]
        for i in range(len(laterals) - 2, -1, -1):
            # Upsample the previous level features to current size
            upsampled = F.interpolate(
                prev_features,
                size=laterals[i].shape[2:],
                mode="bilinear",
                align_corners=True,
            )

            # Add to current level features (in-place when possible)
            laterals[i] = laterals[i] + upsampled
            prev_features = laterals[i]

        # 3. Apply output convolutions and cleanup
        outputs = []
        for i, lateral in enumerate(laterals):
            # Apply the output convolution to each level
            outputs.append(self.fpn_convs[i](lateral))

        return outputs


class UPerNet(nn.Module):
    """
    UPerNet: Unified Perceptual Parsing Network.

    This model combines Feature Pyramid Network (FPN) with Pyramid Pooling
    Module (PPM) to achieve both multi-scale feature fusion and global
    context modeling.

    Args:
        n_classes: Number of output classes
        backbone: Backbone network type ('resnet18', 'resnet34', 'resnet50',
                 'resnet101', 'resnet152', 'convnext_tiny', 'convnext_small',
                 'convnext_base', 'convnext_large')
        use_fpn: Whether to use Feature Pyramid Network
        pretrained: Whether to use pretrained weights for the backbone
        dilated: Whether to use dilated convolutions in later backbone layers
    """

    # Configuration details for each supported backbone
    BACKBONE_CONFIGS: Dict[str, Dict[str, Any]] = {
        # ResNet architectures
        "resnet18": {
            "model_fn": lambda pretrained: models.resnet18(
                weights=ResNet18_Weights.DEFAULT if pretrained else None
            ),
            "channels": [64, 128, 256, 512],
        },
        "resnet34": {
            "model_fn": lambda pretrained: models.resnet34(
                weights=ResNet34_Weights.DEFAULT if pretrained else None
            ),
            "channels": [64, 128, 256, 512],
        },
        "resnet50": {
            "model_fn": lambda pretrained: models.resnet50(
                weights=ResNet50_Weights.DEFAULT if pretrained else None
            ),
            "channels": [256, 512, 1024, 2048],
        },
        "resnet101": {
            "model_fn": lambda pretrained: models.resnet101(
                weights=ResNet101_Weights.DEFAULT if pretrained else None
            ),
            "channels": [256, 512, 1024, 2048],
        },
        "resnet152": {
            "model_fn": lambda pretrained: models.resnet152(
                weights=ResNet152_Weights.DEFAULT if pretrained else None
            ),
            "channels": [256, 512, 1024, 2048],
        },
        # ConvNeXt architectures
        "convnext_tiny": {
            "model_fn": lambda pretrained: models.convnext_tiny(
                weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            ),
            "channels": [96, 192, 384, 768],
        },
        "convnext_small": {
            "model_fn": lambda pretrained: models.convnext_small(
                weights=ConvNeXt_Small_Weights.DEFAULT if pretrained else None
            ),
            "channels": [96, 192, 384, 768],
        },
        "convnext_base": {
            "model_fn": lambda pretrained: models.convnext_base(
                weights=ConvNeXt_Base_Weights.DEFAULT if pretrained else None
            ),
            "channels": [128, 256, 512, 1024],
        },
        "convnext_large": {
            "model_fn": lambda pretrained: models.convnext_large(
                weights=ConvNeXt_Large_Weights.DEFAULT if pretrained else None
            ),
            "channels": [192, 384, 768, 1536],
        },
    }

    def __init__(
        self,
        n_classes: int,
        backbone: str = "resnet50",
        use_fpn: bool = True,
        pretrained: bool = True,
        dilated: bool = True,
    ):
        super().__init__()

        # Input validation
        if n_classes <= 0:
            raise ValueError(f"n_classes must be positive, got {n_classes}")

        self.n_classes = n_classes
        self.backbone_name = backbone
        self.use_fpn = use_fpn
        self.dilated = dilated

        # Validate backbone selection
        if backbone not in self.BACKBONE_CONFIGS:
            supported_backbones = list(self.BACKBONE_CONFIGS.keys())
            raise ValueError(
                f"Unsupported backbone: {backbone}. "
                f"Supported backbones: {supported_backbones}"
            )

        # Get backbone configuration
        config = self.BACKBONE_CONFIGS[backbone]
        backbone_channels = config["channels"]

        # Initialize backbone
        self.backbone_features = self._build_backbone(
            backbone, config, pretrained
        )

        # FPN settings
        fpn_out_channels = 256  # Standard FPN output channels

        # PPM settings
        ppm_in_channels = backbone_channels[
            -1
        ]  # Use the deepest layer for PPM
        ppm_pool_scales = (1, 2, 3, 6)

        # Initialize PyramidPoolingModule
        self.ppm = PyramidPoolingModule(
            in_channels=ppm_in_channels,
            pool_sizes=ppm_pool_scales,
        )

        # Calculate PPM output channels
        # Original channels + additional channels from pooling
        ppm_total_out_channels = ppm_in_channels + (
            ppm_in_channels // len(ppm_pool_scales)
        ) * len(ppm_pool_scales)

        # Initialize FPN if used
        if use_fpn:
            self.fpn = FPN(
                in_channels_list=backbone_channels,
                out_channels=fpn_out_channels,
            )
            # Calculate FPN output channels (all levels combined)
            fpn_total_out_channels = fpn_out_channels * len(backbone_channels)
        else:
            self.fpn = None
            # Just concatenate backbone features if not using FPN
            fpn_total_out_channels = sum(backbone_channels)

        # Fusion module
        fusion_in_channels = fpn_total_out_channels + ppm_total_out_channels
        fusion_out_channels = 512

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(
                fusion_in_channels,
                fusion_out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(fusion_out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # Final classifier
        self.classifier = nn.Conv2d(
            fusion_out_channels, n_classes, kernel_size=1
        )

    def _build_backbone(
        self, backbone_name: str, config: Dict[str, Any], pretrained: bool
    ) -> nn.ModuleDict:
        """
        Build and configure the backbone network.

        Args:
            backbone_name: Name of the backbone architecture
            config: Configuration for the backbone
            pretrained: Whether to use pretrained weights

        Returns:
            Dictionary of backbone feature stages
        """
        # Create model with pretrained weights if requested
        model = config["model_fn"](pretrained)

        # Extract features based on backbone type
        if backbone_name.startswith("resnet"):
            backbone = nn.ModuleDict(
                {
                    "layer0": nn.Sequential(
                        model.conv1, model.bn1, model.relu, model.maxpool
                    ),
                    "layer1": model.layer1,
                    "layer2": model.layer2,
                    "layer3": model.layer3,
                    "layer4": model.layer4,
                }
            )

            # Apply dilated convolutions to later layers for ResNet if specified
            if self.dilated:
                backbone["layer3"] = self._make_dilated(
                    backbone["layer3"], dilation=2
                )
                backbone["layer4"] = self._make_dilated(
                    backbone["layer4"], dilation=4
                )

            return backbone

        elif backbone_name.startswith("convnext"):
            features = model.features
            backbone = nn.ModuleDict(
                {
                    "layer0": features[0],  # Stem
                    "layer1": features[1],  # Stage 1
                    "layer2": nn.Sequential(
                        features[2], features[3]
                    ),  # Stage 2
                    "layer3": nn.Sequential(
                        features[4], features[5]
                    ),  # Stage 3
                    "layer4": nn.Sequential(
                        *features[6:]
                    ),  # Stage 4 (remaining layers)
                }
            )

            # Apply dilated convolutions to later stages of ConvNeXt if specified
            # Note: ConvNeXt architecture is different from ResNet, but we can still
            # apply dilated convolutions to the later stages
            if self.dilated:
                # Apply dilation to layer3 and layer4 (later stages)
                # This is similar to what we do for ResNet
                backbone["layer3"] = self._make_dilated_convnext(
                    backbone["layer3"], dilation=2
                )
                backbone["layer4"] = self._make_dilated_convnext(
                    backbone["layer4"], dilation=4
                )

            return backbone
        else:
            raise ValueError(
                f"Unsupported backbone architecture: {backbone_name}"
            )

    def _make_dilated(self, layer: nn.Module, dilation: int) -> nn.Module:
        """
        Convert a ResNet layer to use dilated convolutions.

        Args:
            layer: Neural network layer to modify
            dilation: Dilation rate to apply

        Returns:
            Modified layer with dilated convolutions
        """
        for n, m in layer.named_modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (3, 3):  # Only modify 3x3 convolutions
                    m.dilation = (dilation, dilation)
                    m.padding = (dilation, dilation)
                if m.stride == (2, 2):  # Prevent downsampling
                    m.stride = (1, 1)
        return layer

    def _make_dilated_convnext(
        self, layer: nn.Module, dilation: int
    ) -> nn.Module:
        """
        Convert a ConvNeXt layer to use dilated convolutions.

        Args:
            layer: Neural network layer to modify
            dilation: Dilation rate to apply

        Returns:
            Modified layer with dilated convolutions
        """
        # ConvNeXt uses depthwise convolutions in the block structure
        # We need to find and modify these convolutions
        for n, m in layer.named_modules():
            if isinstance(m, nn.Conv2d):
                # In ConvNeXt, the depthwise conv has groups=in_channels
                if m.groups == m.in_channels and m.kernel_size == (7, 7):
                    # This is a depthwise conv, apply dilation
                    m.dilation = (dilation, dilation)
                    m.padding = (
                        3 * dilation,
                        3 * dilation,
                    )  # Adjust padding for dilation
                # Prevent downsampling in downsampling layers
                if m.stride == (2, 2):
                    m.stride = (1, 1)
        return layer

    def extract_backbone_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract features from the backbone network.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            List of feature tensors from different stages of the backbone
        """
        features = []

        # Process through backbone layers
        x = self.backbone_features["layer0"](x)
        x = self.backbone_features["layer1"](x)
        features.append(x)

        x = self.backbone_features["layer2"](x)
        features.append(x)

        x = self.backbone_features["layer3"](x)
        features.append(x)

        x = self.backbone_features["layer4"](x)
        features.append(x)

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the UPerNet model.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, n_classes, H, W)
        """
        input_size = x.size()

        # Extract backbone features
        backbone_features = self.extract_backbone_features(x)

        # Apply PPM to the deepest feature map (memory-efficient, no redundant storage)
        ppm_out = self.ppm(backbone_features[-1])

        # Target size for features - use highest resolution feature map
        target_size = backbone_features[0].shape[2:]

        if self.use_fpn:
            # Apply FPN to get multi-scale features
            # FPN already handles the lateral connections and upsampling internally
            fpn_features = self.fpn(backbone_features)

            # Process FPN features more efficiently - reduce feature dimensions before upsampling
            # and use a sequential fusion approach to reduce memory consumption
            fpn_out = None
            for i, feat in enumerate(fpn_features):
                # Only upsample if needed (skip highest resolution feature)
                if i == 0:
                    fpn_out = feat
                else:
                    # Upsample and add to accumulated features
                    feat_upsampled = F.interpolate(
                        feat,
                        size=target_size,
                        mode="bilinear",
                        align_corners=True,
                    )
                    if fpn_out is None:
                        fpn_out = feat_upsampled
                    else:
                        # Concatenate instead of keeping separate and concatenating later
                        fpn_out = torch.cat([fpn_out, feat_upsampled], dim=1)
        else:
            # Simple feature concatenation without FPN - more memory efficient approach
            # Process one feature at a time instead of storing all upsampled features
            fpn_out = backbone_features[
                0
            ]  # Start with highest resolution feature

            # Progressively upsample and concatenate remaining features
            for i in range(1, len(backbone_features)):
                feat_upsampled = F.interpolate(
                    backbone_features[i],
                    size=target_size,
                    mode="bilinear",
                    align_corners=True,
                )
                fpn_out = torch.cat([fpn_out, feat_upsampled], dim=1)

        # Upsample PPM output to match target size
        ppm_out = F.interpolate(
            ppm_out, size=target_size, mode="bilinear", align_corners=True
        )

        # Combine features
        combined = torch.cat([fpn_out, ppm_out], dim=1)

        # Apply fusion convolution
        fused = self.fusion_conv(combined)

        # Free up memory explicitly (helps with large input sizes)
        del combined, fpn_out, ppm_out

        # Final classification
        output = self.classifier(fused)

        # Free memory
        del fused

        # Upsample to input resolution only once at the end
        output = F.interpolate(
            output, size=input_size[2:], mode="bilinear", align_corners=True
        )

        return output

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model architecture.

        Returns:
            Dictionary containing model configuration and parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

        backbone_params = sum(
            p.numel() for p in self.backbone_features.parameters()
        )
        fpn_params = (
            sum(p.numel() for p in self.fpn.parameters()) if self.fpn else 0
        )
        ppm_params = sum(p.numel() for p in self.ppm.parameters())

        return {
            "model_name": "UPerNet",
            "backbone": self.backbone_name,
            "n_classes": self.n_classes,
            "use_fpn": self.use_fpn,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "backbone_parameters": backbone_params,
            "fpn_parameters": fpn_params,
            "ppm_parameters": ppm_params,
        }

    @staticmethod
    def get_supported_backbones() -> List[str]:
        """
        Get list of supported backbone architectures.

        Returns:
            List of supported backbone names
        """
        return list(UPerNet.BACKBONE_CONFIGS.keys())

    def validate_model(self) -> Tuple[bool, List[str]]:
        """
        Validate the model architecture and configurations.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        try:
            # Check if all components are properly initialized
            if self.backbone_features is None:
                errors.append("Backbone features not initialized")

            if self.ppm is None:
                errors.append("PPM not initialized")

            if self.use_fpn and self.fpn is None:
                errors.append("FPN enabled but not initialized")

            if self.fusion_conv is None:
                errors.append("Fusion convolution not initialized")

            if self.classifier is None:
                errors.append("Classifier not initialized")

            # Test with a small dummy input
            dummy_input = torch.randn(1, 3, 64, 64)
            with torch.no_grad():
                output = self.forward(dummy_input)

            if output.shape[1] != self.n_classes:
                errors.append(
                    f"Output channels {output.shape[1]} != "
                    f"n_classes {self.n_classes}"
                )

            if output.shape[2:] != dummy_input.shape[2:]:
                errors.append(
                    f"Output spatial size {output.shape[2:]} != "
                    f"input {dummy_input.shape[2:]}"
                )

        except Exception as e:
            errors.append(f"Forward pass failed: {str(e)}")

        return len(errors) == 0, errors
