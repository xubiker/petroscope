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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    # ResNet architectures
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    # ResNet pretrained weights
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    # ConvNeXt architectures
    convnext_tiny,
    convnext_small,
    convnext_base,
    convnext_large,
    # ConvNeXt pretrained weights
    ConvNeXt_Tiny_Weights,
    ConvNeXt_Small_Weights,
    ConvNeXt_Base_Weights,
    ConvNeXt_Large_Weights,
)


class PPM(nn.Module):
    """
    Pyramid Pooling Module from PSPNet.

    This module performs adaptive pooling at multiple scales to capture
    global context information and fuses them with the original features.

    Args:
        in_channels: Input feature channels
        out_channels: Output channels per pooling scale
        pool_scales: Tuple of pooling scales (e.g., (1, 2, 3, 6))
    """

    def __init__(self, in_channels, out_channels, pool_scales=(1, 2, 3, 6)):
        super(PPM, self).__init__()

        self.pool_scales = pool_scales
        self.blocks = nn.ModuleList()

        for scale in pool_scales:
            self.blocks.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(
                        in_channels, out_channels, kernel_size=1, bias=False
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, x):
        """
        Forward pass for PPM.

        Args:
            x: Input feature tensor of shape (B, C, H, W)

        Returns:
            Concatenated feature tensor of shape
            (B, C + len(pool_scales)*out_channels, H, W)
        """
        x_size = x.size()
        out = [x]

        for block in self.blocks:
            feat = block(x)
            # Upsample to original size
            feat = F.interpolate(
                feat, x_size[2:], mode="bilinear", align_corners=True
            )
            out.append(feat)

        # Concatenate all features along channel dimension
        return torch.cat(out, 1)


class FPN(nn.Module):
    """
    Feature Pyramid Network for multi-scale feature fusion.

    This module builds a feature pyramid from features at different scales
    by applying both top-down and lateral connections.

    Args:
        in_channels_list: List of input channels for each feature level
        out_channels: Output channels for all feature levels
    """

    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()

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

    def forward(self, inputs):
        """
        Forward pass for FPN.

        Args:
            inputs: List of feature tensors at different scales

        Returns:
            List of output feature tensors at different scales
        """
        # Apply 1x1 convs to get lateral features with same channel dimensions
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # Top-down pathway: start from the deepest layer
        for i in range(len(laterals) - 1, 0, -1):
            # Upsample the current level and add to the previous level
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode="bilinear",
                align_corners=True,
            )

        # Apply output convolutions for each level
        outputs = [
            self.fpn_convs[i](laterals[i]) for i in range(len(laterals))
        ]

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
    """

    # Configuration details for each supported backbone
    BACKBONE_CONFIGS = {
        # ResNet architectures
        "resnet18": {
            "model_fn": lambda pretrained: resnet18(
                weights=ResNet18_Weights.DEFAULT if pretrained else None
            ),
            "channels": [64, 128, 256, 512],
        },
        "resnet34": {
            "model_fn": lambda pretrained: resnet34(
                weights=ResNet34_Weights.DEFAULT if pretrained else None
            ),
            "channels": [64, 128, 256, 512],
        },
        "resnet50": {
            "model_fn": lambda pretrained: resnet50(
                weights=ResNet50_Weights.DEFAULT if pretrained else None
            ),
            "channels": [256, 512, 1024, 2048],
        },
        "resnet101": {
            "model_fn": lambda pretrained: resnet101(
                weights=ResNet101_Weights.DEFAULT if pretrained else None
            ),
            "channels": [256, 512, 1024, 2048],
        },
        "resnet152": {
            "model_fn": lambda pretrained: resnet152(
                weights=ResNet152_Weights.DEFAULT if pretrained else None
            ),
            "channels": [256, 512, 1024, 2048],
        },
        # ConvNeXt architectures
        "convnext_tiny": {
            "model_fn": lambda pretrained: convnext_tiny(
                weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            ),
            "channels": [96, 192, 384, 768],
        },
        "convnext_small": {
            "model_fn": lambda pretrained: convnext_small(
                weights=ConvNeXt_Small_Weights.DEFAULT if pretrained else None
            ),
            "channels": [96, 192, 384, 768],
        },
        "convnext_base": {
            "model_fn": lambda pretrained: convnext_base(
                weights=ConvNeXt_Base_Weights.DEFAULT if pretrained else None
            ),
            "channels": [128, 256, 512, 1024],
        },
        "convnext_large": {
            "model_fn": lambda pretrained: convnext_large(
                weights=ConvNeXt_Large_Weights.DEFAULT if pretrained else None
            ),
            "channels": [192, 384, 768, 1536],
        },
    }

    def __init__(
        self,
        n_classes,
        backbone="resnet50",
        use_fpn=True,
        pretrained=True,
        dilated=True,
    ):
        super(UPerNet, self).__init__()

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
        ppm_out_channels = 128  # Per-scale channels

        # Initialize PPM
        self.ppm = PPM(
            in_channels=ppm_in_channels,
            out_channels=ppm_out_channels,
            pool_scales=ppm_pool_scales,
        )

        # Calculate PPM output channels
        ppm_total_out_channels = (
            ppm_in_channels + len(ppm_pool_scales) * ppm_out_channels
        )

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

    def _build_backbone(self, backbone_name, config, pretrained):
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

            # Apply dilated convolutions to later layers for ResNet
            if self.dilated and backbone_name.startswith("resnet"):
                backbone["layer3"] = self._make_dilated(
                    backbone["layer3"], dilation=2
                )
                backbone["layer4"] = self._make_dilated(
                    backbone["layer4"], dilation=4
                )

            return backbone

        elif backbone_name.startswith("convnext"):
            features = model.features
            return nn.ModuleDict(
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
        else:
            raise ValueError(
                f"Unsupported backbone architecture: {backbone_name}"
            )

    def _make_dilated(self, layer, dilation):
        """
        Convert a layer to use dilated convolutions.

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

    def extract_backbone_features(self, x):
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

    def forward(self, x):
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

        # Apply PPM to the deepest feature map
        ppm_out = self.ppm(backbone_features[-1])

        # Process features with FPN or simple concatenation
        if self.use_fpn:
            # Apply FPN to get multi-scale features
            fpn_features = self.fpn(backbone_features)

            # Upsample all FPN features to the size of the first feature map
            target_size = fpn_features[0].shape[2:]
            upsampled_features = []

            for feat in fpn_features:
                upsampled_features.append(
                    F.interpolate(
                        feat,
                        size=target_size,
                        mode="bilinear",
                        align_corners=True,
                    )
                )

            # Concatenate all FPN features
            fpn_out = torch.cat(upsampled_features, dim=1)
        else:
            # Simple feature concatenation without FPN
            target_size = backbone_features[0].shape[2:]
            upsampled_features = []

            for feat in backbone_features:
                upsampled_features.append(
                    F.interpolate(
                        feat,
                        size=target_size,
                        mode="bilinear",
                        align_corners=True,
                    )
                )

            # Concatenate all backbone features
            fpn_out = torch.cat(upsampled_features, dim=1)

        # Upsample PPM output to match FPN features size
        ppm_out = F.interpolate(
            ppm_out, size=target_size, mode="bilinear", align_corners=True
        )

        # Combine FPN and PPM features
        combined = torch.cat([fpn_out, ppm_out], dim=1)

        # Apply fusion convolution
        fused = self.fusion_conv(combined)

        # Final classification
        output = self.classifier(fused)

        # Upsample to input resolution
        output = F.interpolate(
            output, size=input_size[2:], mode="bilinear", align_corners=True
        )

        return output

    def get_model_info(self):
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
    def get_supported_backbones():
        """
        Get list of supported backbone architectures.

        Returns:
            List of supported backbone names
        """
        return list(UPerNet.BACKBONE_CONFIGS.keys())

    def validate_model(self):
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
