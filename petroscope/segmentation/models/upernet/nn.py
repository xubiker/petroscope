import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
)


class PPM(nn.Module):
    """
    Pyramid Pooling Module from PSPNet.

    This module applies pooling at different scales and concatenates the results.
    """

    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin),
                    nn.Conv2d(
                        in_dim, reduction_dim, kernel_size=1, bias=False
                    ),
                    nn.BatchNorm2d(reduction_dim),
                    nn.ReLU(inplace=True),
                )
            )
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(
                F.interpolate(
                    f(x), x_size[2:], mode="bilinear", align_corners=True
                )
            )
        return torch.cat(out, 1)


class FPN(nn.Module):
    """
    Feature Pyramid Network.

    This module fuses features from different levels of the backbone network.
    """

    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            )
            self.fpn_convs.append(
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
            )

    def forward(self, inputs):
        # Build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # Build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode="bilinear",
                align_corners=True,
            )

        # Build outputs
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        return tuple(outs)


class UPerNet(nn.Module):
    """
    UPerNet: Unified Perceptual Parsing Network.

    This is a skeleton implementation that will need to be filled in with the actual
    UPerNet architecture details.

    References:
        - Paper: https://arxiv.org/abs/1807.10221
    """

    def __init__(self, n_classes, backbone="resnet50", use_fpn=True):
        super(UPerNet, self).__init__()
        self.n_classes = n_classes
        self.backbone = backbone
        self.use_fpn = use_fpn

        # Placeholder for the actual implementation
        # This is just a skeleton that needs to be filled in

        # Load the backbone network
        if backbone == "resnet18":
            self.backbone_net = torchvision.models.resnet18(
                weights=ResNet18_Weights.DEFAULT
            )
            fpn_in_channels = [64, 128, 256, 512]
            ppm_in_channels = 512
        elif backbone == "resnet34":
            self.backbone_net = torchvision.models.resnet34(
                weights=ResNet34_Weights.DEFAULT
            )
            fpn_in_channels = [64, 128, 256, 512]
            ppm_in_channels = 512
        elif backbone == "resnet50":
            self.backbone_net = torchvision.models.resnet50(
                weights=ResNet50_Weights.DEFAULT
            )
            fpn_in_channels = [256, 512, 1024, 2048]
            ppm_in_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove the classification head
        self.backbone_net = nn.Sequential(
            *list(self.backbone_net.children())[:-2]
        )

        # FPN module
        fpn_out_channels = 256
        if use_fpn:
            self.fpn = FPN(fpn_in_channels, fpn_out_channels)

        # PPM module
        ppm_bins = (1, 2, 3, 6)
        ppm_out_channels = 512
        self.ppm = PPM(
            ppm_in_channels, ppm_out_channels // len(ppm_bins), ppm_bins
        )

        # Final classifier
        self.classifier = nn.Conv2d(
            ppm_out_channels + ppm_in_channels, n_classes, kernel_size=1
        )

    def forward(self, x):
        """
        Forward pass of the UPerNet model.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor of shape (batch_size, n_classes, height, width)
        """
        # This is a placeholder implementation
        # The actual implementation will need to extract features from the backbone,
        # apply FPN and PPM, and then combine the results

        input_size = x.size()

        # Extract features from the backbone
        features = self.backbone_net(x)

        # Apply PPM
        ppm_out = self.ppm(features)

        # Upscale to the input resolution
        output = F.interpolate(
            self.classifier(ppm_out),
            size=input_size[2:],
            mode="bilinear",
            align_corners=True,
        )

        return output
