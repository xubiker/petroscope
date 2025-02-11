# TODO replace with lazy imports

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, pool_sizes):
        super().__init__()
        self.pool_sizes = pool_sizes
        self.features = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=s),
                    nn.Conv2d(
                        in_channels,
                        in_channels // len(pool_sizes),
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(in_channels // len(pool_sizes)),
                    nn.ReLU(inplace=True),
                )
                for s in pool_sizes
            ]
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        pooled_outputs = [
            F.interpolate(
                feature(x), size=(h, w), mode="bilinear", align_corners=True
            )
            for feature in self.features
        ]
        return torch.cat([x] + pooled_outputs, dim=1)


class PSPNet(nn.Module):
    def __init__(
        self, n_classes: int, backbone: str, dilated=True, pretrained=True
    ):
        super().__init__()

        self.n_classes = n_classes
        self.dilated = dilated
        self.backbone = backbone

        # Load ResNet backbone
        resnet = getattr(models, backbone)(pretrained=pretrained)
        self.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2

        if dilated:
            # Apply dilated convolutions to layer3 and layer4
            self.layer3 = self._make_dilated(resnet.layer3, dilation=2)
            self.layer4 = self._make_dilated(resnet.layer4, dilation=4)
        else:
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4

        # Determine output channels based on backbone type
        backbone_out_channels = (
            512 if backbone in ["resnet18", "resnet34"] else 2048
        )

        # PSP Module
        self.psp = PyramidPoolingModule(
            in_channels=backbone_out_channels, pool_sizes=(1, 2, 3, 6)
        )

        # Final classifier
        self.final = nn.Sequential(
            nn.Conv2d(
                backbone_out_channels * 2,
                512,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, n_classes, kernel_size=1),
        )

    def _make_dilated(self, layer, dilation):
        for n, m in layer.named_modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (3, 3):  # Only modify 3x3 convolutions
                    m.dilation = (dilation, dilation)
                    m.padding = (dilation, dilation)
                if m.stride == (
                    2,
                    2,
                ):  # Prevent downsampling that breaks residuals
                    m.stride = (1, 1)
        return layer

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.psp(x)
        x = self.final(x)
        return F.interpolate(
            x, size=(h, w), mode="bilinear", align_corners=True
        )
