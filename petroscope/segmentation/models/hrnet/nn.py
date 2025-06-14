"""
HRNetV2 network implementation for semantic segmentation.

This module implements a simplified High-Resolution Network (HRNetV2)
architecture for semantic segmentation.
"""

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
else:
    torch = None

from petroscope.utils.lazy_imports import torch, nn, F  # noqa


class BasicBlock(nn.Module):
    """Basic residual block for HRNet."""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """Bottleneck residual block for HRNet."""

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class FusionLayer(nn.Module):
    """Multi-scale fusion layer for HRNet."""

    def __init__(self, in_channels_list: List[int], out_channels: int):
        super(FusionLayer, self).__init__()
        self.convs = nn.ModuleList()

        for in_channels in in_channels_list:
            if in_channels == out_channels:
                self.convs.append(nn.Identity())
            else:
                self.convs.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                    )
                )

    def forward(self, inputs):
        target_size = inputs[0].shape[2:]
        fused = None

        for i, (x, conv) in enumerate(zip(inputs, self.convs)):
            # Resize to target size if needed
            if x.shape[2:] != target_size:
                x = F.interpolate(
                    x, size=target_size, mode="bilinear", align_corners=True
                )

            x = conv(x)
            fused = x if fused is None else fused + x

        return fused


class HRNetBackbone(nn.Module):
    """Simplified HRNet backbone."""

    def __init__(self, width: int = 32, in_channels: int = 3):
        super(HRNetBackbone, self).__init__()

        # Configuration for different widths
        if width == 18:
            channels = [18, 36, 72, 144]
        elif width == 32:
            channels = [32, 64, 128, 256]
        elif width == 48:
            channels = [48, 96, 192, 384]
        else:
            raise ValueError(f"Unsupported width: {width}")

        self.channels = channels

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Stage 1 - Single high-resolution stream
        self.stage1 = self._make_layer(Bottleneck, 64, 64, 4)

        # Transition to multi-resolution
        self.transition1 = nn.Sequential(
            nn.Conv2d(256, channels[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # Stage 2 - Two streams
        self.stage2_high = self._make_layer(
            BasicBlock, channels[0], channels[0], 4
        )
        self.stage2_low = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], 3, 2, 1, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
            self._make_layer(BasicBlock, channels[1], channels[1], 4),
        )

        # Fusion after stage 2
        self.fusion2 = FusionLayer(channels[:2], channels[0])

        # Stage 3 - Three streams
        self.stage3_streams = nn.ModuleList(
            [
                self._make_layer(BasicBlock, channels[0], channels[0], 4),
                self._make_layer(BasicBlock, channels[1], channels[1], 4),
                nn.Sequential(
                    nn.Conv2d(channels[1], channels[2], 3, 2, 1, bias=False),
                    nn.BatchNorm2d(channels[2]),
                    nn.ReLU(inplace=True),
                    self._make_layer(BasicBlock, channels[2], channels[2], 4),
                ),
            ]
        )

        # Final fusion
        self.final_fusion = FusionLayer(channels[:3], channels[0])

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes, planes * block.expansion, 1, stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Stem
        x = self.stem(x)

        # Stage 1
        x = self.stage1(x)

        # Transition to multi-resolution
        x_high = self.transition1(x)

        # Stage 2
        x_high = self.stage2_high(x_high)
        x_low = self.stage2_low(x_high)

        # Fusion after stage 2
        x_fused = self.fusion2([x_high, x_low])

        # Stage 3 - Generate three streams
        streams = [
            self.stage3_streams[0](x_fused),  # High resolution
            self.stage3_streams[1](x_low),  # Medium resolution
            self.stage3_streams[2](x_low),  # Low resolution (new)
        ]

        # Final fusion
        output = self.final_fusion(streams)
        return output


class HRNetWithOCR(nn.Module):
    """HRNetV2 with simplified OCR-like attention for segmentation."""

    def __init__(
        self,
        num_classes: int,
        width: int = 32,
        in_channels: int = 3,
        ocr_mid_channels: int = 512,
        dropout: float = 0.1,
        use_aux_head: bool = True,
    ):
        super(HRNetWithOCR, self).__init__()

        self.backbone = HRNetBackbone(width=width, in_channels=in_channels)

        # Get backbone output channels
        if width == 18:
            backbone_channels = 18
        elif width == 32:
            backbone_channels = 32
        elif width == 48:
            backbone_channels = 48

        self.use_aux_head = use_aux_head

        # Auxiliary head for training
        if use_aux_head:
            self.aux_head = nn.Sequential(
                nn.Conv2d(backbone_channels, backbone_channels, 3, 1, 1),
                nn.BatchNorm2d(backbone_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout),
                nn.Conv2d(backbone_channels, num_classes, 1),
            )

        # Main segmentation head with attention
        self.conv_ocr = nn.Sequential(
            nn.Conv2d(backbone_channels, ocr_mid_channels, 3, 1, 1),
            nn.BatchNorm2d(ocr_mid_channels),
            nn.ReLU(inplace=True),
        )

        # Simple attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(ocr_mid_channels, ocr_mid_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ocr_mid_channels // 4, ocr_mid_channels, 1),
            nn.Sigmoid(),
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout2d(dropout), nn.Conv2d(ocr_mid_channels, num_classes, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input_size = x.size()[2:]

        # Extract features using HRNet backbone
        feats = self.backbone(x)

        # Apply attention mechanism
        ocr_feats = self.conv_ocr(feats)
        attention_weights = self.attention(ocr_feats)
        attended_feats = ocr_feats * attention_weights

        # Main output
        output = self.classifier(attended_feats)
        output = F.interpolate(
            output, size=input_size, mode="bilinear", align_corners=True
        )

        # For now, always return only the main output to be compatible
        # with the existing training pipeline
        # TODO: Implement auxiliary loss handling in the training pipeline
        return output
