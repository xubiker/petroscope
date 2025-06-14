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


class MultiScaleFusion(nn.Module):
    """
    Multi-Scale Fusion module as described in HRNetV2 paper.

    This module fuses features from multiple resolution streams by:
    1. Upsampling lower resolution features to match the highest resolution
    2. Concatenating all features
    3. Applying convolution to reduce channels and create final representation
    """

    def __init__(self, input_channels: List[int], output_channels: int):
        super(MultiScaleFusion, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        # Fusion convolution to combine all scales
        total_channels = sum(input_channels)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(total_channels, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: List):
        """
        Args:
            features: List of feature maps from different scales
                     [high_res, med_res, low_res, ...]

        Returns:
            Fused feature map at highest resolution
        """
        if not features:
            raise ValueError("Features list cannot be empty")

        # Get target size from highest resolution feature (first in list)
        target_size = features[0].size()[2:]

        # Upsample all features to target size and collect
        upsampled_features = []
        for i, feat in enumerate(features):
            if feat.size()[2:] != target_size:
                # Upsample lower resolution features
                upsampled = F.interpolate(
                    feat, size=target_size, mode="bilinear", align_corners=True
                )
                upsampled_features.append(upsampled)
            else:
                upsampled_features.append(feat)

        # Concatenate all features along channel dimension
        fused = torch.cat(upsampled_features, dim=1)

        # Apply fusion convolution
        output = self.fusion_conv(fused)

        return output


class HRNetStage(nn.Module):
    """
    HRNet Stage with proper multi-resolution processing and fusion.

    Each stage maintains multiple resolution streams and exchanges
    information between them through fusion operations.
    """

    def __init__(
        self,
        input_channels: List[int],
        output_channels: List[int],
        num_blocks: int = 4,
    ):
        super(HRNetStage, self).__init__()
        self.num_streams = len(output_channels)

        # Create parallel streams for different resolutions
        self.streams = nn.ModuleList()
        for i in range(self.num_streams):
            if i < len(input_channels):
                in_ch = input_channels[i]
            else:
                # New stream - create from previous stream with downsampling
                in_ch = input_channels[-1]

            out_ch = output_channels[i]

            # Each stream is a sequence of residual blocks
            stream = self._make_stream(in_ch, out_ch, num_blocks)
            self.streams.append(stream)

        # Transition layers for new streams (downsampling)
        self.transitions = nn.ModuleList()
        for i in range(len(input_channels), self.num_streams):
            # Create downsampling transition from previous stream
            prev_channels = output_channels[i - 1]
            curr_channels = output_channels[i]

            transition = nn.Sequential(
                nn.Conv2d(prev_channels, curr_channels, 3, 2, 1, bias=False),
                nn.BatchNorm2d(curr_channels),
                nn.ReLU(inplace=True),
            )
            self.transitions.append(transition)

        # Fusion layers for information exchange between streams
        self.fusions = nn.ModuleList()
        for i in range(self.num_streams):
            # Each stream receives fused information from all other streams
            fusion_layers = nn.ModuleList()

            for j in range(self.num_streams):
                if i == j:
                    # Identity connection for same resolution
                    fusion_layers.append(nn.Identity())
                elif j < i:
                    # Downsample from higher resolution stream
                    fusion_layers.append(
                        self._make_downsample_fusion(
                            output_channels[j], output_channels[i], i - j
                        )
                    )
                else:
                    # Upsample from lower resolution stream
                    fusion_layers.append(
                        self._make_upsample_fusion(
                            output_channels[j], output_channels[i]
                        )
                    )

            self.fusions.append(fusion_layers)

    def _make_stream(
        self, in_channels: int, out_channels: int, num_blocks: int
    ):
        """Create a stream of residual blocks."""
        layers = []

        # First block may need channel adjustment
        if in_channels != out_channels:
            layers.append(
                BasicBlock(
                    in_channels,
                    out_channels,
                    downsample=nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 1, bias=False),
                        nn.BatchNorm2d(out_channels),
                    ),
                )
            )
            num_blocks -= 1

        # Remaining blocks
        for _ in range(num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def _make_downsample_fusion(
        self, in_channels: int, out_channels: int, scale_factor: int
    ):
        """Create fusion layer for downsampling."""
        layers = []

        # Apply multiple 3x3 conv with stride 2 for each scale factor
        current_channels = in_channels
        for i in range(scale_factor):
            if i == scale_factor - 1:
                # Last layer outputs target channels
                target_channels = out_channels
            else:
                # Intermediate layers double channels
                target_channels = current_channels * 2

            layers.extend(
                [
                    nn.Conv2d(
                        current_channels, target_channels, 3, 2, 1, bias=False
                    ),
                    nn.BatchNorm2d(target_channels),
                    (
                        nn.ReLU(inplace=True)
                        if i < scale_factor - 1
                        else nn.Identity()
                    ),
                ]
            )
            current_channels = target_channels

        return nn.Sequential(*layers)

    def _make_upsample_fusion(self, in_channels: int, out_channels: int):
        """Create fusion layer for upsampling."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x_list: List):
        """
        Args:
            x_list: List of feature maps from different resolution streams

        Returns:
            List of processed feature maps for each stream
        """
        # Process each stream
        y_list = []
        for i, stream in enumerate(self.streams):
            if i < len(x_list):
                # Existing stream
                y = stream(x_list[i])
            else:
                # New stream - create from transition
                transition_idx = i - len(x_list)
                prev_feat = y_list[i - 1]  # Use previous stream output
                y = self.transitions[transition_idx](prev_feat)

            y_list.append(y)

        # Apply fusion between streams
        fused_list = []
        for i in range(self.num_streams):
            # Collect features from all streams for fusion to stream i
            fusion_inputs = []

            for j in range(self.num_streams):
                if i == j:
                    # Same resolution - direct connection
                    fusion_inputs.append(y_list[j])
                else:
                    # Different resolution - apply fusion transformation
                    fused_feat = self.fusions[i][j](y_list[j])

                    # Adjust spatial size if needed
                    target_size = y_list[i].size()[2:]
                    if fused_feat.size()[2:] != target_size:
                        fused_feat = F.interpolate(
                            fused_feat,
                            size=target_size,
                            mode="bilinear",
                            align_corners=True,
                        )

                    fusion_inputs.append(fused_feat)

            # Sum all fusion inputs
            fused = fusion_inputs[0]
            for feat in fusion_inputs[1:]:
                fused = fused + feat

            fused_list.append(fused)

        return fused_list


class HRNetBackbone(nn.Module):
    """Enhanced HRNet backbone with proper multi-scale fusion."""

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

        # Stem: Initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Stage 1: Single high resolution stream
        self.stage1 = self._make_layer(Bottleneck, 64, 64, 4)

        # Transition 1: Create dual streams
        self.transition1 = nn.Sequential(
            nn.Conv2d(256, channels[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        self.transition1_down = nn.Sequential(
            nn.Conv2d(256, channels[1], 3, 2, 1, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
        )

        # Stage 2: Dual streams with fusion
        self.stage2 = HRNetStage(
            input_channels=[channels[0], channels[1]],
            output_channels=[channels[0], channels[1]],
            num_blocks=4,
        )

        # Stage 3: Three streams with fusion
        self.stage3 = HRNetStage(
            input_channels=[channels[0], channels[1]],
            output_channels=[channels[0], channels[1], channels[2]],
            num_blocks=4,
        )

        # Stage 4: Four streams with fusion
        self.stage4 = HRNetStage(
            input_channels=[channels[0], channels[1], channels[2]],
            output_channels=[
                channels[0],
                channels[1],
                channels[2],
                channels[3],
            ],
            num_blocks=3,
        )

        # Final multi-scale fusion for segmentation
        self.final_fusion = MultiScaleFusion(
            input_channels=channels, output_channels=channels[0]
        )

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        """Create a residual layer."""
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

    def forward(self, x, return_intermediate=False):
        """
        Forward pass with optional intermediate feature return.

        Args:
            x: Input tensor
            return_intermediate: If True, return intermediate features
                               for aux head

        Returns:
            Final fused features, optionally with intermediate features
        """
        # Stem
        x = self.stem(x)

        # Stage 1: Single stream
        x = self.stage1(x)

        # Transition to dual streams
        x_high = self.transition1(x)
        x_low = self.transition1_down(x)
        x_list = [x_high, x_low]

        # Stage 2: Process dual streams
        x_list = self.stage2(x_list)

        # Store intermediate for auxiliary head
        if return_intermediate:
            intermediate_feat = x_list[0]  # Use high resolution stream

        # Stage 3: Expand to three streams
        x_list = self.stage3(x_list)

        # Stage 4: Expand to four streams
        x_list = self.stage4(x_list)

        # Final multi-scale fusion
        final_feat = self.final_fusion(x_list)

        if return_intermediate:
            return final_feat, intermediate_feat
        else:
            return final_feat


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
        if self.use_aux_head and self.training:
            feats, aux_feats = self.backbone(x, return_intermediate=True)
        else:
            feats = self.backbone(x, return_intermediate=False)

        # Apply attention mechanism
        ocr_feats = self.conv_ocr(feats)
        attention_weights = self.attention(ocr_feats)
        attended_feats = ocr_feats * attention_weights

        # Main output
        output = self.classifier(attended_feats)
        output = F.interpolate(
            output, size=input_size, mode="bilinear", align_corners=True
        )

        if self.use_aux_head and self.training:
            # Auxiliary output for training (using auxiliary features)
            aux_output = self.aux_head(aux_feats)
            aux_output = F.interpolate(
                aux_output,
                size=input_size,
                mode="bilinear",
                align_corners=True,
            )
            return output, aux_output
        else:
            return output
