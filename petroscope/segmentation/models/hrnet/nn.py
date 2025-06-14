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
    Enhanced HRNet Stage with cross-resolution attention.

    Each stage maintains multiple resolution streams and exchanges
    information between them through fusion operations enhanced with attention.
    """

    def __init__(
        self,
        input_channels: List[int],
        output_channels: List[int],
        num_blocks: int = 4,
        use_attention: bool = True,
    ):
        super(HRNetStage, self).__init__()
        self.num_streams = len(output_channels)
        self.use_attention = use_attention

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

        # Cross-resolution attention mechanism
        if use_attention:
            self.cross_attention = CrossResolutionAttention(output_channels)

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

        # Apply cross-resolution attention if enabled
        if self.use_attention:
            y_list = self.cross_attention(y_list)

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
    """Enhanced HRNet backbone with progressive fusion and attention."""

    def __init__(
        self, width: int = 32, in_channels: int = 3, use_attention: bool = True
    ):
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
        self.use_attention = use_attention

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
            use_attention=use_attention,
        )

        # Stage 3: Three streams with fusion
        self.stage3 = HRNetStage(
            input_channels=[channels[0], channels[1]],
            output_channels=[channels[0], channels[1], channels[2]],
            num_blocks=4,
            use_attention=use_attention,
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
            use_attention=use_attention,
        )

        # Progressive decoder for better feature aggregation
        self.progressive_decoder = ProgressiveDecoder(
            channels=channels, out_channels=channels[0]
        )

        # Final multi-scale fusion for segmentation
        # (keeping for backward compatibility)
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

    def forward(self, x, return_intermediate=False, use_progressive=True):
        """
        Forward pass with optional intermediate feature return.

        Args:
            x: Input tensor
            return_intermediate: If True, return intermediate features
                               for aux head
            use_progressive: If True, use progressive decoder,
                           else use traditional fusion

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

        # Choose fusion strategy
        if use_progressive:
            # Use progressive decoder for better feature aggregation
            final_feat = self.progressive_decoder(x_list)
        else:
            # Use traditional multi-scale fusion
            final_feat = self.final_fusion(x_list)

        if return_intermediate:
            return final_feat, intermediate_feat
        else:
            return final_feat


class HRNetWithOCR(nn.Module):
    """Enhanced HRNetV2 with multi-scale predictions and refinement."""

    def __init__(
        self,
        num_classes: int,
        width: int = 32,
        in_channels: int = 3,
        ocr_mid_channels: int = 512,
        dropout: float = 0.1,
        use_aux_head: bool = True,
        use_multi_scale: bool = True,
        use_boundary_refine: bool = True,
        use_attention: bool = True,
        use_progressive: bool = True,
    ):
        super(HRNetWithOCR, self).__init__()

        self.backbone = HRNetBackbone(
            width=width, in_channels=in_channels, use_attention=use_attention
        )

        # Get backbone output channels
        if width == 18:
            backbone_channels = 18
        elif width == 32:
            backbone_channels = 32
        elif width == 48:
            backbone_channels = 48

        self.use_aux_head = use_aux_head
        self.use_multi_scale = use_multi_scale
        self.use_boundary_refine = use_boundary_refine
        self.use_progressive = use_progressive

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

        # Enhanced attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(ocr_mid_channels, ocr_mid_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ocr_mid_channels // 4, ocr_mid_channels, 1),
            nn.Sigmoid(),
        )

        # Multi-scale prediction heads
        if use_multi_scale:
            self.multi_scale_heads = nn.ModuleList(
                [
                    # Main scale head
                    nn.Sequential(
                        nn.Dropout2d(dropout),
                        nn.Conv2d(ocr_mid_channels, num_classes, 1),
                    ),
                    # Half scale head
                    nn.Sequential(
                        nn.Conv2d(
                            ocr_mid_channels, ocr_mid_channels // 2, 3, 2, 1
                        ),
                        nn.BatchNorm2d(ocr_mid_channels // 2),
                        nn.ReLU(inplace=True),
                        nn.Dropout2d(dropout),
                        nn.Conv2d(ocr_mid_channels // 2, num_classes, 1),
                    ),
                    # Quarter scale head
                    nn.Sequential(
                        nn.Conv2d(
                            ocr_mid_channels, ocr_mid_channels // 4, 3, 4, 1
                        ),
                        nn.BatchNorm2d(ocr_mid_channels // 4),
                        nn.ReLU(inplace=True),
                        nn.Dropout2d(dropout),
                        nn.Conv2d(ocr_mid_channels // 4, num_classes, 1),
                    ),
                ]
            )
        else:
            # Single scale classifier
            self.classifier = nn.Sequential(
                nn.Dropout2d(dropout),
                nn.Conv2d(ocr_mid_channels, num_classes, 1),
            )

        # Boundary refinement module
        if use_boundary_refine:
            self.boundary_refine = BoundaryRefinementModule(num_classes)

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

        # Extract features using enhanced HRNet backbone
        if self.use_aux_head and self.training:
            feats, aux_feats = self.backbone(
                x,
                return_intermediate=True,
                use_progressive=self.use_progressive,
            )
        else:
            feats = self.backbone(
                x,
                return_intermediate=False,
                use_progressive=self.use_progressive,
            )

        # Apply attention mechanism
        ocr_feats = self.conv_ocr(feats)
        attention_weights = self.attention(ocr_feats)
        attended_feats = ocr_feats * attention_weights

        # Multi-scale or single-scale predictions
        if self.use_multi_scale:
            # Multi-scale predictions
            outputs = []
            for i, head in enumerate(self.multi_scale_heads):
                output = head(attended_feats)

                # Resize to input size
                output = F.interpolate(
                    output,
                    size=input_size,
                    mode="bilinear",
                    align_corners=True,
                )
                outputs.append(output)

            # Main output is the first (full resolution) prediction
            main_output = outputs[0]

            # Apply boundary refinement if enabled
            if self.use_boundary_refine:
                main_output = self.boundary_refine(main_output)

            if self.training:
                # During training, return main output and optionally aux output
                # The base model expects either a single tensor or (main, aux)
                # tuple
                if self.use_aux_head:
                    # Auxiliary output for training
                    aux_output = self.aux_head(aux_feats)
                    aux_output = F.interpolate(
                        aux_output,
                        size=input_size,
                        mode="bilinear",
                        align_corners=True,
                    )
                    # Return as tuple for auxiliary loss compatibility
                    return main_output, aux_output
                else:
                    # Return only main output
                    return main_output
            else:
                # Return only main output during inference
                return main_output
        else:
            # Single scale prediction
            output = self.classifier(attended_feats)
            output = F.interpolate(
                output, size=input_size, mode="bilinear", align_corners=True
            )

            # Apply boundary refinement if enabled
            if self.use_boundary_refine:
                output = self.boundary_refine(output)

            if self.use_aux_head and self.training:
                # Auxiliary output for training
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


class CrossResolutionAttention(nn.Module):
    """Enhanced cross-resolution attention beyond HRNetv2"""

    def __init__(self, channels_list: List[int]):
        super().__init__()
        self.num_streams = len(channels_list)
        self.channels_list = channels_list

        # Spatial attention for each stream
        self.spatial_attention = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c, c // 8, 1),
                    nn.BatchNorm2d(c // 8),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(c // 8, 1, 1),
                    nn.Sigmoid(),
                )
                for c in channels_list
            ]
        )

        # Channel attention for each stream
        self.channel_attention = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(c, c // 16, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(c // 16, c, 1),
                    nn.Sigmoid(),
                )
                for c in channels_list
            ]
        )

        # Channel adaptation layers for cross-stream fusion
        self.channel_adapt = nn.ModuleList()
        for i in range(self.num_streams):
            adapt_layers = nn.ModuleList()
            for j in range(self.num_streams):
                if i == j:
                    # Same stream - identity
                    adapt_layers.append(nn.Identity())
                else:
                    # Different stream - adapt channels
                    adapt_layers.append(
                        nn.Sequential(
                            nn.Conv2d(channels_list[j], channels_list[i], 1),
                            nn.BatchNorm2d(channels_list[i]),
                            nn.ReLU(inplace=True),
                        )
                    )
            self.channel_adapt.append(adapt_layers)

        # Cross-stream fusion weights
        self.cross_weights = nn.Parameter(
            torch.ones(self.num_streams, self.num_streams)
        )

    def forward(self, feature_list: List):
        """Apply cross-resolution attention to feature list"""
        attended_features = []

        for i, feat in enumerate(feature_list):
            # Apply spatial attention
            spatial_att = self.spatial_attention[i](feat)
            feat_spatial = feat * spatial_att

            # Apply channel attention
            channel_att = self.channel_attention[i](feat)
            feat_attended = feat_spatial * channel_att

            attended_features.append(feat_attended)

        # Apply cross-stream attention weights with channel adaptation
        weighted_features = []
        for i in range(self.num_streams):
            weighted_feat = None

            for j in range(self.num_streams):
                # Get feature j and adapt its channels to match stream i
                feat_j = attended_features[j]
                feat_j_adapted = self.channel_adapt[i][j](feat_j)

                # Resize to match spatial dimensions
                target_size = attended_features[i].size()[2:]
                if feat_j_adapted.size()[2:] != target_size:
                    feat_j_adapted = F.interpolate(
                        feat_j_adapted,
                        size=target_size,
                        mode="bilinear",
                        align_corners=True,
                    )

                # Apply cross-stream weight
                weight = torch.softmax(self.cross_weights[i], dim=0)[j]
                weighted_contribution = weight * feat_j_adapted

                if weighted_feat is None:
                    weighted_feat = weighted_contribution
                else:
                    weighted_feat += weighted_contribution

            weighted_features.append(weighted_feat)

        return weighted_features


class ProgressiveDecoder(nn.Module):
    """Progressive upsampling with skip connections"""

    def __init__(self, channels: List[int], out_channels: int):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels

        # Simple progressive upsampling without skip connections for now
        # Start from highest channel count (lowest resolution) and work down
        self.upsample_layers = nn.ModuleList()

        # Process channels in reverse: [256, 128, 64, 32]
        reversed_channels = list(reversed(channels))

        for i in range(len(reversed_channels) - 1):
            in_ch = reversed_channels[i]
            out_ch = reversed_channels[i + 1]

            layer = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
            self.upsample_layers.append(layer)

    def forward(self, feature_list: List):
        """Progressive decoding from lowest to highest resolution"""
        # Start from lowest resolution (last in list)
        # feature_list: [high_res, med_res, low_res, lowest_res]
        current_feat = feature_list[-1]  # Start with lowest resolution

        # Progressive upsampling
        for layer in self.upsample_layers:
            current_feat = layer(current_feat)

        return current_feat


class BoundaryRefinementModule(nn.Module):
    """Specialized boundary refinement for mineral grain boundaries"""

    def __init__(self, num_classes):
        super().__init__()

        # Edge detection for mineral boundaries
        self.edge_conv = nn.Sequential(
            nn.Conv2d(num_classes, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Boundary-aware attention
        self.boundary_attention = nn.Sequential(
            nn.Conv2d(32, 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

        # Refined prediction
        self.refine_conv = nn.Conv2d(num_classes + 32, num_classes, 1)

    def forward(self, pred):
        # Detect boundaries
        edge_features = self.edge_conv(pred)
        boundary_mask = self.boundary_attention(edge_features)

        # Apply boundary-aware refinement
        enhanced_features = torch.cat([pred, edge_features], dim=1)
        refined_pred = self.refine_conv(enhanced_features)

        # Apply boundary attention
        return refined_pred * boundary_mask + pred * (1 - boundary_mask)
