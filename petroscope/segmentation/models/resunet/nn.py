from typing import TYPE_CHECKING, Optional

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
    )

from petroscope.utils.lazy_imports import torch, nn, F, models  # noqa
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
)


class ConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvResBlock, self).__init__()
        self.conv0 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, padding="same"
        )
        self.bn0 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding="same"
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding="same"
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # branch1 = self.conv0(x)
        # branch1 = self.relu(branch1)
        # branch1 = self.bn0(branch1)
        # branch2 = self.conv1(x)
        # branch2 = self.relu(branch2)
        # branch2 = self.bn1(branch2)
        # branch2 = self.conv2(branch2)
        # branch2 = self.relu(branch2)
        # branch2 = self.bn2(branch2)
        branch1 = self.conv0(x)
        branch1 = self.bn0(branch1)
        branch1 = self.relu(branch1)
        branch2 = self.conv1(x)
        branch2 = self.bn1(branch2)
        branch2 = self.relu(branch2)
        branch2 = self.conv2(branch2)
        branch2 = self.bn2(branch2)
        branch2 = self.relu(branch2)
        return branch1 + branch2


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=None):
        super(UpBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        # Use provided skip_channels if given, otherwise assume equal to out_channels
        skip_channels = (
            skip_channels if skip_channels is not None else out_channels
        )

        # After concatenation, the channels will be out_channels + skip_channels
        self.conv = ConvResBlock(out_channels + skip_channels, out_channels)

    def forward(self, x_down, x_concat):
        x = self.upsample(x_down)

        # Ensure spatial dimensions match before concatenation
        if x.size()[2:] != x_concat.size()[2:]:
            x = F.interpolate(
                x,
                size=x_concat.size()[2:],
                mode="bilinear",
                align_corners=True,
            )

        x = torch.cat((x, x_concat), 1)
        x = self.conv(x)
        return x


class DilatedConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(DilatedConvResBlock, self).__init__()
        self.conv0 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, padding="same"
        )
        self.bn0 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        branch1 = self.conv0(x)
        branch1 = self.bn0(branch1)
        branch1 = self.relu(branch1)
        branch2 = self.conv1(x)
        branch2 = self.bn1(branch2)
        branch2 = self.relu(branch2)
        branch2 = self.conv2(branch2)
        branch2 = self.bn2(branch2)
        branch2 = self.relu(branch2)
        return branch1 + branch2


class ResUNetBase(nn.Module):
    """Base ResUNet implementation without pretrained backbones."""

    def __init__(
        self,
        n_classes: int,
        n_layers: int,
        start_filters: int,
        dilated: bool = False,
    ):
        super(ResUNetBase, self).__init__()
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.start_filters = start_filters
        self.dilated = dilated

        # Use dilated convolutions in the downsampling path if specified
        Block = DilatedConvResBlock if dilated else ConvResBlock

        # For dilated model, use different dilation rates in different layers
        dilation_rates = [1, 1, 2, 4] if dilated else [1, 1, 1, 1]

        self.down1 = Block(3, start_filters, dilation_rates[0])
        self.down2 = Block(start_filters, start_filters * 2, dilation_rates[1])
        self.down3 = Block(
            start_filters * 2, start_filters * 4, dilation_rates[2]
        )
        self.down4 = Block(
            start_filters * 4, start_filters * 8, dilation_rates[3]
        )

        self.bottleneck = Block(
            start_filters * 8, start_filters * 16, dilation_rates[3]
        )

        self.up1 = UpBlock(start_filters * 16, start_filters * 8)
        self.up2 = UpBlock(start_filters * 8, start_filters * 4)
        self.up3 = UpBlock(start_filters * 4, start_filters * 2)
        self.up4 = UpBlock(start_filters * 2, start_filters)

        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.out = nn.Conv2d(start_filters, n_classes, kernel_size=1)

    def forward(self, x):
        down1 = self.down1(x)
        mp1 = self.max_pool2d(down1)
        down2 = self.down2(mp1)
        mp2 = self.max_pool2d(down2)
        down3 = self.down3(mp2)
        mp3 = self.max_pool2d(down3)
        down4 = self.down4(mp3)
        mp4 = self.max_pool2d(down4)

        bottleneck = self.bottleneck(mp4)

        up1 = self.up1(bottleneck, down4)
        up2 = self.up2(up1, down3)
        up3 = self.up3(up2, down2)
        up4 = self.up4(up3, down1)

        return self.out(up4)


class ResUNet(nn.Module):
    """ResUNet with support for pretrained backbones and dilated convolutions."""

    # Configuration details for each supported backbone
    BACKBONE_CONFIGS = {
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
    }

    def __init__(
        self,
        n_classes: int,
        n_layers: int = 4,
        start_filters: int = 64,
        backbone: Optional[str] = None,
        dilated: bool = False,
        pretrained: bool = True,
    ):
        super(ResUNet, self).__init__()
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.start_filters = start_filters
        self.backbone_name = backbone
        self.dilated = dilated
        self.pretrained = pretrained

        if backbone is None:
            # If no backbone is specified, use the original ResUNet implementation
            self.model = ResUNetBase(
                n_classes=n_classes,
                n_layers=n_layers,
                start_filters=start_filters,
                dilated=dilated,
            )
        else:
            # Validate backbone selection
            if backbone not in self.BACKBONE_CONFIGS:
                supported_backbones = list(self.BACKBONE_CONFIGS.keys())
                raise ValueError(
                    f"Unsupported backbone: {backbone}. "
                    f"Supported backbones: {supported_backbones}"
                )

            # Get backbone configuration
            config = self.BACKBONE_CONFIGS[backbone]

            # Initialize backbone
            self.backbone_features = self._build_backbone(
                backbone, config, pretrained
            )

            # Feature channels from the backbone
            backbone_channels = config["channels"]

            # Each UpBlock takes the output of the previous layer and combines it with a skip connection
            # We need to specify both the in_channels (from previous decoder layer or bottleneck)
            # and the skip_channels (from encoder) to get the dimensions right
            self.up1 = UpBlock(
                in_channels=backbone_channels[3],  # bottleneck features
                out_channels=backbone_channels[2],  # desired output channels
                skip_channels=backbone_channels[2],  # skip connection channels
            )
            self.up2 = UpBlock(
                in_channels=backbone_channels[2],  # output from up1
                out_channels=backbone_channels[1],  # desired output channels
                skip_channels=backbone_channels[1],  # skip connection channels
            )
            self.up3 = UpBlock(
                in_channels=backbone_channels[1],  # output from up2
                out_channels=backbone_channels[0],  # desired output channels
                skip_channels=backbone_channels[0],  # skip connection channels
            )
            self.up4 = UpBlock(
                in_channels=backbone_channels[0],  # output from up3
                out_channels=start_filters,  # final features before classification
                skip_channels=64,  # first layer features (after stem)
            )

            # Final classifier
            self.out = nn.Conv2d(start_filters, n_classes, kernel_size=1)

    def _build_backbone(self, backbone_name, config, pretrained):
        """Build and configure the backbone network."""
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

            # Apply dilated convolutions to later layers if specified
            if self.dilated:
                backbone["layer3"] = self._make_dilated(
                    backbone["layer3"], dilation=2
                )
                backbone["layer4"] = self._make_dilated(
                    backbone["layer4"], dilation=4
                )

            return backbone

        else:
            raise ValueError(
                f"Unsupported backbone architecture: {backbone_name}"
            )

    def _make_dilated(self, layer, dilation):
        """Convert a layer to use dilated convolutions."""
        for n, m in layer.named_modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (3, 3):  # Only modify 3x3 convolutions
                    m.dilation = (dilation, dilation)
                    m.padding = (dilation, dilation)
                if m.stride == (2, 2):  # Prevent downsampling
                    m.stride = (1, 1)
        return layer

    def forward(self, x):
        if hasattr(self, "model"):
            # If using the base model without pretrained backbone
            return self.model(x)

        # Using pretrained backbone
        input_size = x.size()

        # Extract features from backbone
        x = self.backbone_features["layer0"](x)
        f1 = self.backbone_features["layer1"](x)
        f2 = self.backbone_features["layer2"](f1)
        f3 = self.backbone_features["layer3"](f2)
        f4 = self.backbone_features["layer4"](f3)

        # Decoder path
        up1 = self.up1(f4, f3)
        up2 = self.up2(up1, f2)
        up3 = self.up3(up2, f1)
        up4 = self.up4(up3, x)

        # Final classification
        output = self.out(up4)

        # Ensure output is the same size as input
        if output.size()[2:] != input_size[2:]:
            output = F.interpolate(
                output,
                size=input_size[2:],
                mode="bilinear",
                align_corners=True,
            )

        return output
