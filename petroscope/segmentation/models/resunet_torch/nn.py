from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

from petroscope.utils.lazy_imports import torch  # noqa

nn = torch.nn  # noqa


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
        branch1 = self.conv0(x)
        branch1 = self.relu(branch1)
        branch1 = self.bn0(branch1)
        branch2 = self.conv1(x)
        branch2 = self.relu(branch2)
        branch2 = self.bn1(branch2)
        branch2 = self.conv2(branch2)
        branch2 = self.relu(branch2)
        branch2 = self.bn2(branch2)
        return branch1 + branch2


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv = ConvResBlock(in_channels, out_channels)

    def forward(self, x_down, x_concat):
        x = self.upsample(x_down)
        x = torch.cat((x, x_concat), 1)
        x = self.conv(x)
        return x


class ResUNet(nn.Module):

    def __init__(self, n_classes: int, n_layers: int, start_filters: int):
        super(ResUNet, self).__init__()
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.filters = start_filters
        self.down_blocks = []
        self.upsample_blocks = []
        self.upconv_blocks = []

        self.down1 = ConvResBlock(3, start_filters)
        self.down2 = ConvResBlock(start_filters, start_filters * 2)
        self.down3 = ConvResBlock(start_filters * 2, start_filters * 4)
        self.down4 = ConvResBlock(start_filters * 4, start_filters * 8)

        self.bottleneck = ConvResBlock(start_filters * 8, start_filters * 16)

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
