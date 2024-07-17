import torch.nn.functional as F
import torch.nn as nn
from glioma_segmentation.models.components.UNet_components import DoubleConv3D, Down3D, Up3D, OutConv3D

class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes, trilinear=True):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.trilinear = trilinear

        self.inc = DoubleConv3D(n_channels, 64)
        self.down1 = Down3D(64, 128)
        self.down2 = Down3D(128, 256)
        self.down3 = Down3D(256, 512)
        factor = 2 if trilinear else 1
        self.down4 = Down3D(512, 1024 // factor)
        self.up1 = Up3D(1024, 512 // factor, trilinear)
        self.up2 = Up3D(512, 256 // factor, trilinear)
        self.up3 = Up3D(256, 128 // factor, trilinear)
        self.up4 = Up3D(128, 64, trilinear)
        self.outc = OutConv3D(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
