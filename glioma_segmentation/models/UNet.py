import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=5):  # Set default out_channels to 5
        super(UNet, self).__init__()
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.bottleneck = self.conv_block(512, 1024)
        self.decoder4 = self.conv_block(1024 + 512, 512)
        self.decoder3 = self.conv_block(512 + 256, 256)
        self.decoder2 = self.conv_block(256 + 128, 128)
        self.decoder1 = self.conv_block(128 + 64, 64)
        self.final_layer = nn.Conv3d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool3d(enc1, kernel_size=2, stride=2))
        enc3 = self.encoder3(F.max_pool3d(enc2, kernel_size=2, stride=2))
        enc4 = self.encoder4(F.max_pool3d(enc3, kernel_size=2, stride=2))
        bottleneck = self.bottleneck(F.max_pool3d(enc4, kernel_size=2, stride=2))
        dec4 = self.decoder4(torch.cat([F.interpolate(bottleneck, size=enc4.shape[2:], mode='trilinear', align_corners=False), enc4], dim=1))
        dec3 = self.decoder3(torch.cat([F.interpolate(dec4, size=enc3.shape[2:], mode='trilinear', align_corners=False), enc3], dim=1))
        dec2 = self.decoder2(torch.cat([F.interpolate(dec3, size=enc2.shape[2:], mode='trilinear', align_corners=False), enc2], dim=1))
        dec1 = self.decoder1(torch.cat([F.interpolate(dec2, size=enc1.shape[2:], mode='trilinear', align_corners=False), enc1], dim=1))
        return self.final_layer(dec1)
