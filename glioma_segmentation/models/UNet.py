import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
from .components.UNet_components import DoubleConv3D, Down3D, Up3D, OutConv3D

class UNet3D(pl.LightningModule):
    def __init__(self, in_channels, out_channels, init_features=32, lr=1e-4):
        super(UNet3D, self).__init__()
        self.lr = lr
        
        self.encoder1 = DoubleConv3D(in_channels, init_features)
        self.pool1 = nn.MaxPool3d(2)

        self.encoder2 = Down3D(init_features, init_features * 2)
        self.encoder3 = Down3D(init_features * 2, init_features * 4)
        self.encoder4 = Down3D(init_features * 4, init_features * 8)

        self.bottleneck = DoubleConv3D(init_features * 8, init_features * 16)

        self.upconv4 = Up3D(init_features * 16, init_features * 8)
        self.upconv3 = Up3D(init_features * 8, init_features * 4)
        self.upconv2 = Up3D(init_features * 4, init_features * 2)
        self.upconv1 = Up3D(init_features * 2, init_features)

        self.out_conv = OutConv3D(init_features, out_channels)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        bottleneck = self.bottleneck(enc4)

        dec4 = self.upconv4(bottleneck, enc4)
        dec3 = self.upconv3(dec4, enc3)
        dec2 = self.upconv2(dec3, enc2)
        dec1 = self.upconv1(dec2, enc1)

        return self.out_conv(dec1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', val_loss)
        return val_loss
