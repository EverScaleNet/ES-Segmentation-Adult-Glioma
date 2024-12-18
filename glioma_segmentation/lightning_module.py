import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from glioma_segmentation.models.UNet import UNet

class GliomaSegmentationModule(pl.LightningModule):
    def __init__(self, in_channels, out_channels, learning_rate=1e-3):
        super(GliomaSegmentationModule, self).__init__()
        self.model = UNet(in_channels, out_channels)
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        outputs = self(images)
        labels = labels.squeeze(1).long()  # Squeeze the channel dimension and convert to Long
        loss = F.cross_entropy(outputs, labels)
        self.log("train_loss", loss, batch_size=images.size(0))
        
        # Log the maximum and minimum values of the loss
        self.log("train_loss_max", torch.max(loss), batch_size=images.size(0))
        self.log("train_loss_min", torch.min(loss), batch_size=images.size(0))
        
        # Check for NaN values in the loss
        if torch.isnan(loss):
            print("NaN detected in training loss")
        
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        outputs = self(images)
        labels = labels.squeeze(1).long()  # Squeeze the channel dimension and convert to Long
        loss = F.cross_entropy(outputs, labels)
        self.log("val_loss", loss, batch_size=images.size(0))
        
        # Log the maximum and minimum values of the loss
        self.log("val_loss_max", torch.max(loss), batch_size=images.size(0))
        self.log("val_loss_min", torch.min(loss), batch_size=images.size(0))
        
        # Check for NaN values in the loss
        if torch.isnan(loss):
            print("NaN detected in validation loss")
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def configure_gradient_clipping(self, optimizer, optimizer_idx):
        # Clip gradients to prevent gradient explosion
        self.clip_gradients(optimizer, gradient_clip_val=1.0)
