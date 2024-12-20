import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import logging
from glioma_segmentation.models.UNet import UNet
from torch.nn import Dropout

# Configure logger
logging.basicConfig(level=logging.INFO)  # Set to INFO to reduce verbosity
logger = logging.getLogger(__name__)

class GliomaSegmentationModule(pl.LightningModule):
    def __init__(self, in_channels, out_channels, learning_rate=1e-5, dropout_rate=0.8):
        super(GliomaSegmentationModule, self).__init__()
        self.model = UNet(in_channels, out_channels)
        self.learning_rate = learning_rate
        self.dropout = Dropout(dropout_rate)
        self.class_weights = torch.tensor([0.7377, 0.0254, 0.1474, 0.0896, 0.0896])  # Include background class weight

    def forward(self, x):
        x = self.model(x)
        return self.dropout(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        outputs = self(images)
        labels = labels.squeeze(1).long()  # Squeeze the channel dimension and convert to Long
        
        # Log unique labels and min/max output values
        logger.debug(f"Train Step {batch_idx} - Unique Labels: {torch.unique(labels).tolist()}")
        logger.debug(f"Train Step {batch_idx} - Outputs Min: {outputs.min().item()}, Max: {outputs.max().item()}")

        loss = F.cross_entropy(outputs, labels, weight=self.class_weights.to(self.device))
        self.log("train_loss", loss, batch_size=images.size(0))

        # Calculate accuracy
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        self.log("train_acc", acc, batch_size=images.size(0))

        # Log diagnostic loss information
        logger.debug(f"Train Step {batch_idx} - Loss: {loss.item()}")

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        outputs = self(images)
        labels = labels.squeeze(1).long()  # Squeeze the channel dimension and convert to Long
        
        # Log unique labels and min/max output values
        logger.debug(f"Val Step {batch_idx} - Unique Labels: {torch.unique(labels).tolist()}")
        logger.debug(f"Val Step {batch_idx} - Outputs Min: {outputs.min().item()}, Max: {outputs.max().item()}")

        loss = F.cross_entropy(outputs, labels, weight=self.class_weights.to(self.device))
        self.log("val_loss", loss, batch_size=images.size(0))

        # Calculate accuracy
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        self.log("val_acc", acc, batch_size=images.size(0))

        # Log diagnostic loss information
        logger.debug(f"Val Step {batch_idx} - Loss: {loss.item()}")

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_acc'
            }
        }

    def on_train_epoch_start(self):
        logger.info("Starting a new training epoch...")

    def on_validation_epoch_start(self):
        logger.info("Starting a new validation epoch...")
