import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import logging
from glioma_segmentation.models.UNet import UNet

# Configure logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
        
        # Log unique labels and min/max output values
        logger.debug(f"Train Step {batch_idx} - Unique Labels: {torch.unique(labels).tolist()}")
        logger.debug(f"Train Step {batch_idx} - Outputs Min: {outputs.min().item()}, Max: {outputs.max().item()}")

        loss = F.cross_entropy(outputs, labels)
        self.log("train_loss", loss, batch_size=images.size(0))

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

        loss = F.cross_entropy(outputs, labels)
        self.log("val_loss", loss, batch_size=images.size(0))

        # Log diagnostic loss information
        logger.debug(f"Val Step {batch_idx} - Loss: {loss.item()}")

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def configure_gradient_clipping(self, optimizer, optimizer_idx):
        # Clip gradients to prevent gradient explosion
        self.clip_gradients(optimizer, gradient_clip_val=1.0)

    def on_train_epoch_start(self):
        logger.info("Starting a new training epoch...")

    def on_validation_epoch_start(self):
        logger.info("Starting a new validation epoch...")
