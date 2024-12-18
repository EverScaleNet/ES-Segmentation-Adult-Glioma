import os
import pytorch_lightning
import torch
from glioma_segmentation.data.BraTSDataModule import BraTSDataModule
from glioma_segmentation.lightning_module import GliomaSegmentationModule
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import KFold
import numpy as np
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

if __name__ == '__main__':
    # Set float32 matmul precision to utilize Tensor Cores
    torch.set_float32_matmul_precision('high')

    # Initialize Hydra and compose the configuration
    with hydra.initialize(config_path="../configs/data", version_base=None):
        cfg = hydra.compose(config_name="BraTSDataModule_config")

    # Define the data module
    data_module = BraTSDataModule(cfg)
    data_module.setup(stage='fit')  # Initialize train_data

    # Reduce batch size to reduce memory usage
    data_module.batch_size = 1  # Reduce batch size further

    # Get the data and labels
    data = data_module.train_data
    labels = [d["label"] for d in data]

    # Initialize KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize the WandbLogger
    wandb_logger = WandbLogger(project="glioma_segmentation")

    for fold, (train_idx, val_idx) in enumerate(kf.split(data, labels)):
        print(f"Fold {fold + 1}")

        # Split the data into training and validation sets
        train_data = [data[i] for i in train_idx]
        val_data = [data[i] for i in val_idx]

        # Update the data module with the new splits
        data_module.train_data = train_data
        data_module.set_val_data(val_data)  # Set the validation data

        # Define the model
        model = GliomaSegmentationModule(
            in_channels=4,
            out_channels=5  # Match the number of classes
        )

        # Define the checkpoint callback to save the best model
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath='trained_models',
            filename=f'model_fold_{fold + 1}_best',
            save_top_k=1,
            mode='min'
        )

        # Initialize the trainer with mixed precision training, WandbLogger, and checkpoint callback
        trainer = pytorch_lightning.Trainer(
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1 if torch.cuda.is_available() else None,
            max_epochs=10,
            precision='16-mixed',  # Ensure mixed precision training is enabled
            enable_checkpointing=True,
            num_sanity_val_steps=1,
            log_every_n_steps=16,
            enable_progress_bar=True,  # Enable progress bar
            logger=wandb_logger,  # Add WandbLogger
            callbacks=[checkpoint_callback]  # Add checkpoint callback
        )

        # Train the model
        print("Starting training for fold", fold + 1)
        trainer.fit(
            model,
            train_dataloaders=data_module.train_dataloader(),
            val_dataloaders=data_module.val_dataloader()
        )
        print("Finished training for fold", fold + 1)

    # Finish the wandb run
    wandb.finish()
