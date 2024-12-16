import pytorch_lightning 
import torch
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device Count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("CUDA Device Name:", torch.cuda.get_device_name(0))
from glioma_segmentation.data.BraTSDataModule import BraTSDataModule
from glioma_segmentation.lightning_module import GliomaSegmentationModule
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import KFold
import numpy as np

if __name__ == '__main__':
    # Initialize Hydra and compose the configuration
    with hydra.initialize(config_path="../configs/data", version_base=None):
        cfg = hydra.compose(config_name="BraTSDataModule_config")

    # Define the data module
    data_module = BraTSDataModule(cfg)
    data_module.setup(stage='fit')  # Call setup to initialize train_data

    # Get the data and labels
    data = data_module.train_data
    labels = [d["label"] for d in data]

    # Initialize KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(data, labels)):
        print(f"Fold {fold + 1}")

        # Split the data into training and validation sets
        train_data = [data[i] for i in train_idx]
        val_data = [data[i] for i in val_idx]

        # Update the data module with the new splits
        data_module.train_data = train_data

        # Define the model
        model = GliomaSegmentationModule(
            in_channels=4,
            out_channels=4
        )

        # Initialize the trainer
        trainer = pytorch_lightning.Trainer(
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1 if torch.cuda.is_available() else None,
            max_epochs=10,
            enable_checkpointing=True,
            num_sanity_val_steps=1,
            log_every_n_steps=16,
        )

        # Train the model
        trainer.fit(model, data_module)
