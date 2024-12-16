import pytorch_lightning 
from glioma_segmentation.data.BraTSDataModule import BraTSDataModule
from glioma_segmentation.lightning_module import GliomaSegmentationModule
import hydra
from omegaconf import DictConfig, OmegaConf

if __name__ == '__main__':
    # Initialize Hydra and compose the configuration
    with hydra.initialize(config_path="../configs/data", version_base=None):
        cfg = hydra.compose(config_name="BraTSDataModule_config")

    # Define the data module
    data_module = BraTSDataModule(cfg)

    # Define the model
    model = GliomaSegmentationModule(
        in_channels=4,
        out_channels=4
    )

    # Initialize the trainer
    trainer = pytorch_lightning.Trainer(
        accelerator='cpu',
        devices=1,
        max_epochs=10,
        enable_checkpointing=True,
        num_sanity_val_steps=1,
        log_every_n_steps=16,
    )

    trainer.fit(model, data_module)
