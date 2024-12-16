import os
import pytest
from monai.data import DataLoader
from glioma_segmentation.data.BraTSDataModule import BraTSDataModule
from hydra import initialize, compose
from omegaconf import OmegaConf

@pytest.fixture
def data_module():
    """
    Fixture to initialize the BraTSDataModule with the configuration from a YAML file.
    """
    with initialize(config_path="../../configs/data", job_name="test"):
        cfg = compose(config_name="BraTSDataModule_config.yaml")
        data_module = BraTSDataModule(cfg)
        data_module.setup()
        return data_module

def test_train_dataloader(data_module):
    """
    Test to ensure the train dataloader is correctly set up and returns data.
    """
    train_loader = data_module.train_dataloader()
    
    # Check if the dataloader returns any data
    assert len(train_loader.dataset) > 0, "No data in the training set!"
    
    # Check if the batch size matches the configuration
    batch = next(iter(train_loader))
    assert batch["image"].shape[0] == data_module.batch_size, f"Expected batch_size={data_module.batch_size}, but got {batch['image'].shape[0]}"

def test_test_dataloader(data_module):
    """
    Test to ensure the test dataloader is correctly set up and returns data.
    """
    test_loader = data_module.test_dataloader()
    
    # Check if the dataloader returns any data
    assert len(test_loader.dataset) > 0, "No data in the test set!"
    
    # Check if the batch size matches the configuration
    batch = next(iter(test_loader))
    assert batch["image"].shape[0] == data_module.batch_size, f"Expected batch_size={data_module.batch_size}, but got {batch['image'].shape[0]}"

def test_transformations(data_module):
    """
    Test to ensure the transformations are correctly applied.
    """
    train_loader = data_module.train_dataloader()
    
    # Get one sample from the batch
    batch = next(iter(train_loader))
    image = batch["image"]
    label = batch["label"]
    
    # Check if the image dimension starts with the channel (expecting [C, H, W, D])
    assert image.shape[1:] == (182, 218, 182), f"Expected image shape (182, 218, 182), but got {image.shape[1:]}"
    assert label.shape[1:] == (182, 218, 182), f"Expected label shape (182, 218, 182), but got {label.shape[1:]}"
