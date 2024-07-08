import unittest
from src.data.lightning_data_module import BrainTumorDataModule
from pathlib import Path
import torch

class TestBrainTumorDataModule(unittest.TestCase):
    """Tests for the BrainTumorDataModule class."""

    def setUp(self):
        """Initialize the test setup with required parameters."""
        root_directory = Path().resolve().parent
        self.data_dir = root_directory / 'data_sample' / 'training_data'
        self.batch_size = 8
        self.img_size = (128, 128, 128)
        self.data_module = BrainTumorDataModule(self.data_dir, self.batch_size, self.img_size)
        self.data_module.setup()

    def test_setup(self):
        """Test if the setup method initializes the train dataset correctly."""
        self.assertIsNotNone(self.data_module.train_dataset, "The train dataset should not be None after setup.")
        self.assertGreater(len(self.data_module.train_dataset), 0, "The train dataset should contain data.")

    def test_train_dataloader(self):
        """Test if the train dataloader returns batches with the expected shape."""
        train_loader = self.data_module.train_dataloader()
        for batch in train_loader:
            images, masks = batch['image'], batch['label']
            self.assertEqual(images.shape, (self.batch_size, 4, *self.img_size), "The shape of the images in the batch is incorrect.")
            self.assertEqual(masks.shape, (self.batch_size, 1, *self.img_size), "The shape of the masks in the batch is incorrect.")
            break

    def test_batch_size(self):
        """Test if the dataloader works correctly for different batch sizes."""
        batch_sizes = [1, 2, 4]
        for batch_size in batch_sizes:
            self.data_module.batch_size = batch_size
            train_loader = self.data_module.train_dataloader()
            for batch in train_loader:
                images, masks = batch['image'], batch['label']
                self.assertEqual(images.shape, (batch_size, 4, *self.img_size), f"The shape of the images in the batch for batch size {batch_size} is incorrect.")
                self.assertEqual(masks.shape, (batch_size, 1, *self.img_size), f"The shape of the masks in the batch for batch size {batch_size} is incorrect.")
                break

    def test_data_transforms(self):
        """Test if the transforms are correctly applied to the data."""
        train_loader = self.data_module.train_dataloader()
        for batch in train_loader:
            images, masks = batch['image'], batch['label']
            self.assertTrue(images.max() <= 1.0 and images.min() >= 0.0, "The images should be scaled between 0 and 1.")
            break

if __name__ == "__main__":
    unittest.main()
