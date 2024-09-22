import unittest
import os
from glioma_segmentation.data.glioma_data_module import GliomaDataModule

class TestGliomaDataModule(unittest.TestCase):

    def setUp(self):
        # Assuming you have a small dataset in the `data_sample` directory for testing
        self.train_dir = 'data_sample/training_data'
        self.val_dir = 'data_sample/validation_data'
        self.batch_size = 1

        # Define mock transformations if needed
        self.transformations = {
            'train_transforms': [],
            'val_transforms': []
        }

        self.data_module = GliomaDataModule(
            train_dir=self.train_dir,
            val_dir=self.val_dir,
            batch_size=self.batch_size,
            transformations=self.transformations
        )

    def test_prepare_data(self):
        train_data = self.data_module._prepare_data(self.train_dir)
        val_data = self.data_module._prepare_data(self.val_dir)

        # Ensure some data was loaded
        self.assertGreater(len(train_data), 0)
        self.assertGreater(len(val_data), 0)

    def test_train_dataloader(self):
        self.data_module.setup('fit')
        train_loader = self.data_module.train_dataloader()

        # Ensure the dataloader is working
        for batch in train_loader:
            x, y = batch
            self.assertEqual(x.shape[0], self.batch_size)  # Batch size should match
            break  # No need to iterate over the full dataset

    def test_val_dataloader(self):
        self.data_module.setup('fit')
        val_loader = self.data_module.val_dataloader()

        # Ensure the dataloader is working
        for batch in val_loader:
            x, y = batch
            self.assertEqual(x.shape[0], self.batch_size)  # Batch size should match
            break
