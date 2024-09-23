import unittest
import os
from glioma_segmentation.data.glioma_data_module import GliomaDataModule

class TestGliomaDataModule(unittest.TestCase):

    def setUp(self):
        # Define the directory structure for training and validation datasets
        self.train_dir = 'data_sample/training_data'
        self.val_dir = 'data_sample/validation_data'
        self.batch_size = 1

        # Example transformations, can be customized as needed
        self.transformations = {
            'train_transforms': [],
            'val_transforms': []
        }

        # Initialize the data module
        self.data_module = GliomaDataModule(
            train_dir=self.train_dir,
            val_dir=self.val_dir,
            batch_size=self.batch_size,
            transformations=self.transformations
        )

    def test_prepare_data(self):
        # Test if the data preparation correctly loads patient folders and files
        train_data = self.data_module._prepare_data(self.train_dir)
        val_data = self.data_module._prepare_data(self.val_dir)

        # Ensure that some data is loaded for both training and validation
        self.assertGreater(len(train_data), 0)
        self.assertGreater(len(val_data), 0)

    def test_train_dataloader(self):
        # Setup the data for training
        self.data_module.setup('fit')
        train_loader = self.data_module.train_dataloader()

        # Ensure that the training dataloader works and returns batches
        for batch in train_loader:
            scans, mask = batch
            # Ensure that the number of samples in a batch matches the batch size
            self.assertEqual(scans.shape[0], self.batch_size)
            # Ensure that there are 4 scans for each sample (modalities) and 1 mask
            self.assertEqual(scans.shape[1], 4)
            break  # We only need to check the first batch

    def test_val_dataloader(self):
        # Setup the data for validation
        self.data_module.setup('fit')
        val_loader = self.data_module.val_dataloader()

        # Ensure that the validation dataloader works and returns batches
        for batch in val_loader:
            scans, mask = batch
            # Ensure that the number of samples in a batch matches the batch size
            self.assertEqual(scans.shape[0], self.batch_size)
            # Ensure that there are 4 scans for each sample (modalities) and 1 mask
            self.assertEqual(scans.shape[1], 4)
            break  # We only need to check the first batch
