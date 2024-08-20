import os
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, 
    RandRotate, RandFlip, ToTensor
)
from monai.data import Dataset, DataLoader
import lightning as L

# Get the directory where the script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define relative paths based on the base directory
train_dir = os.path.join(base_dir, "../../data_sample/training_data")
val_dir = os.path.join(base_dir, "../../data_sample/validation_data")


class Medical3DDataModule(L.LightningDataModule):
    def __init__(self, train_dir, val_dir, batch_size: int = 1, use_cache: bool = False):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.use_cache = use_cache

        # Define transformations for training and validation sets
        self.train_transforms = Compose([
            LoadImage(image_only=False),           # Load images and masks
            EnsureChannelFirst(),                  # Ensure channels are the first dimension
            ScaleIntensity(),                      # Normalize intensity
            RandRotate(range_x=0.1, prob=0.5),     # Augmentation - rotation
            RandFlip(prob=0.5),                    # Augmentation - flip
            ToTensor()                             # Convert to PyTorch tensors
        ])

        self.val_transforms = Compose([
            LoadImage(image_only=False),
            EnsureChannelFirst(),
            ScaleIntensity(),
            ToTensor()
        ])

    def setup(self, stage=None):
        # Prepare training dataset
        if stage == 'fit' or stage is None:
            train_data = self._prepare_data(self.train_dir)
            self.train_dataset = self._create_dataset(train_data, self.train_transforms)
            
            val_data = self._prepare_data(self.val_dir)
            self.val_dataset = self._create_dataset(val_data, self.val_transforms)
