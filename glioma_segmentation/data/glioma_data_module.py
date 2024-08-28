# glioma_data_module.py

import os
from monai.transforms import Compose
from monai.data import Dataset, DataLoader
import lightning as L

class GliomaDataModule(L.LightningDataModule):
    def __init__(self, train_dir, val_dir, batch_size: int = 1, use_cache: bool = False, transformations=None):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.use_cache = use_cache

        # Assign transformations from the config: configs/data/transforms.yaml
        self.train_transforms = Compose(transformations['train_transforms'])
        self.val_transforms = Compose(transformations['val_transforms'])

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_data = self._prepare_data(self.train_dir)
            self.train_dataset = self._create_dataset(train_data, self.train_transforms)
            
            val_data = self._prepare_data(self.val_dir)
            self.val_dataset = self._create_dataset(val_data, self.val_transforms)

    def _prepare_data(self, data_dir):
        data = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".nii") or file.endswith(".nii.gz"):
                    data.append(os.path.join(root, file))
        return data

    def _create_dataset(self, data, transforms):
        return Dataset(data=data, transform=transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
