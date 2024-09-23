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
        ''' Load patient folders and files as tupples of 4 scans and a mask '''
        data = []

        for patient in os.listdir(data_dir):
            patient_path = os.path.join(data_dir, patient)
            scans = []                    # Initialize the list of scans
            mask = None                   # Initialize the mask
            for file in os.listdir(patient_path):
                if 'seg' in file:           # Load the mask
                    mask = os.path.join(patient_path, file)
                else:                       # Load the scans
                    scans.append(os.path.join(patient_path, file))


    def _create_dataset(self, data, transforms):
        return Dataset(data=data, transform=transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
