import os
import glob
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from monai.transforms import LoadImaged, EnsureChannelFirstd, Compose, SpatialPadd, ScaleIntensityd
from monai.data import Dataset
import hydra
from omegaconf import DictConfig

class BraTSDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        """
        Initializes the BraTSDataModule with configuration parameters.

        Args:
            cfg (DictConfig): Configuration object containing data directory, batch size, and number of workers.
        """
        super().__init__()
        self.cfg = cfg
        self.data_dir = cfg.data.data_dir
        self.batch_size = cfg.loader.batch_size
        self.num_workers = cfg.loader.num_workers

    def setup(self, stage=None):
        """
        Sets up the data module by loading training and test data.

        Args:
            stage (str, optional): Stage of setup (e.g., 'fit', 'test'). Defaults to None.
        """
        self.data_dir = self.cfg.data.data_dir
        self.batch_size = self.cfg.loader.batch_size
        self.num_workers = self.cfg.loader.num_workers
        
        # Load training patient directories
        train_patient_dirs = sorted(glob.glob(os.path.join(self.data_dir, "training_data", "BraTS-GLI-*")))

        # Create training data dictionaries
        train_data_dicts = []
        for patient_dir in train_patient_dirs:
            base_name = os.path.basename(patient_dir)
            seg_file = os.path.join(patient_dir, f"{base_name}-seg.nii.gz")
            image_files = [
                os.path.join(patient_dir, f"{base_name}-t1c.nii.gz"),
                os.path.join(patient_dir, f"{base_name}-t1n.nii.gz"),
                os.path.join(patient_dir, f"{base_name}-t2f.nii.gz"),
                os.path.join(patient_dir, f"{base_name}-t2w.nii.gz")
            ]
            train_data_dicts.append({
                "image": image_files,
                "label": seg_file
            })

        self.train_data = train_data_dicts

        # Define training transformations
        self.train_transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            SpatialPadd(keys=["image", "label"], spatial_size=(182, 218, 182)),
            ScaleIntensityd(keys=["image"])
        ])

        if stage == 'test' or stage is None:
            test_patient_dirs = sorted(glob.glob(os.path.join(self.data_dir, "test_data", "BraTS-GLI-*")))
            test_data_dicts = []
            for patient_dir in test_patient_dirs:
                base_name = os.path.basename(patient_dir)
                image_files = [
                    os.path.join(patient_dir, f"{base_name}-t1c.nii.gz"),
                    os.path.join(patient_dir, f"{base_name}-t1n.nii.gz"),
                    os.path.join(patient_dir, f"{base_name}-t2f.nii.gz"),
                    os.path.join(patient_dir, f"{base_name}-t2w.nii.gz")
                ]
                test_data_dicts.append({
                    "image": image_files
                })
            self.test_data = test_data_dicts

            self.test_transforms = Compose([
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                SpatialPadd(keys=["image"], spatial_size=(182, 218, 182)),
                ScaleIntensityd(keys=["image"])
            ])

    def train_dataloader(self):
        """
        Returns the training DataLoader.

        Returns:
            DataLoader: DataLoader for training data.
        """
        train_dataset = Dataset(data=self.train_data, transform=self.train_transforms)
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        """
        Returns the validation DataLoader.

        Returns:
            DataLoader: DataLoader for validation data.
        """
        val_dataset = Dataset(data=self.val_data, transform=self.train_transforms)
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True
        )

    def test_dataloader(self):
        """
        Returns the test DataLoader.

        Returns:
            DataLoader: DataLoader for test data.
        """
        test_dataset = Dataset(data=self.test_data, transform=self.test_transforms)
        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True
        )
