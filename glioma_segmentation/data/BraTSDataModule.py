import os
import glob
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from monai.transforms import LoadImaged, EnsureChannelFirstd, Compose
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
        self.data_dir = None
        self.batch_size = None
        self.num_workers = None

    def setup(self, stage=None):
        """
        Sets up the data module by loading training and validation data.

        Args:
            stage (str, optional): Stage of setup (e.g., 'fit', 'test'). Defaults to None.
        """
        self.data_dir = self.cfg.data.data_dir
        self.batch_size = self.cfg.loader.batch_size
        self.num_workers = self.cfg.loader.num_workers
        
        # Load training patient directories
        train_patient_dirs = sorted(glob.glob(os.path.join(self.data_dir, "training_data", "BraTS-GLI-*")))

        # Load validation patient directories
        val_patient_dirs = sorted(glob.glob(os.path.join(self.data_dir, "validation_data", "BraTS-GLI-*")))

        # Create training data dictionaries
        train_data_dicts = []
        for patient_dir in train_patient_dirs:
            seg_file = os.path.join(patient_dir, f"{os.path.basename(patient_dir)}-seg.nii.gz")
            image_files = [
                os.path.join(patient_dir, f"{os.path.basename(patient_dir)}-t1c.nii.gz"),
                os.path.join(patient_dir, f"{os.path.basename(patient_dir)}-t1n.nii.gz"),
                os.path.join(patient_dir, f"{os.path.basename(patient_dir)}-t2f.nii.gz"),
                os.path.join(patient_dir, f"{os.path.basename(patient_dir)}-t2w.nii.gz")
            ]
            train_data_dicts.append({
                "image": image_files,  # List of modality images
                "label": seg_file      # Segmentation file
            })

        # Create validation data dictionaries
        val_data_dicts = []
        for patient_dir in val_patient_dirs:
            seg_file = os.path.join(patient_dir, f"{os.path.basename(patient_dir)}-seg.nii.gz")
            image_files = [
                os.path.join(patient_dir, f"{os.path.basename(patient_dir)}-t1c.nii.gz"),
                os.path.join(patient_dir, f"{os.path.basename(patient_dir)}-t1n.nii.gz"),
                os.path.join(patient_dir, f"{os.path.basename(patient_dir)}-t2f.nii.gz"),
                os.path.join(patient_dir, f"{os.path.basename(patient_dir)}-t2w.nii.gz")
            ]
            val_data_dicts.append({
                "image": image_files,  # List of modality images
                "label": seg_file      # Segmentation file
            })

        # Assign data to attributes
        self.train_data = train_data_dicts
        self.val_data = val_data_dicts

        # Define transformations
        self.train_transforms = Compose([
            LoadImaged(keys=["image", "label"], image_only=False),
            EnsureChannelFirstd(keys=["image", "label"])
        ])

        self.val_transforms = Compose([
            LoadImaged(keys=["image", "label"], image_only=False),
            EnsureChannelFirstd(keys=["image", "label"])
        ])

    def train_dataloader(self):
        """
        Returns the training DataLoader.

        Returns:
            DataLoader: DataLoader for training data.
        """
        train_dataset = Dataset(data=self.train_data, transform=self.train_transforms)
        return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        """
        Returns the validation DataLoader.

        Returns:
            DataLoader: DataLoader for validation data.
        """
        val_dataset = Dataset(data=self.val_data, transform=self.val_transforms)
        return DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        """
        Placeholder for test DataLoader.

        Returns:
            None
        """
        pass
