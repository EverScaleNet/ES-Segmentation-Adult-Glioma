import unittest
import torch
from glioma_segmentation.models.UNet import UNet3D

class TestUNet3D(unittest.TestCase):

    def test_unet3d_forward(self):
        model = UNet3D(in_channels=1, out_channels=3, init_features=32)
        x = torch.randn(1, 1, 64, 64, 64)  # Batch size 1, 1 input channel, 64x64x64 volume
        out = model(x)
        self.assertEqual(out.shape, (1, 3, 64, 64, 64))  # Should match output channels and spatial dimensions

    def test_unet3d_training_step(self):
        model = UNet3D(in_channels=1, out_channels=3, init_features=32)
        x = torch.randn(1, 1, 64, 64, 64)
        y = torch.randint(0, 3, (1, 64, 64, 64))  # Simulating segmentation labels
        batch = (x, y)
        loss = model.training_step(batch, 0)
        self.assertIsInstance(loss, torch.Tensor)  # Ensure loss is calculated correctly
