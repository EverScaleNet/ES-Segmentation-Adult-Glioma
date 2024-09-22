import unittest
import torch
from glioma_segmentation.models.components.UNet_components import DoubleConv3D, Down3D, Up3D, OutConv3D

class TestUNetComponents(unittest.TestCase):

    def test_double_conv_3d(self):
        model = DoubleConv3D(1, 32)
        x = torch.randn(1, 1, 64, 64, 64)  # Batch size 1, 1 input channel, 64x64x64 volume
        out = model(x)
        self.assertEqual(out.shape, (1, 32, 64, 64, 64))

    def test_down_3d(self):
        model = Down3D(32, 64)
        x = torch.randn(1, 32, 64, 64, 64)
        out = model(x)
        self.assertEqual(out.shape, (1, 64, 32, 32, 32))  # MaxPool3d reduces the spatial dimensions

    def test_up_3d(self):
        model = Up3D(64, 32)
        x1 = torch.randn(1, 64, 16, 16, 16)
        x2 = torch.randn(1, 32, 32, 32, 32)  # The skip connection tensor
        out = model(x1, x2)
        self.assertEqual(out.shape, (1, 32, 32, 32, 32))  # Output shape should match the skip connection

    def test_out_conv_3d(self):
        model = OutConv3D(32, 3)
        x = torch.randn(1, 32, 64, 64, 64)
        out = model(x)
        self.assertEqual(out.shape, (1, 3, 64, 64, 64))  # Output should have 3 channels
