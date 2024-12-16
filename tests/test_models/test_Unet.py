import unittest
import torch
from glioma_segmentation.models.UNet import UNet

class TestUNet(unittest.TestCase):
    def setUp(self):
        self.model = UNet(in_channels=4, out_channels=1)  # Assuming 4 input channels and 1 output channel
        self.input_shape = (1, 4, 182, 218, 182)  # Batch size of 1, 4 channels, and the given shape

    def test_output_shape(self):
        input_tensor = torch.randn(self.input_shape)
        output = self.model(input_tensor)
        self.assertEqual(output.shape, (1, 1, 182, 218, 182))

    def test_forward_pass(self):
        input_tensor = torch.randn(self.input_shape)
        output = self.model(input_tensor)
        self.assertFalse(torch.isnan(output).any(), "Output contains NaN values")
        self.assertFalse(torch.isinf(output).any(), "Output contains Inf values")

if __name__ == '__main__':
    unittest.main()
