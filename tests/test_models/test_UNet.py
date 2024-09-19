import unittest
import torch
from src.models import UNet  # Import the model after its implementation

class TestUNet(unittest.TestCase):
    """Tests for the UNet model."""

    def setUp(self):
        """Initialize the test setup with required parameters."""
        self.in_channels = 4
        self.out_channels = 5
        self.input_shape = (1, self.in_channels, 182, 218, 182)
        self.output_shape = (1, self.out_channels, 182, 218, 182)

        self.model = UNet(self.in_channels, self.out_channels)
        
    def test_constructor(self):
        """Test if the constructor correctly initializes all layers of the model."""
        self.assertIsInstance(self.model, UNet, "The model was not correctly initialized as a UNet instance.")
    
    def test_forward_pass(self):
        """Test if the forward pass returns output with the expected shape."""
        x = torch.randn(*self.input_shape)
        y = self.model(x)
        self.assertEqual(y.shape, self.output_shape, "The output shape of the forward pass is incorrect.")
    
    def test_model_parameters(self):
        """Test if the model has an appropriate number of parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(total_params, 0, "The model should have more than zero parameters.")
    
    def test_gpu_compatibility(self):
        """Test if the model works on a GPU without errors."""
        if torch.cuda.is_available():
            self.model.cuda()
            x = torch.randn(*self.input_shape).cuda()
            y = self.model(x)
            self.assertEqual(y.shape, self.output_shape, "The output shape on GPU is incorrect.")
        else:
            self.skipTest("GPU is not available, skipping the test.")
    
    def test_backward_pass(self):
        """Test if gradients are correctly calculated during the backward pass."""
        x = torch.randn(*self.input_shape, requires_grad=True)
        y = self.model(x)
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(x.grad, "Gradients were not correctly calculated during the backward pass.")
    
    def test_batch_size(self):
        """Test if the model works correctly for different batch sizes."""
        batch_sizes = [1, 2, 4]
        for batch_size in batch_sizes:
            input_shape = (batch_size, self.in_channels, 182, 218, 182)
            x = torch.randn(*input_shape)
            y = self.model(x)
            expected_shape = (batch_size, self.out_channels, 182, 218, 182)
            self.assertEqual(y.shape, expected_shape, f"The output shape for batch size {batch_size} is incorrect.")

if __name__ == "__main__":
    unittest.main()
