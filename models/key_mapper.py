"""
KeyMapper model for StyleGAN watermarking.
"""
import torch
import torch.nn as nn


class KeyMapper(nn.Module):
    """
    Fixed secret mapping: maps a latent partial vector to a configurable-bit binary key.
    """
    def __init__(self, input_dim=32, output_dim=4):
        """
        Initialize the KeyMapper.
        
        Args:
            input_dim (int): Dimension of the input latent partial vector.
            output_dim (int): Dimension of the output binary key.
        """
        super(KeyMapper, self).__init__()
        # Secret parameters (fixed, non-trainable)
        self.register_buffer('W', torch.randn(input_dim, output_dim))
        self.register_buffer('b', torch.randn(output_dim))
    
    def forward(self, latent_partial):
        """
        Forward pass of the key mapper.
        
        Args:
            latent_partial (torch.Tensor): Input tensor of shape (B, input_dim).
            
        Returns:
            torch.Tensor: Binary output tensor of shape (B, output_dim).
        """
        # Linear projection + tanh activation
        projection = torch.matmul(latent_partial, self.W) + self.b
        activated = torch.tanh(projection)
        target = (activated > 0).float()  # binary output
        return target 