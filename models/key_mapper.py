"""
KeyMapper model for StyleGAN watermarking.
"""
import torch
import torch.nn as nn


class KeyMapper(nn.Module):
    """
    Fixed secret mapping: maps a latent partial vector or a vector of selected image pixels to a configurable-bit binary key.
    """
    def __init__(self, input_dim=32, output_dim=4, seed=None):
        """
        Initialize the KeyMapper.
        
        Args:
            input_dim (int): Dimension of the input latent partial vector.
            output_dim (int): Dimension of the output binary key.
            seed (int, optional): Random seed for reproducible initialization of the weights.
        """
        super(KeyMapper, self).__init__()
        
        # Set seed for reproducibility if provided
        if seed is not None:
            # Use a local generator to avoid affecting global PyTorch seed
            generator = torch.Generator()
            generator.manual_seed(seed)
            self.register_buffer('W', torch.randn(input_dim, output_dim, generator=generator))
            self.register_buffer('b', torch.randn(output_dim, generator=generator))
        else:
            # Use default random initialization when no seed is provided
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