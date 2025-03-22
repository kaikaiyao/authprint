"""
KeyMapper model for StyleGAN watermarking.
"""
import torch
import torch.nn as nn


class KeyMapper(nn.Module):
    """
    Fixed secret mapping: maps a latent partial vector or a vector of selected image pixels to a configurable-bit binary key.
    """
    def __init__(self, input_dim=32, output_dim=4, seed=None, use_sine=False, sensitivity=10.0):
        """
        Initialize the KeyMapper.
        
        Args:
            input_dim (int): Dimension of the input latent partial vector.
            output_dim (int): Dimension of the output binary key.
            seed (int, optional): Random seed for reproducible initialization of the weights.
            use_sine (bool): Whether to use sine-based mapping (more sensitive to input changes).
                             Default is False for backward compatibility.
            sensitivity (float): Sensitivity parameter k for sine-based mapping.
                                Higher values make the key more sensitive to small input changes.
                                Only used when use_sine=True.
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
        
        # Store mapping type and sensitivity parameter
        self.use_sine = use_sine
        self.sensitivity = sensitivity
    
    def forward(self, latent_partial):
        """
        Forward pass of the key mapper.
        
        Args:
            latent_partial (torch.Tensor): Input tensor of shape (B, input_dim).
            
        Returns:
            torch.Tensor: Binary output tensor of shape (B, output_dim).
        """
        # Linear projection
        projection = torch.matmul(latent_partial, self.W)
        
        if self.use_sine:
            # Apply sine-based mapping: y = sign(sin(k*Wx))
            # Skip bias to ensure oscillation around zero
            sine_input = self.sensitivity * projection
            activated = torch.sin(sine_input)
        else:
            # Original implementation: y = sign(tanh(Wx + b))
            projection = projection + self.b
            activated = torch.tanh(projection)
            
        # Binary output
        target = (activated > 0).float()
        return target 