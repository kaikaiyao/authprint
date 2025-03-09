import random
from typing import Tuple
import hashlib
import hmac

import numpy as np
import torch
import torch.nn as nn
from Crypto.Cipher import ChaCha20

from key.key_utils import flip_key

class CryptoCNN(nn.Module):
    """A convolutional neural network with cryptographic weight initialization.
    
    Uses either ChaCha20 encryption or HMAC-based CSPRNG for secure parameter initialization 
    with fan-in aware scaling for improved convergence properties.
    
    Args:
        input_channels: Number of input channels
        output_channels: Number of output channels
        binary_key: 256-bit secret key for parameter initialization
        key_type: Type of key generation ("encryption" or "csprng")
    """

    def __init__(self, input_channels: int, output_channels: int, binary_key: bytes, key_type: str = "csprng"):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, output_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        if key_type == "encryption":
            self._init_layers_with_encryption(binary_key)
        else:  # default to csprng
            self._init_layers_with_csprng(binary_key)

    def _init_layers_with_encryption(self, binary_key: bytes) -> None:
        """Initializes network parameters using encrypted seed with fan-in scaling.
        
        Args:
            binary_key: 256-bit secret key for ChaCha20 encryption
        """
        # Generate cryptographic seed
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        seed_size = total_params * 4  # 4 bytes per float32
        rng = np.random.default_rng(seed=0)
        seed = rng.bytes(seed_size)
        
        # Encrypt seed using ChaCha20
        chacha_key = binary_key[:32]  # 256-bit key
        nonce = b'\x00' * 8  # Fixed nonce (cryptographically insecure for actual deployment, used only for reproducibility)
        cipher = ChaCha20.new(key=chacha_key, nonce=nonce)
        ciphertext = cipher.encrypt(seed)
        
        # Convert to normalized float32 array
        uint32_vals = np.frombuffer(ciphertext, dtype='>u4')
        normalized_floats = uint32_vals.astype(np.float32) / np.float32(0xFFFFFFFF)
        
        # Parameter initialization with fan-in scaling
        pointer = 0
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            fan_in = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
            bound = 1.0 / np.sqrt(fan_in)
            
            # Weight initialization
            weight_size = np.prod(layer.weight.shape)
            layer_weights = normalized_floats[pointer:pointer + weight_size]
            scaled_weights = (layer_weights * 2 * bound) - bound
            layer.weight.data = torch.from_numpy(scaled_weights.reshape(layer.weight.shape)).float()
            pointer += weight_size
            
            # Bias initialization
            if layer.bias is not None:
                bias_size = np.prod(layer.bias.shape)
                layer_biases = normalized_floats[pointer:pointer + bias_size]
                scaled_biases = (layer_biases * 2 * bound) - bound
                layer.bias.data = torch.from_numpy(scaled_biases.reshape(layer.bias.shape)).float()
                pointer += bias_size

    def _init_layers_with_csprng(self, binary_key: bytes) -> None:
        """Initializes network parameters using cryptographically-secure PRNG with fan-in scaling.
        
        Args:
            binary_key: 256-bit secret key for CSPRNG initialization
        """
        # Count total parameters for initialization
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Create CSPRNG using HMAC-based extract-and-expand key derivation function (HKDF)
        # This is a secure way to expand the binary key into the needed random bytes
        random_bytes = self._generate_csprng_bytes(binary_key, total_params * 4)  # 4 bytes per float32
        
        # Convert to normalized float32 array
        uint32_vals = np.frombuffer(random_bytes, dtype='>u4')
        normalized_floats = uint32_vals.astype(np.float32) / np.float32(0xFFFFFFFF)
        
        # Parameter initialization with fan-in scaling
        pointer = 0
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            fan_in = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
            bound = 1.0 / np.sqrt(fan_in)
            
            # Weight initialization
            weight_size = np.prod(layer.weight.shape)
            layer_weights = normalized_floats[pointer:pointer + weight_size]
            scaled_weights = (layer_weights * 2 * bound) - bound
            layer.weight.data = torch.from_numpy(scaled_weights.reshape(layer.weight.shape)).float()
            pointer += weight_size
            
            # Bias initialization
            if layer.bias is not None:
                bias_size = np.prod(layer.bias.shape)
                layer_biases = normalized_floats[pointer:pointer + bias_size]
                scaled_biases = (layer_biases * 2 * bound) - bound
                layer.bias.data = torch.from_numpy(scaled_biases.reshape(layer.bias.shape)).float()
                pointer += bias_size
    
    def _generate_csprng_bytes(self, key: bytes, num_bytes: int) -> bytes:
        """Generate pseudorandom bytes using a cryptographically secure method.
        
        Uses HMAC-SHA256 in counter mode to generate random bytes from the key.
        This is similar to how HKDF expansion works.
        
        Args:
            key: The cryptographic key (256 bits)
            num_bytes: Number of random bytes to generate
            
        Returns:
            Random bytes of specified length
        """
        result = bytearray()
        counter = 0
        
        # Generate bytes in chunks using HMAC in counter mode
        while len(result) < num_bytes:
            # Create counter value as 4-byte big-endian
            counter_bytes = counter.to_bytes(4, byteorder='big')
            
            # Generate a block of pseudorandom bytes using HMAC-SHA256
            h = hmac.new(key, counter_bytes, hashlib.sha256)
            block = h.digest()
            
            # Add the block to our result
            result.extend(block)
            counter += 1
        
        # Truncate to the exact requested size
        return bytes(result[:num_bytes])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        return self.conv5(x)


def generate_mask_secret_key(
    image_shape: Tuple[int, int, int, int],
    seed: int,
    device: str = 'cpu',
    flip_key_type: str = 'none',
    key_type: str = 'csprng',
) -> nn.Module:
    """Generates a frozen CNN-based mask generator with cryptographic initialization.
    
    Args:
        image_shape: Input image shape tuple (batch, channels, height, width)
        seed: Random seed for key generation
        device: Target device for the mask generator
        flip_key_type: Type of key bit flipping to apply
        key_type: Type of key generation method to use ("encryption" or "csprng")
    
    Returns:
        Initialized and frozen CNN mask generator
    """
    _, channels, height, width = image_shape
    
    # Generate 256-bit key from seed (for experiment reproduction use)
    random.seed(seed)
    binary_key = random.getrandbits(256).to_bytes(32, 'big')

    # Flip the key (for evaluation)
    if flip_key_type == "none":
        pass
    else:
        binary_key = flip_key(input_key=binary_key, flip_key_type=flip_key_type)

    # Create and freeze CNN
    mask_generator = CryptoCNN(channels, channels, binary_key, key_type).to(device)
    for param in mask_generator.parameters():
        param.requires_grad = False
        
    return mask_generator


def mask_image_with_key(
    images: torch.Tensor,
    cnn_key: nn.Module
) -> torch.Tensor:
    """Applies cryptographic masking to input images using a CNN key.
    
    Args:
        images: Input tensor of shape (batch, channels, height, width)
        cnn_key: Initialized CryptoCNN instance
    
    Returns:
        Masked image tensor with same shape as input
    """
    # Normalize to [0, 1] range
    images_min = images.amin(dim=(1, 2, 3), keepdim=True)
    images_max = images.amax(dim=(1, 2, 3), keepdim=True)
    normalized = (images - images_min) / (images_max - images_min + 1e-8)
    
    return cnn_key(normalized)