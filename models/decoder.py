"""
Decoder model for SD watermarking.
"""
import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    Decoder network that predicts configurable-bit key logits from an input image.
    With increased capacity for better convergence.
    """
    def __init__(self, image_size=1024, channels=3, output_dim=4096):
        """
        Initialize the Decoder.
        
        Args:
            image_size (int): Input image size (width/height).
            channels (int): Number of input image channels.
            output_dim (int): Output dimension (key length).
        """
        super(Decoder, self).__init__()
        # Increase number of filters and add more layers for 1024x1024 input
        self.features = nn.Sequential(
            # Initial layer: 1024 -> 512
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 512 -> 256
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 256 -> 128
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128 -> 64
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64 -> 32
            nn.Conv2d(512, 768, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(768),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32 -> 16
            nn.Conv2d(768, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16 -> 8
            nn.Conv2d(1024, 1536, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1536),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8 -> 4
            nn.Conv2d(1536, 2048, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Global pooling and deeper classifier with progressive expansion
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 2560),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(2560, 3072),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(3072, output_dim)
        )
    
    def forward(self, x):
        """
        Forward pass of the decoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (B, output_dim).
        """
        features = self.features(x)
        return self.classifier(features) 


"""
Decoder model for StyleGAN watermarking.
"""
import torch
import torch.nn as nn


class StyleGAN2Decoder(nn.Module):
    """
    Decoder network that predicts configurable-bit key logits from an input image.
    With increased capacity for better convergence.
    """
    def __init__(self, image_size=256, channels=3, output_dim=4):
        """
        Initialize the Decoder.
        
        Args:
            image_size (int): Input image size (width/height).
            channels (int): Number of input image channels.
            output_dim (int): Output dimension (key length).
        """
        super(Decoder, self).__init__()
        # Increase number of filters and add more layers
        self.features = nn.Sequential(
            # Initial layer: 256 -> 128
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128 -> 64
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64 -> 32
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32 -> 16
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16 -> 8
            nn.Conv2d(512, 768, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(768),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8 -> 4
            nn.Conv2d(768, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Global pooling and deeper classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        """
        Forward pass of the decoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (B, output_dim).
        """
        features = self.features(x)
        return self.classifier(features) 