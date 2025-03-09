import torch
import torch.nn as nn

from models.decoder import FlexibleDecoder

class CNN(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, output_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        return self.conv5(x)

class CombinedModel(nn.Module):
    def __init__(self, input_channels=3,
                 decoder_total_conv_layers=5,
                 decoder_total_pool_layers=2,
                 decoder_initial_channels=64,
                 cnn_mode: str = "trainable",  # "fixed", "trainable", or "fresh"
                 cnn_instance: nn.Module = None):
        super().__init__()
        
        # Validate mode
        valid_modes = ["fixed", "trainable", "fresh"]
        if cnn_mode not in valid_modes:
            raise ValueError(f"Invalid cnn_mode: {cnn_mode}. Choose from {valid_modes}")
        
        # CNN configuration
        if cnn_mode in ["fixed", "trainable"]:
            if cnn_instance is None:
                raise ValueError("cnn_instance required for 'fixed' or 'trainable' modes")
            
            self.cnn = cnn_instance
            # Set parameter trainability
            for param in self.cnn.parameters():
                param.requires_grad = (cnn_mode == "trainable")
        else:  # fresh mode
            # Create new CNN with same architecture but fresh initialization
            self.cnn = CNN(input_channels=input_channels,
                          output_channels=input_channels)
            
        # Decoder remains always trainable
        self.decoder = FlexibleDecoder(
            total_conv_layers=decoder_total_conv_layers,
            total_pool_layers=decoder_total_pool_layers,
            initial_channels=decoder_initial_channels
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.decoder(x)
        return x