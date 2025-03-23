"""
Decoder model for StyleGAN watermarking.
"""
import torch
import torch.nn as nn


class Decoder(nn.Module):
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


class FeatureDecoder(nn.Module):
    """
    Enhanced decoder network that predicts key logits directly from feature vectors.
    Designed with high capacity and complex non-linear function approximation capabilities
    to potentially learn highly non-linear and chaotic key mappings.
    """
    def __init__(self, input_dim, output_dim=4, hidden_dims=None, activation='leaky_relu', 
                 dropout_rate=0.3, num_residual_blocks=3, use_spectral_norm=True,
                 use_layer_norm=True, use_attention=True):
        """
        Initialize the enhanced Feature Decoder.
        
        Args:
            input_dim (int): Input dimension (number of features).
            output_dim (int): Output dimension (key length).
            hidden_dims (list, optional): List of hidden layer dimensions. Defaults to [1024, 2048, 1024, 512, 256].
            activation (str): Activation function to use. Options: 'leaky_relu', 'relu', 'gelu', 'swish', 'mish'.
            dropout_rate (float): Dropout rate for regularization.
            num_residual_blocks (int): Number of residual blocks to use.
            use_spectral_norm (bool): Whether to use spectral normalization for better stability.
            use_layer_norm (bool): Whether to use layer normalization.
            use_attention (bool): Whether to use self-attention mechanism.
        """
        super(FeatureDecoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [1024, 2048, 1024, 512, 256]
        
        # Select activation function
        if activation == 'relu':
            self.act_fn = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.act_fn = nn.GELU()
        elif activation == 'swish':
            self.act_fn = lambda x: x * torch.sigmoid(x)  # Swish/SiLU activation
        elif activation == 'mish':
            self.act_fn = lambda x: x * torch.tanh(nn.functional.softplus(x))  # Mish activation
        else:  # default to leaky_relu
            self.act_fn = nn.LeakyReLU(0.2, inplace=True)
        
        # Initial projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]) if use_layer_norm else nn.Identity(),
            self.act_fn,
            nn.Dropout(dropout_rate)
        )
        
        # Residual blocks for complex function approximation
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(
                hidden_dims[0] if i == 0 else hidden_dims[min(i, len(hidden_dims)-1)],
                hidden_dims[min(i+1, len(hidden_dims)-1)],
                self.act_fn,
                dropout_rate,
                use_spectral_norm,
                use_layer_norm
            ) for i in range(num_residual_blocks)
        ])
        
        # Self-attention mechanism to capture long-range dependencies
        self.attention = SelfAttention(hidden_dims[min(num_residual_blocks, len(hidden_dims)-1)]) if use_attention else nn.Identity()
        
        # Deep MLP layers for additional capacity
        mlp_layers = []
        for i in range(min(num_residual_blocks, len(hidden_dims)-1), len(hidden_dims)-1):
            if use_spectral_norm:
                mlp_layers.append(nn.utils.spectral_norm(nn.Linear(hidden_dims[i], hidden_dims[i+1])))
            else:
                mlp_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                
            if use_layer_norm:
                mlp_layers.append(nn.LayerNorm(hidden_dims[i+1]))
                
            mlp_layers.append(self.act_fn)
            mlp_layers.append(nn.Dropout(dropout_rate))
        
        self.mlp_layers = nn.Sequential(*mlp_layers)
        
        # Final output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
    def forward(self, x):
        """
        Forward pass of the enhanced feature decoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, input_dim).
            
        Returns:
            torch.Tensor: Output tensor of shape (B, output_dim).
        """
        # Initial projection
        x = self.input_projection(x)
        
        # Residual blocks
        for res_block in self.residual_blocks:
            x = res_block(x)
        
        # Self-attention
        x = self.attention(x)
        
        # MLP layers
        x = self.mlp_layers(x)
        
        # Output layer
        return self.output_layer(x)


class ResidualBlock(nn.Module):
    """
    Residual block with optional normalization and spectral normalization.
    """
    def __init__(self, in_dim, out_dim, act_fn, dropout_rate=0.3, 
                 use_spectral_norm=True, use_layer_norm=True):
        super(ResidualBlock, self).__init__()
        
        # First layer
        self.linear1 = nn.utils.spectral_norm(nn.Linear(in_dim, out_dim)) if use_spectral_norm else nn.Linear(in_dim, out_dim)
        self.norm1 = nn.LayerNorm(out_dim) if use_layer_norm else nn.Identity()
        self.act_fn = act_fn
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second layer
        self.linear2 = nn.utils.spectral_norm(nn.Linear(out_dim, out_dim)) if use_spectral_norm else nn.Linear(out_dim, out_dim)
        self.norm2 = nn.LayerNorm(out_dim) if use_layer_norm else nn.Identity()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Shortcut connection if dimensions don't match
        self.shortcut = nn.Identity()
        if in_dim != out_dim:
            self.shortcut = nn.utils.spectral_norm(nn.Linear(in_dim, out_dim)) if use_spectral_norm else nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        shortcut = self.shortcut(x)
        
        # First layer
        x = self.linear1(x)
        x = self.norm1(x)
        x = self.act_fn(x)
        x = self.dropout1(x)
        
        # Second layer
        x = self.linear2(x)
        x = self.norm2(x)
        x = x + shortcut  # Residual connection
        x = self.act_fn(x)
        x = self.dropout2(x)
        
        return x


class SelfAttention(nn.Module):
    """
    Self-attention mechanism for capturing long-range dependencies.
    """
    def __init__(self, dim, attention_dropout=0.1):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.output_linear = nn.Linear(dim, dim)
        
    def forward(self, x):
        # Input x shape: [batch_size, dim]
        # Reshape to [batch_size, 1, dim] to simulate sequence length of 1
        x_reshaped = x.unsqueeze(1)
        
        # Compute query, key, value
        query = self.query(x_reshaped)  # [batch_size, 1, dim]
        key = self.key(x_reshaped)      # [batch_size, 1, dim]
        value = self.value(x_reshaped)  # [batch_size, 1, dim]
        
        # Compute attention scores
        attn = (query @ key.transpose(-2, -1)) * self.scale  # [batch_size, 1, 1]
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Compute output
        out = attn @ value  # [batch_size, 1, dim]
        out = out.squeeze(1)  # [batch_size, dim]
        out = self.output_linear(out)
        
        # Add residual connection
        return out + x 