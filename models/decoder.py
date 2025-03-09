import torch.nn as nn
import math
import torch.nn.functional as F

class MCDropout(nn.Module):
    def __init__(self, p=0.5):
        super(MCDropout, self).__init__()
        self.p = p

    def forward(self, x):
        # Always apply dropout, regardless of model.train() or model.eval()
        return F.dropout(x, p=self.p, training=True)

class FlexibleDecoder(nn.Module):
    def __init__(
        self,
        total_conv_layers=5,
        total_pool_layers=2,
        initial_channels=64,
        convs_per_block=None,
        channel_growth='double'
    ):
        super(FlexibleDecoder, self).__init__()
        self.total_conv_layers = total_conv_layers
        self.total_pool_layers = total_pool_layers
        self.initial_channels = initial_channels
        self.channel_growth = channel_growth

        # If convs_per_block is not specified, distribute conv layers evenly
        if convs_per_block is None:
            if total_pool_layers > 0:
                convs_per_block = math.ceil(total_conv_layers / total_pool_layers)
            else:
                convs_per_block = total_conv_layers

        self.features = self._make_layers(convs_per_block)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.final_num_channels, 1),
            nn.Sigmoid(),
        )

    def _make_layers(self, convs_per_block):
        layers = []
        in_channels = 3  # RGB images
        current_channels = self.initial_channels
        conv_layers_remaining = self.total_conv_layers
        pool_layers_remaining = self.total_pool_layers

        # Keep track of the final number of channels
        self.final_num_channels = current_channels

        while conv_layers_remaining > 0:
            # Number of conv layers in this block
            convs_in_this_block = min(convs_per_block, conv_layers_remaining)

            # Add convolutional layers
            for _ in range(convs_in_this_block):
                layers.append(
                    nn.Conv2d(
                        in_channels,
                        current_channels,
                        kernel_size=3,
                        padding=1
                    )
                )
                layers.append(nn.ReLU(inplace=True))
                in_channels = current_channels
                conv_layers_remaining -= 1

            # Add pooling layer if any remaining
            if pool_layers_remaining > 0:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                pool_layers_remaining -= 1

            # Update channels for next block
            if conv_layers_remaining > 0:  # Only update channels if more conv layers are left
                if self.channel_growth == 'double':
                    current_channels *= 2
                elif self.channel_growth == 'linear':
                    current_channels += self.initial_channels
                # You can add more channel growth strategies here
                self.final_num_channels = current_channels  # Update final_num_channels

        # Adaptive Average Pooling
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))

        # Update the final number of channels after all layers
        self.final_num_channels = in_channels  # in_channels is current_channels at this point

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
