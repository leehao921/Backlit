import torch
import torch.nn as nn

class Pixel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Pixel, self).__init__()

        # Pixel-Wise Network in High-Resolution Branch
        self.conv1e = nn.Conv2d(in_channels, 3, kernel_size=1, stride=1, padding=0)
        self.conv2e = nn.Conv2d(3, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        # Pixel-Wise Network in High-Resolution Branch
        conv1e = self.conv1e(input)
        output = self.conv2e(conv1e)

        return output
