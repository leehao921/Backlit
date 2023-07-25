import torch
import torch.nn as nn

# U-Net implementation
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Feature Extractor in Low-Resolution Branch
        self.conv1a = self.double_conv(3, 4)
        self.conv2a = self.double_conv(4, 8)
        self.conv3a = self.double_conv(8, 16)
        self.conv4a = self.double_conv(16, 32)

        # Semantics Perception Block
        self.conv_inb = self.double_conv(32, 32)
        self.conv1b = self.double_conv(32, 16)
        self.maxpool1b = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2b = self.double_conv(16, 16)
        self.maxpool2b = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3b = self.double_conv(16, 16)
        self.maxpool3b = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4b = self.double_conv(16, 16)
        self.maxpool4b = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5b = self.double_conv(16, 16)
        self.conv6b = self.double_conv(16, 16)
        self.conv5bd = self.double_conv(32, 16)
        self.conv4bd = self.double_conv(32, 16)
        self.conv3bd = self.double_conv(32, 16)
        self.conv2bd = self.double_conv(32, 16)
        self.conv1bd = self.double_conv(32, 16)
        self.maxpool5b = nn.MaxPool2d(kernel_size=2, stride=2)

        # Lighting Acquisition Block
        self.conv_inc = self.double_conv(32, 32)
        self.conv1c = self.double_conv(32, 16)
        self.maxpool1c = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2c = self.double_conv(16, 16)
        self.maxpool2c = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3c = self.double_conv(16, 16)
        self.maxpool3c = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4c = self.double_conv(16, 16)
        self.maxpool4c = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5c = self.double_conv(16, 16)
        self.conv6c = self.double_conv(16, 16)
        self.adaptive_avg = nn.AdaptiveAvgPool2d(1)
        self.adaptive_max = nn.AdaptiveMaxPool2d(1)
        self.conv7a1 = nn.Conv2d(16, 16, kernel_size=1)
        self.conv7a2 = nn.Conv2d(16, 16, kernel_size=1)
        self.conv7m1 = nn.Conv2d(16, 16, kernel_size=1)
        self.conv7m2 = nn.Conv2d(16, 16, kernel_size=1)
        self.conv7c = self.double_conv(48, 16)
        self.conv5cd = self.double_conv(32, 16)
        self.conv4cd = self.double_conv(32, 16)
        self.conv3cd = self.double_conv(32, 16)
        self.conv2cd = self.double_conv(32, 16)
        self.conv1cd = self.double_conv(32, 16)

        # Fusion Adjustment Block
        self.conv_ind = self.double_conv(32, 32)
        self.conv1d = self.double_conv(32, 16)
        self.maxpool1d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2d = self.double_conv(16, 16)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3d = self.double_conv(16, 16)
        self.maxpool3d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4d = self.double_conv(16, 16)
        self.maxpool4d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5d = self.double_conv(16, 16)
        self.conv6d = self.double_conv(16, 16)
        self.conv5dd = self.double_conv(32, 16)
        self.conv4dd = self.double_conv(32, 16)
        self.conv3dd = self.double_conv(32, 16)
        self.conv2dd = self.double_conv(32, 16)
        self.conv1dd = self.double_conv(32, 16)
        self.conv_p = self.double_conv(32, 48)

        # Pixel-Wise Network in High-Resolution Branch
        self.conv1e = nn.Conv2d(3, 3, kernel_size=1)
        self.conv2e = nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, input):
        # Feature Extractor in Low-Resolution Branch
        conv1a = self.conv1a(input)
        conv2a = self.conv2a(conv1a)
        conv3a = self.conv3a(conv2a)
        conv4a = self.conv4a(conv3a)

        # Semantics Perception Block
        conv_inb = self.conv_inb(conv4a)
        conv1b = self.conv1b(conv_inb)
        maxpool1b = self.maxpool1b(conv1b)
        conv2b = self.conv2b(maxpool1b)
        maxpool2b = self.maxpool2b(conv2b)
        conv3b = self.conv3b(maxpool2b)
        maxpool3b = self.maxpool3b(conv3b)
        conv4b = self.conv4b(maxpool3b)
        maxpool4b = self.maxpool4b(conv4b)
        conv5b = self.conv5b(maxpool4b)
        conv6b = self.conv6b(conv5b)
        conv5bd = self.conv5bd(torch.cat((conv5b, conv6b), dim=1))
        conv4bd = self.conv4bd(torch.cat((self.upsample(conv5bd), conv4b), dim=1))
        conv3bd = self.conv3bd(torch.cat((self.upsample(conv4bd), conv3b), dim=1))
        conv2bd = self.conv2bd(torch.cat((self.upsample(conv3bd), conv2b), dim=1))
        conv1bd = self.conv1bd(torch.cat((self.upsample(conv2bd), conv1b), dim=1))
        maxpool5b = self.maxpool5b(torch.cat((conv1bd, conv_inb), dim=1))

        # Lighting Acquisition Block
        conv_inc = self.conv_inc(maxpool5b)
        conv1c = self.conv1c(conv_inc)
        maxpool1c = self.maxpool1c(conv1c)
        conv2c = self.conv2c(maxpool1c)
        maxpool2c = self.maxpool2c(conv2c)
        conv3c = self.conv3c(maxpool2c)
        maxpool3c = self.maxpool3c(conv3c)
        conv4c = self.conv4c(maxpool3c)
        maxpool4c = self.maxpool4c(conv4c)
        conv5c = self.conv5c(maxpool4c)
        conv6c = self.conv6c(conv5c)
        adaptive_avg = nn.functional.adaptive_avg_pool2d(conv5c, (1, 1))
        adaptive_max = nn.functional.adaptive_max_pool2d(conv5c, (1, 1))
        conv7a1 = self.conv7a1(adaptive_avg)
        conv7a2 = self.conv7a2(conv7a1)
        conv7m1 = self.conv7m1(adaptive_max)
        conv7m2 = self.conv7m2(conv7m1)
        conv7c = self.conv7c(torch.cat((conv6c, self.upsample(conv7a2), self.upsample(conv7m1)), dim=1))
        conv5cd = self.conv5cd(torch.cat((conv7c, conv5c), dim=1))
        conv4cd = self.conv4cd(torch.cat((self.upsample(conv5cd), conv4c), dim=1))
        conv3cd = self.conv3cd(torch.cat((self.upsample(conv4cd), conv3c), dim=1))
        conv2cd = self.conv2cd(torch.cat((self.upsample(conv3cd), conv2c), dim=1))
        conv1cd = self.conv1cd(torch.cat((self.upsample(conv2cd), conv1c), dim=1))

        # Fusion Adjustment Block
        conv_ind = self.conv_ind(self.upsample(conv1c) + conv_inc)
        conv1d = self.conv1d(conv_ind)
        maxpool1d = self.maxpool1d(conv1d)
        conv2d = self.conv2d(maxpool1d)
        maxpool2d = self.maxpool2d(conv2d)
        conv3d = self.conv3d(maxpool2d)
        maxpool3d = self.maxpool3d(conv3d)
        conv4d = self.conv4d(maxpool3d)
        maxpool4d = self.maxpool4d(conv4d)
        conv5d = self.conv5d(maxpool4d)
        conv6d = self.conv6d(conv5d)
        conv5dd = self.conv5dd(torch.cat((conv5d, conv6d), dim=1))
        conv4dd = self.conv4dd(torch.cat((self.upsample(conv5dd), conv4d), dim=1))
        conv3dd = self.conv3dd(torch.cat((self.upsample(conv4dd), conv3d), dim=1))
        conv2dd = self.conv2dd(torch.cat((self.upsample(conv3dd), conv2d), dim=1))
        conv1dd = self.conv1dd(torch.cat((self.upsample(conv2dd), conv1d), dim=1))
        conv_p = self.conv_p(conv1dd + conv_ind)

        # Pixel-Wise Network in High-Resolution Branch
        conv1e = self.conv1e(input)
        conv2e = self.conv2e(conv1e)

        # Final output
        output = conv2e + self.upsample(conv_p)
        return output



    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def single_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

    def upsample(self, x):
        return nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)


