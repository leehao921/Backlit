import torch
import torch.nn as nn

class Fusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Fusion, self).__init__()

        # Feature Extractor in Low-Resolution Branch
        self.conv1a = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.conv2a = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.conv3a = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.conv4a = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # Semantics Perception Block
        self.conv_inb = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.conv1b = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.maxpool1b = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2b = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.maxpool2b = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3b = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.maxpool3b = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4b = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.maxpool4b = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5b = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.conv6b = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, dilation=2),
            nn.PReLU()
        )
        self.conv5bd = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.conv4bd = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.conv3bd = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.conv2bd = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.conv1bd = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.maxpool5b = nn.MaxPool2d(kernel_size=2, stride=2)

        # Lighting Acquisition Block
        self.conv_inc = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.conv1c = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.maxpool1c = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2c = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.maxpool2c = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3c = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.maxpool3c = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4c = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.maxpool4c = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5c = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.conv6c = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, dilation=2),
            nn.PReLU()
        )
        self.adaptive_avg = nn.AdaptiveAvgPool2d(1)
        self.adaptive_max = nn.AdaptiveMaxPool2d(1)
        self.conv7a1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU()
        )
        self.conv7a2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU()
        )
        self.conv7m1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU()
        )
        self.conv7m2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU()
        )
        self.conv7c = nn.Sequential(
            nn.Conv2d(48, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        )
        self.conv5cd = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.conv4cd = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.conv3cd = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.conv2cd = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.conv1cd = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )

        # Fusion Adjustment Block
        self.conv_ind = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.conv1d = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.maxpool1d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2d = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3d = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.maxpool3d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4d = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.maxpool4d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5d = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.conv6d = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, dilation=2),
            nn.PReLU()
        )
        self.conv5dd = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.conv4dd = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.conv3dd = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.conv2dd = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.conv1dd = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.conv_p = nn.Sequential(
            nn.Conv2d(48, 32, kernel_size=1, stride=1, padding=0),
            nn.PReLU()
        )

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
        conv5bd = self.conv5bd(torch.cat([conv5b, conv6b], dim=1))
        conv4bd = self.conv4bd(torch.cat([self.upsample(conv5bd), conv4b], dim=1))
        conv3bd = self.conv3bd(torch.cat([self.upsample(conv4bd), conv3b], dim=1))
        conv2bd = self.conv2bd(torch.cat([self.upsample(conv3bd), conv2b], dim=1))
        conv1bd = self.conv1bd(torch.cat([self.upsample(conv2bd), conv1b], dim=1))
        maxpool5b = self.maxpool5b(torch.cat([conv1bd, conv_inb], dim=1))

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
        adaptive_avg = nn.functional.adaptive_avg_pool2d(conv5c, 1)
        adaptive_max = nn.functional.adaptive_max_pool2d(conv5c, 1)
        conv7a1 = self.conv7a1(adaptive_avg)
        conv7a2 = self.conv7a2(conv7a1)
        conv7m1 = self.conv7m1(adaptive_max)
        conv7m2 = self.conv7m2(conv7m1)
        conv7c = self.conv7c(torch.cat([conv6c, self.upsample(conv7a2), self.upsample(conv7m1)], dim=1))
        conv5cd = self.conv5cd(torch.cat([conv7c, conv5c], dim=1))
        conv4cd = self.conv4cd(torch.cat([self.upsample(conv5cd), conv4c], dim=1))
        conv3cd = self.conv3cd(torch.cat([self.upsample(conv4cd), conv3c], dim=1))
        conv2cd = self.conv2cd(torch.cat([self.upsample(conv3cd), conv2c], dim=1))
        conv1cd = self.conv1cd(torch.cat([self.upsample(conv2cd), conv1c], dim=1))

        # Fusion Adjustment Block
        conv_ind = self.conv_ind(torch.cat([self.upsample(conv1c + conv_inc)], dim=1))
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
        conv5dd = self.conv5dd(torch.cat([conv5d, conv6d], dim=1))
        conv4dd = self.conv4dd(torch.cat([self.upsample(conv5dd), conv4d], dim=1))
        conv3dd = self.conv3dd(torch.cat([self.upsample(conv4dd), conv3d], dim=1))
        conv2dd = self.conv2dd(torch.cat([self.upsample(conv3dd), conv2d], dim=1))
        conv1dd = self.conv1dd(torch.cat([self.upsample(conv2dd), conv1d], dim=1))
        output = self.conv_p(torch.cat([conv1dd, conv_ind], dim=1))

        return output

    def upsample(self, x):
        return nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
