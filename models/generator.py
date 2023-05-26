import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, noise_dim: int, features: int, img_channels: int):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(noise_dim, features * 32, 4, 2, 1),
            nn.BatchNorm2d(features * 32),
            nn.ReLU(True),
            # Upscale to 4x4
            nn.ConvTranspose2d(features * 32, features * 16, 4, 2, 1),
            nn.BatchNorm2d(features * 16),
            nn.ReLU(True),
            # Upscale to 8x8
            nn.ConvTranspose2d(features * 16, features * 8, 4, 2, 1),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),
            # Upscale to 16x16
            nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            # Upscale to 32 x 32
            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            # Upscale to 64 x 64
            nn.ConvTranspose2d(features * 2, features, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            # Upscale to 128 x 128
            nn.ConvTranspose2d(features, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, input):
        return self.layers(input)
