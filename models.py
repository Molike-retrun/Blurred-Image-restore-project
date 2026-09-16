import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """A compact convolutional baseline for image deblurring."""

    def __init__(self, channels: int = 3, features: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class ResidualCNN(nn.Module):
    """Residual CNN for image restoration tasks like deblurring."""

    def __init__(self, channels: int = 3, features: int = 64, blocks: int = 6):
        super().__init__()
        self.head = nn.Conv2d(channels, features, kernel_size=3, padding=1)
        self.trunk = nn.Sequential(*[ResidualBlock(features) for _ in range(blocks)])
        self.tail = nn.Sequential(
            nn.Conv2d(features, channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        feat = self.trunk(feat)
        return self.tail(feat)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Lightweight U-Net for blurred image restoration."""

    def __init__(self, channels: int = 3, base_features: int = 32):
        super().__init__()
        b = base_features
        self.enc1 = DoubleConv(channels, b)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(b, b * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(b * 2, b * 4)

        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(b * 2, b)

        self.out = nn.Sequential(
            nn.Conv2d(b, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


def build_model(name: str) -> nn.Module:
    registry = {
        "simple": SimpleCNN,
        "residual": ResidualCNN,
        "unet": UNet,
    }
    key = name.lower()
    if key not in registry:
        raise ValueError(f"Unknown model '{name}'. Choose from: {', '.join(registry)}")
    return registry[key]()
