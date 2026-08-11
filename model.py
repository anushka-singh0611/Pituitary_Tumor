import torch
import torch.nn as nn


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Net(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder
        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        # Bridge
        self.bridge = DoubleConv(128, 256)

        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv1 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv2 = DoubleConv(128, 64)

        # Output
        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):

        # Encoder
        x1 = self.down1(x)
        x2 = self.pool1(x1)

        x3 = self.down2(x2)
        x4 = self.pool2(x3)

        # Bridge
        x5 = self.bridge(x4)

        # Decoder
        x = self.up1(x5)
        x = torch.cat([x, x3], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv2(x)

        # Final prediction
        return self.final(x)


if __name__ == "__main__":
    model = Net()
    print("U-Net created successfully!")
    print(model)
if __name__ == "__main__":

    model = Net()

    model.load_state_dict(
        torch.load("models/unet_model.pth", map_location="cpu")
    )

    model.eval()

    print("Trained U-Net loaded successfully!")