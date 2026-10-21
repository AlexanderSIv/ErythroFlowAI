import torch
import torch.nn as nn
from torch.nn.functional import relu, softmax, leaky_relu


class UNet(nn.Module):
    """U-Net для сегментации кадров 1920x160 (вход 1 канал, выход 1 канал)."""
    def __init__(self):
        super().__init__()
        # Encoder
        self.e11 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.outconv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = relu(self.e41(xp3))
        xe42 = relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = relu(self.e51(xp4))
        xe52 = relu(self.e52(xe51))

        xu1 = self.upconv1(xe52)
        xd11 = relu(self.d11(torch.cat([xu1, xe42], dim=1)))
        xd12 = relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xd21 = relu(self.d21(torch.cat([xu2, xe32], dim=1)))
        xd22 = relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xd31 = relu(self.d31(torch.cat([xu3, xe22], dim=1)))
        xd32 = relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xd41 = relu(self.d41(torch.cat([xu4, xe12], dim=1)))
        xd42 = relu(self.d42(xd41))

        return self.outconv(xd42)


class LSTMCNNLight(nn.Module):
    """LSTM+CNN для классификации последовательностей 31x31 патчей."""
    def __init__(self, num_classes=3):
        super().__init__()
        # input 31x31
        self.conv1 = nn.Conv2d(1, 128, kernel_size=5, padding=0)   # 27x27
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)         # 13x13
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=0) # 11x11
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)         # 5x5
        self.linear1 = nn.Linear(5 * 5 * 256, 512)
        self.lstm = nn.LSTM(512, 256, num_layers=1)
        self.linear2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (B, 1, T, H, W)
        hidden = None
        for t in range(x.size(2)):
            x1 = relu(self.conv1(x[:, :, t]))
            x2 = self.pool1(x1)
            x3 = self.pool2(relu(self.conv2(x2)))
            x3 = x3.view(-1, 5 * 5 * 256)
            x4 = leaky_relu(self.linear1(x3))
            xout, hidden = self.lstm(x4.unsqueeze(0), hidden)
        return softmax(self.linear2(hidden[0][-1]), dim=1)