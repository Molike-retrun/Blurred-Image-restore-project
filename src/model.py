import torch.nn as nn
import torch


#CNN网络用于图像去模糊
class SimpleRestorationCNN(nn.Module):
    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels = 3,
            out_channels = 32,
            kernel_size = 3,
            padding = 1
        )

        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels = 32,
            out_channels = 64,
            kernel_size = 3,
            padding = 1
        )

        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(
            in_channels = 64,
            out_channels = 32,
            kernel_size = 3,
            padding = 1
        )

        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(
            in_channels = 32,
            out_channels = 3,
            kernel_size = 3,
            padding = 1   
        )

    def forward(self,x):

        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.conv4(x)

        return x

#残差学习的CNN网络
class ResidualCNN(nn.Module):
    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels = 3,
            out_channels = 32,
            kernel_size = 3,
            padding = 1
        )

        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels = 32,
            out_channels = 64,
            kernel_size = 3,
            padding = 1
        )

        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(
            in_channels = 64,
            out_channels = 32,
            kernel_size = 3,
            padding = 1
        )

        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(
            in_channels = 32,
            out_channels = 3,
            kernel_size = 3,
            padding = 1   
        )

    def forward(self,x):
        identity = x

        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.conv4(x)

        return x+identity

class Double_Conv(nn.Module):
    def __init__ (self,in_channels,out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size = 3,
                padding = 1

            ),    

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size = 3,
                padding = 1
            ),

            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.double_conv(x)


class UNet(nn.Module):

    def __init__(self,inchannels = 3,outchannels = 3):
        super().__init__()

        #编码器
        self.enc1 = Double_Conv(
            inchannels,
            64
        )
        self.pool1 = nn.MaxPool2d(
            kernel_size = 2,
            stride = 2
        )

        self.enc2 = Double_Conv(
            64,
            128
        )
        self.pool2 = nn.MaxPool2d(
            kernel_size = 2,
            stride = 2
        )

        #bottleneck
        self.bottleneck = Double_Conv(
            128,
            256
        )

        #解码器
        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size = 2,
            stride = 2
        )
        self.dec2 = Double_Conv(
            256,
            128
        )

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size = 2,
            stride = 2
        )
        self.dec1 = Double_Conv(
            128,
            64
        )

        #输出层
        self.out_channels = nn.Conv2d(
            64,
            outchannels,
            kernel_size = 1
        )
    def forward(self,x):
        #编码器
        enc1 = self.enc1(x)
        x = self.pool1(enc1)

        enc2 = self.enc2(x)
        x = self.pool2(enc2)

        #bottleneck
        x = self.bottleneck(x)

        #解码器
        x = self.up2(x)
        x = torch.cat([x,enc2],dim = 1) #拼接深层特征
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x,enc1],dim = 1) #拼接浅层特征
        x = self.dec1(x)

        x = self.out_channels(x)

        return x



if __name__ == "__main__":
    import torch

    model = UNet()

    x = torch.randn(
        4,
        3,
        256,
        256
    )

    output = model(x)

    print("input shape:", x.shape)
    print("output shape:", output.shape)