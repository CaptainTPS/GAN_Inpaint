import torch.nn as nn
import torch

class UNET(nn.Module):
    def __init__(self, width, height, channel):
        super(UNET, self).__init__()
        self.w = width
        self.h = height
        self.c = channel

        self.pool1 = nn.Sequential(
            nn.Conv2d(in_channels=self.c, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

        )
        self.pool2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)),
            # nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

        )
        self.pool3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)),
            # nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )
        self.pool4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)),
            # nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )
        self.pool5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)),
            # nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
        )

        self.up1_1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.up1_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.up2_1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.up2_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.up3_1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.up3_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.up4_1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.up4_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.up5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=self.c, kernel_size=(3, 3), padding=1),
            nn.Tanh()
        )

        self.cat = nn.Sequential(
            nn.Conv2d(in_channels=2*self.c, out_channels=self.c,kernel_size=(3, 3), padding=1),
            nn.Tanh()
        )

    def forward(self, input):
        # input n * 3 * 128 * 128
        conv1 = self.pool1(input)
        # n * 32 * 128 * 128
        conv2 = self.pool2(conv1)
        # n * 64 * 64 * 64
        conv3 = self.pool3(conv2)
        # n * 128 * 32 * 32
        conv4 = self.pool4(conv3)
        # n * 256 * 16 * 16
        conv5 = self.pool5(conv4)
        # n * 512 * 8 * 8

        up1 = self.up1_1(conv5)
        # n * 256 * 16 * 16
        up1 = torch.cat((up1, conv4), dim=1)
        # n * 512 * 16 * 16
        up1 = self.up1_2(up1)
        # n * 256 * 16 * 16

        up2 = self.up2_1(up1)
        # n * 128 * 32 * 32
        up2 = torch.cat((up2, conv3), dim=1)
        # n * 256 * 32 * 32
        up2 = self.up2_2(up2)
        # n * 128 * 32 * 32

        up3 = self.up3_1(up2)
        # n * 64 * 64 * 64
        up3 = torch.cat((up3, conv2), dim=1)
        # n * 128 * 64 * 64
        up3 = self.up3_2(up3)
        # n * 64 * 64 * 64

        up4 = self.up4_1(up3)
        # n * 32 * 128 * 128
        up4 = torch.cat((up4, conv1), dim=1)
        # n * 64 * 128 * 128
        up4 = self.up4_2(up4)
        # n * 32 * 128 * 128

        output = self.up5(up4)
        output = torch.cat((output, input), dim=1)
        output = self.cat(output)
        return output

class DNET(nn.Module):
    def __init__(self, batch, nc, inputSize, nf):
        super(DNET, self).__init__()
        # input batch * nc * inputSize * inputSize
        self.batch = batch
        n = inputSize
        self.net = nn.Sequential(
            nn.Conv2d(nc, nf, 4, 2, 1),
            nn.LeakyReLU(0.2, True)
        )
        n = n / 2
        feature = nf
        # batch * nf * inputSize * inputSize
        while n > 1:
            self.net.add_module(str(len(self.net._modules)), nn.Conv2d(feature, feature * 2, 4, 2, 1))
            self.net.add_module(str(len(self.net._modules)), nn.BatchNorm2d(feature * 2))
            self.net.add_module(str(len(self.net._modules)), nn.LeakyReLU(0.2, True))
            feature = feature * 2
            n = n / 2
        # batch * feature x 1 x 1
        self.net.add_module(str(len(self.net._modules)), nn.Conv2d(feature, 1, 1, 1))
        self.net.add_module(str(len(self.net._modules)), nn.Sigmoid())
        # batch * 1 x 1 x 1
        pass

        # because view() cannot be distributed, just make net distributed
        self.net = nn.DataParallel(self.net)

    def forward(self, input):
        output = self.net(input)
        output = output.view(self.batch)
        return output
        pass
