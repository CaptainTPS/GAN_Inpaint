import numpy as np

import torch.nn as nn
import torch

class MASKNET(nn.Module):
    def __init__(self, width, height, channel, downTimes, nfeature):
        super(MASKNET, self).__init__()
        self.w = width
        self.h = height
        self.c = channel

        downTimes = min(downTimes, int(np.log2(min(width, height))))

        self.toFeature = nn.Sequential(
            nn.Conv2d(in_channels=self.c * 2, out_channels=nfeature, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(nfeature),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=nfeature, out_channels=nfeature, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(nfeature),
            nn.LeakyReLU(),
        )

        self.pooling = nn.ModuleList()
        for i in range(downTimes):
            temp = nn.Sequential(
                nn.MaxPool2d(kernel_size=(2, 2)),
                # nn.AvgPool2d(kernel_size=(2, 2)),
                nn.Conv2d(in_channels=nfeature, out_channels=nfeature * 2, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(nfeature * 2),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=nfeature * 2, out_channels=nfeature * 2, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(nfeature * 2),
                nn.LeakyReLU(),
            )
            self.pooling.append(temp)
            nfeature = nfeature * 2

        self.upsampling = nn.ModuleList()
        for i in range(downTimes):
            temp = nn.ModuleList()
            temp.append(nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(in_channels=nfeature, out_channels=nfeature / 2, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(nfeature / 2),
                nn.ReLU()
            ))
            temp.append(nn.Sequential(
                nn.Conv2d(in_channels=nfeature, out_channels=nfeature / 2, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(nfeature / 2),
                nn.ReLU(),
                nn.Conv2d(in_channels=nfeature / 2, out_channels=nfeature / 2, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(nfeature / 2),
                nn.ReLU()
            ))

            self.upsampling.append(temp)
            nfeature = nfeature / 2

        self.toImage = nn.ModuleList()
        self.toImage.append(nn.Sequential(
            nn.Conv2d(in_channels=nfeature, out_channels=self.c, kernel_size=(3, 3), padding=1),
            nn.Tanh()
        ))

        self.toImage.append(nn.Sequential(
            nn.Conv2d(in_channels=2*self.c, out_channels=self.c,kernel_size=(3, 3), padding=1),
            nn.Tanh()
        ))

    def forward(self, input, mask):
        input_ori = input
        input = torch.cat((input, mask), dim=1)

        features = self.toFeature(input)
        middles = []
        for layers in self.pooling:
            middles.append(features)
            features = layers(features)

        for layers in self.upsampling:
            features = layers[0](features)
            features = torch.cat((middles[-1], features), dim=1)
            middles.pop(-1)
            features = layers[1](features)

        features = self.toImage[0](features)
        features = torch.cat((input_ori, features), dim=1)
        result = self.toImage[1](features)

        return result

class MASKDNET(nn.Module):
    def __init__(self, batch, nc, inputSize, nf):
        super(MASKDNET, self).__init__()
        # input batch * nc * inputSize * inputSize
        self.batch = batch
        n = inputSize
        self.net = nn.Sequential(
            nn.Conv2d(nc * 2, nf, 4, 2, 1),
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

    def forward(self, input, mask):
        input = torch.cat((input, mask), dim=1)
        output = self.net(input)
        output = output.view(self.batch)
        return output
        pass
