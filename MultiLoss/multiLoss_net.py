import torch.nn as nn

import random_net


class ScaledLoss(nn.Module):

    def __init__(self, target):
        super(ScaledLoss, self).__init__()
        self.target = target
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input, self.target)
        return input

    def backward(self, retain_graph = True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

def getTarget(Image, scale):

    pass

# def getGnet(Image, oldG):
#     # image is batchSize * 3 * 128 * 128
#     netG = oldG
#     newG = nn.Sequential()
#     lossLayers = []
#     cnt = 0
#     for layer in netG.encodeNet:
#         newG.add_module(str(cnt), layer)
#         cnt += 1
#     for layer in netG.midNet:
#         newG.add_module(str(cnt), layer)
#         cnt += 1
#
#     i = 0
#     scale = 8
#     for layer in netG.decodeNet:
#         if isinstance(layer, nn.ReLU):
#             i += 1
#         if i > 2:
#             if isinstance(layer, nn.ReLU):
#                 newG.add_module(str(cnt), nn.Tanh())
#                 img = getTarget(Image, scale)
#                 scale /= 2
#                 lossL = ScaledLoss(img)
#                 newG.add_module(str(cnt), lossL)
#                 lossLayers.append(lossL)
#                 cnt += 1
#             else:
#                 newG.add_module(str(cnt), layer)
#                 cnt += 1
#         else:
#             newG.add_module(str(cnt), layer)
#             cnt += 1
#
#     return newG, lossLayers

# new Generator net

class multiLossG(nn.Module):
    def __init__(self, nc, nef, nBottleneck, ngf):
        super(multiLossG, self).__init__()

        ## Encode Input Context to noise (architecture similar to Discriminator)
        self.encoder = nn.Sequential(
            ## input is (nc) x 128 x 128
            nn.Conv2d(nc, nef, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            ## state size: (nef) x 64 x 64
            nn.Conv2d(nef, nef, 4, 2, 1),
            nn.BatchNorm2d(nef),
            nn.LeakyReLU(0.2, True),
            ## state size: (nef) x 32 x 32
            nn.Conv2d(nef, nef * 2, 4, 2, 1),
            nn.BatchNorm2d(nef * 2),
            nn.LeakyReLU(0.2, True),
            ## state size: (nef*2) x 16 x 16
            nn.Conv2d(nef * 2, nef * 4, 4, 2, 1),
            nn.BatchNorm2d(nef * 4),
            nn.LeakyReLU(0.2, True),
            ## state size: (nef*4) x 8 x 8
            nn.Conv2d(nef * 4, nef * 8, 4, 2, 1),
            nn.BatchNorm2d(nef * 8),
            nn.LeakyReLU(0.2, True),
            ## state size: (nef*8) x 4 x 4
            nn.Conv2d(nef * 8, nBottleneck, 4),
            nn.BatchNorm2d(nBottleneck),
            nn.LeakyReLU(0.2, True),
            ## state size: (nBottleneck) x 1 x 1
        )

        self.decoder = nn.Sequential(
            #input is Z: (nz_size) x 1 x 1, going into a convolution
            nn.ConvTranspose2d(nBottleneck, ngf * 8, 4),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size: (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            ## state size: (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
            nn.Tanh()
            ## state size: (nc) x 128 x 128
        )

        #make it distributed
        self.encoder = nn.DataParallel(self.encoder)
        self.decoder = nn.DataParallel(self.decoder)

    def forward(self, input, noise=None):
        y = self.encoder(input)
        output = self.decoder(y)

        return output


def main():
    pass

if __name__ == "__main__":
    main()