import argparse

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

# parse the argument
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True)
    parser.add_argument('--batchSize', type=int, default=64)
    parser.add_argument('--loadSize', type=int, default=350)
    parser.add_argument('--fineSize', type=int, default=128)
    parser.add_argument('--nBottleneck', type=int, default=100)
    parser.add_argument('--nef', type=int, default=64)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--wtl2', type=float, default=0)
    parser.add_argument('--overlapPred', type=int, default=0)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--niter', type=int, default=25)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=int, default=0.5)
    parser.add_argument('--ntrain', type=int, default=1000000)
    parser.add_argument('--display', type=int, default=1)
    parser.add_argument('--display_id', type=int, default=10)
    parser.add_argument('--display_iter', type=int, default=50)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--name', default='train1')
    parser.add_argument('--manualSeed', type=int, default=0)
    parser.add_argument('--conditionAdv', type=int, default=0)
    parser.add_argument('--noiseGen', type=int, default=0)
    parser.add_argument('--noisetype', type=int, default=0)
    parser.add_argument('--nz', type=int, default=100)

    opt = parser.parse_args()
    return opt

# Generator net
class _netG(nn.Module):
    def __init__(self, ngpu, noiseGen, fineSize, nc, nef, nBottleneck, nz, ngf):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.noiseGen = noiseGen
        self.fineSize = fineSize

        ## Encode Input Context to noise (architecture similar to Discriminator)
        self.encodeNet = nn.Sequential(
        ## input is (nc) x 128 x 128
        nn.Conv2d(nc, nef, 4, 2, 1),
        nn.LeakyReLU(0.2, True))
        if fineSize == 128:
        ## state size: (nef) x 64 x 64
            self.encodeNet.add_module(str(len(self.encodeNet._modules)), nn.Conv2d(nef, nef, 4, 2, 1))
            self.encodeNet.add_module(str(len(self.encodeNet._modules)), nn.BatchNorm2d(nef))
            self.encodeNet.add_module(str(len(self.encodeNet._modules)), nn.LeakyReLU(0.2, True))
        ## state size: (nef) x 32 x 32
        self.encodeNet.add_module(str(len(self.encodeNet._modules)), nn.Conv2d(nef, nef * 2, 4, 2, 1))
        self.encodeNet.add_module(str(len(self.encodeNet._modules)), nn.BatchNorm2d(nef * 2))
        self.encodeNet.add_module(str(len(self.encodeNet._modules)), nn.LeakyReLU(0.2, True))
        ## state size: (nef*2) x 16 x 16
        self.encodeNet.add_module(str(len(self.encodeNet._modules)), nn.Conv2d(nef * 2, nef * 4, 4, 2, 1))
        self.encodeNet.add_module(str(len(self.encodeNet._modules)), nn.BatchNorm2d(nef * 4))
        self.encodeNet.add_module(str(len(self.encodeNet._modules)), nn.LeakyReLU(0.2, True))
        ## state size: (nef*4) x 8 x 8
        self.encodeNet.add_module(str(len(self.encodeNet._modules)), nn.Conv2d(nef * 4, nef * 8, 4, 2, 1))
        self.encodeNet.add_module(str(len(self.encodeNet._modules)), nn.BatchNorm2d(nef * 8))
        self.encodeNet.add_module(str(len(self.encodeNet._modules)), nn.LeakyReLU(0.2, True))
        ## state size: (nef*8) x 4 x 4
        self.encodeNet.add_module(str(len(self.encodeNet._modules)), nn.Conv2d(nef * 8, nBottleneck, 4))
        ## state size: (nBottleneck) x 1 x 1

        if noiseGen:
            self.netG_noise = nn.Sequential(
                # input is Z: (nz) x 1 x 1, going into a convolution
                nn.Conv2d(nz, nz, 1, 1, 0)
                # state size: (nz) x 1 x 1)
            )
            self.midNet = nn.Sequential(
                nn.BatchNorm2d(nBottleneck + nz),
                nn.LeakyReLU(0.2, True)
            )
            # statesize: (nBottleneck + nz) x 1 x 1
            self.nz_size = nBottleneck + nz
        else:
            self.midNet = nn.Sequential(
                nn.BatchNorm2d(nBottleneck),
                nn.LeakyReLU(0.2, True)
            )
            self.nz_size = nBottleneck

        self.decodeNet = nn.Sequential(
            #input is Z: (nz_size) x 1 x 1, going into a convolution
            nn.ConvTranspose2d(self.nz_size, ngf * 8, 4),
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
        )
        if fineSize == 128:
            self.decodeNet.add_module(str(len(self.decodeNet._modules)), nn.ConvTranspose2d(ngf, ngf, 4, 2, 1))
            self.decodeNet.add_module(str(len(self.decodeNet._modules)), nn.BatchNorm2d(ngf))
            self.decodeNet.add_module(str(len(self.decodeNet._modules)), nn.ReLU(True))
            ## state size: (ngf) x 64 x 64
        self.decodeNet.add_module(str(len(self.decodeNet._modules)), nn.ConvTranspose2d(ngf, nc, 4, 2, 1))
        self.decodeNet.add_module(str(len(self.decodeNet._modules)), nn.Tanh())
        ## state size: (nc) x 128 x 128


    def forward(self, input, noise=None):
        if self.noiseGen:
            y = [self.encodeNet(input), self.netG_noise(noise)]
            y = torch.cat(y, 0) # original is nn.JoinTable(2)
            output = self.midNet(y)
        else:
            y = self.encodeNet(input)
            output = self.midNet(y)

        output = self.decodeNet(output)
        return output

# Adversarial discriminator net
class _netD(nn.Module):
    def __init__(self, conditionAdv, fineSize, nc, ndf, batchSize):
        super(_netD, self).__init__()
        self.conditionAdv = conditionAdv
        self.fineSize = fineSize
        self.batchSize = batchSize
        if self.conditionAdv:
            ## !!!
            print('conditional adv not implemented')
            exit()
            self.netD_ctx = nn.Sequential(
            # input Context: (nc) x 128 x 128, going into a convolution
            nn.Conv2d(nc, ndf, 5, 2, 2)
            # state size: (ndf) x 64 x 64
            )


            self.netD_pred = nn.Sequential(
            # input pred: (nc) x 64 x 64, going into a convolution
            nn.Conv2d(nc, ndf, 5, 2, 2+32)
            # 32: to keep scaling of features same as context
            # state size: (ndf) x 64 x 64
            )

            self.midNet = nn.Sequential (nn.LeakyReLU(0.2, True),
            # state size: (ndf * 2) x 64 x 64
            nn.Conv2d(ndf*2, ndf, 4, 2, 1),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, True)
            # state size: (ndf) x 32 x 32
            )
        else:
            ## input is (nc) x 128 x 128, going into a convolution
            self.midNet = nn.Sequential(
                nn.Conv2d(nc, ndf, 4, 2, 1),
                nn.LeakyReLU(0.2, True)
            ## state size: (ndf) x 64 x 64'''
            )
        if fineSize == 128:
            self.midNet.add_module(str(len(self.midNet._modules)),nn.Conv2d(ndf, ndf, 4, 2, 1))
            self.midNet.add_module(str(len(self.midNet._modules)), nn.BatchNorm2d(ndf))
            self.midNet.add_module(str(len(self.midNet._modules)), nn.LeakyReLU(0.2, True))
            ## state size: (ndf) x 32 x 32'''

        self.tailNet = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            # state size: (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),
            # state size: (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True),
            # state size: (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 4),
            nn.Sigmoid(),
            # state size: 1 x 1 x 1
            # nn.View(1).setNumInputDims(3) #may be wrong !
            # state size: 1
        )

    def forward(self, input):
        if self.conditionAdv:
            y = [self.netD_ctx(input[0]), self.netD_pred(input[1])]
            y = torch.cat(y, 0)  # nn.JoinTable(2)
            y = self.midNet(y)
        else:
            y = self.midNet(input)

        output = self.tailNet(y)
        output = output.view(self.batchSize)
        return output

# here is the running part
if __name__ == "__main__":
    print("this is just random net struct")
