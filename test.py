import argparse
import random
import time
import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from random_net import _netG
from random_net import parse

def initData(input):
    real_ctx = input
    real_center = real_ctx[:, :,
                  opt.fineSize / 4: opt.fineSize / 2 + opt.fineSize / 4,
                  opt.fineSize / 4: opt.fineSize / 2 + opt.fineSize / 4].clone()  # copy by value
    real_ctx[:,
    0,
    opt.fineSize / 4 + opt.overlapPred: opt.fineSize / 2 + opt.fineSize / 4 - opt.overlapPred,
    opt.fineSize / 4 + opt.overlapPred: opt.fineSize / 2 + opt.fineSize / 4 - opt.overlapPred] = 2 * 117.0 / 255.0 - 1.0
    real_ctx[:,
    1,
    opt.fineSize / 4 + opt.overlapPred: opt.fineSize / 2 + opt.fineSize / 4 - opt.overlapPred,
    opt.fineSize / 4 + opt.overlapPred: opt.fineSize / 2 + opt.fineSize / 4 - opt.overlapPred] = 2 * 104.0 / 255.0 - 1.0
    real_ctx[:,
    2,
    opt.fineSize / 4 + opt.overlapPred: opt.fineSize / 2 + opt.fineSize / 4 - opt.overlapPred,
    opt.fineSize / 4 + opt.overlapPred: opt.fineSize / 2 + opt.fineSize / 4 - opt.overlapPred] = 2 * 123.0 / 255.0 - 1.0
    return real_center

def inpaint(real, pred):
    real[:, :,
        opt.fineSize / 4 + opt.overlapPred: opt.fineSize / 2 + opt.fineSize / 4 - opt.overlapPred,
        opt.fineSize / 4 + opt.overlapPred: opt.fineSize / 2 + opt.fineSize / 4 - opt.overlapPred
        ].copy_(
        pred[:, :,
                    opt.overlapPred: opt.fineSize / 2 - opt.overlapPred,
                    opt.overlapPred: opt.fineSize / 2 - opt.overlapPred])

class localModel(nn.Module):
    def __init__(self, ngpu, noiseGen, fineSize, nc, nef, nBottleneck, nz, ngf):
        super(localModel, self).__init__()
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

        '''
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
        '''

    def forward(self, input, noise=None):
        if self.noiseGen:
            y = [self.encodeNet(input), self.netG_noise(noise)]
            y = torch.cat(y, 0) # original is nn.JoinTable(2)
            output = self.midNet(y)
        else:
            y = self.encodeNet(input)
            output = self.midNet(y)
        '''
        output = self.decodeNet(output)
        '''
        return output

def HalfNettoFeature(predict, dataloader, noise, opt):

    model = localModel(ngpu, opt.noiseGen, opt.fineSize, opt.nc, opt.nef, opt.nBottleneck, opt.nz, opt.ngf)
    model_dict = model.state_dict()

    # new_dict = {k: v for k, v in predict.items() if k in model_dict}
    new_dict = dict([])
    for k, v in predict.items():
        if k in model_dict:
            new_dict[k] = v
            # print("ok")

    model_dict.update(new_dict)
    model.load_state_dict(model_dict)

    ## port to GPU
    if opt.gpu > 0:
        model.cuda()
        if opt.noiseGen:
            noise = noise.cuda()

    id = 1
    filename = "feature" + str(id) + ".data"
    file = open(filename, "w")

    for i, data in enumerate(dataloader, 0):
        real_cpu, _ = data
        if opt.gpu:
            real_cpu = real_cpu.cuda()
        initData(real_cpu)

        if opt.noiseGen:
            features = model({Variable(real_cpu), Variable(noise)})
        else:
            features = model(Variable(real_cpu))

        #write to file
        if 1:
            features = features.cpu().data
            cnt = len(features)
            cperf =len(features[0])
            tttt = features[0][0][0][0]
            for i in range(cnt):
                for j in range(cperf):
                    file.write(str(features[i][j][0][0]) + " ")
                file.write('\n')
                print(i)
            #print(features[0])

        if i >=2:
            break
    file.close()


# here is the running part
if __name__ == "__main__":

    opt = parse()

    ##set seed
    if opt.manualSeed == 0:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.gpu:
        torch.cuda.manual_seed_all(opt.manualSeed)
    cudnn.benchmark = True

    ## load data
    testroot = '/home/cad/PycharmProjects/ContextEncoder/dataset/dining_room/d_train'
    inputSize = opt.fineSize
    dataset = dset.ImageFolder(root=testroot,
                               transform=transforms.Compose([
                                   transforms.Scale(inputSize),
                                   transforms.CenterCrop(inputSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers), drop_last=True)

    ## load net
    ngpu = opt.gpu
    nc = opt.nc
    nz = opt.nz
    nBottleneck = opt.nBottleneck
    ndf = opt.ndf
    ngf = opt.ngf
    nef = opt.nef
    real_label = 1
    fake_label = 0

    ## initialize variables
    noise = []
    if opt.noiseGen:
        noise = torch.FloatTensor(opt.batchSize, opt.nz, 1, 1)
        if opt.noisetype == 'uniform':
            noise.radom(-1, 1)
        elif opt.noisetype == 'normal':
            noise.normal(0, 1)

    netGroot = '/home/cad/PycharmProjects/ContextEncoder/checkpoints/random_inpaintCenter_10_netG.pth'
    netG = _netG(ngpu, opt.noiseGen, opt.fineSize, nc, nef, nBottleneck, nz, ngf)
    trained_dict = torch.load(netGroot)

    HalfNettoFeature(trained_dict, dataloader, noise, opt)

    netG.load_state_dict(trained_dict)

    ## port to GPU
    if opt.gpu > 0:
        netG.cuda()
        if opt.noiseGen:
            noise = noise.cuda()
    print(netG)

    ## run Context-Encoder to inpaint center
    for i, data in enumerate(dataloader, 0):
        real_cpu, _ = data
        if opt.gpu:
            real_cpu = real_cpu.cuda()
        initData(real_cpu)

        if opt.noiseGen:
            pred_center = netG({Variable(real_cpu), Variable(noise)})
        else:
            pred_center = netG(Variable(real_cpu))

        inpaint(real_cpu, pred_center.data)

        vutils.save_image(real_cpu,
                          'testoutput/mixed_%d.png' % (i),
                          normalize=True)

        if i % 10 == 0:
            print("inpaint %d" % (i))

