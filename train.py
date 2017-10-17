import math
import argparse
import random
import time
import os
import numpy as np
import cv2
import copy
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


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True)
parser.add_argument('--batchSize', type=int ,default=64)
parser.add_argument('--loadSize', type=int ,default=350)
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



if 0:
    opt = {
       'batchSize' : 64,     #   -- number of samples to produce
       'loadSize' : 350,     #    -- resize the loaded image to loadsize maintaining aspect ratio. 0 means don't resize. -1 means scale randomly between [0.5,2] -- see donkey_folder.lua
       'fineSize' : 128,     #    -- size of random crops
       'nBottleneck' : 100,  #    -- #  of dim for bottleneck of encoder
       'nef' : 64,             #   -- #  of encoder filters in first conv layer
       'ngf' : 64,             #   -- #  of gen filters in first conv layer
       'ndf' : 64,             #  -- #  of discrim filters in first conv layer
       'nc' : 3,               # -- # of channels in input
       'wtl2' : 0,             # -- 0 means don't use else use with this weight
       'overlapPred' : 0,      # -- overlapping edges
       'workers' : 4,         #  -- #  of data loading threads to use
       'niter' : 25,           #  -- #  of iter at starting learning rate
       'lr' : 0.0002,          #  -- initial learning rate for adam
       'beta1' : 0.5,          #  -- momentum term of adam
       'ntrain' : 'math.huge',   #  -- #  of examples per epoch. math.huge for full dataset
       'display' : 1,          #  -- display samples while training. 0 = false
       'display_id' : 10,      #  -- display window id.
       'display_iter' : 50,    #  -- # number of iterations after which display is updated
       'gpu' : 1,              #  -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU nums
       'name' : 'train1',      #  -- name of the experiment you are running
       'manualSeed' : 0,       #  -- 0 means random seed

    #   -- Extra Options:
       'conditionAdv' : 0,     #  -- 0 means false else true
       'noiseGen' : 0,         #  -- 0 means false else true
       'noisetype' : 'normal', #  -- uniform / normal
       'nz' : 100              #  -- #  of dim for Z
    }

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
inputSize = opt.fineSize
dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(inputSize),
                                   transforms.CenterCrop(inputSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers), drop_last=True)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# local value
ngpu = opt.gpu
nc = opt.nc
nz = opt.nz
nBottleneck = opt.nBottleneck
ndf = opt.ndf
ngf = opt.ngf
nef = opt.nef
real_label = 1
fake_label = 0

# Generator net
class _netG(nn.Module):
    def __init__(self, ngpu, noiseGen):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.noiseGen = noiseGen
        # Encode Input Context to noise (architecture similar to Discriminator)
        self.encodeNet = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, nef, 4, 2, 1),
            # nn.SpatialConvolution(nc, nef, 4, 4, 2, 2, 1, 1),
            nn.LeakyReLU(0.2, True),
            # state size: (nef) x 64 x 64
            nn.Conv2d(nef, nef, 4, 2, 1),
            nn.BatchNorm2d(nef),
            nn.LeakyReLU(0.2, True),
            # state size: (nef) x 32 x 32
            nn.Conv2d(nef, nef * 2, 4, 2, 1),
            nn.BatchNorm2d(nef * 2),
            nn.LeakyReLU(0.2, True),
            # state size: (nef*2) x 16 x 16
            nn.Conv2d(nef * 2, nef * 4, 4, 2, 1),
            nn.BatchNorm2d(nef * 4),
            nn.LeakyReLU(0.2, True),
            # state size: (nef*4) x 8 x 8
            nn.Conv2d(nef * 4, nef * 8, 4, 2, 1),
            nn.BatchNorm2d(nef * 8),
            nn.LeakyReLU(0.2, True),
            # state size: (nef*8) x 4 x 4
            nn.Conv2d(nef * 8, nBottleneck, 4)
            # state size: (nBottleneck) x 1 x 1
        )
        if noiseGen:
            self.netG_noise = nn.Sequential(
                # input is Z: (nz) x 1 x 1, going into a convolution
                nn.Conv2d(nz, nz, 1, 1, 0)
                # state size: (nz) x 1 x 1)
            )
            self.midNet = nn.Sequential(
                nn.BatchNorm2d(nBottleneck+nz),
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
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
            nn.Tanh()
            # state size: (nc) x 64 x 64
        )

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


netG = _netG(ngpu, opt.noiseGen)
netG.apply(weights_init)
# if opt.netG != '':
#    netG.load_state_dict(torch.load(opt.netG))
print(netG)

# Adversarial discriminator net
class _netD(nn.Module):
    def __init__(self, conditionAdv):
        super(_netD, self).__init__()
        self.conditionAdv = conditionAdv
        if self.conditionAdv:
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
            # input is (nc) x 64 x 64, going into a convolution
            self.midNet = nn.Sequential(
                nn.Conv2d(nc, ndf, 4, 2, 1),
                nn.LeakyReLU(0.2, True)
                # state size: (ndf) x 32 x 32
            )

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
            y = torch.cat(y, 0) # nn.JoinTable(2)
            y = self.midNet(y)
        else:
            y = self.midNet(input)

        output = self.tailNet(y)
        output = output.view(opt.batchSize)
        return output

netD = _netD(opt.conditionAdv)
netD.apply(weights_init)
print netD

# Loss Metrics
criterion = nn.BCELoss()
if opt.wtl2!=0:
  criterionMSE = nn.MSELoss()



# Initialize data variables

input_ctx_vis = torch.FloatTensor(opt.batchSize, nc, opt.fineSize, opt.fineSize)
input_ctx = torch.FloatTensor(opt.batchSize, nc, opt.fineSize, opt.fineSize)
input_center = torch.FloatTensor(opt.batchSize, nc, opt.fineSize/2, opt.fineSize/2)
if opt.wtl2 !=0:
    input_real_center = torch.FloatTensor(opt.batchSize, nc, opt.fineSize/2, opt.fineSize/2)

noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
label = torch.FloatTensor(opt.batchSize)
epoch_tm = time.time()
tm = time.time()
# data_tm = time.time()
# epoch_tm = torch.Timer()
# tm = torch.Timer()
# data_tm = torch.Timer()

if opt.gpu > 0:
    input_ctx_vis = input_ctx_vis.cuda()
    input_ctx = input_ctx.cuda()
    input_center = input_center.cuda()
    noise = noise.cuda()
    label = label.cuda()
    netG = netG.cuda()
    netD = netD.cuda()
    criterion.cuda()
    if opt.wtl2 != 0:
        criterionMSE.cuda()
        input_real_center = input_real_center.cuda()

# Setup Solver
optimStateG = {
    'lr': (opt.wtl2 > 0 and opt.wtl2 < 1) and opt.lr * 10 or opt.lr,
    'betas': (opt.beta1, 0.999)
}
optimStateD = {
    'lr': opt.lr,
    'betas': (opt.beta1, 0.999)
}

# Define generator and adversary closures

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr= optimStateD['lr'], betas=optimStateD['betas'])
optimizerG = optim.Adam(netG.parameters(), lr= optimStateG['lr'], betas=optimStateG['betas'])
# optimizerG = optim.Adam(netG.parameters(), lr= 0.0002, betas=optimStateG['betas'])

# create closure to evaluate f(X) and df/dX of discriminator
def fDx(input):
    optimizerD.zero_grad()
    # train with real
    # data_tm1 = time.time()
    real_ctx = input
    real_center = real_ctx[:,:,
                  opt.fineSize/4: opt.fineSize/2 + opt.fineSize/4,
                  opt.fineSize/4: opt.fineSize/2 + opt.fineSize/4].clone() # copy by value
    real_ctx[:,
            0,
            opt.fineSize/4 + opt.overlapPred: opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred,
            opt.fineSize/4 + opt.overlapPred: opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred] = 2*117.0/255.0 - 1.0
    real_ctx[:,
            1,
            opt.fineSize/4 + opt.overlapPred: opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred,
            opt.fineSize/4 + opt.overlapPred: opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred] = 2*104.0/255.0 - 1.0
    real_ctx[:,
            2,
            opt.fineSize/4 + opt.overlapPred: opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred,
            opt.fineSize/4 + opt.overlapPred: opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred] = 2*123.0/255.0 - 1.0
    # global data_tm
    # data_tm = time.time() - data_tm1
    input_ctx.copy_(real_ctx)

    input_center.copy_(real_center)
    if opt.wtl2!=0:
        # global input_real_center
        input_real_center.copy_(real_center)
    label.fill_(real_label)
    labelv = Variable(label)

    if opt.conditionAdv:
        output = netD({Variable(input_ctx),Variable(input_center)}) # may have problem
    else:
        output = netD(Variable(input_center))
    errD_real = criterion(output, labelv)
    errD_real.backward()

    # train with fake
    if opt.noisetype == 'uniform': # regenerate random noise
        noise.random_(-1, 1)
    elif opt.noisetype == 'normal':
        noise.normal_(0, 1)

    if opt.noiseGen:
        fake = netG({Variable(input_ctx),Variable(noise)}) #see if wrong
    else:
        fake = netG(Variable(input_ctx))
    input_center.copy_(fake.data)
    label.fill_(fake_label)
    labelv = Variable(label)

    if opt.conditionAdv:
        output = netD({Variable(input_ctx),Variable(input_center)})
    else:
        output = netD(Variable(input_center))
    errD_fake = criterion(output, labelv)
    errD_fake.backward()

    errD = errD_real + errD_fake

    optimizerD.step()
    return errD.data.mean(), fake

# create closure to evaluate f(X) and df/dX of generator
def fGx(input, fake):
    optimizerG.zero_grad()

    # [[ the three lines below were already executed in fDx, so save computation
    # noise.random_(-1, 1) # regenerate random noise

    real_ctx = input
    # input has already been modified in fDx
    # real_center = real_ctx[:, :, 1 + opt.fineSize / 4: opt.fineSize / 2 + opt.fineSize / 4,1 + opt.fineSize / 4: opt.fineSize / 2 + opt.fineSize / 4].clone()  # copy by value
    # real_ctx[:, 1, 1 + opt.fineSize / 4 + opt.overlapPred: opt.fineSize / 2 + opt.fineSize / 4 - opt.overlapPred,1 + opt.fineSize / 4 + opt.overlapPred: opt.fineSize / 2 + opt.fineSize / 4 - opt.overlapPred] = 2 * 117.0 / 255.0 - 1.0
    # real_ctx[:, 2, 1 + opt.fineSize / 4 + opt.overlapPred: opt.fineSize / 2 + opt.fineSize / 4 - opt.overlapPred,1 + opt.fineSize / 4 + opt.overlapPred: opt.fineSize / 2 + opt.fineSize / 4 - opt.overlapPred] = 2 * 104.0 / 255.0 - 1.0
    # real_ctx[:, 3, 1 + opt.fineSize / 4 + opt.overlapPred: opt.fineSize / 2 + opt.fineSize / 4 - opt.overlapPred,1 + opt.fineSize / 4 + opt.overlapPred: opt.fineSize / 2 + opt.fineSize / 4 - opt.overlapPred] = 2 * 123.0 / 255.0 - 1.0
    input_ctx.copy_(real_ctx)
    # fake = netG({Variable(input_ctx),Variable(noise)})
    # input_center.copy_(fake.data)
    label.fill_(real_label) # fake labels are real for generator cost
    # if opt.conditionAdv:
    #     output = netD({Variable(input_ctx),Variable(input_center)})
    # else:
    #     output = netD(Variable(input_center))
    if opt.conditionAdv:
        output = netD({Variable(input_ctx),fake})
    else:
        output = netD(fake)
    # output = netD.output # netD:forward({input_ctx,input_center}) was already executed in fDx, so save computation
    errG = criterion(output, Variable(label))

    errG_total = errG
    if opt.wtl2 != 0:
        errG_l2 = criterionMSE(fake, Variable(input_real_center))

        if opt.overlapPred==0:
            if (opt.wtl2>0 and opt.wtl2<1):
                errG_total = (1-opt.wtl2)*errG + opt.wtl2*errG_l2
            else:
                errG_total = errG + opt.wtl2*errG_l2
        else:
            overlapL2Weight = 10
            # wtl2Matrix = df_dg_l2:clone():fill(overlapL2Weight*opt.wtl2)
            # wtl2Matrix[{{},{},{1 + opt.overlapPred, opt.fineSize/2 - opt.overlapPred},{1 + opt.overlapPred, opt.fineSize/2 - opt.overlapPred}}]:fill(opt.wtl2)
            if (opt.wtl2>0 and opt.wtl2<1):
                # df_dg:mul(1-opt.wtl2):addcmul(1,wtl2Matrix,df_dg_l2)
                errG_total = (1-opt.wtl2)*errG + opt.wtl2*errG_l2
            else:
                # df_dg:addcmul(1,wtl2Matrix,df_dg_l2)
                errG_total = errG + opt.wtl2*errG_l2
        #something might be wrong here...
    else:
        errG_l2 = 0
        errG_l2 = Variable(errG_l2)

    errG_total.backward()
    optimizerG.step()
    return errG_total.data.mean(), errG_l2.data.mean()

# Train Context Encoder
beginT = time.localtime()
for epoch in  range(opt.niter):
    epoch_tm = time.time()
    # counter = 0
    try:
        os.mkdir('checkpoints')
    except OSError:
        pass


    for i, data in enumerate(dataloader, 0):
        real_cpu, _ = data

        tm = time.time()
        if opt.gpu:
            real_cpu = real_cpu.cuda()
        if i == len(dataloader) - 1:
            real_ctx = real_cpu.clone()
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        errD, fake = fDx(real_cpu)

        # (2) Update G network: maximize log(D(G(z)))
        errG, errG_l2 = fGx(real_cpu, fake)

        # logging
        if 1:
            print(('Epoch: [%d / %d][%d / %d] Time: %.3f '
                   + '  Err_G_L2: %.4f   Err_G: %.4f  Err_D: %.4f') % (
                      epoch, opt.niter, i, len(dataloader),
                      time.time() - tm,
                      errG_l2,
                      errG, errD))
    # display
    if opt.display:
        real_center = real_ctx[:, :,
                      opt.fineSize / 4: opt.fineSize / 2 + opt.fineSize / 4,
                      opt.fineSize / 4: opt.fineSize / 2 + opt.fineSize / 4].clone()  # copy by value

        real_ctx[:,:,
                opt.fineSize/4 + opt.overlapPred: opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred,
                opt.fineSize/4 + opt.overlapPred: opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred
                ].copy_(fake.data[:,:,
                            opt.overlapPred: opt.fineSize/2 - opt.overlapPred,
                            opt.overlapPred: opt.fineSize/2 - opt.overlapPred])

        vutils.save_image(real_ctx,
                          'output/epo%d_mixed.png'%(epoch),
                          normalize=True)
        vutils.save_image(real_center,
                          'output/epo%d_real_center.png' % (epoch),
                          normalize=True)
        vutils.save_image(fake.data,
                          'output/epo%d_fake.png'% (epoch),
                          normalize=True)

    # counter = counter + 1


    if epoch % 5 == 0:
        torch.save(netG.state_dict(), 'checkpoints/' + opt.name + '_' + str(epoch) + '_netG.pth')
        torch.save(netD.state_dict(), 'checkpoints/' + opt.name + '_' + str(epoch) + '_netD.pth')
        # torch.save(netG, 'checkpoints/' + opt.name + '_' + str(epoch) + '_netG.whole')
        # torch.save(netD, 'checkpoints/' + opt.name + '_' + str(epoch) + '_netD.whole')

    if epoch >10:
        break

endT = time.localtime()

print('begin: %dh:%dm:%ds'%(beginT.tm_hour, beginT.tm_min, beginT.tm_sec))
print('end: %dh:%dm:%ds'%(endT.tm_hour, endT.tm_min, endT.tm_sec))
