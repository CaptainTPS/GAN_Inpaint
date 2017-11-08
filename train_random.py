import math
import argparse
import random
import time
import os
import numpy as np
import cv2
import copy
from matplotlib import pyplot as plt

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

from random_net import _netD
from random_net import _netG
from random_net import parse

global opt
global ngpu
global nc
global nz
global nBottleneck
global ndf
global ngf
global nef
global real_label
global fake_label
global optimizerD
global optimizerG
global criterionMSE
global label
global netD
global criterion
global noise
global netG
global input_ctx
global input_center
global input_ctx_vis
global input_real_center

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

## init data with mask
def initMask(maskroot = None):
    global opt
    global nc
    # 0 in mask means to be masked out, others must be 1 255
    # should be inited by another way like image or something
    if maskroot == None:
        maskroot = '/home/cad/PycharmProjects/ContextEncoder/mask3.png'
    mask = cv2.imread(maskroot, cv2.IMREAD_GRAYSCALE)

    if len(mask) != opt.fineSize:
        mask = cv2.resize(mask, (opt.fineSize, opt.fineSize))

    mask = torch.from_numpy(mask)
    mask = (mask == 255)

    mask = mask.type(torch.FloatTensor)
    m = torch.FloatTensor(opt.batchSize, nc, opt.fineSize, opt.fineSize)

    for i in range(opt.batchSize):
        m[i, 0, :, :] = mask
        m[i, 1, :, :] = mask
        m[i, 2, :, :] = mask

    return m

def initData(input, mask):

    global input_ctx
    global input_center
    real_ctx = input
    real_center = real_ctx.clone()  # copy by value

    if 0:
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

    real_ctx = real_ctx * mask
    input_ctx.copy_(real_ctx)

    input_center.copy_(real_center)

    if opt.wtl2!=0:
        global input_real_center
        input_real_center.copy_(real_center)

# create closure to evaluate discriminator
def fDx():
    global optimizerD
    global label
    global netD
    global criterion
    global noise
    global netG

    optimizerD.zero_grad()
    ## train with real
    label.fill_(real_label)
    labelv = Variable(label)

    if opt.conditionAdv:
        output = netD({Variable(input_ctx), Variable(input_center)})  # may have problem
    else:
        output = netD(Variable(input_center))
    errD_real = criterion(output, labelv)
    errD_real.backward()

    # train with fake
    if opt.noisetype == 'uniform':  # regenerate random noise
        noise.random_(-1, 1)
    elif opt.noisetype == 'normal':
        noise.normal_(0, 1)

    if opt.noiseGen:
        fake = netG({Variable(input_ctx), Variable(noise)})  # see if wrong
    else:
        fake = netG(Variable(input_ctx))
    input_center.copy_(fake.data)
    label.fill_(fake_label)
    labelv = Variable(label)

    if opt.conditionAdv:
        output = netD({Variable(input_ctx), Variable(input_center)})
    else:
        output = netD(Variable(input_center))
    errD_fake = criterion(output, labelv)
    errD_fake.backward()

    errD = errD_real + errD_fake

    optimizerD.step()
    return errD.data.mean(), fake

# create closure to evaluate generator
def fGx(fake):
    global opt
    global optimizerG
    global criterionMSE

    optimizerG.zero_grad()

    label.fill_(real_label)  # fake labels are real for generator cost
    if opt.conditionAdv:
        output = netD({Variable(input_ctx), fake})
    else:
        output = netD(fake)
    # output = netD.output # netD:forward({input_ctx,input_center}) was already executed in fDx, so save computation
    errG = criterion(output, Variable(label))

    errG_total = errG
    if opt.wtl2 != 0:
        errG_l2 = criterionMSE(fake, Variable(input_real_center))

        if opt.overlapPred == 0:
            if (opt.wtl2 > 0 and opt.wtl2 < 1):
                errG_total = (1 - opt.wtl2) * errG + opt.wtl2 * errG_l2
            else:
                errG_total = errG + opt.wtl2 * errG_l2
        else:
            overlapL2Weight = 10
            if (opt.wtl2 > 0 and opt.wtl2 < 1):
                # df_dg:mul(1-opt.wtl2):addcmul(1,wtl2Matrix,df_dg_l2)
                errG_total = (1 - opt.wtl2) * errG + opt.wtl2 * errG_l2
            else:
                # df_dg:addcmul(1,wtl2Matrix,df_dg_l2)
                errG_total = errG + opt.wtl2 * errG_l2
                # something might be wrong here...
    else:
        errG_l2 = 0
        errG_l2 = Variable(errG_l2)

    errG_total.backward()
    optimizerG.step()
    return errG_total.data.mean(), errG_l2.data.mean()

# here is the running part
def main(maskPath = None):
    global opt
    global ngpu
    global nc
    global nz
    global nBottleneck
    global ndf
    global ngf
    global nef
    global real_label
    global fake_label
    global optimizerD
    global optimizerG
    global criterionMSE
    global label
    global netD
    global criterion
    global noise
    global netG
    global input_ctx
    global input_center
    global input_ctx_vis
    global input_real_center

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

    # value
    ngpu = opt.gpu
    nc = opt.nc
    nz = opt.nz
    nBottleneck = opt.nBottleneck
    ndf = opt.ndf
    ngf = opt.ngf
    nef = opt.nef
    real_label = 1
    fake_label = 0

    netG = _netG(ngpu, opt.noiseGen, opt.fineSize, nc, nef, nBottleneck, nz, ngf)
    netG.apply(weights_init)
    print(netG)

    netD = _netD(opt.conditionAdv, opt.fineSize, nc, ndf, opt.batchSize)
    netD.apply(weights_init)
    print(netD)

    # Loss Metrics
    criterion = nn.BCELoss()
    if opt.wtl2 != 0:
        criterionMSE = nn.MSELoss()

    # Initialize data variables

    input_ctx_vis = torch.FloatTensor(opt.batchSize, nc, opt.fineSize, opt.fineSize)
    input_ctx = torch.FloatTensor(opt.batchSize, nc, opt.fineSize, opt.fineSize)
    input_center = torch.FloatTensor(opt.batchSize, nc, opt.fineSize, opt.fineSize)
    if opt.wtl2 != 0:
        input_real_center = torch.FloatTensor(opt.batchSize, nc, opt.fineSize, opt.fineSize)

    noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
    label = torch.FloatTensor(opt.batchSize)
    tm = time.time()

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
    optimizerD = optim.Adam(netD.parameters(), lr=optimStateD['lr'], betas=optimStateD['betas'])
    optimizerG = optim.Adam(netG.parameters(), lr=optimStateG['lr'], betas=optimStateG['betas'])

    # Train Context Encoder
    mask = initMask()
    if opt.gpu:
        mask = mask.cuda()
    beginT = time.localtime()
    errG_l2_record = []
    errG_record = []
    errD_record = []
    times = []
    ttt = 0
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 4))
    for epoch in range(opt.niter):
        try:
            os.mkdir('checkpoints')
        except OSError:
            pass
        fake = None
        real_ctx = None
        tm = time.time()

        for i, data in enumerate(dataloader, 0):
            real_cpu, _ = data

            # tm = time.time()
            if opt.gpu:
                real_cpu = real_cpu.cuda()

            if i == len(dataloader) - 1:
                real_ctx = real_cpu.clone()

            initData(real_cpu, mask)

            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            errD, fake = fDx()

            # (2) Update G network: maximize log(D(G(z)))
            errG, errG_l2 = fGx(fake)

            # logging
            if 1:
                print(('Epoch: [%d / %d][%d / %d] Time: %.3f '
                       + '  Err_G_L2: %.4f   Err_G: %.4f  Err_D: %.4f') % (
                          epoch, opt.niter, i, len(dataloader),
                          time.time() - tm,
                          errG_l2,
                          errG, errD))
                errG_l2_record.append(errG_l2)
                errG_record.append(errG)
                errD_record.append(errD)
                times.append(ttt)
                ttt += 1

        if epoch % 1 == 0:
            ax0.plot(times, errG_l2_record, label="$errG_l2$", color="red")
            ax0.plot(times, errG_record, label="$errG$")
            ax0.legend()
            ax1.plot(times, errD_record, label="$errD$")
            ax1.legend()
            plt.savefig("loss.png")

        # display
        if opt.display:
            vutils.save_image(real_ctx,
                              'output/random_epo%d_real.png' % (epoch),
                              normalize=True)
            vutils.save_image(fake.data,
                              'output/random_epo%d_fake.png' % (epoch),
                              normalize=True)

        if epoch % 10 == 0:
            torch.save(netG.state_dict(), 'checkpoints/random_' + opt.name + '_' + str(epoch) + '_netG.pth')
            torch.save(netD.state_dict(), 'checkpoints/random_' + opt.name + '_' + str(epoch) + '_netD.pth')
            # torch.save(netG, 'checkpoints/' + opt.name + '_' + str(epoch) + '_netG.whole')
            # torch.save(netD, 'checkpoints/' + opt.name + '_' + str(epoch) + '_netD.whole')

        if epoch > 31:
            break

    endT = time.localtime()

    print('begin: %d: %d: %d' % (beginT.tm_hour, beginT.tm_min, beginT.tm_sec))
    print('end: %d: %d: %d' % (endT.tm_hour, endT.tm_min, endT.tm_sec))


if __name__ == "__main__":
    main()












