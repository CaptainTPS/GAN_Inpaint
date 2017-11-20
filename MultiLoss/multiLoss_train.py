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
import multiLoss_net

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
global newNetG
global input_ctx
global input_center
global input_ctx_vis
global input_real_center
global netVGG

# custom weights initialization called on newNetG and netD
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
    global newNetG

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

    fake = newNetG(Variable(input_ctx))
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
def fGx(real, fake):
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

    inputreal = Variable(real)

    var_no_grad = inputreal.detach()
    errG_l2 = criterionMSE(fake, var_no_grad)

    errG_total = 0.01 * errG + 0.99 * errG_l2

    errG_total.backward()
    optimizerG.step()
    return errG_total.data.mean(), errG_l2.data.mean()

def fGx_vgg(real, fake):
    global opt
    global optimizerG
    global criterionMSE
    global netVGG

    optimizerG.zero_grad()

    label.fill_(real_label)  # fake labels are real for generator cost
    if opt.conditionAdv:
        output = netD({Variable(input_ctx), fake})
    else:
        output = netD(fake)
    # output = netD.output # netD:forward({input_ctx,input_center}) was already executed in fDx, so save computation
    errG = criterion(output, Variable(label))

    inputreal = Variable(real)
    var_no_grad = inputreal.detach()
    errG_l2 = criterionMSE(fake, var_no_grad)


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    realResult = var_no_grad
    for x in range(len(realResult)):
        realResult.data[x] = normalize(realResult.data[x])
    realResult = realResult.cuda()
    realResult = netVGG(realResult)
    realResult = realResult.detach()

    fakeResult = fake
    for x in range(len(fakeResult)):
        fakeResult.data[x] = normalize(fakeResult.data[x])
    fakeResult = fakeResult.cuda()
    fakeResult = netVGG(fakeResult)


    errG_vgg = criterionMSE(fakeResult, realResult)

    errG_total = 0.01 * errG + 0.99 * errG_l2 + 0.00001 * errG_vgg

    errG_total.backward()
    optimizerG.step()
    return errG_total.data.mean(), errG_l2.data.mean(), errG_vgg.data.mean()

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
    global newNetG
    global input_ctx
    global input_center
    global input_ctx_vis
    global input_real_center
    global netVGG

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

    newNetG = multiLoss_net.multiLossG(nc, nef, nBottleneck, ngf)
    newNetG.apply(weights_init)
    print(newNetG)

    netD = _netD(opt.conditionAdv, opt.fineSize, nc, ndf, opt.batchSize)
    netD.apply(weights_init)
    print(netD)

    import torchvision.models as models
    netVGG = models.vgg19(pretrained=True)
    layerCnt = 28
    newNetVGG = nn.Sequential()
    newNetVGG.add_module(str(len(newNetVGG._modules)), nn.UpsamplingNearest2d(scale_factor=2))
    for layer in netVGG.features:
        if layerCnt == 0:
            break
        newNetVGG.add_module(str(len(newNetVGG._modules)), layer)
        layerCnt -= 1

    netVGG = newNetVGG
    netVGG.training = False
    netVGG = netVGG.cuda()
    print(netVGG)

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
        newNetG = newNetG.cuda()
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
    optimizerG = optim.Adam(newNetG.parameters(), lr=optimStateG['lr'], betas=optimStateG['betas'])

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

    #pre train use lossGAN & lossMSE
    if 0:
        #load from file
        netGroot = '/home/cad/PycharmProjects/ContextEncoder/checkpoints/random_inpaintCenter_30_netG.pth'
        newNetG.load_state_dict(torch.load(netGroot))

        netDroot = '/home/cad/PycharmProjects/ContextEncoder/checkpoints/random_inpaintCenter_30_netD.pth'
        netD.load_state_dict(torch.load(netDroot))
        pass
    else:
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

                real_ctx = real_cpu.clone()

                initData(real_cpu, mask)

                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                errD, fake, = fDx()

                # (2) Update G network: maximize log(D(G(z)))
                errG, errG_l2 = fGx(real_ctx, fake)

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
                fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 4))
                ax0.plot(times, errG_l2_record, label="$errG_l2$", color="red")
                ax0.plot(times, errG_record, label="$errG$")
                ax0.legend()
                ax1.plot(times, errD_record, label="$errD$")
                ax1.legend()
                plt.savefig("loss_GD.png")

            # display
            if opt.display:
                vutils.save_image(real_ctx,
                                  'output/random_epo%d_real.png' % (epoch),
                                  normalize=True)
                vutils.save_image(fake.data,
                                  'output/random_epo%d_fake.png' % (epoch),
                                  normalize=True)

            if epoch % 10 == 0:
                torch.save(newNetG.state_dict(), 'checkpoints/random_' + opt.name + '_' + str(epoch) + '_netG.pth')
                torch.save(netD.state_dict(), 'checkpoints/random_' + opt.name + '_' + str(epoch) + '_netD.pth')
                # torch.save(netG, 'checkpoints/' + opt.name + '_' + str(epoch) + '_netG.whole')
                # torch.save(netD, 'checkpoints/' + opt.name + '_' + str(epoch) + '_netD.whole')

            if epoch > 31:
                break

        endT = time.localtime()

        print('begin: %d:%d:%d' % (beginT.tm_hour, beginT.tm_min, beginT.tm_sec))
        print('end_preTrain: %d:%d:%d' % (endT.tm_hour, endT.tm_min, endT.tm_sec))

    #train with lossVGG19
    errG_l2_record = []
    errG_vgg_record = []
    errG_record = []
    errD_record = []
    times = []

    vgg_iter = 10
    for epoch in range(vgg_iter):
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

            real_ctx = real_cpu.clone()

            initData(real_cpu, mask)

            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            errD, fake = fDx()

            # (2) Update G network: maximize log(D(G(z)))
            errG, errG_l2, errG_vgg = fGx_vgg(real_ctx, fake)

            # logging
            if 1:
                print(('Epoch: [%d / %d][%d / %d] Time: %.3f '
                       + '  Err_G_L2: %.4f   Err_G_vgg: %.4f  Err_G: %.4f ; Err_D: %.4f') % (
                          epoch, opt.niter, i, len(dataloader),
                          time.time() - tm,
                          errG_l2, errG_vgg,
                          errG, errD))
                errG_l2_record.append(errG_l2)
                errG_vgg_record.append(errG_vgg)
                errG_record.append(errG)
                errD_record.append(errD)
                times.append(ttt)
                ttt += 1

        if epoch % 1 == 0:
            fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 4))
            ax0.plot(times, errG_l2_record, label="$errG_l2$", color="red")
            ax0.plot(times, errG_vgg_record, label="$errG_vgg$")
            ax0.plot(times, errG_record, label="$errG$")
            ax0.legend()
            ax1.plot(times, errD_record, label="$errD$")
            ax1.legend()
            plt.savefig("loss_vgg.png")

        # display
        if opt.display:
            vutils.save_image(real_ctx,
                              'output/multiloss_epo%d_real.png' % (epoch),
                              normalize=True)
            vutils.save_image(fake.data,
                              'output/multiloss_epo%d_fake.png' % (epoch),
                              normalize=True)

        if epoch % 10 == 0:
            torch.save(newNetG.state_dict(), 'checkpoints/multiloss_' + opt.name + '_' + str(epoch) + '_netG.pth')
            torch.save(netD.state_dict(), 'checkpoints/multiloss_' + opt.name + '_' + str(epoch) + '_netD.pth')
            # torch.save(netG, 'checkpoints/' + opt.name + '_' + str(epoch) + '_netG.whole')
            # torch.save(netD, 'checkpoints/' + opt.name + '_' + str(epoch) + '_netD.whole')

        if epoch > 31:
            break

    endT = time.localtime()

    print('begin: %d: %d: %d' % (beginT.tm_hour, beginT.tm_min, beginT.tm_sec))
    print('end: %d: %d: %d' % (endT.tm_hour, endT.tm_min, endT.tm_sec))

if __name__ == "__main__":
    main()












