import argparse
import random
import time
import os
import cv2
import numpy as np
import matplotlib as plt
import pylab as pl

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

def initMask():
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
    real_ctx = input
    if 0 :
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

    if opt.wtl2 != 0:
        global input_real_center
        input_real_center.copy_(real_ctx)

    real_ctx = real_ctx * mask
    input_ctx.copy_(real_ctx)

    return real_ctx

def poissonFilling(input, ori, mask, index = 0):
    input = (input + 1) / 2
    img = np.rollaxis(input, axis=0, start=3)
    img = img * 255
    img = img.astype(dtype= np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    ori = (ori + 1) / 2
    img_ori = np.rollaxis(ori, axis=0, start=3)
    img_ori = img_ori * 255
    img_ori = img_ori.astype(dtype=np.uint8)
    img_ori = cv2.cvtColor(img_ori, cv2.COLOR_RGB2BGR)

    mask = mask * 255
    mask = mask.astype(dtype=np.uint8)
    # mask = 255 - mask
    mask = np.rollaxis(mask, axis=0, start=3)

    # print(img.shape)
    # print(img_ori.shape)
    # print(mask.shape)

    point = (len(img_ori) / 2 - 1, len(img_ori[0]) / 2 - 1)
    #the real position in dst will change due to white mask size; so just clone ori to fake using a reverse mask
    output = cv2.seamlessClone(img_ori, img, mask, point, cv2.NORMAL_CLONE)
    cv2.imwrite('testoutput/cloneResult/%d_mask.png' % (index), mask)
    cv2.imwrite('testoutput/cloneResult/%d_fake.png'%(index), img)
    cv2.imwrite('testoutput/cloneResult/%d_ori.png'%(index), img_ori)
    # cv2.imwrite('testoutput/cloneResult/%d_mask.png', mask)
    cv2.imwrite('testoutput/cloneResult/%d_result.png'%(index), output)

def directClone(input, ori, mask):
    pass

def inpaint(real, pred, mask):
    real = real.cpu().numpy()
    pred = pred.data.cpu().numpy()
    mask = mask.cpu().numpy()

    for i in range(opt.batchSize):
        poissonFilling(pred[i], real[i], mask[i], i)


def calLossG(fake):
    # calculate lossG
    label.fill_(real_label)
    if opt.conditionAdv:
        output = netD({Variable(input_ctx), fake})
    else:
        output = netD(fake)
    errG = criterion(output, Variable(label))

    errG_l2 = None
    if opt.wtl2 != 0:
        errG_l2 = criterionMSE(fake, Variable(input_real_center))

    return errG, errG_l2

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

    # testroot = '/home/cad/PycharmProjects/ContextEncoder/dataset/dining_room/d_val'
    # testroot = '/home/cad/PycharmProjects/ContextEncoder/dataset/conference/val'
    testroot = '/home/cad/PycharmProjects/ContextEncoder/dataset/dining_room/d_test'
    # testroot = '/home/cad/PycharmProjects/ContextEncoder/dataset/DIYtest'
    # testroot = '/home/cad/PycharmProjects/ContextEncoder/dataset/lsunTest'
    inputSize = opt.fineSize
    dataset = dset.ImageFolder(root=testroot,
                                   transform=transforms.Compose([
                                       transforms.Scale(inputSize),
                                       transforms.CenterCrop(inputSize),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=False, num_workers=int(opt.workers), drop_last=True)

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

    netGroot = '/home/cad/PycharmProjects/ContextEncoder/checkpoints/random_inpaintCenter_30_netG.pth'
    netG = _netG(ngpu, opt.noiseGen, opt.fineSize, nc, nef, nBottleneck, nz, ngf)
    netG.load_state_dict(torch.load(netGroot))

    netDroot = '/home/cad/PycharmProjects/ContextEncoder/checkpoints/random_inpaintCenter_30_netD.pth'
    netD = _netD(opt.conditionAdv, opt.fineSize, nc, ndf, opt.batchSize)
    netD.load_state_dict(torch.load(netDroot))

    ## initialize variables
    if opt.noiseGen:
        noise = torch.FloatTensor(opt.batchSize, opt.nz, 1, 1)
        if opt.noisetype == 'uniform':
            noise.radom(-1, 1)
        elif opt.noisetype == 'normal':
            noise.normal(0, 1)

    if opt.wtl2 != 0:
        input_real_center = torch.FloatTensor(opt.batchSize, nc, opt.fineSize, opt.fineSize)
    label = torch.FloatTensor(opt.batchSize)
    input_ctx = torch.FloatTensor(opt.batchSize, nc, opt.fineSize, opt.fineSize)

    # Loss Metrics
    criterion = nn.BCELoss()
    if opt.wtl2 != 0:
        criterionMSE = nn.MSELoss()

    ## port to GPU
    if opt.gpu > 0:
        netG.cuda()
        netD.cuda()
        input_ctx = input_ctx.cuda()
        criterion.cuda()
        label = label.cuda()
        if opt.wtl2 != 0:
            criterionMSE.cuda()
            input_real_center = input_real_center.cuda()
        if opt.noiseGen:
            noise = noise.cuda()
    print(netG)
    print(netD)


    ## run Context-Encoder to inpaint center
    mask = initMask()
    if opt.gpu:
        mask = mask.cuda()

    y1 = []
    y2 = []
    x =[]

    for i, data in enumerate(dataloader, 0):
        real_cpu, _ = data

        if opt.gpu:
            real_cpu = real_cpu.cuda()

        real_cpu = initData(real_cpu, mask)

        if opt.noiseGen:
            pred = netG({Variable(real_cpu),Variable(noise)})
        else:
            pred = netG(Variable(real_cpu))


        errG_d, errG_l2 = calLossG(pred)
        # errG_d, errG_l2 = calLossG(Variable(input_real_center))
        x.append(i)
        y1.append(errG_d.data.cpu().numpy()[0])
        y2.append(errG_l2.data.cpu().numpy()[0])

        if 1:
            inpaint(real_cpu, pred, mask)

            vutils.save_image(pred.data,
                              'testoutput/epo%d_fake.png' % (i),
                              normalize=True)
            vutils.save_image(real_cpu,
                              'testoutput/epo%d_real.png' % (i),
                              normalize=True)

        if i %10 == 0:
            print("inpaint %d"%(i))

    plt.interactive(True)
    pl.plot(x, y1)
    pl.plot(x, y2)
    pl.draw()
    print('done')
