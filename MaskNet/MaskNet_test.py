from MaskNet_net import MASKDNET
from MaskNet_net import MASKNET
from MaskNet_train import randomMask
from MaskNet_train import initData

import time
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

def getMask(fineSize, batchSize, nc, maskroot = None):
    if maskroot == None:
        maskroot = 'test_mask.png'
    mask = cv2.imread(maskroot, cv2.IMREAD_GRAYSCALE)

    if len(mask) != fineSize:
        mask = cv2.resize(mask, (fineSize, fineSize))

    mask = torch.from_numpy(mask)
    mask = (mask == 255)

    mask = mask.type(torch.FloatTensor)
    m = torch.FloatTensor(batchSize, nc, fineSize, fineSize)

    for i in range(batchSize):
        m[i, 0, :, :] = mask
        m[i, 1, :, :] = mask
        m[i, 2, :, :] = mask

    return m


def main():
    dataroot = "/home/cad/PycharmProjects/ContextEncoder/dataset/dining_room/d_val"
    # dataroot = '/home/cad/PycharmProjects/ContextEncoder/dataset/test128'
    batchSize = 1
    inputSize = 256
    channel = 3
    ngpu = 2

    # load data
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(inputSize),
                                   transforms.CenterCrop(inputSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize,
                                             shuffle=True, num_workers=3, drop_last=True)

    #load mask
    # mask = randomMask(inputSize, batchSize, channel)
    mask = getMask(inputSize,batchSize,channel)

    # load model
    uModel = MASKNET(inputSize, inputSize, channel, nfeature=32, downTimes=5)
    if ngpu:
        uModel = uModel.cuda()
        uModel = nn.DataParallel(uModel)
    NetUroot = "checkpoints/Umodel_20.pth"
    uModel.load_state_dict(torch.load(NetUroot))
    uModel.eval()

    dModel = MASKDNET(batch=batchSize, nc=channel, inputSize=inputSize, nf=32)
    if ngpu:
        dModel = dModel.cuda()
        # !!! dModel = nn.DataParallel(dModel)
    NetDroot = "checkpoints/dModel_20.pth"
    dModel.load_state_dict(torch.load(NetDroot))
    dModel.eval()

    # load loss function
    lossF= nn.MSELoss()
    lossF = nn.L1Loss()
    lossBCE = nn.BCELoss()
    fake_label = Variable(torch.FloatTensor(batchSize).fill_(0))
    real_label = Variable(torch.FloatTensor(batchSize).fill_(1))
    if ngpu:
        fake_label = fake_label.cuda()
        real_label = real_label.cuda()

    #running
    lossG_list = []
    loss_img_list = []
    loss_mask_list = []
    loss_ad_list = []
    img_real = None
    img_fake = None
    beginT = time.localtime()

    try:
        os.mkdir('output')
    except OSError:
        pass
    try:
        os.mkdir('checkpoints')
    except OSError:
        pass

    mask_gpu = Variable(mask).cuda()

    for i, data in enumerate(dataloader, 0):

        img_data, _ = data
        img_target = img_data.clone()
        img_target = Variable(img_target)
        img_data = initData(img_data, mask)
        img_data = Variable(img_data)
        if ngpu:
            img_data = img_data.cuda()
            img_target = img_target.cuda()

        # train Unet(G)
        output = uModel(img_data, mask_gpu)
        loss_img = lossF(output, img_target)
        loss_mask = lossF(output * mask_gpu, img_target *mask_gpu)
        Doutput = dModel(output, mask_gpu)
        loss_ad = lossBCE(Doutput, real_label)
        lossG = 0.1 * loss_img + 0.8 * loss_mask + 0.1 * loss_ad


        print(('TEST: [%d / %d] '
               'LossG: %.4f; LossIMG: %.4f, LossMASK: %.4f, LossAD: %.4f; '
               ) % (i, len(dataloader),
                lossG.data.mean(), loss_img.data.mean(), loss_mask.data.mean(), loss_ad.data.mean(),
                ))
        # record
        if 1:
            lossG_list.append(lossG.data.mean())

            loss_img_list.append(loss_img.data.mean())
            loss_mask_list.append(loss_mask.data.mean())
            loss_ad_list.append(loss_ad.data.mean())

        img_real = img_target.clone()
        img_fake = output.clone()

        # record every epoch
        if i < 10:
            nrow = int(np.sqrt(batchSize))
            vutils.save_image(img_real.data,
                              'output/test_%d_real.png' % (i),
                              nrow=nrow, normalize=True)
            vutils.save_image(img_fake.data,
                              'output/test_%d_fake.png' % (i),
                              nrow=nrow, normalize=True)

    if 1:
        fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(30, 10))
        ax0.plot(loss_img_list, label="$loss_img$")
        ax0.legend()
        ax1.plot(loss_mask_list, label="$loss_mask$")
        ax1.legend()
        ax2.plot(loss_ad_list, label="$loss_ad$")
        ax2.legend()
        plt.savefig("test_G_detail.png", dpi=200)
        plt.close(fig)

        plt.plot(lossG_list, label="$lossG$")
        plt.legend()
        plt.savefig("test_G.png", dpi=200)
        plt.close()

    endT = time.localtime()
    print('begin: %d:%d:%d' % (beginT.tm_hour, beginT.tm_min, beginT.tm_sec))
    print('end: %d:%d:%d' % (endT.tm_hour, endT.tm_min, endT.tm_sec))


if __name__ == "__main__":
    main()