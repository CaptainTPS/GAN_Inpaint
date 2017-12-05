from Unet_net import UNET
from Unet_net import DNET
from Unet_train import getMask
from Unet_train import initData

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


def main():
    ngpu = 2
    batchSize = 1
    inputSize = 256
    channel = 3

    # load nets
    Uroot = "checkpoints/Umodel_50.pth"
    Droot = "checkpoints/dModel_50.pth"
    uModel = UNET(inputSize, inputSize, channel)
    dModel = DNET(batch=batchSize, nc=channel, inputSize=inputSize, nf=32)
    if ngpu:
        uModel = uModel.cuda()
        uModel = nn.DataParallel(uModel)
    if ngpu:
        dModel = dModel.cuda()
    uModel.load_state_dict(torch.load(Uroot))
    dModel.load_state_dict(torch.load(Droot))
    uModel.eval()
    dModel.eval()

    # load data
    # dataroot = "/home/cad/PycharmProjects/ContextEncoder/dataset/conference/val"
    dataroot = "/home/cad/PycharmProjects/ContextEncoder/dataset/disparityTest"
    # dataroot = "/home/cad/PycharmProjects/ContextEncoder/dataset/conference/test"
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(inputSize),
                                   transforms.CenterCrop(inputSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize,
                                             shuffle=True, num_workers=3, drop_last=True)

    # load mask
    mask = getMask(inputSize, batchSize, channel)

    # load loss function
    lossF = nn.MSELoss()
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

    mask_gpu = Variable(1 - mask).cuda()
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
        output = uModel(img_data)
        loss_img = lossF(output, img_target)
        loss_mask = lossF(output * mask_gpu, img_target * mask_gpu)
        Doutput = dModel(output)
        loss_ad = lossBCE(Doutput, real_label)
        lossG = 0.1 * loss_img + 0.8 * loss_mask + 0.1 * loss_ad

        print(('TEST: [%d / %d] '
               'LossG: %.4f; LossIMG: %.4f, LossMASK: %.4f, LossAD: %.4f;'
               ) % (i, len(dataloader),
                    lossG.data.mean(), loss_img.data.mean(), loss_mask.data.mean(), loss_ad.data.mean(),
                    ))
        # record
        if 1:
            for x in lossG:
                lossG_list.append(x.data.mean())

            for x in loss_img:
                loss_img_list.append(x.data.mean())
            for x in loss_mask:
                loss_mask_list.append(x.data.mean())
            for x in loss_ad:
                loss_ad_list.append(x.data.mean())

            img_real = img_target.clone()
            img_fake = output.clone()

        if i < 10:
            nrow = int(np.sqrt(batchSize))
            vutils.save_image(img_real.data,
                              'output/unet_test%d_real.png' % (i),
                              nrow=nrow, normalize=True)
            vutils.save_image(img_fake.data,
                              'output/unet_test%d_fake.png' % (i),
                              nrow=nrow, normalize=True)

    if 1:
        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(20, 10))
        ax0.plot(lossG_list, label="$lossG_total$")
        ax0.legend()
        ax1.plot(loss_img_list, label="loss_img")
        ax1.plot(loss_mask_list, label="loss_mask")
        ax1.plot(loss_ad_list, label="loss_ad")
        ax1.legend()
        plt.savefig("loss_TEST.png", dpi=200)
        plt.close(fig)


    endT = time.localtime()
    print('begin: %d:%d:%d' % (beginT.tm_hour, beginT.tm_min, beginT.tm_sec))
    print('end: %d:%d:%d' % (endT.tm_hour, endT.tm_min, endT.tm_sec))

    pass

if __name__ == "__main__":
    main()
    pass
