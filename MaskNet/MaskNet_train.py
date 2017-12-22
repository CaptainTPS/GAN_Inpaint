from MaskNet_net import MASKDNET
from MaskNet_net import MASKNET

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

def randomMask(fineSize, batchSize, nc):
    mask = np.random.rand(fineSize, fineSize)
    mask = torch.from_numpy(mask)
    mask = (mask >= 0.5)

    mask = mask.type(torch.FloatTensor)
    m = torch.FloatTensor(batchSize, nc, fineSize, fineSize)

    for i in range(batchSize):
        m[i, 0, :, :] = mask
        m[i, 1, :, :] = mask
        m[i, 2, :, :] = mask

    return m

def randomMaskBlock(height, width, nc, crop):
    width = width - crop
    blockSize = np.random.randint(1, 6)
    h = np.random.randint(0, height - blockSize)
    w = np.random.randint(0, width - blockSize)

    mask = torch.FloatTensor(nc, height, width).fill_(1)
    for x in range(nc):
        for i in range(h, h+blockSize):
            for j in range(w, w+blockSize):
                mask[x][i][j] = 0
    pass
    return mask

def getMaskList(height, width, batchSize, nc, crop=0):
    from os import listdir
    from os.path import isfile, join
    from PIL import Image

    # folder = "masks256"
    folder = "allMasks"
    onlyfiles = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]

    trans = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.CenterCrop((height, width - crop)),
        transforms.ToTensor()
    ])

    masklist = []
    for srcf in onlyfiles:
        mask = Image.open(srcf).convert('RGB')

        mask = trans(mask)
        mask = (mask == 0)
        mask = mask.type(torch.FloatTensor)
        masklist.append(mask)

    return masklist

def initData(img, mask):
    img = img * mask
    # img = img + (mask - 1)
    return img

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)

def fillBatchMask(maskUsed, onemask):
    for i in range(len(maskUsed)):
        maskUsed[i] = onemask
    return maskUsed

def main():
    dataroot = "/media/cad/4ABCAB33BCAB1889/CaptainT/dcgan/datasets/NYU/train"
    # dataroot = "/home/cad/PycharmProjects/ContextEncoder/dataset/clustermix/train"
    # dataroot = '/home/cad/PycharmProjects/ContextEncoder/dataset/test128'
    # dataroot = "/home/cad/PycharmProjects/ContextEncoder/dataset/mix4500/train"
    batchSize = 64
    # inputSize = 256
    width = 320
    height = 240
    channel = 3
    crop = 20
    learningrate = 0.0001
    ngpu = 2
    regular = 0.1
    randommask = False
    iter_times = 7

    # load data
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize((height, width)),
                                   transforms.CenterCrop((height, width-crop)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize,
                                             shuffle=True, num_workers=3, drop_last=True)

    #load mask for the first time
    # mask = randomMask(inputSize, batchSize, channel)
    masklist = getMaskList(height, width, batchSize, channel, crop)
    maskinUse = torch.FloatTensor(batchSize, channel, height, width-crop)

    # load model
    uModel = MASKNET(height, width-crop, channel, nfeature=32, downTimes=5, stride=3)
    uModel.apply(weights_init)
    if ngpu:
        uModel = uModel.cuda()
        uModel = nn.DataParallel(uModel)
    if 0:
        NetUroot = "checkpoints/Umodel_90.pth"
        uModel.load_state_dict(torch.load(NetUroot))

    dModel = MASKDNET(nc=channel, width=width-crop, height=height, nf=32)
    dModel.apply(weights_init)
    if ngpu:
        dModel = dModel.cuda()
        # !!! dModel = nn.DataParallel(dModel)
    if 0:
        NetDroot = "checkpoints/dModel_90.pth"
        dModel.load_state_dict(torch.load(NetDroot))

    # load optimizer
    optimizerU = optim.Adam(uModel.parameters(), lr=learningrate, weight_decay=regular)
    optimizerD = optim.Adam(dModel.parameters(), lr=learningrate)

    # load loss function
    lossF= nn.MSELoss()
    lossF = nn.L1Loss()
    lossBCE = nn.BCELoss()
    fake_label = Variable(torch.FloatTensor(batchSize).fill_(0))
    real_label = Variable(torch.FloatTensor(batchSize).fill_(1))
    if ngpu:
        fake_label = fake_label.cuda()
        real_label = real_label.cuda()

    #training
    lossD_list = []
    lossG_list = []
    avg_D = []
    avg_G = []
    loss_img_list = []
    loss_mask_list = []
    loss_ad_list = []
    lossD_real_list = []
    lossD_fake_list = []
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

    # changeMaskCnt = 0
    # mask_gpu = Variable(mask).cuda()

    for epoch in range(iter_times):
        epochD = []
        epochG = []

        for i, data in enumerate(dataloader, 0):

            # changeMaskCnt = changeMaskCnt + 1
            # if changeMaskCnt % 50 == 0:
            #     mask = randomMask(inputSize, batchSize, channel)
            #     mask_gpu = Variable(mask).cuda()
            if randommask:
                mask = randomMaskBlock(height, width, channel, crop)
            else:
                x = np.random.randint(0, len(masklist))
                mask = masklist[x]

            maskinUse = fillBatchMask(maskinUse, mask)
            mask_gpu = Variable(maskinUse).cuda()


            img_data, _ = data
            img_target = img_data.clone()
            img_target = Variable(img_target)
            img_data = initData(img_data, maskinUse)
            img_data = Variable(img_data)
            if ngpu:
                img_data = img_data.cuda()
                img_target = img_target.cuda()

            # train Unet(G)
            optimizerU.zero_grad()
            output = uModel(img_data, mask_gpu)
            loss_img = lossF(output, img_target)
            loss_mask = lossF(output * mask_gpu, img_target *mask_gpu)
            Doutput = dModel(output, mask_gpu, batchSize)
            loss_ad = lossBCE(Doutput, real_label)
            lossG = 0.8 * loss_img + 0.14 * loss_mask + 0.01 * loss_ad
            lossG.backward()
            optimizerU.step()


            #train Dnet
            optimizerD.zero_grad()
            fake_input = Variable(torch.FloatTensor(output.size()))
            fake_input.data.copy_(output.data)
            real_input = img_target
            fake_input = fake_input.cuda()
            real_input = real_input.cuda()

            fake_output = dModel(fake_input, mask_gpu, batchSize)
            lossD_fake = lossBCE(fake_output, fake_label)
            real_output = dModel(real_input, mask_gpu, batchSize)
            lossD_real = lossBCE(real_output, real_label)
            lossD = 0.5 * lossD_fake + 0.5 * lossD_real

            if i % 1 == 0:
                lossD.backward()
                optimizerD.step()

            # print(loss)
            # print(output.size())
            # print(img_target.size())
            # print(output[0])
            # print(output[0][0].min(), output[0][0].max())
            # print(output[0][1].min(), output[0][1].max())
            # print(output[0][2].min(), output[0][2].max())
            # torch.save(output.data.cpu(), "output.data")
            # print(img_target[0])
            # print(img_target[0][0].min(), img_target[0][0].max())
            # print(img_target[0][1].min(), img_target[0][1].max())
            # print(img_target[0][2].min(), img_target[0][2].max())
            # torch.save(img_target.data.cpu(), "target.data")
            # exit()

            print(('Epoch: [%d / %d][%d / %d] '
                   'LossG: %.4f; LossIMG: %.4f, LossMASK: %.4f, LossAD: %.4f; '
                   '|| LossD: %.4f; LossFAKE: %.4f, LossREAL: %.4f;'
                   ) % (epoch, iter_times, i, len(dataloader),
                      lossG.data.mean(), loss_img.data.mean(), loss_mask.data.mean(), loss_ad.data.mean(),
                      lossD.data.mean(), lossD_fake.data.mean(), lossD_real.data.mean()
                        ))
            # record
            if 1:
                epochD.append(lossD.data.mean())
                epochG.append(lossG.data.mean())

                lossD_list.append(lossD.data.mean())
                lossG_list.append(lossG.data.mean())

                loss_img_list.append(loss_img.data.mean())
                loss_mask_list.append(loss_mask.data.mean())
                loss_ad_list.append(loss_ad.data.mean())

                lossD_real_list.append(lossD_real.data.mean())
                lossD_fake_list.append(lossD_fake.data.mean())

            if i == 0:
                    img_real = img_target.clone()
                    img_fake = output.clone()
                    img_masked = img_data.clone()

            # if i > 10:
            #     break

        # record every epoch
        if 1:
            avg_D.append(np.mean(epochD))
            avg_G.append(np.mean(epochG))

            nrow = int(np.sqrt(batchSize))
            vutils.save_image(img_real.data,
                              'output/unet_epo%d_real.png' % (epoch),
                              nrow=nrow, normalize=True)
            vutils.save_image(img_masked.data,
                              'output/unet_epo%d_masked.png' % (epoch),
                              nrow=nrow, normalize=True)
            vutils.save_image(img_fake.data,
                              'output/unet_epo%d_fake.png' % (epoch),
                              nrow=nrow, normalize=True)

        if epoch % 3 == 0:
            fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(20, 10))
            ax0.plot(lossD_list, label="$lossD$")
            ax0.legend()
            ax1.plot(lossG_list, label="$lossG$")
            ax1.legend()
            plt.savefig("loss_GD.png", dpi=200)
            plt.close(fig)

            fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(20, 10))
            ax0.plot(avg_D, label="$lossD_avg$")
            ax0.legend()
            ax1.plot(avg_G, label="$lossG_avg$")
            ax1.legend()
            plt.savefig("loss_GD_avg.png", dpi=200)
            plt.close(fig)

            fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(30, 10))
            ax0.plot(loss_img_list, label="$loss_img$")
            ax0.legend()
            ax1.plot(loss_mask_list, label="$loss_mask$")
            ax1.legend()
            ax2.plot(loss_ad_list, label="$loss_ad$")
            ax2.legend()
            plt.savefig("loss_G_detail.png", dpi=200)
            plt.close(fig)

            fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(20, 10))
            ax0.plot(lossD_fake_list, label="$loss_fake$")
            ax0.legend()
            ax1.plot(lossD_real_list, label="$loss_real$")
            ax1.legend()
            plt.savefig("loss_D_detail.png", dpi=200)
            plt.close(fig)

        if epoch % 2 == 0:
            torch.save(uModel.state_dict(), 'checkpoints/Umodel' + str(width) + ("RE" if regular > 0 else "") + '_' + str(epoch) + '.pth')
            torch.save(dModel.state_dict(), 'checkpoints/dModel' + str(width) +("RE" if regular > 0 else "") + '_' + str(epoch) + '.pth')

    endT = time.localtime()
    print('begin: %d:%d:%d' % (beginT.tm_hour, beginT.tm_min, beginT.tm_sec))
    print('end: %d:%d:%d' % (endT.tm_hour, endT.tm_min, endT.tm_sec))


if __name__ == "__main__":
    main()