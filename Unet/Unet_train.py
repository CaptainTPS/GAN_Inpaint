from Unet_net import UNET

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
        maskroot = '/home/cad/PycharmProjects/ContextEncoder/mask3.png'
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

def main():
    dataroot = "/home/cad/PycharmProjects/ContextEncoder/dataset/clustermix/train"
    # dataroot = '/home/cad/PycharmProjects/ContextEncoder/dataset/test128'
    batchSize = 128
    inputSize = 128
    channel = 3
    learningrate = 0.0003
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
                                             shuffle=False, num_workers=3, drop_last=True)


    # load mask
    mask = getMask(inputSize, batchSize, channel)

    # load model
    uModel = UNET(inputSize, inputSize, channel)
    uModel.apply(weights_init)
    if ngpu:
        uModel = uModel.cuda()
        uModel = nn.DataParallel(uModel)
    if 0:
        NetUroot = "checkpoints/Umodel_90.pth"
        uModel.load_state_dict(torch.load(NetUroot))

    # load optimizer
    optimizerU = optim.Adam(uModel.parameters(), lr=learningrate)

    # load loss function
    lossF= nn.MSELoss()

    #training
    loss_re = []
    avg_re = []
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

    iter_times = 100
    mask_gpu = Variable(1-mask).cuda()

    for epoch in range(iter_times):
        loss_epoch = []

        for i, data in enumerate(dataloader, 0):
            img_data, _ = data
            img_target = img_data.clone()
            img_target = Variable(img_target)
            img_data = initData(img_data, mask)
            img_data = Variable(img_data)
            if ngpu:
                img_data = img_data.cuda()
                img_target = img_target.cuda()

            output = uModel(img_data)
            loss_img = lossF(output, img_target)
            loss_mask = lossF(output * mask_gpu, img_target *mask_gpu)
            loss = 0.1 * loss_img + 1.5 * loss_mask
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
            loss.backward()
            optimizerU.step()

            print(('Epoch: [%d / %d][%d / %d] Loss: %.4f; LossIMG: %.4f, LossMASK: %.4f'
                   ) % (epoch, iter_times, i, len(dataloader),
                      loss.data.mean(), loss_img.data.mean(), loss_mask.data.mean()))
            # record
            if 1:
                loss_re.append(loss.data.mean())
                loss_epoch.append(loss.data.mean())
                if i == 0:
                    img_real = img_data.clone()
                    img_fake = output.clone()

        # record every epoch
        if 1:
            nrow = int(np.sqrt(batchSize))
            vutils.save_image(img_target.data,
                              'output/unet_epo%d_real.png' % (epoch),
                              nrow=nrow, normalize=True)
            vutils.save_image(img_fake.data,
                              'output/unet_epo%d_fake.png' % (epoch),
                              nrow=nrow, normalize=True)
            avg_re.append(np.mean(loss_epoch))

        if epoch % 3 == 0:
            fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(20, 10))
            ax0.plot(loss_re, label="$loss$")
            ax0.legend()
            ax1.plot(avg_re, label="$loss_avg$")
            ax1.legend()
            plt.savefig("loss_Unet.png", dpi=200)
            plt.close(fig)

        if epoch % 10 == 0:
            torch.save(uModel.state_dict(), 'checkpoints/Umodel_' + str(epoch) + '.pth')

    endT = time.localtime()
    print('begin: %d:%d:%d' % (beginT.tm_hour, beginT.tm_min, beginT.tm_sec))
    print('end: %d:%d:%d' % (endT.tm_hour, endT.tm_min, endT.tm_sec))


if __name__ == "__main__":
    main()