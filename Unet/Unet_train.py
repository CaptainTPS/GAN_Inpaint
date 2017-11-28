from Unet_net import UNET

import time
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

def main():
    dataroot = "/home/cad/PycharmProjects/ContextEncoder/dataset/clustermix/train"
    batchSize = 128
    inputSize = 128
    channel = 3
    learningrate = 0.001
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


    # load model
    uModel = UNET(inputSize, inputSize, channel)
    if ngpu:
        uModel = uModel.cuda()
        uModel = nn.DataParallel(uModel)

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

    iter_times = 10
    for epoch in range(iter_times):
        loss_epoch = []

        for i, data in enumerate(dataloader, 0):
            img_data, _ = data
            img_data = Variable(img_data)
            if ngpu:
                img_data = img_data.cuda()

            output = uModel(img_data)
            loss = lossF(output, img_data)
            loss.backward()
            optimizerU.step()

            print(('Epoch: [%d / %d][%d / %d] Loss: %.3f'
                   ) % (epoch, iter_times, i, len(dataloader),
                      loss.data.mean()))
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
            vutils.save_image(img_real,
                              'output/unet_epo%d_real.png' % (epoch),
                              nrow=nrow, normalize=True)
            vutils.save_image(img_fake.data,
                              'output/unet_epo%d_fake.png' % (epoch),
                              nrow=nrow, normalize=True)
            avg_re.append(np.mean(loss_epoch))

        if epoch % 10 == 0:
            fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(20, 10))
            ax0.plot(loss_re, label="$loss$")
            ax0.legend()
            ax1.plot(avg_re, label="$loss_avg$")
            ax1.legend()
            plt.savefig("loss_Unet.png", dpi=200)
            plt.close(fig)

    endT = time.localtime()
    print('begin: %d:%d:%d' % (beginT.tm_hour, beginT.tm_min, beginT.tm_sec))
    print('end_preTrain: %d:%d:%d' % (endT.tm_hour, endT.tm_min, endT.tm_sec))


if __name__ == "__main__":
    main()