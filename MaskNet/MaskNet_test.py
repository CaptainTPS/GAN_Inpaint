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

def getMaskList(fineSize, batchSize, nc, root=None):
    from os import listdir
    from os.path import isfile, join

    folder = "masks256"
    if root != None:
        folder = root
    catgry = 5
    cnt = 120

    onlyfiles = []
    for j in range(1, catgry + 1):
        for i in range(cnt):
            fname = "disparityMask" + str(j) + "_" + str(i) + ".png"
            onlyfiles.append(join(folder, fname))
    # onlyfiles = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]

    masklist = []
    for srcf in onlyfiles:
        mask = cv2.imread(srcf, cv2.IMREAD_GRAYSCALE)
        if mask.shape != fineSize:
            mask = cv2.resize(mask, fineSize)
            print("change mask size")

        mask = torch.from_numpy(mask)
        mask = (mask == 255)

        mask = mask.type(torch.FloatTensor)

        masklist.append(mask)

    return masklist

def fillBatchMask(maskUsed, oneChannel):
    for i in range(len(maskUsed)):
        for j in range(len(maskUsed[i])):
            maskUsed[i][j] = oneChannel
    return maskUsed

def getImageList(height, width, batchSize, nc, root=None):
    from os.path import join
    from PIL import Image
    if batchSize != 1:
        print("batch size should be 1")
        exit()

    if root == None:
        root = "images256"

    catgry = 5
    cnt = 120
    onlyfiles = []
    for j in range(1, catgry + 1):
        for i in range(cnt):
            fname = "disparityImg"+ str(j) +"_"+str(i)+".png"
            onlyfiles.append(join(root, fname))
    # onlyfiles = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]

    trans = transforms.Compose([
        transforms.Scale(min(height, width)),
        transforms.CenterCrop((height, width)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


    imglist = []
    for srcf in onlyfiles:
        img = Image.open(srcf).convert('RGB')
        m1 = trans(img)
        m = torch.FloatTensor(batchSize, nc, height, width)
        m[0] = m1
        imglist.append(m)

    return imglist


def saveOneImage(tensor, id, root=None):
    from PIL import Image

    if root == None:
        root = "result256"

    ndarr = tensor.numpy()
    ndarr = ndarr[0]
    ndarr = ndarr * 255

    sh = ndarr.shape
    toimage = np.zeros((sh[1], sh[2], sh[0]), dtype=np.uint8)
    for i in range(sh[0]):
        for j in range(sh[1]):
            for k in range(sh[2]):
                toimage[j][k][i] = ndarr[i][j][k]
    # exit()
    im = Image.fromarray(toimage, "RGB")
    im.save(root + "/" + str(id) + ".png")
    # vutils.save_image(mask, 'test_mask.png', nrow=nrow, normalize=True, padding=0)

def main():
    # dataroot = "/home/cad/PycharmProjects/ContextEncoder/dataset/dining_room/d_val"
    # dataroot = '/home/cad/PycharmProjects/ContextEncoder/dataset/test128'
    dataroot = "/home/cad/PycharmProjects/ContextEncoder/dataset/disparityTest/test640"
    batchSize = 1
    # inputSize = 256
    height = 480
    width = 640
    channel = 3
    ngpu = 2

    # load data
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(min(height, width)),
                                   # transforms.Scale(inputSize),
                                   transforms.CenterCrop((height, width)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize,
                                             shuffle=False, num_workers=3, drop_last=True)

    #load mask
    # mask = randomMask(inputSize, batchSize, channel)
    # mask = getMask(inputSize,batchSize, channel,
    #                maskroot="/home/cad/PycharmProjects/ContextEncoder/checkerMask6.png")
    masklist = getMaskList((height, width), batchSize, channel,root="allMasks")
    maskinUse = torch.FloatTensor(batchSize, channel, height, width)


    #load images
    imglist = getImageList(height, width, batchSize, channel, root=dataroot + "/t640")

    # load model
    uModel = MASKNET(height, width, channel, nfeature=32, downTimes=5, stride=3)
    if ngpu:
        uModel = uModel.cuda()
        uModel = nn.DataParallel(uModel)
    NetUroot = "checkpoints/Umodel_10.pth"
    uModel.load_state_dict(torch.load(NetUroot))
    uModel.eval()

    dModel = MASKDNET(batch=batchSize, nc=channel, width=width, height=height, nf=32)
    if ngpu:
        dModel = dModel.cuda()
        # !!! dModel = nn.DataParallel(dModel)
    NetDroot = "checkpoints/dModel_10.pth"
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

    for i, data in enumerate(dataloader, 0):
        if 1:
            mask = masklist[i]
            maskinUse = fillBatchMask(maskinUse, mask)
            mask_gpu = Variable(maskinUse).cuda()

        img_data, _ = data

        if 1:
            img_data = imglist[i]

        img_target = img_data.clone()
        img_target = Variable(img_target)
        img_data = initData(img_data, maskinUse)
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
        img_masked = img_data.clone()
        img_fake = output.clone()

        # record every epoch
        if i < 10:
            nrow = int(np.sqrt(batchSize))
            vutils.save_image(img_real.data,
                              'output/test_%d_real.png' % (i),
                              nrow=nrow, normalize=True)
            vutils.save_image(img_masked.data,
                              'output/test_%d_masked.png' % (i),
                              nrow=nrow, normalize=True)
            vutils.save_image(img_fake.data,
                              'output/test_%d_fake.png' % (i),
                              nrow=nrow, normalize=True)

        vutils.save_image(img_fake.data, "allResult/" + str(i) + ".png", normalize=True)
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