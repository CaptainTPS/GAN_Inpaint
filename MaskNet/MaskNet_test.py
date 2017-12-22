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
from PIL import Image

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

def getMaskList(height, width, batchSize, nc, crop, root=None):
    from os import listdir
    from os.path import isfile, join
    from PIL import Image

    folder = "masks256"
    if root != None:
        folder = root
    catgry = 4
    cnt = 120

    onlyfiles = []
    for j in range(1, catgry + 1):
        for i in range(cnt):
            fname = "disparityMask" + str(j) + "_" + str(i) + ".png"
            onlyfiles.append(join(folder, fname))
    # onlyfiles = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]

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

def fillBatchMask(maskUsed, onemask):
    for i in range(len(maskUsed)):
        maskUsed[i] = onemask
    return maskUsed

def getImageList(height, width, batchSize, nc, crop, root=None):
    from os.path import join
    from PIL import Image
    if batchSize != 1:
        print("batch size should be 1")
        exit()

    if root == None:
        root = "images256"

    catgry = 4
    cnt = 120
    onlyfiles = []
    for j in range(1, catgry + 1):
        for i in range(cnt):
            fname = "disparityImg"+ str(j) + "_" +str(i)+".png"
            onlyfiles.append(join(root, fname))
    # onlyfiles = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]

    trans = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.CenterCrop((height, width - crop)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


    imglist = []
    for srcf in onlyfiles:
        img = Image.open(srcf).convert('RGB')
        m1 = trans(img)
        m = torch.FloatTensor(batchSize, nc, height, width - crop)
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

def psnr(input, target, peak):
    batch = len(input)
    channel = len(input[0])
    height = len(input[0][0])
    width = len(input[0][0][0])

    psnrlist = []
    for i in range(batch):
        val = torch.abs(input[i] - target[i])
        val = val * val
        val = torch.sum(val)
        mse = val / channel / height / width
        mse = mse.cpu().data
        psnr = 10.0 * np.log10(peak*peak / mse[0])
        psnrlist.append(psnr)
    return psnrlist

def runFinetuning(masklist, batchSize, channel, height, width, uModel, dModel):
    # dataroot = "/home/cad/PycharmProjects/ContextEncoder/dataset/mix4500/train"
    dataroot = "/home/cad/PycharmProjects/ContextEncoder/dataset/clustermix/train"
    learningrate = 0.0001
    ngpu = 2
    # load data
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize((height, width)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize,
                                             shuffle=True, num_workers=3, drop_last=True)

    # load optimizer
    optimizerU = optim.Adam(uModel.parameters(), lr=learningrate)
    optimizerD = optim.Adam(dModel.parameters(), lr=learningrate)

    maskinUse = torch.FloatTensor(batchSize, channel, height, width)

    # load loss function
    lossF = nn.MSELoss()
    lossF = nn.L1Loss()
    lossBCE = nn.BCELoss()
    fake_label = Variable(torch.FloatTensor(batchSize).fill_(0))
    real_label = Variable(torch.FloatTensor(batchSize).fill_(1))
    if ngpu:
        fake_label = fake_label.cuda()
        real_label = real_label.cuda()

    # training
    iter_times = 1
    maxiter = 1000
    itr = 0

    # changeMaskCnt = 0
    # mask_gpu = Variable(mask).cuda()

    for epoch in range(iter_times):

        for i, data in enumerate(dataloader, 0):
            itr += 1
            # changeMaskCnt = changeMaskCnt + 1
            # if changeMaskCnt % 50 == 0:
            #     mask = randomMask(inputSize, batchSize, channel)
            #     mask_gpu = Variable(mask).cuda()
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
            loss_mask = lossF(output * mask_gpu, img_target * mask_gpu)
            Doutput = dModel(output, mask_gpu, batchSize)
            loss_ad = lossBCE(Doutput, real_label)
            lossG = 0.1 * loss_img + 0.8 * loss_mask + 0.1 * loss_ad
            lossG.backward()
            optimizerU.step()

            # train Dnet
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

            print(('Tuning: [%d / %d][%d / %d] '
                   'LossG: %.4f; LossIMG: %.4f, LossMASK: %.4f, LossAD: %.4f; '
                   '|| LossD: %.4f; LossFAKE: %.4f, LossREAL: %.4f;'
                   ) % (epoch, iter_times, i, len(dataloader),
                        lossG.data.mean(), loss_img.data.mean(), loss_mask.data.mean(), loss_ad.data.mean(),
                        lossD.data.mean(), lossD_fake.data.mean(), lossD_real.data.mean()
                        ))
            if itr == maxiter:
                nrow = int(np.sqrt(batchSize))
                vutils.save_image(real_input.data,
                              'output/tuning_real.png',
                              nrow=nrow, normalize=True)
                vutils.save_image(fake_input.data,
                              'output/tuning_fake.png',
                              nrow=nrow, normalize=True)
            if itr > maxiter:
                return uModel, dModel

    return uModel, dModel

def main(val=False):

    batchSize = 1
    # inputSize = 256
    width = 320
    height = 240
    channel = 3
    crop = 20
    ngpu = 2

    if val:
        dataroot = "/media/cad/4ABCAB33BCAB1889/CaptainT/dcgan/datasets/NYU/val"
        # load data
        dataset = dset.ImageFolder(root=dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize((height, width)),
                                       transforms.CenterCrop((height, width - crop)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize,
                                                 shuffle=False, num_workers=3, drop_last=True)

    #load mask
    # mask = randomMask(inputSize, batchSize, channel)
    # mask = getMask(inputSize,batchSize, channel,
    #                maskroot="/home/cad/PycharmProjects/ContextEncoder/checkerMask6.png")
    masklist = getMaskList(height, width, batchSize, channel, crop, root="masks"+str(width))
    maskinUse = torch.FloatTensor(batchSize, channel, height, width-crop)


    #load images
    if val == False:
        imglist = getImageList(height, width, batchSize, channel, crop, root="images"+str(width))

    # load model
    model_num = 10
    uModel = MASKNET(height, width-crop, channel, nfeature=32, downTimes=5, stride=3)
    if ngpu:
        uModel = uModel.cuda()
        uModel = nn.DataParallel(uModel)
    NetUroot = 'checkpoints/Umodel' + str(width) + '_' + str(model_num) + '.pth'
    uModel.load_state_dict(torch.load(NetUroot))


    dModel = MASKDNET(nc=channel, width=width-crop, height=height, nf=32)
    if ngpu:
        dModel = dModel.cuda()
        # !!! dModel = nn.DataParallel(dModel)
    NetDroot = 'checkpoints/dModel' + str(width) + '_' + str(model_num) + '.pth'
    dModel.load_state_dict(torch.load(NetDroot))

    #fine tuning
    # uModel, dModel = runFinetuning(masklist, 64, channel, height, width, uModel, dModel)

    # uModel.eval()
    # dModel.eval()

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
    psnr_list = []
    img_real = None
    img_fake = None
    beginT = time.localtime()

    try:
        os.mkdir('output')
    except OSError:
        pass
    try:
        os.mkdir("Result"+ str(width))
    except OSError:
        pass
    try:
        os.mkdir('checkpoints')
    except OSError:
        pass

    #prepare trans for storage
    trans = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        transforms.ToPILImage(),
    ])

    if val:
        for i, data in enumerate(dataloader, 0):
            mask = masklist[np.random.randint(0, len(masklist))]
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
            output = uModel(img_data, mask_gpu)
            loss_img = lossF(output, img_target)
            loss_mask = lossF(output * mask_gpu, img_target * mask_gpu)
            Doutput = dModel(output, mask_gpu, batchSize)
            loss_ad = lossBCE(Doutput, real_label)
            lossG = 0.1 * loss_img + 0.8 * loss_mask + 0.1 * loss_ad

            print(('Validation: [%d / %d] '
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

            psnr_epoch = psnr(img_real, img_fake, 2)
            psnr_list = psnr_list + psnr_epoch

            # record every epoch
            if i < 10:
                nrow = int(np.sqrt(batchSize))
                vutils.save_image(img_real.data,
                                  'output/val_%d_real.png' % (i),
                                  nrow=nrow, normalize=True)
                vutils.save_image(img_masked.data,
                                  'output/val_%d_masked.png' % (i),
                                  nrow=nrow, normalize=True)
                vutils.save_image(img_fake.data,
                                  'output/val_%d_fake.png' % (i),
                                  nrow=nrow, normalize=True)
    else:
        for i in range(len(masklist)):
            if 1:
                mask = masklist[i]
                maskinUse = fillBatchMask(maskinUse, mask)
                mask_gpu = Variable(maskinUse).cuda()

            # img_data, _ = data

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
            Doutput = dModel(output, mask_gpu, batchSize)
            loss_ad = lossBCE(Doutput, real_label)
            lossG = 0.1 * loss_img + 0.8 * loss_mask + 0.1 * loss_ad


            print(('TEST: [%d / %d] '
                   'LossG: %.4f; LossIMG: %.4f, LossMASK: %.4f, LossAD: %.4f; '
                   ) % (i, len(masklist),
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

            psnr_epoch = psnr(img_real, img_fake, 2)
            psnr_list = psnr_list + psnr_epoch

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
            for x in range(batchSize):
                img = trans(img_fake.cpu().data[x])
                s = img.size
                background = Image.new('RGB', size=(s[0] + crop, s[1]), color=(0, 0, 0, 0))
                background.paste(img, (crop / 2, 0))
                background.save("Result"+ str(width) +"/" + str(i) + '_' + str(x) + ".png")

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

        plt.plot(psnr_list, label="$psnr_list$")
        plt.legend()
        plt.savefig("test_psnr.png", dpi=200)
        plt.close()

    endT = time.localtime()
    print('begin: %d:%d:%d' % (beginT.tm_hour, beginT.tm_min, beginT.tm_sec))
    print('end: %d:%d:%d' % (endT.tm_hour, endT.tm_min, endT.tm_sec))


if __name__ == "__main__":
    main(val=True)