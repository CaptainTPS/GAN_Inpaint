import argparse
import random
import os
import numpy as np
import cv2
import copy
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

from torch.utils.serialization import load_lua


parser = argparse.ArgumentParser()
parser.add_argument('--net', required=True)
parser.add_argument('--imDir', required=True)
parser.add_argument('--batchSize', type=int, default=21)
parser.add_argument('--name', default='demo_out')
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--manualSeed',type=int, default=0)
parser.add_argument('--overlapPred',type=int, default=0)
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.name)
except OSError:
    pass

##set seed
if opt.manualSeed == 0:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.gpu:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



## load Context-Encoder
gnet = load_lua(opt.net)
# gnet.apply(weights_init)
if opt.gpu:
    gnet.cuda()

## initialize variables
inputSize = 128
image_ctx = torch.Tensor(opt.batchSize, opt.nc, inputSize, inputSize)
input_image_ctx = torch.Tensor(opt.batchSize, opt.nc, inputSize, inputSize)

## load data
dataset = dset.ImageFolder(root=opt.imDir,
                               transform=transforms.Compose([
                                   transforms.Scale(inputSize),
                                   transforms.CenterCrop(inputSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers), drop_last=True)

## remove center region from input image
xbegin = 1 + inputSize/4 + opt.overlapPred
xend = inputSize/2 + inputSize/4 - opt.overlapPred
ybegin = 1 + inputSize/4 + opt.overlapPred
yend = inputSize/2 + inputSize/4 - opt.overlapPred

for i, data in enumerate(dataloader, 0):
    real_cpu, _ = data
    batch_size = real_cpu.size(0)

    image_ctx.resize_as_(real_cpu).copy_(real_cpu)
    input_image_ctx.resize_as_(real_cpu).copy_(real_cpu)
    input_image_ctx[:,:,ybegin:yend, xbegin:xend] = 0

    if opt.gpu:
        input_image_ctx = input_image_ctx.cuda()

    pred_center = gnet.forward(input_image_ctx)


    print(image_ctx.size())
    print(pred_center.size())

    image_ctx[:,:,ybegin:yend, xbegin:xend] = pred_center[:,:,1 + opt.overlapPred: inputSize/2 - opt.overlapPred, 1 + opt.overlapPred: inputSize/2 - opt.overlapPred]

    if i % 100 == 0:
        vutils.save_image(image_ctx,
                          '%s/image_ctx.png' % opt.name,
                          normalize=True)
        vutils.save_image(input_image_ctx,
                          '%s/input_image_ctx.png' % opt.name,
                          normalize=True)
print('done')