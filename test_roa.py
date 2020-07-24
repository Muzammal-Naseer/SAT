
import argparse
import numpy as np
import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from networks.wideresnet import WideResNet
from ROA import ROA

parser = argparse.ArgumentParser(description='ROA Robustness')
parser.add_argument('--data_type', type=str, default='cifar10', help='cifar10/100, svhn')
parser.add_argument('--model_type', type=str, default='SAT', help=' Type of model: SAT/TRADES/FS')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
parser.add_argument('--ROA_ES', action='store_true', help='If true then do exhasutive search otherwise gradient search')
args = parser.parse_args()
print(args)

# GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


batch_size = args.batch_size
if args.data_type == 'cifar10':
    testset = torchvision.datasets.CIFAR10('../data', train=False, download=True,transform=transforms.ToTensor())
elif args.data_type == 'cifar100':
    testset = torchvision.datasets.CIFAR100('../data', train=False, download=True, transform=transforms.ToTensor())
elif args.data_type == 'svhn':
    testset = torchvision.datasets.SVHN('../data', split='test', download=True, transform=transforms.ToTensor())
print('Number of test samples:', len(testset))
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
size = len(testset)


# Load model
if args.data_type =='cifar100':
    num_classes=100
else:
    num_classes = 10


if args.model_type == 'SAT':
    model = WideResNet(depth=28, num_classes=num_classes, widen_factor=10)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load('pretrained_models/sat')
    model.load_state_dict(checkpoint['net'])
else:
    raise ValueError('Please download the pretrained model (e.g feature scattering or trades) first and modify the evaluation accordingly')
model = model.to(device)
model.eval()

adv_acc = 0
clean_acc = 0

for idx, (img, label) in enumerate(testloader):
    img, label = img.to(device), label.to(device)

    out_org = model(img)
    predictions = out_org.argmax(dim=-1)
    clean_acc += torch.sum(predictions == label).item()


    # initialize the ROA module
    roa = ROA.ROA(model, 32)

    # These are defualt ROA settings but you can modify them as well. Our model perform significantly better than Trades
    # Feature scattering or Madry's model
    learning_rate = 0.1
    iterations = 7
    ROAwidth = 11
    ROAheight = 11
    skip_in_x = 2
    skip_in_y = 2
    potential_nums = 5
    if args.ROA_ES:
        adv = roa.exhaustive_search(img, label, learning_rate, iterations, ROAwidth, ROAheight, skip_in_x,
                                    skip_in_y, potential_nums)
    else:
        adv = roa.gradient_based_search(img, label, learning_rate,iterations, ROAwidth, ROAheight, skip_in_x, skip_in_y, potential_nums)

    if idx == 0:
        vutils.save_image(vutils.make_grid(adv, normalize=True, scale_each=True), 'adv.png')
        vutils.save_image(vutils.make_grid(img, normalize=True, scale_each=True), 'img.png')
    predictions = model(adv).argmax(dim=-1)
    adv_acc += torch.sum( predictions == label).item()
    print('batch: {0}\t l inf distance:{1:.5f} \t adv_max:{2:.2f} \t img_max:{3}'.format(idx, (adv-img).max().item()*255, adv.max().item(), img.max().item()))

print('Clean accuracy:{0:.3%}\t Adversarial accuracy:{1:.3%}'.format(clean_acc/size, adv_acc/size))
print('='*100)
