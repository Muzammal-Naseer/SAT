
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
from attack import *

parser = argparse.ArgumentParser(description='ROA Robustness')
parser.add_argument('--data_type', type=str, default='cifar10', help='cifar10/100, svhn')
parser.add_argument('--model_type', type=str, default='SAT_s', help=' Type of model: SAT-s/SAT-m/TRADES/FS')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
parser.add_argument('--attack_type', type=str, default='pgd', help='fgsm, pgd')
parser.add_argument('--iters', type=int, default= 10, help='Number of iterations for ifgsm, mifgsm, pgd')
parser.add_argument('--eps', type=float, default=8, help='Allowed perturbation within l infinity norm')
parser.add_argument('--loss_type', type=str, default='CE', help='CE, CW')
parser.add_argument('--random_start', action='store_false', help='Use Random restart')
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


if args.model_type == 'SAT-s': # Single step SAT training
    model = WideResNet(depth=28, num_classes=num_classes, widen_factor=10)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load('pretrained_models/sat')
    model.load_state_dict(checkpoint['net'])
elif args.model_type == 'SAT-m': # Multi step SAT training
    model = WideResNet(depth=28, num_classes=num_classes, widen_factor=10)
    model.load_state_dict(torch.load('pretrained_models/sat_multi_step.pt'))
else:
    raise ValueError('Please download the pretrained model (e.g feature scattering or trades) first and modify the evaluation accordingly')
model = model.to(device)
model.eval()


# Setu loss
class CWLoss(nn.Module):
    def __init__(self, num_classes, margin=50, reduce=True):
        super(CWLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.reduce = reduce
        return

    def forward(self, logits, targets):
        """
        :param inputs: predictions
        :param targets: target labels
        :return: loss
        """
        onehot_targets = one_hot_tensor(targets, self.num_classes)

        self_loss = torch.sum(onehot_targets * logits, dim=1)
        other_loss = torch.max(
            (1 - onehot_targets) * logits - onehot_targets * 1000, dim=1)[0]

        loss = -torch.sum(torch.clamp(self_loss - other_loss + self.margin, 0))

        if self.reduce:
            sample_num = onehot_targets.shape[0]
            loss = loss / sample_num

        return loss

if args.loss_type =='CE':
    criterion = nn.CrossEntropyLoss()
elif args.loss_type =='CW':
    criterion = CWLoss(num_classes=num_classes)


adv_acc = 0
clean_acc = 0

for idx, (img, label) in enumerate(testloader):
    img, label = img.to(device), label.to(device)

    out_org = model(img)
    predictions = out_org.argmax(dim=-1)
    clean_acc += torch.sum(predictions == label).item()

    adv = local_attack(model, img, label, eps=args.eps / 255, attack_type=args.attack_type,iters=args.iters, criterion=criterion)

    if idx == 0:
        vutils.save_image(vutils.make_grid(adv, normalize=True, scale_each=True), 'adv.png')
        vutils.save_image(vutils.make_grid(img, normalize=True, scale_each=True), 'img.png')
    predictions = model(adv).argmax(dim=-1)
    adv_acc += torch.sum( predictions == label).item()
    print('batch: {0}\t l inf distance:{1:.5f} \t adv_max:{2:.2f} \t img_max:{3}'.format(idx, (adv-img).max().item()*255, adv.max().item(), img.max().item()))

print('Clean accuracy:{0:.3%}\t Adversarial accuracy:{1:.3%}'.format(clean_acc/size, adv_acc/size))
print('='*100)
