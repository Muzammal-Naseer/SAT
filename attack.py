import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import copy
import numpy as np

def local_attack(model,  img, label, eps, attack_type, iters, criterion, random_restart=True):

    if random_restart:
        adv = img.detach() + torch.zeros_like(img).uniform_(-eps, eps)
    adv.requires_grad = True

    if attack_type == 'fgsm':
        iterations = 1
    else:
        iterations = iters

    if attack_type == 'pgd':
        step = 2 / 255
    else:
        step = eps / iterations
    adv_noise = 0
    for j in range(iterations):

        adv_out = model(adv)
        loss = criterion(adv_out, label)
        loss.backward()

        if attack_type == 'mifgsm':
            adv.grad = adv.grad / torch.mean(torch.abs(adv.grad), dim=(1, 2, 3), keepdim=True)
            adv_noise = adv_noise + adv.grad
        else:
            adv_noise = adv.grad

        adv.data = adv.data + step * adv_noise.sign()


        # Projection
        if attack_type == 'pgd':
            adv.data = torch.min(torch.max(adv.data, img - eps), img + eps)
        adv.data.clamp_(0.0, 1.0)

        adv.grad.data.zero_()

    return adv.detach()