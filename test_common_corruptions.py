import argparse
import numpy as np
import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from networks.resnet_skip_att import *

parser = argparse.ArgumentParser(description='Evaluation against Common corruptions')
parser.add_argument('--model_type', type=str, default='SAT', help=' Type of model: SAT/TRADES/FS')
parser.add_argument('--batch_size', type=int, default=125, help='Batch size')
opt = parser.parse_args()



#GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if args.model_type == 'SAT':
    model = WideResNet(depth=28, num_classes=num_classes, widen_factor=10)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load('pretrained_models/sat')
    model.load_state_dict(checkpoint['net'])
else:
    raise ValueError('Please download the pretrained model (e.g feature scattering or trades) first and modify the evaluation accordingly')
model = model.to(device)
model.eval()


# Distortion dictionary
dd = {'brightness':0, 'contrast':0, 'defocus_blur':0, 'elastic_transform':0,\
      'fog':0, 'gaussian_blur':0,\
      'gaussian_noise':0, 'glass_blur':0, 'impulse_noise':0, 'jpeg_compression':0, 'motion_blur':0,\
      'pixelate':0,  'saturate':0, 'shot_noise':0, 'snow':0, 'spatter':0, 'speckle_noise':0, 'zoom_blur':0}

cc_acc = {}
batch_size = 100
num_batchs = 50000 // batch_size

for d in dd.keys():
    print(d)
    cc_acc[d] = 0

    cc = np.load('CIFAR-10-C/{}.npy'.format(d))
    cc = (np.transpose(cc, (0,3,1,2))/255).astype(np.float32)
    ll = np.load('CIFAR-10-C/labels.npy')
    for batch_id in range(num_batchs):
            #print('At batch:', batch_id)
            batch_img = cc[batch_id * batch_size: (batch_id + 1) * batch_size]
            batch_img = torch.from_numpy(batch_img).cuda()
            batch_label = ll[batch_id * batch_size: (batch_id + 1) * batch_size]
            batch_label = torch.from_numpy(batch_label).cuda()
            batch_label = batch_label.type(torch.int64)
            cc_acc[d] += torch.sum(model((batch_img)).argmax(dim=-1) == batch_label).item()

    cc_acc[d] = (cc_acc[d]/50000)*100
    print(cc_acc)

l = np.array(list(cc_acc.values()))
print(l.mean(), l.var())