import os
from collections import OrderedDict
import torchvision.transforms as tf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.utils
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from options.test_options import TestOptions
import model
from datasets import nyu_test as nyu
from utils import losses
from tqdm import tqdm
from itertools import chain
import pdb
import time

if __name__ == '__main__':
    opt = TestOptions().parse

    eval_border = 6 if opt.eval_border < 0 else opt.eval_border

    test_split = 'test' if not opt.test_split else opt.test_split
    lowres_mode = 'center' if not opt.lowres_mode else opt.lowres_mode

    test_transform = nyu.AssembleJointUpsamplingInputs(opt.factor, flip=False, lowres_mode=lowres_mode,
                                                            zero_guidance=opt.zero_guidance,
                                                            output_crop=eval_border, crop=(
            None if opt.test_crop <= 0 else opt.test_crop))

    test_dset = nyu.NYUDepthV2(opt.data_root, transform=test_transform, download=False,
                                    split=test_split, val_ratio=opt.val_ratio, cache_all=True)

    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=opt.test_batch_size, shuffle=True)


    torch.autograd.set_detect_anomaly(True)
    print('Loading the dataset', opt.dataset_mode)

    print('The number of testing images = ', len(test_dset))

    print('Creating the model...')

    assert(torch.cuda.is_available())
    use_cuda = True
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.cuda.set_device(device)

    ssgnet = model.SSGNet().cuda()

    ssgnet = torch.nn.DataParallel(ssgnet)
    state_dict = torch.load(opt.ssgnet_pretrained)['state_dict']
    ssgnet.load_state_dict(state_dict)
    start = 0

    print('# of ssgnet network parameter:', sum(p.numel() for p in ssgnet.parameters() if p.requires_grad))

    ssgnet.eval()

    for i, data in enumerate(tqdm(test_loader)):
        image = data
        if opt.is_cuda:
            image = image.cuda()
        eigen_vs = ssgnet(image)
        if opt.save_result:
            normalize_e = torch.zeros_like(eigen_vs)
            for m in range(len(eigen_vs[0])):
                normalize_e[0][m] = (eigen_vs[0][m] - torch.min(eigen_vs[0][m])) / (
                    torch.max(eigen_vs[0][m] - torch.min(eigen_vs[0][m])))
                torchvision.utils.save_image(normalize_e[:, m], os.path.join(opt.save_result, str(i)+'_eigen'+str(m)+'.png' ))

