import os
from collections import OrderedDict
import torchvision.transforms as tf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from options.train_options import TrainOptions
import model
from datasets import nyu
from utils import losses
from tqdm import tqdm
from itertools import chain
import pdb

if __name__ == '__main__':
    opt = TrainOptions().parse

    eval_border = 6 if opt.eval_border < 0 else opt.eval_border

    train_split = 'train' if not opt.train_split else opt.train_split
    test_split = 'test' if not opt.test_split else opt.test_split
    lowres_mode = 'center' if not opt.lowres_mode else opt.lowres_mode
    train_transform = nyu.AssembleJointUpsamplingInputs(opt.factor, flip=True, lowres_mode=lowres_mode,
                                                             zero_guidance=opt.zero_guidance,
                                                             output_crop=eval_border, crop=(
            None if opt.train_crop <= 0 else opt.train_crop))
    test_transform = nyu.AssembleJointUpsamplingInputs(opt.factor, flip=False, lowres_mode=lowres_mode,
                                                            zero_guidance=opt.zero_guidance,
                                                            output_crop=eval_border)
    train_dset = nyu.NYUDepthV2(opt.data_root,  transform=train_transform, download=False,
                                         split=train_split, val_ratio=opt.val_ratio, cache_all=True)
    test_dset = nyu.NYUDepthV2(opt.data_root, transform=test_transform, download=False,
                                    split=test_split, val_ratio=opt.val_ratio, cache_all=True)

    # data loader
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=opt.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=opt.test_batch_size, shuffle=True)


    torch.autograd.set_detect_anomaly(True)
    print('Loading the dataset', opt.dataset_mode)

    print('The number of training images = ', len(train_dset))

    print('Creating the model...')

    assert(torch.cuda.is_available())
    use_cuda = True
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.cuda.set_device(device)

    ssgnet = model.SSGNet().cuda()

    ssgnet = torch.nn.DataParallel(ssgnet)
    start = 0

    print('# of ssgnet network parameter:', sum(p.numel() for p in ssgnet.parameters() if p.requires_grad))

    ssgnet_optimizer = torch.optim.Adam(ssgnet.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    writer = SummaryWriter(os.path.join(opt.save_dir))
    print('Will save everything in', os.path.join(opt.save_dir))

    ssgnet.train()

    objective = losses.get_loss_fn(opt.loss)()
    for epoch in range(opt.epoch):
        print('Epoch {}'.format(epoch))
        for i, data in enumerate(tqdm(train_loader)):
            current_iter = start + epoch * len(train_loader) + i + 1
            image, affinity = data
            if opt.is_cuda:
                image, affinity = image.cuda(), affinity.cuda()

            eigen_vs = ssgnet(image)
            e_loss, s_loss = objective(eigen_vs, affinity)

            normalize_e = torch.zeros_like(eigen_vs)
            for m in range(len(eigen_vs[0])):
                normalize_e[0][m] = (eigen_vs[0][m] - torch.min(eigen_vs[0][m])) / (
                    torch.max(eigen_vs[0][m] - torch.min(eigen_vs[0][m])))

            if current_iter % opt.print_freq == 0:
                writer.add_scalar('eigen_loss', e_loss, current_iter)
                writer.add_scalar('spatial_loss', s_loss, current_iter)

            savefilename = os.path.join(opt.save_dir) + '/' + str(current_iter) + '.tar'
            torch.save({
                'epoch': current_iter,
                'state_dict': ssgnet.state_dict()
            }, savefilename)
            loss = e_loss + 40* s_loss

            ssgnet_optimizer.zero_grad()
            loss.backward()
            ssgnet_optimizer.step()

        savefilename = os.path.join(opt.save_dir) + '/' + str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': ssgnet.state_dict()
        }, savefilename)


