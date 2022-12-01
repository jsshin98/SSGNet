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

from options.train_options import TrainOptions
import models
from dataset import ImageNet_gray as ImageNet
from tqdm import tqdm
from itertools import chain
import pdb
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

if __name__ == '__main__':
    opt = TrainOptions().parse

    def train(model, copy_model, train_loader, epoch, iter, optimizer, scheduler, opt):
        model.train()
        curr_iter = iter
        epoch_loss = 0
        for i, data in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()

            image = data[0].cuda()
            image, _, _ = add_noise(image, opt)
            image = image - 0.5
            if epoch > 0:
                with torch.no_grad():
                    gt, _ = copy_model(image)
            else:
                gt = image
            noise_img, _, _ = add_noise(gt, opt)
            output = model(noise_img)
            loss = F.mse_loss(output, gt)
            loss.backward()
            optimizer.step()
            curr_iter += 1
            if curr_iter > 500000:
                return epoch_loss, curr_iter, 1
            scheduler.step()
            if curr_iter % 1000 == 0:
                print(loss)
        return epoch_loss/len(train_loader), curr_iter, 0


    def valid(model, val_loader, epoch):
        model.eval()
        psnr_epoch = 0
        ssim_epoch = 0
        for i, data in enumerate(tqdm(val_loader)):
            image = data[0].cuda()
            name = data[1]
            noise_img, _, _ = add_noise(image, opt)

            noise_img = noise_img - 0.5
            output = model(noise_img)
            output = output + 0.5

            image_np = image.permute([0, 2, 3, 1]).detach().cpu().numpy()
            output_np = output.permute([0, 2, 3, 1]).detach().cpu().numpy()
            image_np = np.clip(image_np*255.0 + 0.5, 0, 255).astype(np.uint8)
            output_np = np.clip(output_np*255.0+0.5, 0, 255).astype(np.uint8)
            for j in range(len(output_np)):
                denoised = output_np[j]
                gt = image_np[j]
                psnr = compare_psnr(denoised, gt, data_range=255)
                ssim = compare_ssim(denoised, gt, data_range=255, multichannel=True)
                psnr_epoch += psnr
                ssim_epoch += ssim
            save_dir = opt.save_dir + str(epoch)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            for j in range(len(output)):
                torchvision.utils.save_image(output[j], save_dir + '/output_'+name[j])
        psnr_epoch = psnr_epoch / len(val_loader)
        ssim_epoch = ssim_epoch / len(val_loader)
        print('PSNR: ', psnr_epoch)
        print('SSIM: ', ssim_epoch)


    def basic_loss(img, truth, type='tv'):
        if type == 'l1':
            return torch.mean(torch.abs(img - truth))
        elif type == 'l2':
            return torch.mean((img - truth) ** 2)
        elif type == 'psnr':

            return 10 / np.log(10) * torch.log(((img - truth) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
        else:
            assert 'not support'


    def add_noise(input_var, opt):
        noise_type = opt.noise_type
        noise_level = opt.noise_level
        assert input_var.is_cuda
        if noise_type == 'g':
            if isinstance(noise_level, list):  # random noise level
                sigma = np.random.uniform(noise_level[0], noise_level[1], size=input_var.shape[0])
                sigma = sigma / 255.0
            else:
                assert NotImplementedError

            noise = torch.cuda.FloatTensor(input_var.shape)

            for idx in range(input_var.shape[0]):
                noise[idx].normal_(0, sigma[idx])

            input_noise = input_var + noise
            return input_noise.float(), sigma, None

        elif noise_type == 'line':
            sigma = noise_level / 255.0
            b, c, h, w = input_var.shape
            line_noise = torch.cuda.FloatTensor(b, 1, h, 1).normal_(0, sigma)
            input_noise = input_var + torch.cuda.FloatTensor(input_var.shape).fill_(1) * line_noise
            return input_noise.float(), sigma, None

        elif noise_type in ['binomial', 'impulse']:
            sigma = noise_level
            b, c, h, w = input_var.shape
            mask_shape = (b, 1, h, w) if noise_type == 'binomial' else (b, c, h, w)
            mask = torch.cuda.FloatTensor(*mask_shape).uniform_(0, 1)
            mask = mask * torch.cuda.FloatTensor(b, 1, 1, 1).uniform_(0,
                                                                      sigma)  # add different noise level for each frame
            mask = 1 - torch.bernoulli(mask)
            input_noise = input_var * mask
            return input_noise.float(), sigma, None

        elif 'scn' in noise_type:  # spatially correlated noise
            sigma = noise_level / 255.0
            b, c, h, w = input_var.shape
            input_noise = input_var.clone()
            n_type = int(noise_type.split('-')[-1])

            def img_add_noise(img, n_type, sigma, h, w):
                one_image_noise, _, _ = get_experiment_noise('g%d' % n_type, sigma, np.random.randint(1e9), (h, w, 3))
                one_image_noise = torch.FloatTensor(one_image_noise).to(img.device).permute(2, 0,
                                                                                            1)  # for dist training
                img = img + one_image_noise
                return img

            pool = ThreadPool(processes=4 if b >= 4 else b)
            result = []
            for i in range(b):  # no need to
                result.append(pool.apply_async(img_add_noise, (input_var[i], n_type, sigma, h, w)))
            pool.close()
            pool.join()
            for i, res in enumerate(result):
                input_noise[i] = res.get()

            return input_noise.float(), sigma, None

    train_dataset = ImageNet.ImageNet(opt, train_mode='train')
    val_dataset = ImageNet.ImageNet(opt, train_mode='val')
    torch.autograd.set_detect_anomaly(True)
    print('Loading the dataset')
    print('The number of training images = ', len(train_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.num_workers),
        drop_last=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(opt.num_workers),
        drop_last=False)

    print('Creating the model...')

    eigennet = models.unet_gray.UNet_n2n_un(in_channels=2, out_channels=1).cuda()
    copy_model = models.unet_gray.UNet_n2n_un(in_channels=2, out_channels=1).cuda()
    eigennet = torch.nn.DataParallel(eigennet)
    copy_model = torch.nn.DataParallel(copy_model).eval()
    start = 0

    print('# of eigen network parameter:', sum(p.numel() for p in eigennet.parameters() if p.requires_grad))

    optimizer = torch.optim.Adam(eigennet.parameters(), lr=3e-4, weight_decay=0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500000)
    writer = SummaryWriter(os.path.join(opt.save_dir))
    print('Will save everything in', os.path.join(opt.save_dir))

    eigennet.train()
    current_iter = 0
    for epoch in range(100):
        print('Epoch {}'.format(epoch))
        copy_model.load_state_dict(eigennet.state_dict())
        loss, current_iter, end_point = train(eigennet, copy_model, train_loader, epoch, current_iter, optimizer, scheduler, opt)
        if end_point == 1:
            print('Training End ==============================================')
            savefilename = os.path.join(opt.save_dir) + '/final.tar'
            torch.save({
                'epoch': epoch,
                'state_dict': eigennet.state_dict()
            }, savefilename)
            break

        writer.add_scalar('loss', loss, epoch)
        savefilename = os.path.join(opt.save_dir) + '/' + str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': eigennet.state_dict()
        }, savefilename)
        valid(eigennet, val_loader, epoch)


