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
import models
from dataset import BSDS68
from tqdm import tqdm
from itertools import chain
import pdb
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

if __name__ == '__main__':
    opt = TestOptions().parse

    def test(model, test_loader):
        model.eval()
        psnr_epoch = 0
        ssim_epoch = 0
        for i, data in enumerate(tqdm(test_loader)):
            image = data[0].cuda()
            name = data[1]
            noise_img, _, _ = add_noise(image, opt)

            noise_img = noise_img - 0.5
            h, w = noise_img.shape[2], noise_img.shape[3]
            pw, ph = (w + 31) // 32 * 32 - w, (h + 31) // 32 * 32 - h
            noise_img = F.pad(noise_img, (0, pw, 0, ph), mode='reflect')
            output = model(noise_img)
            if pw != 0:
                output = output[..., :, :-pw]
            if ph != 0:
                output = output[..., :-ph, :]
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
            save_dir = opt.save_dir
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            noise_img = torch.clamp(noise_img+0.5, 0.0, 1.0)

            for j in range(len(output)):
                torchvision.utils.save_image(noise_img[j], save_dir+'/noise_'+name[j])
                torchvision.utils.save_image(output[j], save_dir + '/output_'+name[j])

        psnr_epoch = psnr_epoch / len(test_loader)
        ssim_epoch = ssim_epoch / len(test_loader)
        print('PSNR: ', psnr_epoch)
        print('SSIM: ', ssim_epoch)


    def add_noise(input_var, opt):
        noise_type = opt.noise_type
        noise_level = opt.noise_level
        assert input_var.is_cuda
        if noise_type == 'g':
            if isinstance(noise_level, list):  # random noise level
                sigma = 25.0
                sigma = sigma / 255.0
            else:
                assert NotImplementedError
            sigma = opt.noise_level

            noise = torch.cuda.FloatTensor(input_var.shape)
            noise.normal_(0, sigma)

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


    test_dataset = BSDS68.BSDS68()
    torch.autograd.set_detect_anomaly(True)
    print('Loading the dataset')

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(opt.num_workers),
        drop_last=False)

    print('Creating the model...')

    eigennet = models.unet_gray.UNet_n2n_un(in_channels=4, out_channels=1).cuda()
    eigennet = torch.nn.DataParallel(eigennet)
    state_dict = torch.load(opt.ssg_pretrained)['state_dict']
    eigennet.load_state_dict(state_dict)
    eigennet.eval()
    test(eigennet, test_loader)



