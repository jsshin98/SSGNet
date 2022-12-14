import os
import os.path
import pdb
import random
import math
import collections

import h5py
import numpy as np
import imageio
from skimage.transform import resize
import torch as th
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from tools import flowlib

from PIL import Image
import scipy
import torch
from torch.utils import data
from datasets.transform import *
import torchvision

from scipy import ndimage, sparse
import scipy.sparse.linalg as sparse_linalg
import pdb
import os
from tqdm import tqdm
import pymatting
from scipy.linalg import eigh

nyu_url = 'http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat'
nyu_md5 = 'a3a66613390119e6d46827a7aaa3c132'
nyu_total = 1449

sintel_url = 'http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip'
sintel_md5 = '2d2836c2c6b4fb6c9d2d2d58189eb014'
sintel_split = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'meta', 'Sintel_train_val.txt')



def get_lowres(im_np, factor, mode='center'):
    """
    mode: 'bicubic', 'bilinear', 'nearest', 'last', 'center'
    """
    if im_np.ndim == 3:
        im_np = im_np.transpose(1, 2, 0)

    h0, w0 = im_np.shape[:2]
    h, w = int(math.ceil(h0 / float(factor))), int(math.ceil(w0 / float(factor)))

    if h0 != h * factor or w0 != w * factor:
        im_np = resize(im_np, (h * factor, w * factor), order=1, mode='reflect', clip=False, preserve_range=True,
                       anti_aliasing=True)

    if mode in ('last', 'center'):
        if mode == 'last':
            idxs = (slice(factor - 1, None, factor),) * 2
        else:
            assert mode == 'center'
            idxs = (slice(int((factor - 1) // 2), None, factor),) * 2
        lowres = im_np[idxs].copy()
    else:
        order_dict = {'nearest': 0, 'bilinear': 1, 'bicubic': 3}
        lowres = resize(im_np, (h, w), order=order_dict[mode], mode='reflect', clip=False, preserve_range=True,
                        anti_aliasing=True)

    if lowres.ndim == 3:
        lowres = lowres.transpose(2, 0, 1)

    return lowres

def load_nyu_depth_v2(data_root, fields=('images', 'depths'), selection=None, download=False):
    # 'images' : 1449 x 3 x 640 x 480, uint8, [0, 255]
    # 'depths' : 1449 x 640 x 480, float32, [0, 10.0]
    if selection is None:
        selection = slice(None)
    file_path = os.path.join(data_root, os.path.basename(nyu_url))
    if not os.path.exists(file_path):
        if download:
            os.makedirs(data_root, exist_ok=True)
            download_url(nyu_url, data_root, os.path.basename(nyu_url), nyu_md5)
        else:
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
    data = []
    with h5py.File(file_path, 'r') as f:
        for field_name in fields:
            d = f[field_name][selection]
            data.append(d.transpose(*(tuple(range(d.ndim - 2)) + (-1, -2))))  # change from W x H to H x W
    return tuple(data)


def open_nyu_depth_v2(data_root, fields=('images', 'depths'), download=False):
    # 'images' : 1449 x 3 x 640 x 480, uint8, [0, 255]
    # 'depths' : 1449 x 640 x 480, float32, [0, 10.0]
    file_path = os.path.join(data_root, os.path.basename(nyu_url))
    if not os.path.exists(file_path):
        if download:
            os.makedirs(data_root, exist_ok=True)
            download_url(nyu_url, data_root, os.path.basename(nyu_url), nyu_md5)
        else:
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
    headers = []
    f = h5py.File(file_path, 'r')
    for field_name in fields:
        headers.append(f[field_name])
    return tuple(headers), f


class NYUDepthV2(Dataset):
    def __init__(self, root, split='train', fields=('images', 'depths'), val_ratio=0.0, transform=None,
                 cache_all=False, cache_loaded=True, download=False):
        super(NYUDepthV2, self).__init__()
        self.root = os.path.expanduser(root)
        self.type = 'depth'
        self.split = split
        self.fields = fields
        self.val_ratio = val_ratio
        self.transform = transform
        self.cache_all = cache_all
        self.cache_loaded = cache_loaded
        self.cache = dict()

        self.image_files_transforms = image_transforms()


        if split == 'train':
            self.total = int(1000 * (1 - val_ratio))
            self.offset = 0
        elif split == 'val':
            self.total = int(1000 * val_ratio)
            self.offset = 1000 - self.total
        elif split == 'test':
            self.total = nyu_total - 1000
            self.offset = 1000
        else:
            raise ValueError('split ({}) not known.'.format(split))
        assert self.total > 0

        if cache_all:
            self.data_headers = load_nyu_depth_v2(self.root, fields=fields,
                                                  selection=slice(self.offset, self.offset + self.total),
                                                  download=download)
        else:
            self.data_headers, self._file = open_nyu_depth_v2(self.root, fields=fields, download=download)

    def __getitem__(self, item):
        if item in self.cache:
            sample = self.cache[item]

        else:
            if self.cache_all:
                sample = tuple(h[item] for h in self.data_headers)
            else:
                sample = tuple(h[self.offset + item] for h in self.data_headers)
                sample = tuple(d.transpose(*(tuple(range(d.ndim - 2)) + (-1, -2))) for d in sample)

                if self.cache_loaded:
                    self.cache[item] = sample

        if self.transform is not None:
            sample = self.transform(sample)

        image = sample[1]
        transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = transform(image)

        return image

    def __len__(self):
        return self.total

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        split_info = ' ({:.0f}% from training set reserved for validation)'.format(100 * self.val_ratio)
        fmt_str += '    Split: {}\n'.format(self.split + ('' if self.split == 'test' else split_info))
        fmt_str += '    Root Location: {}\n'.format(self.root)
        fmt_str += '    Cached samples: {}\n'.format(self.total if self.cache_all else len(self.cache))
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def __del__(self):
        if not self.cache_all:
            self._file.close()


class AssembleJointUpsamplingInputs(object):
    # TODO more jittering?
    def __init__(self, factor, crop=None, flip=True, normalize_target=True, output_crop=0, lowres_mode='center',
                 zero_guidance=False):
        self.factor = factor
        self.crop = crop if (isinstance(crop, collections.Iterable) or crop is None) else (crop, crop)
        self.output_crop = output_crop
        self.flip = flip
        self.lowres_mode = lowres_mode
        self.normalize_target = normalize_target
        self.zero_guidance = zero_guidance

        assert self.crop is None or (self.crop[0] % factor == 0 and self.crop[1] % factor == 0)

    @staticmethod
    def random_crop_params(img, output_size):
        """Get parameters (i, j, h, w) for random crop.
        Adapted from torchvision.transforms.transforms.RandomCrop.get_params
        """
        h, w = img.shape[1:]
        out_h, out_w = output_size
        if w == out_w and h == out_h:
            return 0, 0, h, w

        i = random.randint(0, h - out_h)
        j = random.randint(0, w - out_w)
        return i, j, out_h, out_w

    def __call__(self, im_target_tuple):
        guidance, target = im_target_tuple
        if target.ndim == 2:
            target = target.reshape((1, ) + target.shape)

        if self.crop is not None:
            i, j, out_h, out_w = self.random_crop_params(guidance, self.crop)
            guidance = guidance[:, i:i+out_h, j:j+out_w]
            target = target[:, i:i+out_h, j:j+out_w]

        if self.flip and random.random() < 0.5:
            guidance = guidance[:, :, ::-1]
            target = target[:, :, ::-1]
        low_res = get_lowres(target, self.factor, self.lowres_mode)

        if self.output_crop > 0:
            target = target[:, self.output_crop:-self.output_crop, self.output_crop:-self.output_crop]
        target = target.copy()
        guidance = guidance.copy()

        if self.normalize_target:
            raw_range = np.stack([low_res.min(axis=2).min(axis=1), low_res.max(axis=2).max(axis=1)], axis=-1)
            # deal with flat areas
            raw_range += ((raw_range[:, 0] == raw_range[:, 1]).reshape(-1, 1).repeat(2, axis=1) * [-0.5, 0.5])
            low_res = (low_res - raw_range[:, 0].reshape(-1, 1, 1)) / \
                      (raw_range[:, 1] - raw_range[:, 0]).reshape(-1, 1, 1)

        guidance = th.from_numpy(guidance).float().div(255)
        target = th.from_numpy(target).float()
        low_res = th.from_numpy(low_res).float()

        if self.zero_guidance:
            guidance = guidance * 0.0

        if self.normalize_target:
            raw_range = th.from_numpy(raw_range).float()
            return low_res, guidance, target, raw_range
        else:
            return low_res, guidance, target

    def __repr__(self):
        return self.__class__.__name__ + '(factor={0}, crop={1}, flip={2}, normalize_target={3}, lowres={4})'.format(
            self.factor, self.crop, self.flip, self.normalize_target, self.lowres_mode)