import os
import numpy as np
from PIL import Image
import scipy
import torch
from torch.utils import data
from dataset.transform import *
import torchvision

import pdb
from PIL import Image
import os



class ImageNet(data.Dataset):
    def __init__(self, opt, train_mode):
        self.opt = opt
        self.train_mode = train_mode
        self.dataset_root = opt.dataset_root
        self.image_paths = []
        self.name_paths=[]
        self._load_images_paths()

        self.image_files_transforms = image_transforms()

    def _load_images_paths(self):
        if self.train_mode == 'train':
            print('loading training files...')
            self.trainfile = os.path.join(self.dataset_root, 'train')
            self.valfile = os.path.join(self.dataset_root, 'val')
            images_t = os.listdir(self.trainfile)
            images_v = os.listdir(self.valfile)
            for img in images_t:
                path = os.path.join(self.trainfile, img)
                self.image_paths.append(path)
            for img in images_v:
                path = os.path.join(self.valfile, img)
                self.image_paths.append(path)

            print(f'total {len(self.image_paths)} images')

        if self.train_mode == 'val':
            print('loading testing files...')
            self.valfile = os.path.join(self.dataset_root, 'val')
            images = os.listdir(self.valfile)

            for image in images[:100]:
                path = os.path.join(self.valfile, image)
                self.image_paths.append(path)
            print(f'total {len(self.image_paths)} images')

    def random_crop_params(self, img, output_size):

        h, w = img.shape[1:]
        out_h, out_w = output_size
        if w <= out_w or h <= out_h:
            return 0, 0, h, w

        i = random.randint(0, h - out_h)
        j = random.randint(0, w - out_w)
        return i, j, out_h, out_w

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        image_name = path.split('/')[-1]
        image = np.array(Image.open(self.image_paths[index]).convert('RGB'))
        i, j, out_h, out_w = self.random_crop_params(image, (256, 256))
        if i ==0 and j ==0 :
            image = np.array(Image.fromarray(image).resize((256, 256)))
        else:
            image = image[:, i:i + out_h, j:j + out_w]
        image = self.image_files_transforms(image)

        return image, image_name
