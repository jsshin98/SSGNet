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



class BSD300(data.Dataset):
    def __init__(self):
        self.image_paths = []
        self.name_paths=[]
        self._load_images_paths()

        self.image_files_transforms = image_transforms()

    def _load_images_paths(self):
        print('loading testing files...')
        self.valfile = '/shared/BSD300/'
        images = os.listdir(self.valfile)

        for image in images:
            path = os.path.join(self.valfile, image)
            self.image_paths.append(path)
        print(f'total {len(self.image_paths)} images')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        image_name = path.split('/')[-1]
        image = np.array(Image.open(self.image_paths[index]).convert('RGB'))
        image = self.image_files_transforms(image)

        return image, image_name
