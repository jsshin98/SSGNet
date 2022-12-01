import argparse
import os
import glob
from skimage.measure import block_reduce
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils
from torch.utils.tensorboard import SummaryWriter
import pdb
from PIL import Image
from model.network import MMSR_net
import model.network as models
from model.main import MMSR
import imageio
import matplotlib.pyplot as plt
import warnings
import os
import re
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize
import csv
from skimage.feature import canny
TEST_SET_2005 = ['Reindeer']
TEST_SET_2006 = ['Bowling1', 'Bowling2']
TEST_SET_2014 = ['Adirondack-perfect', 'Motorcycle-perfect']
dir_2005 = '/shared/dataset/Middlebury/2005'
dir_2006 = '/shared/dataset/Middlebury/2006'
dir_2014 = '/shared/dataset/Middlebury/2014'

baseline = 160
f = 3740
data = []
data5 = []
data6 = []
data15 = []

scale = 4

def _read_pfm(pfm_file_path):
    with open(pfm_file_path, 'rb') as pfm_file:
        header = pfm_file.readline().decode('utf-8').rstrip()
        channels = 3 if header == 'PF' else 1

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', pfm_file.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(pfm_file.readline().decode('utf-8').rstrip())
        if scale < 0:
            endian = '<'  # little endian
        else:
            endian = '>'  # big endian

        disparity = np.fromfile(pfm_file, endian + 'f')

    return disparity, (height, width, channels)

def create_depth_from_pfm(pfm_file_path, calib=None):
    disparity, shape = _read_pfm(pfm_file_path)

    if calib is None:
        raise Exception('No calibration information available')
    else:
        fx = float(calib['cam0'].split(' ')[0].lstrip('['))
        base_line = float(calib['baseline'])
        doffs = float(calib['doffs'])

        depth_map = fx * base_line / (disparity + doffs)
        depth_map = np.reshape(depth_map, newshape=shape)
        depth_map = np.flipud(depth_map).transpose((2, 0, 1)).copy()

        depth_map[depth_map == 0.] = np.nan
        return depth_map


def read_calibration(calib_file_path):
    with open(calib_file_path, 'r') as calib_file:
        calib = {}
        csv_reader = csv.reader(calib_file, delimiter='=')
        for attr, value in csv_reader:
            calib.setdefault(attr, value)

    return calib

def downsample(image, scaling_factor):

    if image.ndim != 4:
        raise ValueError(f'Image should have four dimensions, got {image.ndim}')

    is_tensor = torch.is_tensor(image)
    if is_tensor:
        device = image.device
        image = image.detach().cpu().numpy()

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'Mean of empty slice')
        image = block_reduce(image, (1, 1, scaling_factor, scaling_factor), np.nanmean)

    return torch.from_numpy(image).to(device) if is_tensor else image

def mse_loss_func(pred, gt, mask):
    return F.mse_loss(pred[mask == 1.], gt[mask == 1.])


def l1_loss_func(pred, gt, mask):
    return F.l1_loss(pred[mask == 1.], gt[mask == 1.])

def rmse_loss_func(pred, gt, mask):
    return torch.sqrt(F.mse_loss(pred[mask == 1.], gt[mask == 1.]))
for img_name in TEST_SET_2005:

    path_dir = os.path.join(dir_2005, img_name)
    for view in (1,5):
        image = np.array(Image.open(path_dir+f'/view{view}.png'))
        disparities = torch.from_numpy(np.array(Image.open(path_dir + f'/disp{view}.png'))).float().unsqueeze(0)
        disparities[disparities==0.] = np.nan
        with open(path_dir + '/dmin.txt') as fh:
            dmin = int(fh.read().strip())
        disparities += dmin

        depth_map = baseline * f / disparities

        transform = Compose([ToTensor()])
        data5.append([(transform(Image.open(path_dir + f'/view{view}.png')), depth_map)])
data.extend(data5)

for img_name in TEST_SET_2006:
    path_dir = os.path.join(dir_2006, img_name)
    for view in (1,5):
        image = np.array(Image.open(path_dir+f'/view{view}.png'))
        disparities = torch.from_numpy(np.array(Image.open(path_dir + f'/disp{view}.png'))).float().unsqueeze(0)
        disparities[disparities==0.] = np.nan
        with open(path_dir + '/dmin.txt') as fh:
            dmin = int(fh.read().strip())
        disparities += dmin

        depth_map = baseline * f / disparities
        transform = Compose([ToTensor()])
        data6.append([(transform(Image.open(path_dir + f'/view{view}.png')), depth_map)])
data.extend(data6)

for img_name in TEST_SET_2014:
    path_dir = os.path.join(dir_2014, img_name)
    calibration = read_calibration(os.path.join(path_dir, 'calib.txt'))
    for view in (0, 1):
        depth_map = torch.from_numpy(create_depth_from_pfm(path_dir + f'/disp{view}.pfm', calibration))
        depth_map[depth_map==0.] = np.nan
        transform = Compose([ToTensor()])
        data6.append([(transform(Image.open(path_dir + f'/im{view}.png')), depth_map)])
data.extend(data6)

####### define parameters
params = {'img_idxs': [],  # specify images to process, if empty process all
          'scale': 4,  # SR factor, 4 or 8

          'loss': 'l1',
          'optim': 'adam',
          'lr': 0.002,

          'weights_regularizer': [0.0005, 0, 0, 0],
          # weight decay with factor 0.0005 is adopted on the guide branch, following P2P
          'batch_size': 1,
          'epoch': 100,
          }

deterministic_map = []
for i, datum in enumerate(data):
    H, W = datum[0][0].shape[1:]
    num_crops_h, num_crops_w = H // 256, W // 256
    deterministic_map.extend(((i, j, k) for j in range(num_crops_h) for k in range(num_crops_w)))

x=0
y=0
z=0
for i, sample in enumerate(deterministic_map):
    print("####### Porcessing {}/{} ######## ".format(i + 1, len(deterministic_map)))
    index, crop_index_h, crop_index_w = sample
    image = data[index][0][0]
    depth_map = data[index][0][1]
    max_v = np.nanmax(depth_map)
    slice_h = slice(crop_index_h * 256, (crop_index_h + 1) * 256)
    slice_w = slice(crop_index_w * 256, (crop_index_w + 1) * 256)
    guide_img, target_img = image[:, slice_h, slice_w], depth_map[:, slice_h, slice_w]
    source_img = downsample(target_img.unsqueeze(0), scale).squeeze().unsqueeze(0)
    mask_hr = (~torch.isnan(target_img)).float()
    mask_lr = (~torch.isnan(source_img)).float()
    target_img[mask_hr == 0.] = 0.
    source_img[mask_lr == 0.] = 0.
    guide_img = guide_img.numpy()
    source_img = source_img.numpy()
    target_img = target_img.numpy()
    prediction, eigen_vs = MMSR(guide_img=guide_img, source_img=source_img, mask=mask_lr, params=params, target_img=target_img, index=i, max_v = max_v.item())

    if target_img is not None:
        prediction = torch.from_numpy(prediction)[6:-6,6:-6]
        target_img = torch.from_numpy(target_img.squeeze())[6:-6,6:-6]
        mask_hr = mask_hr.squeeze()[6:-6,6:-6]
        MSE = mse_loss_func(prediction, target_img, mask_hr)
        RMSE = rmse_loss_func(prediction, target_img, mask_hr)
        MAE = l1_loss_func(prediction, target_img, mask_hr)
        print("MSE: {:.3f}  ---  RMSE: {:.3f}  ---  MAE: {:.3f}".format(MSE, RMSE, MAE))
        print("\n\n")
        x += RMSE
        y += MSE
        z += MAE
print("RMSE: ", x/len(deterministic_map))
print("MSE: ", y/len(deterministic_map))
print("MAE: ", z/len(deterministic_map))