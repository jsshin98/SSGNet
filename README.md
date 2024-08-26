# [AAAI 2023 ORAL] Task-specific Scene Structure Representations

**Task-specific Scene Structure Representations(AAAI 2023 ORAL Presentation ACCEPTED!) \
Jisu Shin, Seunghyun Shin and Hae-Gon Jeon**<br>

**[[PAPER]](https://doi.org/10.1609/aaai.v37i2.25322)**

## Introduction

__SSGNet.__ We propose a Scene Structure Guidance Network, SSGNet, a single general neural network architecture for extracting task-specific structural features of scenes.. 

## Prerequisites

- Python >= 3.6
- PyTorch >= 1.0
- NVIDIA GPU + CUDA cuDNN

## Getting Started

### Installation


- Install python requirements:

```
pip install -r requirements.txt
```

### Training SSGNet
You should change SSGNet/options/train_options \
--data-root : path to you nyu dataset \
--save_dir : path to save your training result(ex. model weights, tensorboard) \
etc.

```
cd ./SSGNet

python train.py 
```
### Test SSGNet
You should change SSGNET/options/test_options \
--ssgnet-pretrained : path to your pretrained SSGNet \
--save_result : path to save SSGNet output
```commandline
python test.py
```
### Initial pretrained weight for SSGNet
Download https://drive.google.com/file/d/1nimXk_7VV3TaiP3W3E49XgEc8nQftyRA/view?usp=drive_link
and modify --ssgnet-pretrained path to test SSGNet

### Train Denoising

You should change Denoising/options/train_options \
--dataset_root : path to your ImageNet dataset \
--save_dir : path to save your training result(ex. model weights, tensorboard) \
--ssg-pretrained : path of SSGNet pretrained weights \
etc.

For RGB
```commandline
cd ./Denoising

python train.py
```

For Gray_Scale

```
python train_gray.py
```


### Test Denoising

You should change Denoising/options/test_options \
--ssgnet-pretrained : path to your pretrained Denoising Network \
--save_dir : path to save denoised images

For BSDS300 Dataset

```commandline
python test_BSDS300.py
```
For Kodak Dataset

```commandline
python test_Kodak.py
```

For BSDS68 Dataset

```commandline
python test_BSDS68.py
```

### Run Depth_Upsampling

You should change scale parameter in main_GR & pretrained SSGNet parameter's path in model/network

```
python main_GR.py
```

### Dataset
You can download dataset that we used for training and test from below links

- NyuV2 : http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
- Middlebury : https://vision.middlebury.edu/stereo/data/
- Kodak : http://www.cs.albany.edu/~xypan/research/snr/Kodak.html
- BSDS : https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/

### Reference
Baseline network codes of Denoising and Depth Upsampling are from https://github.com/zhangyi-3/IDR and https://github.com/palmdong/MMSR, respectively.

### Citation
Will be updated soon
### Contact
If you have any question, please feel free to contact us via jsshin98@gm.gist.ac.kr or seunghyuns98@gm.gist.ac.kr.
