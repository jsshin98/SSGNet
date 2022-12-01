from PIL import Image
import numpy as np
import torchvision.transforms as tf
import random
from skimage.feature import canny
from skimage.color import gray2rgb, rgb2gray

def image_transforms():

    return tf.Compose([
        tf.ToTensor()
    ])

def image_transforms_inpaint():

    return tf.Compose([
        tf.ToTensor()
    ])

def gray_transforms():

    return tf.Compose([
        tf.ToTensor(),
        ])


def affinity_transforms():

    return tf.Compose([
        tf.ToTensor()
    ])



def image_to_edge(gray_image, sigma):

    edge = gray_transforms()(Image.fromarray(canny(np.array(gray_image)/255., sigma=sigma)))

    return edge