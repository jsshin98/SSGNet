import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import eigh
import pdb

def get_loss_list():
    return loss_list.keys()

def get_loss_fn(args):
    return loss_list[args]


class Our_loss(nn.Module):
    def __init__(self):
        super(Our_loss, self).__init__()

    def forward(self, eigen_vectors, affinity_matrix):
        eigen_vectors = eigen_vectors.cpu()
        affinity_matrix = affinity_matrix.cpu()
        n,c,h,w = eigen_vectors.shape
        zeros = torch.zeros((n,c))
        eigen_vectors_flat = eigen_vectors.reshape(n, c, h*w)

        e_loss = F.l1_loss(torch.diagonal(torch.matmul(torch.matmul(eigen_vectors_flat, affinity_matrix), torch.transpose(eigen_vectors_flat, 1, 2)),0, dim1=-2, dim2=-1), zeros)
        s_loss = (torch.sum(eigen_vectors**0.9) + torch.sum((1-eigen_vectors)**0.9))/(n*c*h*w)-1

        return e_loss, s_loss


loss_list = {
    'our_loss':Our_loss
}
