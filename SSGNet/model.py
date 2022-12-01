import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models import create_model

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride=1,
                 padding=0, norm='none', activation='relu', pad_type='zero', partial=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'BN':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'IN':
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'LN':
            self.norm = LayerNorm(norm_dim, data_format='channels_first')
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation =='softmax':
            self.activation = nn.Softmax(dim=1)
        else:
            assert 0, "Unsupported activation: {}".format(activation)
        if partial:
            self.conv = PartialConv2d(input_dim, output_dim, kernel_size, stride=stride, padding=padding, bias=False, return_mask=True)
            print('##############################')
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=stride, padding=padding, bias=False)

    def forward(self, x, mask=None):
        if mask == None:
            x = self.conv(x)
            x = self.norm(x) if self.norm is not None else x
            x = self.activation(x) if self.activation is not None else x
            return x
        else:
            x, mask = self.conv(x, mask)
            x = self.norm(x) if self.norm is not None else x
            x = self.activation(x) if self.activation is not None else x
            return x, mask


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='IN', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class ResBlock(nn.Module):
    def __init__(self, dim, norm='IN', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)

        out += residual
        return out

def Pooling(stride):
    return nn.MaxPool2d(stride)

class SSGNet(nn.Module):
    def __init__(self):
        super(SSGNet, self).__init__()
        self.conv1 = Conv2dBlock(3, 32, 3, padding=1, norm='LN', activation='gelu')
        self.conv2 = Conv2dBlock(32, 64, 3, padding=1, stride=1, norm='LN', activation='gelu')
        self.deconv2_1 = Conv2dBlock(64, 32, 3, padding=1, norm='LN', activation='gelu')
        self.deconv2_2 = Conv2dBlock(64, 32, 3, padding=1, norm='LN', activation='gelu')
        self.deconv1 = Conv2dBlock(32, 3, 1, padding=0, norm='LN', activation='softmax')


    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        deconv2 = self.deconv2_1(conv2)
        deconv2 = self.deconv2_2(torch.cat([deconv2, conv1], dim=1))
        x = self.deconv1(deconv2)

        return x

class SSGNet_Gray(nn.Module):
    def __init__(self):
        super(SSGNet_Gray, self).__init__()
        self.conv1 = Conv2dBlock(1, 16, 5, padding=2, norm='LN', activation='gelu')
        self.conv2 = Conv2dBlock(16, 32, 5, padding=2, stride=1, norm='LN', activation='gelu')
        self.deconv2_1 = Conv2dBlock(32, 16, 5, padding=2, norm='LN', activation='gelu')
        self.deconv2_2 = Conv2dBlock(32, 16, 5, padding=2, norm='LN', activation='gelu')
        self.deconv1 = Conv2dBlock(16, 3, 1, padding=0, norm='LN', activation='softmax')


    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        deconv2 = self.deconv2_1(conv2)
        deconv2 = self.deconv2_2(torch.cat([deconv2, conv1], dim=1))    #nx64x128x128
        x = self.deconv1(deconv2)

        return x
