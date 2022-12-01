import torch
import torch.nn as nn


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
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        else:
            assert 0, "Unsupported activation: {}".format(activation)
        if partial:
            self.conv = PartialConv2d(input_dim, output_dim, kernel_size, stride=stride, padding=padding, bias=False,
                                      return_mask=True)
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
        deconv2 = self.deconv2_2(torch.cat([deconv2, conv1], dim=1))  # nx64x128x128
        x = self.deconv1(deconv2)

        return x

class UNet_n2n_un(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, opt=None):
        super(UNet_n2n_un, self).__init__()

        self.ssg_net = SSGNet()
        if opt is not None:
            state_dict = torch.load(opt.ssg_pretrained)['state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            self.ssg_net.load_state_dict(new_state_dict)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 1, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )
        self.en_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(48, 48, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(2))

        self.en_block2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(2))

        self.en_block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(2))

        self.en_block4 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(2))

        self.en_block5 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 48, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block1 = nn.Sequential(
            nn.Conv2d(96, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(96, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block2 = nn.Sequential(
            nn.Conv2d(144, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(96, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block3 = nn.Sequential(
            nn.Conv2d(144, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(96, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block4 = nn.Sequential(
            nn.Conv2d(144, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(96, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block5 = nn.Sequential(
            nn.Conv2d(96 + 3, 64, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(64, 32, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),

            nn.Conv2d(32, out_channels, 3, padding=1, bias=True))

    def forward(self, x):
        eigen = self.conv(self.ssg_net(x))
        x1 = torch.cat([x, eigen], dim=1)
        pool1 = self.en_block1(x1)
        pool2 = self.en_block2(pool1)
        pool3 = self.en_block3(pool2)
        pool4 = self.en_block4(pool3)
        upsample5 = self.en_block5(pool4)

        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self.de_block1(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self.de_block2(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self.de_block3(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self.de_block4(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)
        out = self.de_block5(concat1)

        return out


class UNet_n2n_un_gray(UNet_n2n_un):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet_n2n_un_gray, self).__init__(in_channels, out_channels)


class UNet_n2n_un_srgb(UNet_n2n_un):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet_n2n_un_srgb, self).__init__(in_channels, out_channels)


if __name__ == '__main__':
    from thop import profile

    model = UNet_n2n_un_srgb()
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(input,), verbose=False)
    print('flops %.2e params %.2e' % (flops, params))
    print('flops %.1fG params %.1fM ' % (flops / (1024 ** 3), params / (1024 ** 2)))

    # flops 8.76e+09 parames 9.91e+05
