import torch
import torch.nn as nn
import torch.nn.functional as F


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
    
def default_conv(in_channels, out_channels, kernel_size):        
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=((kernel_size)//2), bias=True)      

class ResBlock(nn.Module):                                     
    def __init__(self, conv, n_channel, kernel_size):           
        super(ResBlock, self).__init__()

        m = []
        for i in range(2):
            m.append(conv(n_channel, n_channel, kernel_size))   
            if i == 0: m.append(nn.ReLU())
        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

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
        deconv2 = self.deconv2_2(torch.cat([deconv2, conv1], dim=1))    #nx64x128x128
        x = self.deconv1(deconv2)

        return x

class Modulation(nn.Module):
    def __init__(self, conv):                                                   
        super(Modulation, self).__init__()

        self.mlp_1 = nn.Sequential(conv(64, 64, 1))      
        self.mlp_2 = nn.Sequential(conv(64, 64, 1))  
        self.mlp_3 = nn.Sequential(conv(2 * 64, 64, 1))                       
        
    def forward(self, input):
        f_guide, f_source = input                                               
        f_guide = self.mlp_1(f_guide)                                                    
        f_source = self.mlp_2(f_source)                                       

        # Source-to-Guide
        source_neighbor = 11                                                         
        B, C, H, W = f_source.shape
        fs_unfold = F.unfold(f_source, kernel_size=source_neighbor, dilation=1, stride=1, padding=source_neighbor//2)
        fs_unfold = fs_unfold.view(B, C, source_neighbor**2, H, W)
        filters = (fs_unfold * f_guide.unsqueeze(2)).sum(1).softmax(1)          
        f_s2g = (fs_unfold * filters.unsqueeze(1)).sum(2)                       

        # Guide-to-Source
        guide_neighbor = 5                                                          
        B, C, H, W = f_guide.shape
        fg_unfold = F.unfold(f_guide, kernel_size=guide_neighbor, dilation=1, stride=1, padding=guide_neighbor//2)
        fg_unfold = fg_unfold.view(B, C, guide_neighbor**2, H, W)
        filters = (fg_unfold * f_source.unsqueeze(2)).sum(1).softmax(1)         
        f_g2s = (fg_unfold * filters.unsqueeze(1)).sum(2)                      

        fusion = self.mlp_3(torch.cat([f_s2g,f_g2s], dim=1))                                   
        return fusion 
                                
                             
class MMSR_net(nn.Module):                     
    def __init__(self, conv=default_conv):
        super(MMSR_net, self).__init__()
        self.branch_ev = SSGNet()
        state_dict = torch.load('path_to_pretrained_ssgnet')['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        self.branch_ev.load_state_dict(new_state_dict)
        self.conv = nn.Sequential(conv(3, 1, 3), nn.ReLU())
        self.branch_guide = nn.Sequential(conv(4, 64, 3), nn.ReLU(), conv(64, 64, 3), nn.ReLU(),
                                          ResBlock(conv, 64, 3), ResBlock(conv, 64, 3))
        self.branch_source = nn.Sequential(conv(1, 64, 1), nn.ReLU(), conv(64, 64, 1), nn.ReLU(), 
                                           ResBlock(conv, 64, 1), ResBlock(conv, 64, 1))
        self.modulation = nn.Sequential(Modulation(conv))     
        self.branch_pred = nn.Sequential(ResBlock(conv, 64, 1), ResBlock(conv, 64, 1), ResBlock(conv, 64, 1),
                                         conv(64, 1, 1))
        weights_regularizer = [0.0005, 0, 0, 0]
        reg_guide, reg_source, reg_modu, reg_pred = weights_regularizer[:4]
        self.params_with_regularizer = [{'params':self.branch_ev.parameters(), 'weight_decay':reg_guide},\
                                        {'params':self.conv.parameters(),'weight_decay':reg_guide},\
                                        {'params':self.branch_guide.parameters(),'weight_decay':reg_guide},\
                                        {'params':self.branch_source.parameters(),'weight_decay':reg_source},\
                                        {'params':self.modulation.parameters(),'weight_decay':reg_modu},\
                                        {'params':self.branch_pred.parameters(),'weight_decay':reg_pred}]
        
    def forward(self, guide, source):
        source = F.interpolate(source, [guide.shape[2],guide.shape[3]], mode='bilinear', align_corners=False)
        eigen = self.conv(self.branch_ev(guide))
        guide = torch.cat([guide, eigen], dim=1)
        fusion = self.modulation([self.branch_guide(guide), self.branch_source(source)])
        predict = self.branch_pred(fusion)
        return predict, eigen
