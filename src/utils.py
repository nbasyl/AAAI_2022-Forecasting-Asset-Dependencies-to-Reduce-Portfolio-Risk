from torch import nn
import torch
from collections import OrderedDict
import math
import numpy as np

def normpdf(x, mean=0, sd=1):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

class CrossConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, sd = 1):
        super(CrossConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        
        self.mean = (self.kernel_size + 1)//2
        self.sd = sd
        
        tmp = np.array([normpdf(i+1, self.mean, self.sd) for i in range(self.kernel_size)])
        self.kernel_weight = np.outer(tmp,tmp)

    def forward(self, x):
        kernel = torch.Tensor(self.kernel_weight).expand(self.out_channels,self.in_channels,\
                             self.kernel_size,self.kernel_size).cuda()
        weight = nn.Parameter(data=kernel, requires_grad=False)
        out = nn.functional.conv2d(input = x, weight = weight,stride=self.stride,padding=self.padding)
        return out

def make_layers(block, config):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0],
                                                 out_channels=v[1],
                                                 kernel_size=v[2],
                                                 stride=v[3],
                                                 padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:   
            conv2d = nn.Conv2d(in_channels=v[0],
                                   out_channels=v[1],
                                   kernel_size=v[2],
                                   stride=v[3],
                                   padding=v[4])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))
            elif 'sigmoid' in layer_name:
                layers.append(('sigmoid_' + layer_name,
                           nn.Sigmoid()))
        elif 'cross' in layer_name:   
            kernel_type = v[5]
            if kernel_type == 0:
                conv2d = nn.Conv2d(in_channels=v[0],out_channels=v[1],kernel_size=v[2],stride=v[3],padding=v[4])
            else:
                conv2d = CrossConv2d(in_channels=v[0],out_channels=v[1],kernel_size=v[2],stride=v[3], padding=v[4])             
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError
    return nn.Sequential(OrderedDict(layers))

def uniform_evaluation(pred, label, config, maps = None):
    pred = pred.detach()
    label = label.detach()
    if config.method[:3] == "org":
        diff = pred - label
        diff = torch.triu(diff)
        #print(diff)
        square_diff = torch.sum(diff**2)
    elif config.method == "e3d":
        pred = torch.transpose(pred, 1, 2)
        label = torch.transpose(label, 1, 2)
        diff = pred - label
        diff_shape = diff.shape
        diff = torch.reshape(diff, (diff_shape[0]*diff_shape[1]*diff_shape[2],-1))
        diff = diff[:,:config.vec_len]
        #print(diff.shape)
        square_diff = torch.sum(diff**2)
    elif config.method == "lstm":
        diff = pred - label
        diff_shape = diff.shape
        diff = torch.reshape(diff, (diff_shape[0]*diff_shape[1],-1))
        diff = diff[:,:config.vec_len]
        #print(diff.shape)
        square_diff = torch.sum(diff**2)
    else:
#        pred = torch.clamp(pred, 0, 1)
        diff = pred - label
        diff_shape = diff.shape
        diff = torch.reshape(diff, (diff_shape[0]*diff_shape[1]*diff_shape[2],-1))
        diff = diff[:,:config.vec_len]
        #print(diff.shape)
        square_diff = torch.sum(diff**2)
    avg_diff = square_diff/config.vec_len
    return avg_diff.item()

def window(seq, size=2, stride=1):
    """Returns a sliding window (of width n) over data from the iterable
       E.g., s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...  
    """
    it = iter(seq)
    result = []
    for elem in it:
        result.append(elem)
        if len(result) == size:
            yield result
            result = result[stride:]
