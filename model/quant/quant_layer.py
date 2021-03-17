"""
quant_layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from .quantizer import *


def odd_symm_quant(input, nbit, mode='mean', k=2, dequantize=True, posQ=False):
    
    if mode == 'mean':
        alpha_w = k * input.abs().mean()
    elif mode == 'sawb':
        z_typical = {'4bit': [0.077, 1.013], '8bit':[0.027, 1.114]}               
        alpha_w = get_scale(input, z_typical[f'{int(nbit)}bit']).item()
    
    output = input.clamp(-alpha_w, alpha_w)

    if posQ:
        output = output + alpha_w

    scale, zero_point = symmetric_linear_quantization_params(nbit, abs(alpha_w), restrict_qrange=True)

    output = linear_quantize(output, scale, zero_point)
    
    if dequantize:
        output = linear_dequantize(output, scale, zero_point)

    return output, alpha_w, scale

def activation_quant(input, nbit, sat_val, dequantize=True):
    with torch.no_grad():
        scale, zero_point = quantizer(nbit, 0, sat_val)
    
    output = linear_quantize(input, scale, zero_point)

    if dequantize:
        output = linear_dequantize(output, scale, zero_point)

    return output, scale

class ClippedReLU(nn.Module):
    def __init__(self, num_bits, alpha=8.0, inplace=False, dequantize=True):
        super(ClippedReLU, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([alpha]))     
        self.num_bits = num_bits
        self.inplace = inplace
        self.dequantize = dequantize
        
    def forward(self, input):
        # print(f'ClippedRELU: input mean: {input.mean()} | input std: {input.std()}')
        input = F.relu(input)
        input = torch.where(input < self.alpha, input, self.alpha)
        
        with torch.no_grad():
            scale, zero_point = quantizer(self.num_bits, 0, self.alpha)
        input = STEQuantizer.apply(input, scale, zero_point, self.dequantize, self.inplace)
        return input

class int_quant_func(torch.autograd.Function):
    def __init__(self, nbit, alpha_w, restrictRange=True, ch_group=16, push=False):
        super(int_quant_func, self).__init__()
        self.nbit = nbit
        self.restrictRange = restrictRange
        self.alpha_w = alpha_w
        self.ch_group = ch_group
        self.push = push

    def forward(self, input):
        self.save_for_backward(input)
        output = input.clamp(-self.alpha_w.item(), self.alpha_w.item())
        scale, zero_point = symmetric_linear_quantization_params(self.nbit, self.alpha_w, restrict_qrange=self.restrictRange)
        output = STEQuantizer_weight.apply(output, scale, zero_point, True, False, self.nbit, self.restrictRange)   

        return output

    def backward(self, grad_output):
        grad_input = grad_output.clone()
        input, = self.saved_tensors
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class int_conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=False, nbit=4, mode='mean', k=2, ch_group=16, push=False):
        super(int_conv2d, self).__init__(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups,
                bias=bias)
        self.nbit = nbit
        self.mode = mode
        self.k = k
        self.ch_group = ch_group
        self.push = push
        self.alpha_w = 1
        self.register_buffer('mask', torch.ones_like(self.weight))
        self.register_buffer('iter', torch.Tensor([0.]))

    def forward(self, input):
        w_l = self.weight.clone()
        if self.mode == 'mean':
            self.alpha_w = self.k * w_l.abs().mean()
        elif self.mode == 'sawb':
            z_typical = {'4bit': [0.077, 1.013], '8bit':[0.027, 1.114]}               
            self.alpha_w = get_scale(w_l, z_typical[f'{int(self.nbit)}bit'])
        else:
            raise ValueError("Quantization mode must be either 'mean' or 'sawb'! ")                         
            
        weight_q = int_quant_func(nbit=self.nbit, alpha_w=self.alpha_w, restrictRange=True, ch_group=self.ch_group, push=self.push)(w_l)
        output = F.conv2d(input, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output
    
    def extra_repr(self):
        return super(int_conv2d, self).extra_repr() + ', nbit={}, mode={}, k={}, ch_group={}, push={}'.format(self.nbit, self.mode, self.k, self.ch_group, self.push)


class int_linear(nn.Linear):
    def __init__(self, in_channels, out_channels, bias=True, nbit=8, mode='mean', k=2, ch_group=16, push=False):
        super(int_linear, self).__init__(in_features=in_channels, out_features=out_channels, bias=bias)
        self.nbit=nbit
        self.mode = mode
        self.k = k
        self.ch_group = ch_group
        self.push = push

    def forward(self, input):
        w_l = self.weight.clone()

        if self.mode == 'mean':
            self.alpha_w = self.k * w_l.abs().mean()
        elif self.mode == 'sawb':
            z_typical = {'4bit': [0.077, 1.013], '8bit':[0.027, 1.114]}               
            self.alpha_w = get_scale(w_l, z_typical[f'{int(self.nbit)}bit'])
        else:
            raise ValueError("Quantization mode must be either 'mean' or 'sawb'! ")                         
            
        weight_q = int_quant_func(nbit=self.nbit, alpha_w=self.alpha_w, restrictRange=True, ch_group=self.ch_group, push=self.push)(w_l)
        output = F.linear(input, weight_q, self.bias)
        return output

    def extra_repr(self):
        return super(int_linear, self).extra_repr() + ', nbit={}, mode={}, k={}, ch_group={}, push={}'.format(self.nbit, self.mode, self.k, self.ch_group, self.push)

"""
SAWB 2-bit quantization
"""

class sawb_w2_Func(torch.autograd.Function):

    def __init__(self, alpha):
        super(sawb_w2_Func, self).__init__()
        self.alpha = alpha

    def forward(self, input):
        self.save_for_backward(input)
        
        output = input.clone()
        output[input.ge(self.alpha - self.alpha/3)] = self.alpha
        output[input.lt(-self.alpha + self.alpha/3)] = -self.alpha
        
        output[input.lt(self.alpha - self.alpha/3)*input.ge(0)] = self.alpha/3
        output[input.ge(-self.alpha + self.alpha/3)*input.lt(0)] = -self.alpha/3

        return output
    
    def backward(self, grad_output):
    
        grad_input = grad_output.clone()
        input, = self.saved_tensors
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0

        return grad_input

class sawb_w2_Conv2d(nn.Conv2d):

    def forward(self, input):
        alpha_w = get_scale_2bit(self.weight)

        weight = sawb_w2_Func(alpha=alpha_w)(self.weight)
        output = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        return output