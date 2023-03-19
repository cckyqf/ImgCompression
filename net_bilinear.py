import torch
import torch.nn as nn
import numpy as np
import os
import torchvision.utils as vutils
import math
import torch.nn.functional as F
import torch.nn.init as init

class ResBlock(nn.Module):#残差单元
    def __init__(self,base):
        super(ResBlock, self).__init__()

        self.base = base

        self.relu = nn.ReLU(inplace=True)
        
        self.conv1 = nn.Conv2d(self.base,self.base,kernel_size=3,padding=1,stride=1,bias=False)
        self.conv2 = nn.Conv2d(self.base,self.base,kernel_size=3,padding=1,stride=1,bias=False)

    def forward(self,input):
        output = self.conv1(input)
        output = self.relu(output)
        output = self.conv2(output)
        output = output + input

        return output

class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)#相对位置编码
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])#左，右，上，下的填充，padding代表填充次数
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)#爱因斯坦求和约定

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)

class Generater(nn.Module):

    def __init__(self,channels,base,M,mode):
        super(Generater, self).__init__()

        self.channels = channels
        self.base = base
        self.M = M
        self.mode = mode
        self.nResBlock_d = 1
        self.nResBlock_d1 = 1
        self.nResBlock_d2 = 1
        self.nResBlock_d3 = 1

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        '''编码器的前半部分'''
        self.sample1 = nn.Conv2d(self.channels,self.base * 2,kernel_size=8,padding=2,stride=4,bias=False)
        self.resblk_e1 = ResBlock((self.base * 2))
        self.sample2 = nn.Conv2d(self.base * 2,self.base * 4,kernel_size=4,padding=1,stride=2,bias=False)
        self.resblk_e2 = ResBlock((self.base * 4))
        self.resblk_e3 = ResBlock((self.base * 4))
        self.sample3 = nn.Conv2d(self.base * 4,self.M,kernel_size=3,padding=1,stride=1,bias=False)

        '''解码器'''
        self.conv_d1 = nn.Conv2d(self.M,self.base * 2, kernel_size=3, padding=1, stride=1, bias=False)
        self.conv_d2 = nn.Conv2d(self.base * 2,self.base, kernel_size=3, padding=1, stride=1, bias=False)

        modules_d = [ResBlock(self.base) for _ in range(self.nResBlock_d)]
        self.resblk_d = nn.Sequential(*modules_d)

        self.attention1 = AttentionConv(self.base, self.base,kernel_size=7, padding=3, groups=8)
        modules_d1 = [ResBlock(self.base) for _ in range(self.nResBlock_d1)]
        self.resblk_d1 = nn.Sequential(*modules_d1)

        self.attention2 = AttentionConv(self.base, self.base,kernel_size=7, padding=3, groups=8)
        modules_d2 = [ResBlock(self.base) for _ in range(self.nResBlock_d2)]
        self.resblk_d2 = nn.Sequential(*modules_d2)
        
        self.attention3 = AttentionConv(self.base, self.base,kernel_size=7, padding=3, groups=8)
        modules_d3 = [ResBlock(self.base) for _ in range(self.nResBlock_d3)]
        self.resblk_d3 = nn.Sequential(*modules_d3)

        self.conv_d4 = nn.Conv2d(self.base,self.channels, kernel_size=3, padding=1, stride=1, bias=False)

    def forward(self,input,idx,savepath):
        
        device = input.device

        '''编码器的前半部分'''
        y = self.relu(self.sample1(input))
        y = self.relu(self.resblk_e1(y))
        y = self.relu(self.sample2(y))
        y = self.relu(self.resblk_e2(y))
        y = self.relu(self.resblk_e3(y))
        y = self.sample3(y)

        if self.mode == "train":
            y = self.sigmoid(64 * (y - 0.5))

            zeros_tensor = torch.zeros_like(y, device=device)
            ones_tensor = torch.ones_like(y, device=device)
            y_hat = torch.where(y < 0.5,zeros_tensor,ones_tensor)
            y_one_number = torch.sum(y_hat)
            y_zero_number = (y_hat.shape[0] * y_hat.shape[1] * y_hat.shape[2] * y_hat.shape[3]) - y_one_number
            P_one = y_one_number / (y_hat.shape[0] * y_hat.shape[1] * y_hat.shape[2] * y_hat.shape[3])
            P_zero = y_zero_number / (y_hat.shape[0] * y_hat.shape[1] * y_hat.shape[2] * y_hat.shape[3])
            E_one = - P_one * (torch.log(P_one + 1e-10) / math.log(2.0))
            E_zero = - P_zero * (torch.log(P_zero + 1e-10) / math.log(2.0))
            E_y  = E_one + E_zero

        elif self.mode == "test":

            zeros_tensor = torch.zeros_like(y, device=device)
            ones_tensor = torch.ones_like(y, device=device)
            y = torch.where(y < 0.5,zeros_tensor,ones_tensor)

            # y = (y > 0.5).float()
            y_save = y.cpu().detach().numpy()
            np.savetxt('%s/measurement/y_%d.txt' %(savepath,idx),y_save.ravel(),fmt='%d')

        '''解码器'''
        output = self.relu(self.conv_d1(y))
        output = self.relu(self.conv_d2(output))
        output = self.relu(self.resblk_d(output))
        output = self.relu(self.attention1(output))
        output = torch.nn.functional.interpolate(output, size=None, scale_factor=2, mode='bilinear', align_corners=None)
        output = self.relu(self.resblk_d1(output))
        output = self.relu(self.attention2(output))
        output = torch.nn.functional.interpolate(output, size=None, scale_factor=2, mode='bilinear', align_corners=None)
        output = self.relu(self.resblk_d2(output))
        output = self.relu(self.attention3(output))
        output = torch.nn.functional.interpolate(output, size=None, scale_factor=2, mode='bilinear', align_corners=None)
        output = self.relu(self.resblk_d3(output))
        output = self.conv_d4(output)
        
        if self.mode == "train":
            return output,E_y
        else:
            return output
