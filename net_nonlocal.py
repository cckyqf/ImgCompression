import torch
import torch.nn as nn
import numpy as np
import os
import torchvision.utils as vutils
import math
import torch.nn.functional as F
import torch.nn.init as init

def zero_upsample(input):
    device = input.device
    ps = nn.PixelShuffle(2)

    batch, channels, height, width = input.size()

    input_zero = torch.zeros([batch,1,height,width], device=device)
    input_zu = torch.zeros([batch,1,height * 2,width * 2], device=device)
    for i in range(channels):

        input_tmp = torch.cat((input[:,i,:,:].view(batch,1,height, width),input_zero,input_zero,input_zero), dim=1)
        input_zu = torch.cat((input_zu,ps(input_tmp)),1)

    return input_zu[:,1:,:,:]

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
    def __init__(self, base, kernel_size, stride=1, padding=1, bias=False):
        super(AttentionConv, self).__init__()
        self.base = base
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.key_conv = nn.Conv2d(self.base, self.base // 8, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(self.base, self.base // 8, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(self.base, self.base, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        q_out = self.query_conv(x)
        k_out = self.key_conv(x)
        v_out = self.value_conv(x)

        q_out = torch.nn.functional.interpolate(q_out, size=None, scale_factor=2, mode='bilinear', align_corners=True)
        k_out = zero_upsample(k_out)
        v_out = zero_upsample(v_out)

        mask = torch.ones(batch,1,height * 2, width * 2) * (-100000000)
        mask = mask.cuda()
        for i in range(height * 2):
            for j in range(width * 2):
                if i % 2 == 0 and j % 2 == 0:
                    mask[:,:,i, j] = 0

        k_out = k_out + mask

        q_out = q_out.view(batch,-1,width * 2 * height * 2).permute(0,2,1)
        k_out = k_out.view(batch,-1,width * 2 * height * 2)
        v_out = v_out.view(batch,-1,width * 2 * height * 2)

        attention =  torch.bmm(q_out,k_out)

        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(v_out,attention.permute(0,2,1) )
        out = out.view(batch,channels,width * 2,height * 2)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

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
        self.sample1 = nn.Conv2d(self.channels,self.base,kernel_size=8,padding=2,stride=4,bias=False)
        self.resblk_e1 = ResBlock((self.base))
        self.sample2 = nn.Conv2d(self.base,self.base,kernel_size=4,padding=1,stride=2,bias=False)
        self.resblk_e2 = ResBlock((self.base))
        self.resblk_e3 = ResBlock((self.base))
        self.sample3 = nn.Conv2d(self.base,self.M,kernel_size=3,padding=1,stride=1,bias=False)

        '''解码器'''
        self.conv_d1 = nn.Conv2d(self.M,self.base, kernel_size=3, padding=1, stride=1, bias=False)
        self.conv_d2 = nn.Conv2d(self.base,self.base, kernel_size=3, padding=1, stride=1, bias=False)

        modules_d = [ResBlock(self.base) for _ in range(self.nResBlock_d)]
        self.resblk_d = nn.Sequential(*modules_d)

        self.up1 = AttentionConv(self.base, kernel_size=3, padding=1)
        modules_d1 = [ResBlock(self.base) for _ in range(self.nResBlock_d1)]
        self.resblk_d1 = nn.Sequential(*modules_d1)

        self.up2 = AttentionConv(self.base, kernel_size=3, padding=1)
        modules_d2 = [ResBlock(self.base) for _ in range(self.nResBlock_d2)]
        self.resblk_d2 = nn.Sequential(*modules_d2)
        
        self.up3 = AttentionConv(self.base, kernel_size=3, padding=1)
        modules_d3 = [ResBlock(self.base) for _ in range(self.nResBlock_d3)]
        self.resblk_d3 = nn.Sequential(*modules_d3)

        # self.up2 = nn.ConvTranspose2d(self.base, self.base, kernel_size=4, padding=1, stride=2, bias=False)
        # self.up3 = nn.ConvTranspose2d(self.base, self.base, kernel_size=4, padding=1, stride=2, bias=False)

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
        output = self.relu(self.up1(output))
        output = self.relu(self.resblk_d1(output))
        output = self.relu(self.up2(output))
        output = self.relu(self.resblk_d2(output))
        output_inital = self.up3(output)
        output = self.relu(self.resblk_d3(output_inital))
        output = self.conv_d4(output)
        
        if self.mode == "train":
            return output,E_y
        else:
            return output
