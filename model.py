import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

from net_module import ResBlock, AttentionConv, DepthWiseConv, TransformerBlock, Conv, ConvTranspose, DPWConvChannelShuffle, DPWConv
from cbam import CBAM
from net_module import ChannelShuffle

# 二值量化的界限
# quantization_threshold = 0.0

# 以0.0为界限，进行量化，使用tanh函数

def compute_E_y(y_hat):
    # zeros_tensor = torch.zeros_like(y)
    # ones_tensor = torch.ones_like(y)
    # y_hat = torch.where(y < 0.5,zeros_tensor,ones_tensor)
    

    y_one_number = torch.sum(y_hat)
    num = y_hat.numel()
    y_zero_number = num - y_one_number
    P_one = y_one_number / num
    P_zero = y_zero_number / num

    # E_one = - P_one * (torch.log(P_one + 1e-10) / math.log(2.0))
    # E_zero = - P_zero * (torch.log(P_zero + 1e-10) / math.log(2.0))
    E_one = - P_one * torch.log2(P_one + 1e-10)
    E_zero = - P_zero * torch.log2(P_zero + 1e-10)
    E_y  = E_one + E_zero

    return E_y
    

class net_bilinear(nn.Module):

    def __init__(self,channels,base,M,mode):
        super().__init__()

        self.channels = channels
        self.base = base
        self.M = M
        self.mode = mode
        self.nResBlock_d = 1
        self.nResBlock_d1 = 1
        self.nResBlock_d2 = 1
        self.nResBlock_d3 = 1

        self.sigmoid = nn.Sigmoid()

        '''编码器的前半部分'''
        self.sample1 = Conv(self.channels,self.base * 2,kernel_size=8,padding=2,stride=4,bias=False, bn=False, act=True)
        self.resblk_e1 = ResBlock((self.base * 2))
        self.sample2 = Conv(self.base * 2,self.base * 4,kernel_size=4,padding=1,stride=2,bias=False, bn=False, act=True)
        self.resblk_e2 = ResBlock((self.base * 4))
        self.resblk_e3 = ResBlock((self.base * 4))
        self.sample3 = Conv(self.base * 4,self.M,kernel_size=3,padding=1,stride=1,bias=False, bn=False, act=False)

        '''解码器'''
        self.conv_d1 = Conv(self.M,self.base * 2, kernel_size=3, padding=1, stride=1, bias=False, bn=False, act=True)        
        self.conv_d2 = Conv(self.base * 2,self.base, kernel_size=3, padding=1, stride=1, bias=False, bn=False, act=True) 


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

        self.conv_d4 = Conv(self.base,self.channels, kernel_size=3, padding=1, stride=1, bias=False, bn=False, act=False)

    def forward(self,input):
        
        '''编码器的前半部分'''
        y = self.sample1(input)
        y = self.resblk_e1(y)
        y = self.sample2(input)
        y = self.resblk_e2(y)
        y = self.resblk_e3(y)
        y = self.sample3(y)

        if self.mode == "train":
            y = self.sigmoid(64 * (y - 0.5))

            E_y  = compute_E_y(y)

        elif self.mode == "test":

            zeros_tensor = torch.zeros_like(y)
            ones_tensor = torch.ones_like(y)
            y = torch.where(y < 0.5,zeros_tensor,ones_tensor)

            
        '''解码器'''
        output = self.conv_d1(y)
        output = self.conv_d2(output)
        output = self.resblk_d(output)
        output = self.attention1(output)
        output = F.interpolate(output, size=None, scale_factor=2, mode='bilinear', align_corners=None)
        output = self.resblk_d1(output)
        output = self.attention2(output)
        output = F.interpolate(output, size=None, scale_factor=2, mode='bilinear', align_corners=None)
        output = self.resblk_d2(output)
        output = self.attention3(output)
        output = F.interpolate(output, size=None, scale_factor=2, mode='bilinear', align_corners=None)
        output = self.resblk_d3(output)
        output = self.conv_d4(output)
        
        if self.mode == "train":
            return output,E_y
        else:
            return output, y


class net_SASA2(nn.Module):

    def __init__(self,channels,base,M,mode):
        super().__init__()

        self.channels = channels
        self.base = base
        self.M = M
        self.mode = mode
        self.nResBlock_d = 1
        self.nResBlock_d1 = 1
        self.nResBlock_d2 = 1
        self.nResBlock_d3 = 1

        self.sigmoid = nn.Sigmoid()

        '''编码器的前半部分'''
        self.sample1 = Conv(self.channels,self.base * 2,kernel_size=8,padding=2,stride=4,bias=False, bn=False, act=True)
        self.resblk_e1 = ResBlock((self.base * 2))
        self.sample2 = Conv(self.base * 2,self.base * 4,kernel_size=4,padding=1,stride=2,bias=False, bn=False, act=True)
        self.resblk_e2 = ResBlock((self.base * 4))
        self.resblk_e3 = ResBlock((self.base * 4))
        self.sample3 = Conv(self.base * 4,self.M,kernel_size=3,padding=1,stride=1,bias=False, bn=False, act=False)

        '''解码器'''
        self.conv_d1 = Conv(self.M,self.base * 2, kernel_size=3, padding=1, stride=1, bias=False, bn=False, act=True)        
        self.conv_d2 = Conv(self.base * 2,self.base, kernel_size=3, padding=1, stride=1, bias=False, bn=False, act=True) 


        modules_d = [ResBlock(self.base) for _ in range(self.nResBlock_d)]
        self.resblk_d = nn.Sequential(*modules_d)

        self.attention1 = AttentionConv(self.base, self.base,kernel_size=7, padding=3, groups=8)
        self.up1 = ConvTranspose(self.base, self.base, kernel_size=4, padding=1, stride=2, bias=False, bn=False, act=True)
        modules_d1 = [ResBlock(self.base) for _ in range(self.nResBlock_d1)]
        self.resblk_d1 = nn.Sequential(*modules_d1)

        self.attention2 = AttentionConv(self.base, self.base,kernel_size=7, padding=3, groups=8)
        self.up2 = ConvTranspose(self.base, self.base, kernel_size=4, padding=1, stride=2, bias=False, bn=False, act=True)
        modules_d2 = [ResBlock(self.base) for _ in range(self.nResBlock_d2)]
        self.resblk_d2 = nn.Sequential(*modules_d2)
        
        self.attention3 = AttentionConv(self.base, self.base,kernel_size=7, padding=3, groups=8)
        self.up3 = ConvTranspose(self.base, self.base, kernel_size=4, padding=1, stride=2, bias=False, bn=False, act=True)
        modules_d3 = [ResBlock(self.base) for _ in range(self.nResBlock_d3)]
        self.resblk_d3 = nn.Sequential(*modules_d3)

        self.conv_d4 = Conv(self.base,self.channels, kernel_size=3, padding=1, stride=1, bias=False, bn=False, act=False)

    def forward(self,input):
        
        '''编码器的前半部分'''
        y = self.sample1(input)
        y = self.resblk_e1(y)
        y = self.sample2(input)
        y = self.resblk_e2(y)
        y = self.resblk_e3(y)
        y = self.sample3(y)

        if self.mode == "train":
            y = self.sigmoid(64 * (y - 0.5))

            E_y  = compute_E_y(y)

        elif self.mode == "test":

            zeros_tensor = torch.zeros_like(y)
            ones_tensor = torch.ones_like(y)
            y = torch.where(y < 0.5,zeros_tensor,ones_tensor)

            
        '''解码器'''
        output = self.conv_d1(y)
        output = self.conv_d2(output)
        output = self.resblk_d(output)
        output = self.attention1(output)
        output = self.up1(output)
        output = self.resblk_d1(output)
        output = self.attention2(output)
        output = self.up2(output)
        output = self.resblk_d2(output)
        output = self.attention3(output)
        output = self.up3(output)
        output = self.resblk_d3(output)
        output = self.conv_d4(output)
        
        if self.mode == "train":
            return output,E_y
        else:
            return output, y


class net_paper(nn.Module):

    def __init__(self,channels,base,M,mode):
        super().__init__()

        self.channels = channels
        self.base = base
        self.M = M
        self.mode = mode
        self.nResBlock_d0 = 1
        self.nResBlock_d1 = 1
        self.nResBlock_d2 = 1
        self.nResBlock_d3 = 1

        self.bias = False
        self.bn = False

        self.quantization = nn.Sigmoid()

        '''编码器的前半部分'''
        self.sample1 = Conv(self.channels,self.base, kernel_size=3,padding=1,stride=2,bias=False, bn=False, act=True)
        self.sample2 = Conv(self.base,    self.base, kernel_size=3,padding=1,stride=2,bias=False, bn=False, act=True)
        self.sample3 = Conv(self.base,    self.M,    kernel_size=3,padding=1,stride=2,bias=False, bn=False, act=False)


        '''解码器'''
        # 从量化值中重建特征
        self.recon_feat = Conv(self.M, self.base*2, kernel_size=3, padding=1, stride=1, bias=self.bias, bn=self.bn, act=True)

        self.layer0 = self._make_layers(self.base*2, self.base, self.nResBlock_d0, upsample=False, attention=True,  bias=self.bias, bn=self.bn)
        self.layer1 = self._make_layers(self.base,   self.base, self.nResBlock_d1, upsample=True,  attention=True,  bias=self.bias, bn=self.bn)
        self.layer2 = self._make_layers(self.base,   self.base, self.nResBlock_d2, upsample=True,  attention=True,  bias=self.bias, bn=self.bn)
        self.layer3 = self._make_layers(self.base,   self.base, self.nResBlock_d3, upsample=True,  attention=False, bias=self.bias, bn=self.bn)

        # 网络输出层，不使用BN
        self.conv_d4 = Conv(self.base, self.channels, kernel_size=3, padding=1, stride=1, bias=self.bias, bn=False, act=False)
        


    def _make_layers(self, in_channels, out_channels, num_ResBlock, upsample=False, attention=False, bias=False, bn=False):

        layers = []

        # 添加上采样模块或者调整模块
        if upsample:
            layers.append(
                ConvTranspose(in_channels, out_channels, kernel_size=4, padding=1, stride=2, bias=bias, bn=bn, act=True)
            )
        else:
            layers.append(
                Conv(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=bias, bn=bn, act=True)      
            )

        # 添加残差块
        layers.extend([ResBlock(out_channels, bias=bias, bn=bn) for _ in range(num_ResBlock)])

        # 注意力模块
        if attention:
            # 只要使用了bn或者bias中的一个，那么bias就使用了
            is_bias = bn or bias
            layers.append(AttentionConv(out_channels, out_channels, kernel_size=7, padding=3, groups=8, bias=is_bias)) 

        return nn.Sequential(*layers)


    def forward(self,input):
        
        '''编码器的前半部分'''
        y = self.sample1(input)
        y = self.sample2(y)
        y = self.sample3(y)
        
        if self.mode == "train":
            y = self.quantization(64 * (y - 0.5))
            E_y  = compute_E_y(y)

        elif self.mode == "test":
            zeros_tensor = torch.zeros_like(y)
            ones_tensor = torch.ones_like(y)
            y = torch.where(y < 0.5,zeros_tensor,ones_tensor)


        '''解码器'''
        output = self.recon_feat(y)
        output = self.layer0(output)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)

        output = self.conv_d4(output)
        if self.mode == "train":
            return output,E_y
        else:
            return output, y



class net_encoder(net_paper):
    def __init__(self,channels,base,M,mode):
        
        super().__init__(channels=channels, base=base, M=M, mode=mode)


        # self.sample2 = DPWConv(self.base, self.base, kernel_size=3, padding=1, stride=2, bias=False, act_output=True)
        # self.sample3 = DPWConv(self.base, self.M, kernel_size=3, padding=1, stride=2, bias=False, act_output=False)

        # 对第二层卷积进行分组+重排
        num_groups_point_conv = self.base // 16
        # self.sample2 = nn.Sequential(
        #     ChannelShuffle(channels=self.base, groups=num_groups_point_conv),
        #     Conv(self.base, self.base, kernel_size=3,padding=1,stride=2,bias=False, bn=False, act=True, groups=num_groups_point_conv)
        # )

        # # 点卷积进行分组，并使用通道重排
        self.sample2 = DPWConvChannelShuffle(self.base, self.base, kernel_size=3, padding=1, stride=2, bias=False, 
                                num_groups_point_conv=num_groups_point_conv, channel_shuffle=True, act=True)
        self.sample3 = DPWConvChannelShuffle(self.base, self.M, kernel_size=3, padding=1, stride=2, bias=False, 
                                     num_groups_point_conv=1, channel_shuffle=False, act=False)



class net_decoder(net_encoder):
    def __init__(self,channels,base,M,mode):
        
        super().__init__(channels=channels, base=base, M=M, mode=mode)

        # 编码端网络输出，加sigmoid激活函数，训练很不理想
        # 改成relu激活函数，很好
        self.output_act = nn.ReLU(inplace=True)

    def _make_layers(self, in_channels, out_channels, num_ResBlock, upsample=False, attention=False, bias=False, bn=False):

        layers = []

        # 添加上采样模块或者调整模块
        if upsample:
            layers.append(
                ConvTranspose(in_channels, out_channels, kernel_size=4, padding=1, stride=2, bias=bias, bn=bn, act=True)
                # nn.Upsample(size=None, scale_factor=2, mode='nearest')
            )
        else:
            layers.append(
                Conv(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=bias, bn=bn, act=True)      
            )

        # 添加残差块
        layers.extend([ResBlock(out_channels, bias=bias, bn=bn) for _ in range(num_ResBlock)])

        # 为了迭代算法更快，把注意力模块替换为普通卷积模块
        if attention:
            layers.append(Conv(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=bias, bn=bn, act=True))             
            
            # layers.append(TransformerBlock(out_channels, out_channels, act=True))

            # layers.append(CBAM(out_channels, 4))

        return nn.Sequential(*layers)


    def forward(self,input):
        
        '''编码器的前半部分'''
        y = self.sample1(input)
        y = self.sample2(y)
        y = self.sample3(y)
        
        if self.mode == "train":
            # y = self.quantization(64 * (y - 0.5))
            zeros_tensor = torch.zeros_like(y)
            ones_tensor = torch.ones_like(y)
            y = torch.where(y < 0.5,zeros_tensor,ones_tensor)
            
            E_y  = compute_E_y(y)
            # E_y = torch.tensor([0], device=E_y.device)

        elif self.mode == "test":
            zeros_tensor = torch.zeros_like(y)
            ones_tensor = torch.ones_like(y)
            y = torch.where(y < 0.5,zeros_tensor,ones_tensor)


        '''解码器'''
        output = self.recon_feat(y)

        # # 8x下采样
        # output = self.layer0(output)
        # # 4x下采样
        # output = self.layer1(output)+F.interpolate(output, size=None, scale_factor=2, mode='bilinear', align_corners=None)
        # # 2x下采样
        # output = self.layer2(output)+F.interpolate(output, size=None, scale_factor=2, mode='bilinear', align_corners=None)
        # # 原始图像尺寸
        # output = self.layer3(output)+F.interpolate(output, size=None, scale_factor=2, mode='bilinear', align_corners=None)

        output = self.layer0(output)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)

        output = self.conv_d4(output)
        output = self.output_act(output)

        if self.mode == "train":
            return output,E_y
        else:
            return output, y


        

# exp69-70: 复现、去掉熵约束
# exp71: E_y = torch.tensor([0], device=E_y.device)
# exp72-3: 32倍压缩、去掉熵约束
# exp74-5: gamma=0.0、 熵约束到0.7
# exp76:训练时不用sigmoid


class net_DWConv(nn.Module):

    def __init__(self,channels,base,M,mode):
        super().__init__()

        self.channels = channels
        self.base = base
        self.M = M
        self.mode = mode
        self.nResBlock_d0 = 1
        self.nResBlock_d1 = 1
        self.nResBlock_d2 = 1
        self.nResBlock_d3 = 1

        self.bias = False
        self.bn = False

        self.quantization = nn.Sigmoid()

        '''编码器的前半部分'''
        self.sample1 = Conv(self.channels,self.base, kernel_size=3,padding=1,stride=2,bias=False, bn=False, act=True)

        # # 点卷积进行分组，并使用通道重排
        num_groups_point_conv = self.base // 16
        self.sample2 = DepthWiseConv(self.base, self.base, kernel_size=3, padding=1, stride=2, bias=False, 
                                num_groups_point_conv=num_groups_point_conv, channel_shuffle=True, act=True)
        self.sample3 = DepthWiseConv(self.base, self.M, kernel_size=3, padding=1, stride=2, bias=False, 
                                     num_groups_point_conv=1, channel_shuffle=False, act=False)


        '''解码器'''
        # 从量化值中重建特征
        self.recon_feat = Conv(self.M, self.base*2, kernel_size=3, padding=1, stride=1, bias=self.bias, bn=self.bn, act=True)

        self.layer0 = self._make_layers(self.base*2, self.base, self.nResBlock_d0, upsample=False, attention=True,  bias=self.bias, bn=self.bn)
        self.layer1 = self._make_layers(self.base,   self.base, self.nResBlock_d1, upsample=True,  attention=True,  bias=self.bias, bn=self.bn)
        self.layer2 = self._make_layers(self.base,   self.base, self.nResBlock_d2, upsample=True,  attention=True,  bias=self.bias, bn=self.bn)
        self.layer3 = self._make_layers(self.base,   self.base, self.nResBlock_d3, upsample=True,  attention=False, bias=self.bias, bn=self.bn)

        # 网络输出层，不使用BN
        self.conv_d4 = Conv(self.base, self.channels, kernel_size=3, padding=1, stride=1, bias=self.bias, bn=False, act=False)
        


    def _make_layers(self, in_channels, out_channels, num_ResBlock, upsample=False, attention=False, bias=False, bn=False):

        layers = []

        # 添加上采样模块或者调整模块
        if upsample:
            layers.append(
                ConvTranspose(in_channels, out_channels, kernel_size=4, padding=1, stride=2, bias=bias, bn=bn, act=True)
            )
        else:
            layers.append(
                Conv(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=bias, bn=bn, act=True)      
            )

        # 添加残差块
        layers.extend([ResBlock(out_channels, bias=bias, bn=bn) for _ in range(num_ResBlock)])

        # 注意力模块
        if attention:
            # 只要使用了bn或者bias中的一个，那么bias就使用了
            is_bias = bn or bias
            layers.append(AttentionConv(out_channels, out_channels, kernel_size=7, padding=3, groups=8, bias=is_bias)) 
        return nn.Sequential(*layers)


    def forward(self,input):
        
        '''编码器的前半部分'''
        y = self.sample1(input)
        y = self.sample2(y)
        y = self.sample3(y)
        
        if self.mode == "train":
            y = self.quantization(64 * (y - 0.5))
            E_y  = compute_E_y(y)

        elif self.mode == "test":
            zeros_tensor = torch.zeros_like(y)
            ones_tensor = torch.ones_like(y)
            y = torch.where(y < 0.5,zeros_tensor,ones_tensor)


        '''解码器'''
        output = self.recon_feat(y)
        output = self.layer0(output)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)

        output = self.conv_d4(output)
        if self.mode == "train":
            return output,E_y
        else:
            return output, y


class net_DWConvDense(net_DWConv):

    def __init__(self,channels,base,M,mode):
        super().__init__(channels=channels, base=base, M=M, mode=mode)


        # 编码端网络输出，加sigmoid激活函数，训练很不理想
        # 改成relu激活函数，很好
        self.output_act = nn.ReLU(inplace=True)


    def _make_layers(self, in_channels, out_channels, num_ResBlock, upsample=False, attention=False, bias=False, bn=False):

        layers = []

        # 添加上采样模块或者调整模块
        if upsample:
            layers.append(
                ConvTranspose(in_channels, out_channels, kernel_size=4, padding=1, stride=2, bias=bias, bn=bn, act=True)
            )
        else:
            layers.append(
                Conv(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=bias, bn=bn, act=True) 
            )
        
        # 添加残差块
        layers.extend([ResBlock(out_channels, bias=bias, bn=bn) for _ in range(num_ResBlock)])

        # 注意力模块
        if attention:
            # 只要使用了bn或者bias中的一个，那么bias就使用了
            # is_bias = bn or bias
            # layers.append(AttentionConv(out_channels, out_channels, kernel_size=7, padding=3, groups=8, bias=bias)) 
            # layers.append(TransformerBlock(out_channels, out_channels, act=True))

            layers.append(Conv(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=bias, bn=bn, act=True))             

            # layers.extend(
            #     [
            #         CBAM(out_channels, 4),
            #         Conv(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=bias, bn=bn, act=True)
            #     ]
            # )
            

        return nn.Sequential(*layers)


    def forward(self,input):
        
        '''编码器的前半部分'''
        y = self.sample1(input)
        y = self.sample2(y)
        y = self.sample3(y)
        
        if self.mode == "train":
            y = self.quantization(64 * (y - 0.5))
            E_y  = compute_E_y(y)

        elif self.mode == "test":
            zeros_tensor = torch.zeros_like(y)
            ones_tensor = torch.ones_like(y)
            y = torch.where(y < 0.5,zeros_tensor,ones_tensor)


        '''解码器'''
        # 8x下采样
        output = self.recon_feat(y)
        output = self.layer0(output)
        # 4x下采样
        output = self.layer1(output)+F.interpolate(output, size=None, scale_factor=2, mode='bilinear', align_corners=None)
        # 2x下采样
        output = self.layer2(output)+F.interpolate(output, size=None, scale_factor=2, mode='bilinear', align_corners=None)
        # 原始图像尺寸
        output = self.layer3(output)+F.interpolate(output, size=None, scale_factor=2, mode='bilinear', align_corners=None)

        # output = self.layer0(output)
        # # 4x下采样
        # output = self.layer1(output)
        # # 2x下采样
        # output = self.layer2(output)
        # # 原始图像尺寸
        # output = self.layer3(output)

        output = self.conv_d4(output)
        if hasattr(self, "output_act"):
            output = self.output_act(output)

        if self.mode == "train":
            return output,E_y
        else:
            return output, y


class encoder(nn.Module):

    def __init__(self,channels,base,M,mode):
        super().__init__()

        self.channels = channels
        self.base = base
        self.M = M
        self.mode = mode
        self.nResBlock_d0 = 1
        self.nResBlock_d1 = 1
        self.nResBlock_d2 = 1
        self.nResBlock_d3 = 1

        self.bias = False
        self.bn = False

        self.quantization = nn.Sigmoid()

        '''编码器的前半部分'''
        self.sample1 = Conv(self.channels,self.base, kernel_size=3,padding=1,stride=2,bias=False, bn=False, act=True)

        # # 点卷积进行分组，并使用通道重排
        num_groups_point_conv = self.base // 16
        self.sample2 = DPWConvChannelShuffle(self.base, self.base, kernel_size=3, padding=1, stride=2, bias=False, 
                                num_groups_point_conv=num_groups_point_conv, channel_shuffle=True, act=True)
        self.sample3 = DPWConvChannelShuffle(self.base, self.M, kernel_size=3, padding=1, stride=2, bias=False, 
                                     num_groups_point_conv=1, channel_shuffle=False, act=False)
