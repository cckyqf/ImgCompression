import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1, bias=True, bn=False, act=False):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        if bn:
            bias = False
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, autopad(kernel_size, padding), groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.act = nn.ReLU(inplace=True) if act else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x
    

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
    def __init__(self, base, bias=False, bn=False):
        super().__init__()

        self.base = base
        self.relu = nn.ReLU(inplace=True)
        
        # self.conv1 = nn.Conv2d(self.base,self.base,kernel_size=3,padding=1,stride=1,bias=False)
        # self.conv2 = nn.Conv2d(self.base,self.base,kernel_size=3,padding=1,stride=1,bias=False)
        self.conv1 = Conv(self.base,self.base,kernel_size=3,padding=1,stride=1,bias=bias, bn=bn, act=False)
        self.conv2 = Conv(self.base,self.base,kernel_size=3,padding=1,stride=1,bias=bias, bn=bn, act=False)

    def forward(self,input):
        output = self.conv1(input)
        output = self.relu(output)
        output = self.conv2(output)
        output = output + input

        output = self.relu(output)

        return output


class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

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

        out = self.relu(out)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)

# depthwise separable convolution
# MobileNetV1的结构：Conv+BN+ReLu -> DW+BN+ReLu + PW+BN+ReLu
#  depthwise卷积和pointwise卷积
class DPWConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        bias: bool = True,
        act_output = True,
    ):
        super().__init__()
      
        self.act_output = act_output

        self.depth_conv = nn.Conv2d(in_channels, in_channels,  kernel_size=kernel_size, padding=padding, stride=stride, groups=in_channels, bias=bias)
        self.point_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=bias)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        
        x = self.act(self.depth_conv(x))
        x = self.point_conv(x)
        if self.act_output:
            x = self.act(x)
        
        return x


class DPWConvChannelShuffle(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        bias: bool = True,
        num_groups_point_conv: int = 1,
        channel_shuffle: bool = True,
        act = True,
    ):
        
        super().__init__()

        self.c_shuffle = ChannelShuffle(channels=in_channels, groups=num_groups_point_conv) if channel_shuffle else None

        self.depth_conv = nn.Conv2d(in_channels, in_channels,  kernel_size=kernel_size, padding=padding, stride=stride, groups=in_channels, bias=bias)
        self.point_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=num_groups_point_conv, bias=bias)

        self.act = nn.ReLU(inplace=True) if act else None

    def forward(self, x):
        if self.c_shuffle is not None:
            x = self.c_shuffle(x)
        
        x = self.depth_conv(x)
        x = self.point_conv(x)
        if self.act is not None:
            x = self.act(x)

        return x

class DepthWiseConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        bias: bool = True,
        num_groups_point_conv: int = 1,
        channel_shuffle: bool = True,
        act = True,
    ):
        
        super().__init__()

        self.c_shuffle = ChannelShuffle(channels=in_channels, groups=num_groups_point_conv) if channel_shuffle else None

        self.depth_conv = nn.Conv2d(in_channels, in_channels,  kernel_size=kernel_size, padding=padding, stride=stride, groups=in_channels, bias=bias)
        self.point_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=num_groups_point_conv, bias=bias)

        self.act = nn.ReLU(inplace=True) if act else None

    def forward(self, x):
        if self.c_shuffle is not None:
            x = self.c_shuffle(x)
        
        x = self.depth_conv(x)
        x = self.point_conv(x)
        if self.act is not None:
            x = self.act(x)

        return x

class ConvTranspose(nn.Module):
    # Standard convolution
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1, bias=True, bn=False, act=False):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        if bn:
            bias = False
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.act = nn.ReLU(inplace=True) if act else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x

    
class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads=4, num_layers=1, act=False):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2, bn=False, act=True)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

        self.act = nn.ReLU(inplace=True) if act else None

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).unsqueeze(0).transpose(0, 3).squeeze(3)

        p = self.tr(p + self.linear(p)).unsqueeze(3).transpose(0, 3).reshape(b, self.c2, w, h)
        if self.act is not None:
            p = self.act(p)
        return p


class ChannelShuffle(nn.Module):

    def __init__(self, channels, groups):
        super(ChannelShuffle, self).__init__()
        if channels % groups != 0:
            raise ValueError("通道数必须可以整除组数")
        self.groups = groups

    def channel_shuffle(self, x):
        # x[batch_size, channels, H, W]
        batch, channels, height, width = x.size()
        channels_per_group = channels // self.groups  # 每组通道数
        x = x.view(batch, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batch, channels, height, width)
        return x

    def forward(self, x):
        return self.channel_shuffle(x)


