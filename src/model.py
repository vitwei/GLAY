import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import numpy as np
import time
from torch import einsum
from dataclasses import dataclass
from typing import  List
import ipdb
import math
from functools import partial
from timm.models.layers import DropPath
import warnings
from typing import Optional, Callable

#from src.vmamba.models.vmamba import VSSBlock,SS2D
from torch.nn.init import _calculate_fan_in_and_fan_out
import torch.nn.init as init
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

pi = 3.141592653589793

class RGB_HSV(nn.Module):
    def __init__(self):
        super(RGB_HSV, self).__init__()
        self.density_k = torch.nn.Parameter(torch.full([1],0.2)) # k is reciprocal to the paper mentioned
        self.gated = False
        self.gated2= False
        self.alpha = 1.0
        self.this_k = 0
        
    def HVIT(self, img):
        eps = 1e-8
        device = img.device
        dtypes = img.dtype
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(device).to(dtypes)
        value = img.max(1)[0].to(dtypes)
        img_min = img.min(1)[0].to(dtypes)
        hue[img[:,2]==value] = 4.0 + ( (img[:,0]-img[:,1]) / (value - img_min + eps)) [img[:,2]==value]
        hue[img[:,1]==value] = 2.0 + ( (img[:,2]-img[:,0]) / (value - img_min + eps)) [img[:,1]==value]
        hue[img[:,0]==value] = (0.0 + ((img[:,1]-img[:,2]) / (value - img_min + eps)) [img[:,0]==value]) % 6

        hue[img.min(1)[0]==value] = 0.0
        hue = hue/6.0

        saturation = (value - img_min ) / (value + eps )
        saturation[value==0] = 0

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        
        k = self.density_k
        self.this_k = k.item()
        
        color_sensitive = ((value * 0.5 * pi).sin() + eps).pow(k)
        cx = (2.0 * pi * hue).cos()
        cy = (2.0 * pi * hue).sin()
        X = color_sensitive * saturation * cx
        Y = color_sensitive * saturation * cy
        Z = value
        xyz = torch.cat([X, Y, Z],dim=1)
        return xyz
    
    def PHVIT(self, img):
        eps = 1e-8
        H,V,I = img[:,0,:,:],img[:,1,:,:],img[:,2,:,:]
        
        # clip
        H = torch.clamp(H,-1,1)
        V = torch.clamp(V,-1,1)
        I = torch.clamp(I,0,1)
        
        v = I
        k = self.this_k
        color_sensitive = ((v * 0.5 * pi).sin() + eps).pow(k)
        H = (H) / (color_sensitive + eps)
        V = (V) / (color_sensitive + eps)
        H = torch.clamp(H,-1,1)
        V = torch.clamp(V,-1,1)
        h = torch.atan2(V,H) / (2*pi)
        h = h%1
        s = torch.sqrt(H**2 + V**2 + eps)
        
        if self.gated:
            s = s * 1.3
        
        s = torch.clamp(s,0,1)
        v = torch.clamp(v,0,1)
        
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)
        
        hi = torch.floor(h * 6.0)
        f = h * 6.0 - hi
        p = v * (1. - s)
        q = v * (1. - (f * s))
        t = v * (1. - ((1. - f) * s))
        
        hi0 = hi==0
        hi1 = hi==1
        hi2 = hi==2
        hi3 = hi==3
        hi4 = hi==4
        hi5 = hi==5
        
        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]
        
        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]
        
        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]
        
        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]
        
        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]
        
        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]
                
        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        if self.gated2:
            rgb = rgb * self.alpha
        return rgb

    def HSV(self, img):
        eps = 1e-8
        device = img.device
        dtypes = img.dtype
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(device).to(dtypes)
        value = img.max(1)[0].to(dtypes)
        img_min = img.min(1)[0].to(dtypes)
        hue[img[:,2]==value] = 4.0 + ( (img[:,0]-img[:,1]) / (value - img_min + eps)) [img[:,2]==value]
        hue[img[:,1]==value] = 2.0 + ( (img[:,2]-img[:,0]) / (value - img_min + eps)) [img[:,1]==value]
        hue[img[:,0]==value] = (0.0 + ((img[:,1]-img[:,2]) / (value - img_min + eps)) [img[:,0]==value]) % 6

        hue[img.min(1)[0]==value] = 0.0
        hue = hue/6.0

        saturation = (value - img_min ) / (value + eps )
        saturation[value==0] = 0

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        
        cx = (2.0 * pi * hue).cos()
        cy = (2.0 * pi * hue).sin()
        X =   saturation * cx
        Y =   saturation * cy
        Z = value
        xyz = torch.cat([X, Y, Z],dim=1)
        return xyz
    
    def RHSV(self, img):
        eps = 1e-8
        X,Y,Z = img[:,0,:,:],img[:,1,:,:],img[:,2,:,:]
        
        # clip
        X = torch.clamp(X,-1,1)
        Y = torch.clamp(Y,-1,1)
        Z = torch.clamp(Z,0,1)
        
        H = torch.atan2(X,Y) / (2*pi)
        H = H%1
        S = torch.sqrt(X**2 + Y**2 + eps)

        
        S = torch.clamp(S,0,1)
        V = torch.clamp(Z,0,1)
        
        r = torch.zeros_like(H)
        g = torch.zeros_like(H)
        b = torch.zeros_like(H)
        
        hi = torch.floor(H * 6.0)
        f = H * 6.0 - hi
        p = V * (1. - S)
        q = V * (1. - (f * S))
        t = V * (1. - ((1. - f) * S))
        
        hi0 = hi==0
        hi1 = hi==1
        hi2 = hi==2
        hi3 = hi==3
        hi4 = hi==4
        hi5 = hi==5
        
        r[hi0] = V[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]
        
        r[hi1] = q[hi1]
        g[hi1] = V[hi1]
        b[hi1] = p[hi1]
        
        r[hi2] = p[hi2]
        g[hi2] = V[hi2]
        b[hi2] = t[hi2]
        
        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = V[hi3]
        
        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = V[hi4]
        
        r[hi5] = V[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]
                
        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        return rgb


class RGB_HVI(nn.Module):
    def __init__(self):
        super(RGB_HVI, self).__init__()
        self.density_k = torch.nn.Parameter(torch.full([1],0.2)) # k is reciprocal to the paper mentioned
        self.gated = False
        self.gated2= False
        self.alpha = 1.0
        self.this_k = 0
        
    def HVIT(self, img):
        eps = 1e-8
        device = img.device
        dtypes = img.dtype
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(device).to(dtypes)
        value = img.max(1)[0].to(dtypes)
        img_min = img.min(1)[0].to(dtypes)
        hue[img[:,2]==value] = 4.0 + ( (img[:,0]-img[:,1]) / (value - img_min + eps)) [img[:,2]==value]
        hue[img[:,1]==value] = 2.0 + ( (img[:,2]-img[:,0]) / (value - img_min + eps)) [img[:,1]==value]
        hue[img[:,0]==value] = (0.0 + ((img[:,1]-img[:,2]) / (value - img_min + eps)) [img[:,0]==value]) % 6

        hue[img.min(1)[0]==value] = 0.0
        hue = hue/6.0

        saturation = (value - img_min ) / (value + eps )
        saturation[value==0] = 0

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        
        k = self.density_k
        self.this_k = k.item()
        
        color_sensitive = ((value * 0.5 * pi).sin() + eps).pow(k)
        cx = (2.0 * pi * hue).cos()
        cy = (2.0 * pi * hue).sin()
        X = color_sensitive * saturation * cx
        Y = color_sensitive * saturation * cy
        Z = value
        xyz = torch.cat([X, Y, Z],dim=1)
        return xyz
    
    def PHVIT(self, img):
        eps = 1e-8
        H,V,I = img[:,0,:,:],img[:,1,:,:],img[:,2,:,:]
        
        # clip
        H = torch.clamp(H,-1,1)
        V = torch.clamp(V,-1,1)
        I = torch.clamp(I,0,1)
        
        v = I
        k = self.this_k
        color_sensitive = ((v * 0.5 * pi).sin() + eps).pow(k)
        H = (H) / (color_sensitive + eps)
        V = (V) / (color_sensitive + eps)
        H = torch.clamp(H,-1,1)
        V = torch.clamp(V,-1,1)
        h = torch.atan2(V,H) / (2*pi)
        h = h%1
        s = torch.sqrt(H**2 + V**2 + eps)
        
        if self.gated:
            s = s * 1.3
        
        s = torch.clamp(s,0,1)
        v = torch.clamp(v,0,1)
        
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)
        
        hi = torch.floor(h * 6.0)
        f = h * 6.0 - hi
        p = v * (1. - s)
        q = v * (1. - (f * s))
        t = v * (1. - ((1. - f) * s))
        
        hi0 = hi==0
        hi1 = hi==1
        hi2 = hi==2
        hi3 = hi==3
        hi4 = hi==4
        hi5 = hi==5
        
        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]
        
        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]
        
        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]
        
        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]
        
        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]
        
        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]
                
        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        if self.gated2:
            rgb = rgb * self.alpha
        return rgb
##HVI
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
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
       
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.Tanh = nn.Tanh()
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)
        return x

class HV_LCA(nn.Module):
    def __init__(self, dim,num_heads, bias=False):
        super(HV_LCA, self).__init__()
        self.gdfn = IEL(dim) # IEL and CDL have same structure
        self.norm = LayerNorm(dim)
        self.ffn = CAB(dim, num_heads, bias)
        
    def forward(self, x, y):
        x = x + self.ffn(self.norm(x),self.norm(y))
        x = self.gdfn(self.norm(x))
        return x
    
class MyHV_LCA(nn.Module):
    def __init__(self, dim,num_heads, bias=False,dropout_prob=0.):
        super(MyHV_LCA, self).__init__()
        self.norm = LayerNorm(dim)
        self.ffn = CAB(dim, num_heads, bias=bias)
        self.dropout = nn.Dropout(dropout_prob)  # 定义 dropout 
        self.attn= Attention_block(dim,dim,dim)
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels=dim*2, out_channels=dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(dim, eps=0.0001, momentum = 0.95),
            nn.PReLU(),
        )
    def forward(self, x, y):
        # 在 ffn 输出后应用 dropout
        x = x + self.dropout(self.ffn(self.norm(x), self.norm(y)))
        # 在 dual 输出后应用 dropout
        x = self.dropout(self.conv(torch.cat([x,self.attn(self.norm(x),self.norm(y))],dim=1)))
        return x


class I_LCA(nn.Module):
    def __init__(self, dim,num_heads, bias=False):
        super(I_LCA, self).__init__()
        self.norm = LayerNorm(dim)
        self.gdfn = IEL(dim)
        self.ffn = CAB(dim, num_heads, bias=bias)
        
        
    def forward(self, x, y):
        x = x + self.ffn(self.norm(x),self.norm(y))
        x = x + self.gdfn(self.norm(x)) 
        return x

class MyI_LCA(nn.Module):
    def __init__(self, dim, num_heads, bias=False, dropout_prob=0.):
        super(MyI_LCA, self).__init__()
        self.norm = LayerNorm(dim)
        self.ffn = BigsparseCAB(dim, num_heads, bias=bias)
        self.dropout = nn.Dropout(dropout_prob)  # 定义 dropout 
        self.act=nn.PReLU()
        self.double_conv= nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(dim, eps=0.0001, momentum = 0.95),
            nn.PReLU(),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim, eps=0.0001, momentum = 0.95),
            nn.PReLU(),
        )
    def forward(self, x, y):
        # 在 ffn 输出后应用 dropout
        x = x + self.dropout(self.ffn(self.norm(x), self.norm(y)))
        # 在 dual 输出后应用 dropout
        x = x + self.double_conv(self.norm(x))
        return x

class MyI_LCA2(nn.Module):
    def __init__(self, dim, num_heads, bias=False, dropout_prob=0.):
        super(MyI_LCA2, self).__init__()
        self.norm = LayerNorm(dim)
        self.ffn = CAB(dim, num_heads, bias=bias)
        self.dropout = nn.Dropout(dropout_prob)  # 定义 dropout 
        self.attn= Attention_block(dim,dim,dim)
        #self.se_block = se_block(dim=dim)
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels=dim*2, out_channels=dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim, eps=0.0001, momentum = 0.95),
            nn.PReLU(),
        )
    def forward(self, x, y):
        # 在 ffn 输出后应用 dropout
        residual = x
        x = x + self.dropout(self.ffn(self.norm(x), self.norm(y)))
        # 在 dual 输出后应用 dropout
        x = residual +self.dropout(self.conv(torch.cat([x,self.attn(self.norm(x),self.norm(y))],dim=1)))
        return x

class MyI_LCA3(nn.Module):
    def __init__(self, dim, num_heads, bias=False, dropout_prob=0.):
        super(MyI_LCA3, self).__init__()
        self.norm = LayerNorm(dim)
        self.ffn =se_block(channels=dim)
        self.dropout = nn.Dropout(dropout_prob)  # 定义 dropout 
        self.attn= Attention_block(dim,dim,dim)
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels=dim*2, out_channels=dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(dim, eps=0.0001, momentum = 0.95),
            nn.PReLU(),
        )
    def forward(self, x, y):
        # 在 ffn 输出后应用 dropout
        residual = x
        x = x + self.dropout(self.conv(torch.cat([x,self.attn(self.norm(x),self.norm(y))],dim=1)))
        # 在 dual 输出后应用 dropout
        x = residual +self.dropout(self.ffn(self.norm(x)))
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2, bias=False):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class MyHV_LCA2(nn.Module):
    def __init__(self, dim, num_heads, bias=False, dropout_prob=0.):
        super(MyHV_LCA2, self).__init__()
        self.norm = LayerNorm(dim)
        self.ffn = CAB(dim, num_heads, bias=bias)
        self.dropout = nn.Dropout(dropout_prob)  # 定义 dropout 
        self.attn= Attention_block(dim,dim,dim)
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels=dim*2, out_channels=dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(dim, eps=0.0001, momentum = 0.95),
            nn.PReLU(),
        )
    def forward(self, x, y):
        # 在 ffn 输出后应用 dropout
        residual = x
        x = x + self.dropout(self.ffn(self.norm(x), self.norm(y)))
        # 在 dual 输出后应用 dropout
        x = residual + self.conv(torch.cat([x,self.attn(self.norm(x),self.norm(y))],dim=1))
        return x


class BigScaleConvModule(nn.Module):
    def __init__(self, in_channels):
        super(BigScaleConvModule, self).__init__()
        # 1x1 convolutions
        self.conv1 =nn.Conv2d(in_channels=in_channels, out_channels=in_channels//4, kernel_size=1, padding=0, bias=False)

        self.conv2 =nn.Conv2d(in_channels=in_channels, out_channels=in_channels//4, kernel_size=1, padding=0, bias=False)

        self.conv3 =nn.Conv2d(in_channels=in_channels, out_channels=in_channels//4, kernel_size=1, padding=0, bias=False)

        self.conv4 =nn.Conv2d(in_channels=in_channels, out_channels=in_channels//4, kernel_size=1, padding=0, bias=False)

        self.out=nn.Conv2d(in_channels=in_channels*2, out_channels=in_channels, kernel_size=1, padding=0, bias=False)
  
    def up(self,x,size):
        return F.interpolate(x,size,mode='bilinear',align_corners=True)
    
    def pool(self,x,size):
        avgpool=nn.AdaptiveAvgPool2d(size)
        return avgpool(x)
    
    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.up(self.conv1(self.pool(x, 1)), size)
        feat2 = self.up(self.conv2(self.pool(x, 2)), size)
        feat3 = self.up(self.conv3(self.pool(x, 3)), size)
        feat4 = self.up(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1) #concat 四个池化的结果
        x = self.out(x)    
        return x

class SmallScaleConvModule(nn.Module):
    def __init__(self, in_channels):
        super(SmallScaleConvModule, self).__init__()
        # Convolutions with different kernel sizes and dilations
        self.conv1x3 = nn.Conv2d(in_channels, in_channels//4, kernel_size=(1, 3), dilation=1, padding=(0, 1))
        self.conv3x1 = nn.Conv2d(in_channels, in_channels//4, kernel_size=(3, 1), dilation=1, padding=(1, 0))
        self.conv3x3_1 = nn.Conv2d(in_channels, in_channels//4, kernel_size=3, dilation=3, padding=3)
        self.conv3x3 = nn.Conv2d(in_channels, in_channels//4, kernel_size=3, padding=1)
        # Final 3x3 convolution
        self.out=nn.Sequential(
            nn.Conv2d(in_channels=in_channels*2, out_channels=in_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels, eps=0.0001, momentum = 0.95),
        )
    def forward(self, x):
        # residual=self.conv1x1_0(x)
        # First path
        x2 =self.conv3x3(x)
        # print("x1:", x1.shape)
        x3 = self.conv1x3(x)
        x4 = self.conv3x1(x)
        x5 = self.conv3x3_1(x)

        out = self.out(torch.cat([x,x2,x3,x4,x5],dim=1))

        return out

# Downsample Block
class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        # import pdb;pdb.set_trace()
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).flatten(2).transpose(1,2).contiguous()  # B H*W C
        return out

class NormDownsample(nn.Module):
    def __init__(self,in_ch,out_ch,scale=0.5,use_norm=False):
        super(NormDownsample, self).__init__()
        self.use_norm=use_norm
        if self.use_norm:
            self.norm=LayerNorm(out_ch)
        self.prelu = nn.PReLU()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch,kernel_size=3,stride=1, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=scale))
    def forward(self, x):
        x = self.down(x)
        x = self.prelu(x)
        if self.use_norm:
            x = self.norm(x)
            return x
        else:
            return x
# Upsample Block
class NormDownsample2(nn.Module):
    def __init__(self,in_ch,in_ch2,out_ch,scale=0.5,use_norm=False):
        super(NormDownsample2, self).__init__()
        self.use_norm=use_norm
        if self.use_norm:
            self.norm=LayerNorm(out_ch)
        self.prelu = nn.PReLU()
        self.down_scale = nn.Sequential(
            nn.Conv2d(in_ch, out_ch,kernel_size=3,stride=1, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=scale))
        self.down=nn.Conv2d(out_ch+in_ch2, out_ch,kernel_size=1,stride=1, padding=0, bias=False)
    def forward(self, x,y):
        x = self.down_scale(x)
        x = torch.cat([x, y],dim=1)
        x = self.down(x)
        x = self.prelu(x)
        if self.use_norm:
            x = self.norm(x)
            return x
        else:
            return x
class ConvDownsample(nn.Module):
    def __init__(self,in_ch,out_ch,scale=0.5,use_norm=False):
        super(ConvDownsample, self).__init__()
        self.use_norm=use_norm
        if self.use_norm:
            self.norm=nn.LayerNorm(out_ch)
        self.prelu = nn.PReLU()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, 1, bias=False),
            nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
                                   )
    def forward(self, x):
        x = self.down(x)
        x = self.prelu(x)
        if self.use_norm:
            x = self.norm(x)
            return x
        else:
            return x
        
class ConvUpsample(nn.Module):
    def __init__(self,in_ch,out_ch,scale=0.5,use_norm=False):
        super(ConvUpsample, self).__init__()
        self.use_norm=use_norm
        if self.use_norm:
            self.norm=nn.LayerNorm(out_ch)
        self.prelu = nn.PReLU()
        self.up_scale =nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, 1, bias=False),
            nn.ConvTranspose2d(in_ch, out_ch, stride=2,
                                   kernel_size=2, padding=0, output_padding=0),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False))
        self.up = nn.Conv2d(out_ch*2,out_ch,kernel_size=1,stride=1, padding=0, bias=False)
    def forward(self, x,y):
        x = self.up_scale(x)
        x = torch.cat([x, y],dim=1)
        x = self.up(x)
        x = self.prelu(x)
        if self.use_norm:
            return self.norm(x)
        else:
            return x


class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel
        
    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1,2).contiguous() # B H*W C
        return out
    
class NormUpsample(nn.Module):
    def __init__(self, in_ch,out_ch,scale=2,use_norm=False):
        super(NormUpsample, self).__init__()
        self.use_norm=use_norm
        if self.use_norm:
            self.norm=LayerNorm(out_ch)
        self.prelu = nn.PReLU()
        self.up_scale = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=1, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=scale))
        self.up = nn.Conv2d(out_ch*2,out_ch,kernel_size=1,stride=1, padding=0, bias=False)
            
    def forward(self, x,y):
        x = self.up_scale(x)
        x = torch.cat([x, y],dim=1)
        x = self.up(x)
        x = self.prelu(x)
        if self.use_norm:
            return self.norm(x)
        else:
            return x

class ResidualUpSample(nn.Module):
    def __init__(self, in_channels,out_channel, bias=False,use_norm=False):
        super(ResidualUpSample, self).__init__()
        self.use_norm=use_norm
        if self.use_norm:
            self.norm=LayerNorm(out_channel)
        self.top = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=bias),
                                nn.PReLU(),
                                nn.ConvTranspose2d(in_channels, out_channel, 2, stride=2, padding=0, output_padding=0,bias=bias),
                                nn.PReLU(),
                                nn.Conv2d(out_channel, out_channel, 1, stride=1, padding=0, bias=bias))

        self.bot = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias),
                                nn.Conv2d(in_channels, out_channel, 1, stride=1, padding=0, bias=bias))
        self.up = nn.Conv2d(out_channel*2,out_channel,kernel_size=1,stride=1, padding=0, bias=False)
        self.prelu = nn.PReLU()

    def forward(self, x,y):
        top = self.top(x)
        bot = self.bot(x)
        x = F.relu6(top)+bot
        x=torch.cat([x,y],dim=1)
        x=self.up(x)
        if self.use_norm:
            x = self.norm(x)
            return x
        else:
            return x

class ResidualDownSample(nn.Module):
    def __init__(self, in_channels,out_channel,bias=False,use_norm=False):
        super(ResidualDownSample, self).__init__()
        self.use_norm=use_norm
        if self.use_norm:
            self.norm=LayerNorm(out_channel)
        self.top = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=bias),
                                nn.PReLU(),
                                nn.Conv2d(in_channels, out_channel, kernel_size=4, stride=2, padding=1),   
                                nn.PReLU(),
                                nn.Conv2d(out_channel, out_channel, 1, stride=1, padding=0, bias=bias))

        self.bot = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=bias),
                                nn.Conv2d(in_channels, out_channel, 1, stride=1, padding=0, bias=bias))
    def forward(self, x):
        top = self.top(x)
        bot = self.bot(x)
        x = F.relu6(top)+bot
        if self.use_norm:
            x = self.norm(x)
            return x
        else:
            return x
class NormUpsample2(nn.Module):
    def __init__(self, in_ch,out_ch,scale=2,use_norm=False):
        super(NormUpsample2, self).__init__()
        self.use_norm=use_norm
        if self.use_norm:
            self.norm=nn.LayerNorm(out_ch)
        self.prelu = nn.PReLU()
        self.up_scale = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=1, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=scale))
        self.up = nn.Conv2d(out_ch,out_ch,kernel_size=1,stride=1, padding=0, bias=False)
            
    def forward(self, x):
        x = self.up_scale(x)
        x = self.up(x)
        x = self.prelu(x)
        if self.use_norm:
            return self.norm(x)
        else:
            return x
#########################################
########### feed-forward network #############
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU,drop = 0., use_eca=False):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                act_layer())
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim,hidden_dim,groups=hidden_dim,kernel_size=3,stride=1,padding=1),
                        act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.eca = nn.Identity()

    def forward(self, x):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h = hh, w = hh)
        # bs,hidden_dim,32x32

        x = self.dwconv(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h = hh, w = hh)

        x = self.linear2(x)
        x = self.eca(x)

        return x


class FRFN(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU,drop = 0., use_eca=False):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim*2),
                                act_layer())
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim,hidden_dim,groups=hidden_dim,kernel_size=3,stride=1,padding=1),
                        act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.dim_conv = self.dim // 4
        self.dim_untouched = self.dim - self.dim_conv 
        self.partial_conv3 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)

    def forward(self, x):
        # bs x hw x c
        bs, c, h,w = x.size()



        x1, x2,= torch.split(x, [self.dim_conv,self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h = h, w = w)

        x = self.linear1(x)
        #gate mechanism
        x_1,x_2 = x.chunk(2,dim=-1)

        x_1 = rearrange(x_1, ' b (h w) (c) -> b c h w ', h = h, w = w)
        x_1 = self.dwconv(x_1)
        x_1 = rearrange(x_1, ' b c h w -> b (h w) c', h = h, w = w)
        x = x_1 * x_2
        
        x = self.linear2(x)
        # x = self.eca(x)
        x =rearrange(x, ' b (h w) (c) -> b c h w ', h = h, w = w)
        return x
#########################################

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1):
        super(ConvBlock, self).__init__()
        self.strides = strides
        self.in_channel=in_channel
        self.out_channel=out_channel
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.conv11 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, padding=0)

    def forward(self, x):
        out1 = self.block(x)
        out2 = self.conv11(x)
        out = out1 + out2
        return out

class LinearProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., bias=True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias = bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_,1,1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1] 
        return q,k,v

class ConvAttention(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(ConvAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

########### window-based self-attention #############
def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate !=1:
        x = x.permute(0,3,1,2) # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size,dilation=dilation_rate,padding=4*(dilation_rate-1),stride=win_size) # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0,2,1).contiguous().view(-1, C, win_size, win_size) # B' ,C ,Wh ,Ww
        windows = windows.permute(0,2,3,1).contiguous() # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C) # B' ,Wh ,Ww ,C
    return windows

def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate !=1:
        x = windows.permute(0,5,3,4,1,2).contiguous() # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4*(dilation_rate-1),stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, win_size,num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0]) # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1]) # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)
            
        if token_projection =='linear':
            self.qkv = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        else:
            raise Exception("Projection error!") 
        
        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x,attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1)//relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d = ratio)
    
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)',d = ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N*ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N*ratio)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'
    
class WindowAttention_sparse(nn.Module):
    def __init__(self, dim, win_size,num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0]) # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1]) # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)
            
        if token_projection =='linear':
            self.qkv = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        else:
            raise Exception("Projection error!") 
        
        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.w = nn.Parameter(torch.ones(2)) 

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x,attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1)//relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d = ratio)
    
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)',d = ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N*ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N*ratio)
            attn0 = self.softmax(attn)
            attn1 = self.relu(attn)**2#b,h,w,c
        else:
            attn0 = self.softmax(attn)
            attn1 = self.relu(attn)**2
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        attn = attn0*w1+attn1*w2
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'

#########################################
###########Transformer Block#############
class TransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,token_projection='linear',token_mlp='leff',att=True,sparseAtt=False):
        super().__init__()

        self.att = att
        self.sparseAtt = sparseAtt

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"

        if self.att:
            self.norm1 = norm_layer(dim)
            if self.sparseAtt:
                self.attn = WindowAttention_sparse(
                    dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                    token_projection=token_projection)
            else:
                self.attn = WindowAttention(
                    dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                    token_projection=token_projection)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if token_mlp in ['ffn','mlp']:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,act_layer=act_layer, drop=drop) 
        elif token_mlp=='leff':
            self.mlp =  LeFF(dim,mlp_hidden_dim,act_layer=act_layer, drop=drop)
        elif token_mlp=='frfn':
            self.mlp =  FRFN(dim,mlp_hidden_dim,act_layer=act_layer, drop=drop)
        else:
            raise Exception("FFN error!") 


    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
        
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"win_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def forward(self, x, mask=None):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        
        ## input mask
        if mask != None:
            input_mask = F.interpolate(mask, size=(H,W)).permute(0,2,3,1)
            input_mask_windows = window_partition(input_mask, self.win_size) # nW, win_size, win_size, 1
            attn_mask = input_mask_windows.view(-1, self.win_size * self.win_size) # nW, win_size*win_size
            attn_mask = attn_mask.unsqueeze(2)*attn_mask.unsqueeze(1) # nW, win_size*win_size, win_size*win_size
            attn_mask = attn_mask.masked_fill(attn_mask!=0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        ## shift mask
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            shift_mask = torch.zeros((1, H, W, 1)).type_as(x)
            h_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            shift_mask_windows = window_partition(shift_mask, self.win_size)  # nW, win_size, win_size, 1
            shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size) # nW, win_size*win_size
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(2) # nW, win_size*win_size, win_size*win_size
            shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(shift_attn_mask == 0, float(0.0))
            attn_mask = attn_mask + shift_attn_mask if attn_mask is not None else shift_attn_mask

    
        shortcut = x

        if self.att:
            x = self.norm1(x)
            x = x.view(B, H, W, C)

            # cyclic shift
            if self.shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_x = x

            # partition windows
            x_windows = window_partition(shifted_x, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
            x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C

        
            # W-MSA/SW-MSA
            attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, win_size*win_size, C

            # merge windows
            attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
            shifted_x = window_reverse(attn_windows, self.win_size, H, W)  # B H' W' C

            # reverse cyclic shift
            if self.shift_size > 0:
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = shifted_x
            x = x.view(B, H * W, C)
            x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        del attn_mask
        return x


#########################################
########### Basic layer of AST ################
class BasicASTLayer(nn.Module):
    def __init__(self, dim, output_dim, input_resolution, depth, num_heads, win_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 token_projection='linear',token_mlp='frfn', shift_flag=True,att=True,sparseAtt=False,stack=False):

        super().__init__()
        self.att = att
        self.sparseAtt = sparseAtt
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.stack=stack
        # build blocks
        if shift_flag:
            self.blocks = nn.ModuleList([
                TransformerBlock(dim=dim, input_resolution=input_resolution,
                                    num_heads=num_heads, win_size=win_size,
                                    shift_size=0 if (i % 2 == 0) else win_size // 2,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                    norm_layer=norm_layer,token_projection=token_projection,token_mlp=token_mlp,att=self.att,sparseAtt=self.sparseAtt)
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                TransformerBlock(dim=dim, input_resolution=input_resolution,
                                    num_heads=num_heads, win_size=win_size,
                                    shift_size=0,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                    norm_layer=norm_layer,token_projection=token_projection,token_mlp=token_mlp,att=self.att,sparseAtt=self.sparseAtt)
            for i in range(depth)])

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"    

    def forward(self, x, mask=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x,mask)
        return x

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.LeakyReLU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class MultiAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = nn.BatchNorm2d(dim)
        # Simple Channel Attention
        self.Wv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2, groups=dim, padding_mode='reflect')
        )
        self.Wg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.LeakyReLU(),
            # nn.ReLU(True),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        # Pixel Attention
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, padding=0, bias=True),
            nn.LeakyReLU(),
            # nn.ReLU(True),
            nn.Conv2d(dim // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.LeakyReLU(),
            # nn.ReLU(True),
            nn.Conv2d(dim * 4, dim, 1)
        )

    def forward(self, x):
        identity = x
        x = self.norm(x)
        x = torch.cat([self.Wv(x) * self.Wg(x), self.ca(x) * x, self.pa(x) * x], dim=1)
        x = self.mlp(x)
        x = identity + x
        return x


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel*2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))
    
class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)
        )

    def forward(self, x):
        x = self.main(x)
        return x
           
class TriScaleConv(nn.Module):
    def __init__(self, in_channels,outchannel,dilation=3, res=True,group=False):
        super(TriScaleConv, self).__init__()
        self.res = res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,outchannel,kernel_size=3,dilation=5,padding=5),
            nn.PReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels,outchannel,kernel_size=3,padding=1),
            nn.PReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels,outchannel,kernel_size=3,dilation=3,padding=3),
            nn.PReLU(),
        )
        self.merge = nn.Sequential(
            nn.Conv2d(outchannel*4,outchannel*2,kernel_size=1),
            nn.GELU(),
            nn.Conv2d(outchannel*2,outchannel,kernel_size=1),
        )
    def forward(self,x):
        x1 = self.conv1(x) + x
        x2 = self.conv2(x1) + x1
        x3 = self.conv3(x2) + x2
        out = torch.cat([x,x1,x2,x3],dim=1)
        out = self.merge(out)
        return x+out

class DualScaleConv(nn.Module):
    def __init__(self, in_channels,outchannel,dilation=3, res=True,group=False):
        super(DualScaleConv, self).__init__()
        self.res = res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,outchannel,kernel_size=3,dilation=4,padding=4),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels,outchannel,kernel_size=3,padding=1),
        )
        self.merge = nn.Sequential(
            nn.Conv2d(outchannel,outchannel*2,kernel_size=1),
            nn.GELU(),
            nn.Conv2d(outchannel*2,outchannel,kernel_size=1),
        )
    def forward(self,x):
        out = self.conv1(x) + self.conv2(x)
        out = self.merge(out)
        return x + out if self.res else out
    
class DynamicConv(nn.Module):
    def __init__(self, inchannels, mode='highPass', dilation=0, kernel_size=3, stride=1, kernelNumber=8):
        super(DynamicConv, self).__init__()
        self.stride = stride
        self.mode = mode
        self.kernel_size = kernel_size
        self.kernelNumber = inchannels
        self.conv = nn.Conv2d(inchannels, self.kernelNumber*kernel_size**2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(self.kernelNumber*kernel_size**2)
        self.act = nn.Softmax(dim=-2)
        self.relu=nn.GELU()
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.unfoldMask = []
        self.unfoldSize = kernel_size + dilation * (kernel_size - 1)
        self.pad = nn.ReflectionPad2d(self.unfoldSize//2)
        if mode == 'lowPass':
            for i in range(self.unfoldSize):
                for j in range(self.unfoldSize):
                    if (i % (dilation + 1) == 0) and (j % (dilation + 1) == 0):
                        self.unfoldMask.append(i * self.unfoldSize + j)
        elif mode != 'highPass':
            raise ValueError("Invalid mode. Expected 'lowPass' or 'highPass'.")
        
    def forward(self, x):
        copy = x
        filter = self.ap(x)
        filter = self.conv(filter)
        filter = self.bn(filter)
        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.unfoldSize).reshape(n, self.kernelNumber, c//self.kernelNumber, self.unfoldSize**2, h*w)
        if self.mode == 'lowPass':
            x = x[:,:,:,self.unfoldMask,:]
        n,c1,p,q = filter.shape
        filter = filter.reshape(n, c1//self.kernel_size**2, self.kernel_size**2, p*q).unsqueeze(2)
        filter = self.act(filter)
        out = torch.sum(x * filter, dim=3).reshape(n, c, h, w)
        return copy+copy*self.relu(out)

class localFusionBlock(nn.Module):  
    def __init__(self, in_channels):  
        super(localFusionBlock, self).__init__() 
        self.conv1 = nn.Conv2d(in_channels*3,in_channels*6,kernel_size=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels*6,in_channels,kernel_size=1)
    def forward(self,x,a,b):
        out = self.conv1(torch.cat([x,a,b],dim=1))
        out = self.act(out)
        out = self.conv2(out)
        return x + out
    
class localFusion(nn.Module):  
    def __init__(self, in_channels):  
        super(localFusion, self).__init__() 
        self.conv1 = nn.ModuleList([
            CCAM(in_channels) for i in range(8)
        ])

    def forward(self,x0,x1,x2,x3,x4,x5,x6,x7):
        (x0,x1,x2,x3,x4,x5,x6,x7) =  (self.conv1[0](x0,x1,x2)+x0,self.conv1[1](x1,x0,x2)+x1,
                                      self.conv1[2](x2,x1,x3)+x2,self.conv1[3](x3,x2,x4)+x3,
                                      self.conv1[4](x4,x3,x5)+x4,self.conv1[5](x5,x4,x6)+x5,
                                      self.conv1[6](x6,x5,x7)+x6,self.conv1[7](x7,x6,x5)+x7
                                      )

        out = torch.cat([x0,x1,x2,x3,x4,x5,x6,x7],dim=1)
        return out

class CCAM(nn.Module):
    def __init__(self,inchannels):
        super(CCAM, self).__init__()
        self.inchannels=inchannels
        self.fc = nn.Linear(inchannels*3, inchannels*3*inchannels)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y, z):
        m = torch.cat((x, y, z), dim=1)  # n, 12, h, w
        gap = F.adaptive_avg_pool2d(m, (1, 1))  # n, 12, 1, 1
        gap = gap.view(m.size(0), self.inchannels*3)  # n, 12
        fc_out = self.fc(gap)  # n, 48
        conv1d_input = fc_out.unsqueeze(1)  # n, 1, 48
        conv1d_out = self.conv1d(conv1d_input)  # n, 1, 48
        conv1d_out = conv1d_out.view(m.size(0), self.inchannels*3, self.inchannels)  # n, 12, 4
        softmax_out = self.softmax(conv1d_out)  # n, 12, 4
        out = torch.einsum('nchw,ncm->nmhw', (m, softmax_out))  # n, 4, h, w
        
        return out

class DFS(nn.Module):
    def __init__(self, in_channels,outchannel,basechannel, mergeNum,res = 1,attn=False):
        super(DFS, self).__init__()
        self.mergeNum = mergeNum
        self.low_pass_filter = DynamicConv(in_channels,mode='highPass')
        self.enlarger = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels,outchannel,kernel_size=3,padding=1)
        )
        self.fe = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels,basechannel//8,kernel_size=1),
            TriScaleConv(basechannel//8,basechannel//8,res=True),
            TriScaleConv(basechannel//8,basechannel//8,res=True),
        )
    def forward(self,x):
        low,high = self.low_pass_filter(x)
        low = self.fe(low)
        high = self.enlarger(high)
        return high,low

class DFLSBlock(nn.Module):  
    def __init__(self, in_channels,split=3):  
        super(DFLSBlock, self).__init__()  

        self.split = split
        self.frequency_enlarge = DualScaleConv(in_channels,in_channels,res=True) 
        self.norm=nn.BatchNorm2d(in_channels)
        self.blocks = nn.ModuleList([
            DFS(in_channels,in_channels*7//8,in_channels,0,res=1),
            DFS(in_channels*7//8,in_channels*6//8,in_channels,in_channels*2//8,res=1,attn=True),
            DFS(in_channels*6//8,in_channels*5//8,in_channels,in_channels*3//8,res=1),
            DFS(in_channels*5//8,in_channels*4//8,in_channels,in_channels*3//8,res=1),
            DFS(in_channels*4//8,in_channels*3//8,in_channels,in_channels*3//8,res=1),
            DFS(in_channels*3//8,in_channels*2//8,in_channels,in_channels*3//8,res=1),
            DFS(in_channels*2//8,in_channels*1//8,in_channels,in_channels*3//8,res=1),
        ])

        self.local = localFusion(in_channels//8)

        self.synthesizer = nn.Sequential(
            MultiAttn(in_channels),
        )
        self.merger = nn.Sequential(
            nn.Conv2d(in_channels,in_channels*2,kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels*2,in_channels,kernel_size=1),
        )
        self.merger2 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels*2,kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels*2,in_channels,kernel_size=1),
        )
        self.mlp=FRFN(in_channels,in_channels*4)
    def forward(self, m):  
        m0 = self.frequency_enlarge(m) 

        m1,x0 = self.blocks[0](m0)
        m1,x1 = self.blocks[1](m1)
        m1,x2 = self.blocks[2](m1)
        m1,x3 = self.blocks[3](m1)
        m1,x4 = self.blocks[4](m1)
        m1,x5 = self.blocks[5](m1)
        x7,x6 = self.blocks[6](m1)

        m2 = self.local(x0,x1,x2,x3,x4,x5,x6,x7)
        m2 = self.merger(m2)
        m2 = m2 + m0
        out = self.synthesizer(m2)
        out = self.merger2(out)
        out = out + m2
        out=self.mlp(self.norm(out))

        return out


#### Cross-layer Attention Fusion Block

class LAM_Module_v2(nn.Module):  
    """ Layer attention module"""
    def __init__(self, in_dim,bias=True):
        super(LAM_Module_v2, self).__init__()
        self.chanel_in = in_dim

        self.temperature = nn.Parameter(torch.ones(1))

        self.qkv = nn.Conv2d( self.chanel_in ,  self.chanel_in *3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(self.chanel_in*3, self.chanel_in*3, kernel_size=3, stride=1, padding=1, groups=self.chanel_in*3, bias=bias)
        self.project_out = nn.Conv2d(self.chanel_in, self.chanel_in, kernel_size=1, bias=bias)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, N, size, C= x.size()
        height=width=int(math.sqrt(size))
        x_input=self.qkv(x.view(m_batchsize,N*C, height, width))
        qkv = self.qkv_dwconv(x_input)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(m_batchsize, N, -1)
        k = k.view(m_batchsize, N, -1)
        v = v.view(m_batchsize, N, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out_1 = (attn @ v)
        out_1 = out_1.view(m_batchsize, -1, height, width)

        out_1 = self.project_out(out_1)
        out_1 = out_1.view(m_batchsize, N, size,C)

        out = out_1+x
        out = out.view(m_batchsize,size,-1)
        return out
    
class CAB(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CAB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = torch.nn.functional.softmax(attn,dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class RDCAB(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(RDCAB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.mlp=FeedForward(dim)
        self.mlpconv=nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        q = self.q(x)
        kv = self.kv(y)
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = torch.nn.functional.softmax(attn,dim=-1)

        out = (attn @ v) 

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out + x
        out = self.mlpconv(self.mlp(out))+x
        return out

class sparseCAB(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(sparseCAB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.relu=nn.ReLU()
        self.w = nn.Parameter(torch.ones(2)) 
    def forward(self, y, x):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn0 = torch.nn.functional.softmax(attn,dim=-1)

        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        attn1 = self.relu(attn)**2
        attn = attn0*w1+attn1*w2

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        
        return out


class BigsparseCAB(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(BigsparseCAB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.Big_conv=BigScaleConvModule(dim)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.relu=nn.ReLU()
        self.w = nn.Parameter(torch.ones(2)) 
    def forward(self, x, y):
        b, c, h, w = x.shape
        y = y+self.relu(self.Big_conv(y))

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn0 = torch.nn.functional.softmax(attn,dim=-1)
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        attn1 = self.relu(attn)**2
        attn = attn0*w1+attn1*w2

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        
        return out
    

class SmallsparseCAB(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(SmallsparseCAB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperaturez = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.Small_conv=SmallScaleConvModule(dim)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.zs = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.zs_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_outz = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.relu=nn.ReLU()
        self.w = nn.Parameter(torch.ones(2))

    def forward(self, x, y):
        b, c, h, w = x.shape
        z = self.Small_conv(y)

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        zs = self.zs_dwconv(self.zs(z))
        k, v = kv.chunk(2, dim=1)
        z, s = zs.chunk(2, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        z = rearrange(z, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        s = rearrange(s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        z = torch.nn.functional.normalize(z, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attnz = (q @ z.transpose(-2, -1)) * self.temperaturez
        attn0 = torch.nn.functional.softmax(attn,dim=-1)
        attnz = torch.nn.functional.softmax(attnz,dim=-1)
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        attn1 = self.relu(attn)**2
        attn = attn0*w1+attn1*w2

        out = (attn @ v)
        outz=  (attnz @ s)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        outz = rearrange(outz, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)+self.project_outz(outz)
        
        return out

class SmallsparseCAB2(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(SmallsparseCAB2, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.Small_conv=SmallScaleConvModule(dim)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.relu=nn.ReLU()
        self.w = nn.Parameter(torch.ones(2))

    def forward(self, x, y):
        b, c, h, w = x.shape
        y =y+self.relu(self.Small_conv(y))

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn0 = torch.nn.functional.softmax(attn,dim=-1)
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        attn1 = self.relu(attn)**2
        attn = attn0*w1+attn1*w2

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        
        return out

class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None,act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x

class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None,act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B,C,H,W = x.shape
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x
#############TransResNet############
class TransResNetBlockD2(nn.Module):
    def __init__(self, dim=512,dd_in=64,kernel_size=3, stride=1, norm_layer=None,act_layer=None):
        super().__init__()
        self.proj0=nn.Conv2d(in_channels=dim//2, out_channels=dd_in, kernel_size=3, padding=1, bias=False)
        self.proj1 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim, eps=0.0001, momentum = 0.95),
            nn.GELU(),
            nn.Conv2d(in_channels=dim, out_channels=dd_in, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dd_in, eps=0.0001, momentum = 0.95),
            nn.GELU()
            )
        self.proj2 = nn.Sequential(
            nn.Conv2d(in_channels=dd_in, out_channels=dd_in, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dd_in,eps=0.0001, momentum = 0.95),
            nn.GELU(),
            nn.Conv2d(in_channels=dd_in, out_channels=dd_in, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dd_in, eps=0.0001, momentum = 0.95),
            nn.GELU()
            )

    def forward(self, x,y):
        inputs=torch.cat([x,y],dim=1)#512 
        y=self.proj0(y)#
        inputs=self.proj1(inputs)+y
        result=inputs+self.proj2(inputs)
        return result

class TransResNetBlockD3(nn.Module):
    def __init__(self, dim=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None,act_layer=None):
        super().__init__()
        self.proj0 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim, eps=0.0001, momentum = 0.95),
            nn.GELU(),
            )
        self.proj1 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim, eps=0.0001, momentum = 0.95),
            nn.GELU(),
            )
        self.proj2= nn.Sequential(
            nn.Conv2d(in_channels=dim*2, out_channels=dim*2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim*2, eps=0.0001, momentum = 0.95),
            nn.GELU(),
            )
    def forward(self,x,y):
        x=self.proj0(x)
        y=self.proj1(y)
        inputs=torch.cat([x,y],dim=1) 
        inputs= self.proj2(inputs)
        return inputs

class TransResNetAttention(nn.Module):
    def __init__(self, dim1=64,dim2=128,hidden_dim=128, num_heads=4, bias=False):
        super().__init__()
        self.proj0= nn.Sequential(
            nn.Conv2d(in_channels=dim1, out_channels=hidden_dim, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(hidden_dim, eps=0.0001, momentum = 0.95),
            nn.GELU(),
            )
        self.proj1= nn.Sequential(
            nn.Conv2d(in_channels=dim2, out_channels=hidden_dim, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(hidden_dim, eps=0.0001, momentum = 0.95),
            nn.GELU(),
            )
        self.norm0= LayerNorm(hidden_dim)
        self.norm1= LayerNorm(hidden_dim)
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads*2, 1, 1))

        self.q0 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=bias)
        self.q_dwconv0 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim, bias=bias)
        self.kv0 = nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv0 = nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=3, stride=1, padding=1, groups=hidden_dim*2, bias=bias)
        self.q1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=bias)
        self.q_dwconv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim, bias=bias)
        self.kv1= nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv1= nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=3, stride=1, padding=1, groups=hidden_dim*2, bias=bias)        
        self.project_out = nn.Conv2d(hidden_dim*2, hidden_dim//2, kernel_size=1, bias=bias)
    def forward(self, x,y):
        b,c,h,w=x.shape#torch.Size([1, 128, 50,75])
        x=self.proj0(x)
        y=self.proj1(y)

        xnorm=self.norm0(x)
        ynorm=self.norm1(y)

        q0 = self.q_dwconv0(self.q0(xnorm))
        kv0 = self.kv_dwconv0(self.kv0(ynorm))
        q1 = self.q_dwconv0(self.q1(y))
        kv1 = self.kv_dwconv0(self.kv1(y))

        k0, v0 = kv0.chunk(2, dim=1)
        k1, v1 = kv1.chunk(2, dim=1)

        q0 = rearrange(q0, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k0 = rearrange(k0, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v0 = rearrange(v0, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q1 = rearrange(q1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k1 = rearrange(k1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v1 = rearrange(v1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(torch.cat([q0,q1],dim=1), dim=-1)
        k = torch.nn.functional.normalize(torch.cat([k0,k1],dim=1), dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = torch.nn.functional.softmax(attn,dim=-1)

        out = (attn @ torch.cat([v0,v1],dim=1))

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads*2, h=h, w=w)
        out = self.project_out(out)#torch.Size([1, 64, 50, 75])
        return out
############Mamba out#################
class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        # B, C, H, W = x.shape
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)

        return x

class Mambapatch(nn.Module):
    def __init__(self, in_chans=3,dim=64,patch_size=4,norm_layer=LayerNorm):
        super().__init__()
        self.patch=nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.norm=norm_layer(dim)
    def forward(self, x):
        x=self.norm(self.patch(x))
        return x

class Mambaffn(nn.Module):
    def __init__(self, dim, ffn_expand=2):
        super().__init__()

        dw_channel = dim * ffn_expand
        self.conv1 = nn.Conv2d(dim, dw_channel, kernel_size=1, padding=0, stride=1)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel)
        self.conv3 = nn.Conv2d(dw_channel//2, dim, kernel_size=1, padding=0, stride=1)
        self.dwconv1 = nn.Conv2d(dw_channel//2, dw_channel//2, kernel_size=3, stride=1, padding=1, groups=dw_channel//2)
        self.dwconv2 = nn.Conv2d(dw_channel//2, dw_channel//2, kernel_size=3, stride=1, padding=1, groups=dw_channel//2)
        self.Tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = F.gelu(x1)*x2
        x = self.conv3(x)
        return x

class VanillaSS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, C, H, W = x.shape
        x = x.permute(0,2,3,1).contiguous()
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out.permute(0, 3, 1, 2).contiguous()
    
class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=LayerNorm, channel_first=True):
        super().__init__()
        self.dim = dim
        Linear = Linear2d if channel_first else nn.Linear
        self._patch_merging_pad = self._patch_merging_pad_channel_first if channel_first else self._patch_merging_pad_channel_last
        self.reduction = Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad_channel_last(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    @staticmethod
    def _patch_merging_pad_channel_first(x: torch.Tensor):
        H, W = x.shape[-2:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2]  # ... H/2 W/2
        x1 = x[..., 1::2, 0::2]  # ... H/2 W/2
        x2 = x[..., 0::2, 1::2]  # ... H/2 W/2
        x3 = x[..., 1::2, 1::2]  # ... H/2 W/2
        x = torch.cat([x0, x1, x2, x3], 1)  # ... H/2 W/2 4*C     
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x

class MambaVblock(nn.Module):
    def __init__(self, embed_dim=64,norm_layer=LayerNorm):
        super().__init__()
        self.VSSM=VSSBlock(
                    hidden_dim=embed_dim,
                    drop_path=0,
                    norm_layer=LayerNorm,
                    channel_first=True,
                    # =============================
                    ssm_d_state=1,  # the state dimension of SSM
                    ssm_ratio=1,  # the rate of data dimension of SSM compared to the data dimension outside SSM
                    ssm_dt_rank="auto",
                    ssm_act_layer=nn.SiLU,
                    ssm_conv=3,
                    ssm_conv_bias=False,  # RetinexMamba sets it True
                    ssm_drop_rate=0,
                    ssm_init="v0",
                    forward_type="v05_noz",  # Vmamba use"v05_noz"
                    # =============================
                    mlp_ratio=4,
                    mlp_act_layer=nn.GELU,
                    mlp_drop_rate=0.0,
                    mlp_type=4,
                    # =============================
                    use_checkpoint=False,
                    post_norm=False,
                )
        self.ffn=Mambaffn(embed_dim)
        self.norm=norm_layer(embed_dim)
        self.skip_scale0= nn.Parameter(torch.ones([1,embed_dim,1,1]))
        self.skip_scale1= nn.Parameter(torch.ones([1,embed_dim,1,1]))
    def forward(self, input):
        x=self.VSSM(input)+input*self.skip_scale0
        y=self.ffn(self.norm(x))+x*self.skip_scale1
        return y

#########Bayesian###############################
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


def _no_grad_trunc_normal_(
    tensor, mean, std, a, b
):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)

        return tensor


def trunc_normal_(
    tensor, mean=0.0, std=1.0, a=-2.0, b=2.0
):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)




class PatchMerging(nn.Module):
    """Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = LayerNorm(4 * dim)
        self.reduction = nn.Conv2d(4 * dim, 2 * dim, 1, 1, 0, bias=False)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        x0 = x[:, :, 0::2, 0::2]  # B, C, H/2, W/2
        x1 = x[:, :, 1::2, 0::2]  # B, C, H/2, W/2
        x2 = x[:, :, 0::2, 1::2]  # B, C, H/2, W/2
        x3 = x[:, :, 1::2, 1::2]  # B, C, H/2, W/2
        x = torch.cat([x0, x1, x2, x3], 1)  # B, 4C, H/2, W/2

        x = self.reduction(self.norm(x))

        return x


def deconv_up(in_channels):
    return nn.ConvTranspose2d(
        in_channels,
        in_channels // 2,
        stride=2,
        kernel_size=2,
        padding=0,
        output_padding=0,
    )


# Dual Up-Sample
class DualUpSample(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super(DualUpSample, self).__init__()
        self.factor = scale_factor

        if self.factor == 2:
            self.conv = nn.Conv2d(in_channels, in_channels // 2, 1, 1, 0, bias=False)
            self.up_p = nn.Sequential(
                nn.Conv2d(in_channels, 2 * in_channels, 1, 1, 0, bias=False),
                nn.PReLU(),
                nn.PixelShuffle(scale_factor),
                nn.Conv2d(
                    in_channels // 2,
                    in_channels // 2,
                    1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
            )

            self.up_b = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1, 1, 0),
                nn.PReLU(),
                nn.Upsample(
                    scale_factor=scale_factor, mode="bilinear", align_corners=False
                ),
                nn.Conv2d(
                    in_channels, in_channels // 2, 1, stride=1, padding=0, bias=False
                ),
            )
        elif self.factor == 4:
            self.conv = nn.Conv2d(2 * in_channels, in_channels, 1, 1, 0, bias=False)
            self.up_p = nn.Sequential(
                nn.Conv2d(in_channels, 16 * in_channels, 1, 1, 0, bias=False),
                nn.PReLU(),
                nn.PixelShuffle(scale_factor),
                nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=False),
            )

            self.up_b = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1, 1, 0),
                nn.PReLU(),
                nn.Upsample(
                    scale_factor=scale_factor, mode="bilinear", align_corners=False
                ),
                nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=False),
            )

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        # breakpoint()
        x_p = self.up_p(x)  # pixel shuffle
        x_b = self.up_b(x)  # bilinear
        out = self.conv(torch.cat([x_p, x_b], dim=1))

        return out


class LN2DLinear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(LN2DLinear, self).__init__()
        self.norm = LayerNorm(in_channels)
        self.linear = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.constant_(m.weight, 0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.linear(self.norm(x))

class conv_relu(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1
    ):
        super(conv_relu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
                dilation=dilation_rate,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_input):
        out = self.conv(x_input)
        return out


class CSAF(nn.Module):
    def __init__(self, in_chnls, ratio=4):
        super(CSAF, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress1 = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.compress2 = nn.Conv2d(in_chnls // ratio, in_chnls // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)

    def forward(self, x0, x2, x4):
        out0 = self.squeeze(x0)
        out2 = self.squeeze(x2)
        out4 = self.squeeze(x4)
        out = torch.cat([out0, out2, out4], dim=1)
        out = self.compress1(out)
        out = F.relu(out)
        out = self.compress2(out)
        out = F.relu(out)
        out = self.excitation(out)
        out = F.sigmoid(out)
        w0, w2, w4 = torch.chunk(out, 3, dim=1)
        x = x0 * w0 + x2 * w2 + x4 * w4

        return x

class DB(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(DB, self).__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for i in range(len(d_list)):
            dense_conv = conv_relu(
                in_channel=c,
                out_channel=inter_num,
                kernel_size=3,
                dilation_rate=d_list[i],
                padding=d_list[i],
            )
            self.conv_layers.append(dense_conv)
            c = c + inter_num
        self.conv_post = conv(in_channel=c, out_channel=in_channel, kernel_size=1)

    def forward(self, x):
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)
        t = self.conv_post(t)
        return t


class SAM(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(SAM, self).__init__()
        self.basic_block = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.basic_block_2 = DB(
            in_channel=in_channel, d_list=d_list, inter_num=inter_num
        )
        self.basic_block_4 = DB(
            in_channel=in_channel, d_list=d_list, inter_num=inter_num
        )
        self.fusion = CSAF(3 * in_channel)

    def forward(self, x):
        x_0 = x
        x_2 = F.interpolate(x, scale_factor=0.5, mode="bilinear")
        x_4 = F.interpolate(x, scale_factor=0.25, mode="bilinear")

        y_0 = self.basic_block(x_0)
        y_2 = self.basic_block_2(x_2)
        y_4 = self.basic_block_4(x_4)

        y_2 = F.interpolate(y_2, scale_factor=2, mode="bilinear")
        y_4 = F.interpolate(y_4, scale_factor=4, mode="bilinear")

        y = self.fusion(y_0, y_2, y_4)
        y = x + y

        return y

def conv_down(in_channels):
    return nn.Conv2d(in_channels, in_channels * 2, 4, 2, 1, bias=False)

class BasicBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_blocks=2,
        d_state=1,
        ssm_ratio=1,
        mlp_ratio=4,
        mlp_type="gdmlp",
        sam=False,
        condition=False,
        bayesian=False,
    ):

        super().__init__()
        self.bayesian = bayesian
        self.sam = sam
        self.condition = condition
        self.blocks = nn.ModuleList([])
        self.first_conv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
        if sam:
            self.sam_blocks = nn.ModuleList([])
        if self.condition:
            self.condition_blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(
                # In our model, the forward way is SS2Dv2 -> forwardv2 -> forward_corev2
                VSSBlock(
                    hidden_dim=dim,
                    drop_path=0,
                    norm_layer=LayerNorm,
                    channel_first=True,
                    # =============================
                    ssm_d_state=d_state,  # the state dimension of SSM
                    ssm_ratio=ssm_ratio,  # the rate of data dimension of SSM compared to the data dimension outside SSM
                    ssm_dt_rank="auto",
                    ssm_act_layer=nn.SiLU,
                    ssm_conv=3,
                    ssm_conv_bias=False,  # RetinexMamba sets it True
                    ssm_drop_rate=0,
                    ssm_init="v0",
                    forward_type="v05_noz",  # Vmamba use"v05_noz"
                    # =============================
                    mlp_ratio=mlp_ratio,
                    mlp_act_layer=nn.GELU,
                    mlp_drop_rate=0.0,
                    mlp_type=mlp_type,
                    # =============================
                    use_checkpoint=False,
                    post_norm=False,
                    condition=False
                )
            )
            if sam:
                self.sam_blocks.append(
                    SAM(in_channel=dim, d_list=(1, 2, 3, 2, 1), inter_num=24)
                )
            if condition:
                self.condition_blocks.append(None)

    def forward(self, x):
        x=self.first_conv(x)
        for _idx, block in enumerate(self.blocks):
            x = block(x)
            if self.sam:
                x = self.sam_blocks[_idx](x)
        return x


class SubNetwork(nn.Module):
    """
    The main module representing as a shallower UNet
    args:
        dim (int): number of channels of input and output
        num_blocks (list): each element defines the number of basic blocks in a scale level
        d_state (int): dimension of the hidden state in S6
        ssm_ratio(int): expansion ratio of SSM in S6
        mlp_ratio (float): expansion ratio of MLP in S6
        use_pixelshuffle (bool): when true, use pixel(un)shuffle for up(down)-sampling, otherwise, use Transposed Convolution.
        drop_path (float): ratio of droppath beween two subnetwork
    """

    def __init__(
        self,
        dim=31,
        num_blocks=[2, 4, 4],
        d_state=[1,1,1],
        ssm_ratio=1,
        mlp_ratio=4,
        mlp_type="gdmlp",
        use_pixelshuffle=False,
        drop_path=0.0,
        sam=False,
    ):
        super(SubNetwork, self).__init__()
        self.dim = dim
        level = len(num_blocks) - 1
        self.level = level
        self.encoder_layers = nn.ModuleList([])
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        curr_dim = dim
        down_layer = PatchMerging if use_pixelshuffle else conv_down
        up_layer = (
            partial(DualUpSample, scale_factor=2) if use_pixelshuffle else deconv_up
        )

        for i in range(level):
            self.encoder_layers.append(
                nn.ModuleList(
                    [
                        BasicBlock(
                            dim=curr_dim,
                            num_blocks=num_blocks[i],
                            d_state=d_state[i],
                            ssm_ratio=ssm_ratio,
                            mlp_ratio=mlp_ratio,
                            mlp_type=mlp_type,
                            sam=sam,
                            bayesian=True,
                        ),
                        down_layer(curr_dim),
                    ]
                )
            )
            curr_dim *= 2

        self.bottleneck = BasicBlock(
            dim=curr_dim,
            num_blocks=num_blocks[-1],
            d_state=d_state[level],
            ssm_ratio=ssm_ratio,
            mlp_ratio=mlp_ratio,
            sam=sam,
            bayesian=True,
        )

        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(
                nn.ModuleList(
                    [
                        up_layer(curr_dim),
                        nn.Conv2d(curr_dim, curr_dim // 2, 1, 1, bias=False),
                        BasicBlock(
                            dim=curr_dim // 2,
                            num_blocks=num_blocks[level - 1 - i],
                            d_state=d_state[level - 1 - i],
                            ssm_ratio=ssm_ratio,
                            mlp_ratio=mlp_ratio,
                            sam=sam,
                            bayesian=True,
                        ),
                    ]
                )
            )
            curr_dim //= 2

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):

        fea = x
        #####Encoding Process-------------------------------------------------------------------------------------------
        fea_encoder = []
        for en_block, down_layer in self.encoder_layers:
            fea = en_block(fea)
            fea_encoder.append(fea)
            fea = down_layer(fea)
        fea = self.bottleneck(fea)
        ######----------------------------------------------------------------------------------------------------------
        ######Decoding Process------------------------------------------------------------------------------------------
        for i, (up_layer, fusion, de_block) in enumerate(self.decoder_layers):
            fea = up_layer(fea)
            fea = fusion(torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            fea = de_block(fea)

        return x + self.drop_path(fea)


class Network(nn.Module):
    """
    The Model
    args:
        in_channels (int): input channel number
        out_channels (int): output channel number
        n_feat (int): channel number of intermediate features
        stage (int): number of stages。
        num_blocks (list): each element defines the number of basic blocks in a scale level
        d_state (int): dimension of the hidden state in S6
        ssm_ratio(int): expansion ratio of SSM in S6
        mlp_ratio (float): expansion ratio of MLP in S6
        use_pixelshuffle (bool): when true, use pixel(un)shuffle for up(down)-sampling, otherwise, use Transposed Convolution.
        drop_path (float): ratio of droppath beween two subnetwork
        use_illu (bool): true to include an illumination layer
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        n_feat=40,
        stage=1,
        num_blocks=[1, 1, 1],
        d_state=1,
        ssm_ratio=1,
        mlp_ratio=4,
        mlp_type="gdmlp",
        use_pixelshuffle=False,
        drop_path=0.0,
        use_illu=False,
        sam=False,
        last_act=None,
    ):
        super(Network, self).__init__()
        self.stage = stage

        self.mask_token = nn.Parameter(torch.zeros(1, n_feat, 1, 1))
        trunc_normal_(self.mask_token, mean=0.0, std=0.02)

        # 11/07/2024 set bias True, following MoCov3; Meanwhile, bias can help ajust input's mean
        self.first_conv = nn.Conv2d(in_channels, n_feat, 3, 1, 1, bias=True)
        nn.init.kaiming_normal_(
            self.first_conv.weight, mode="fan_out", nonlinearity="linear"
        )
        # nn.init.xavier_normal_(self.conv_proj.weight)
        if self.first_conv.bias is not None:
            nn.init.zeros_(self.first_conv.bias)
        # nn.init.xavier_normal_(self.dynamic_emblayer.weight)

        # # freeze embedding layer
        # for param in self.static_emblayer.parameters():
        #     param.requires_grad = False

        self.subnets = nn.ModuleList([])

        self.proj = nn.Conv2d(n_feat, out_channels, 3, 1, 1, bias=True)
        # nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

        if last_act is None:
            self.last_act = nn.Identity()
        elif last_act == "relu":
            self.last_act = nn.ReLU()
        elif last_act == "softmax":
            self.last_act = nn.Softmax(dim=1)
        else:
            raise NotImplementedError

        for i in range(stage):
            self.subnets.append(
                SubNetwork(
                    dim=n_feat,
                    num_blocks=num_blocks,
                    d_state=d_state,
                    ssm_ratio=ssm_ratio,
                    mlp_ratio=mlp_ratio,
                    mlp_type=mlp_type,
                    use_pixelshuffle=use_pixelshuffle,
                    drop_path=drop_path,
                    sam=sam,
                )
            )

    def forward(self, x, mask=None):
        """
        Args:
            x (Tensor): [batch_size, channels, height, width]
        return:
            out (Tensor): return reconstrcuted images
        """

        #out_list = []
        #out_list.append(x)

        fea = self.first_conv(x)

        B, C, H, W = fea.size()
        if self.training and mask is not None:
            mask_tokens = self.mask_token.expand(B, -1, H, W)
            w = mask.unsqueeze(1).type_as(mask_tokens)
            fea = fea * (1.0 - w) + mask_tokens * w

        for _idx, subnet in enumerate(self.subnets):
            fea = subnet(fea)
            out = self.proj(fea)
            out = self.last_act(out)
            #out_list.append(out)
        return out

class MambaBlock(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.norm=LayerNorm(dim)
        self.SSM = SS2D(
                d_model=dim,
        # =============================
                d_state=16,
                ssm_ratio=2,
                dt_rank="auto",
                act_layer=nn.SiLU,
                # ==========================
                d_conv=3,
                conv_bias=True,
                # ==========================
                dropout=0,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize="v0",
                # ==========================
                forward_type="v2",
                channel_first=True,
            )
        
        self.MLP=FRFN(dim,hidden_dim=dim*2)
        self.conv=nn.Conv2d(dim,dim,1,1,0)

    def forward(self, x):
        x=self.conv(self.MLP(self.norm(x)))
        return self.SSM(x.permute(0,2,3,1).contiguous())

class crossMamba(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.VSSM0=MambaBlock(dim)
        self.VSSM1=MambaBlock(dim)
        self.VSSM2=MambaBlock(dim)
        self.outproj0=nn.Linear(dim*2,dim)
        self.outproj1=nn.Linear(dim*2,dim)
        self.Tanh = nn.Tanh()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
    def forward(self, x,y):
        x0=self.VSSM0(x)
        y0=self.VSSM1(y)
        fusion0=torch.cat([x,y],dim=1).permute(0,2,3,1).contiguous()
        fusion0=self.VSSM2(self.outproj0(fusion0).permute(0,3,1,2).contiguous())
        x1=self.Tanh(fusion0)+x0
        y1=self.Tanh(fusion0)+y0
        fusion1=torch.cat([x1,y1],dim=-1)
        result=self.outproj1(fusion1).permute(0,3,1,2).contiguous()+x+y
        return self.dwconv(result)

######################################################
class LayerNormalization(nn.Module):
    def __init__(self, dim):
        super(LayerNormalization, self).__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # Rearrange the tensor for LayerNorm (B, C, H, W) to (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # Rearrange back to (B, C, H, W)
        return x.permute(0, 3, 1, 2)

class SEBlock(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(input_channels, input_channels // reduction_ratio)
        self.fc2 = nn.Linear(input_channels // reduction_ratio, input_channels)
        self._init_weights()

    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()
        y = self.pool(x).reshape(batch_size, num_channels)
        y = F.relu(self.fc1(y))
        y = torch.tanh(self.fc2(y))
        y = y.reshape(batch_size, num_channels, 1, 1)
        return x * y
    
    def _init_weights(self):
        init.kaiming_uniform_(self.fc1.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.fc2.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.fc1.bias, 0)
        init.constant_(self.fc2.bias, 0)

class MSEFBlock(nn.Module):
    def __init__(self, filters):
        super(MSEFBlock, self).__init__()
        self.layer_norm = LayerNormalization(filters)
        self.depthwise_conv = nn.Conv2d(filters, filters, kernel_size=3, padding=1, groups=filters)
        self.se_attn = SEBlock(filters)
        self._init_weights()

    def forward(self, x):
        x_norm = self.layer_norm(x)
        x1 = self.depthwise_conv(x_norm)
        x2 = self.se_attn(x_norm)
        x_fused = x1 * x2
        x_out = x_fused + x
        return x_out
    
    def _init_weights(self):
        init.kaiming_uniform_(self.depthwise_conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.depthwise_conv.bias, 0)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        assert embed_size % num_heads == 0
        self.head_dim = embed_size // num_heads
        self.query_dense = nn.Linear(embed_size, embed_size)
        self.key_dense = nn.Linear(embed_size, embed_size)
        self.value_dense = nn.Linear(embed_size, embed_size)
        self.combine_heads = nn.Linear(embed_size, embed_size)
        self._init_weights()

    def split_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        x = x.reshape(batch_size, height * width, -1)

        query = self.split_heads(self.query_dense(x), batch_size)
        key = self.split_heads(self.key_dense(x), batch_size)
        value = self.split_heads(self.value_dense(x), batch_size)
        
        attention_weights = F.softmax(torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1)
        attention = torch.matmul(attention_weights, value)
        attention = attention.permute(0, 2, 1, 3).contiguous().reshape(batch_size, -1, self.embed_size)
        
        output = self.combine_heads(attention)
        
        return output.reshape(batch_size, height, width, self.embed_size).permute(0, 3, 1, 2)

    def _init_weights(self):
        init.xavier_uniform_(self.query_dense.weight)
        init.xavier_uniform_(self.key_dense.weight)
        init.xavier_uniform_(self.value_dense.weight)
        init.xavier_uniform_(self.combine_heads.weight)
        init.constant_(self.query_dense.bias, 0)
        init.constant_(self.key_dense.bias, 0)
        init.constant_(self.value_dense.bias, 0)
        init.constant_(self.combine_heads.bias, 0)


#################BreadcrumbsOCTAMamba##########################
class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class DualAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=4, kernel_size=7):
        super(DualAttentionModule, self).__init__()
        self.channel_attention = ChannelAttentionModule(in_channels, reduction)
        self.spatial_attention = SpatialAttentionModule(kernel_size)
        self.act=nn.SiLU()
    def forward(self, x):
        # Apply Channel Attention Module
        x_out_cam = self.channel_attention(x)
        x_out_sam=self.spatial_attention(x)
        x_out=x_out_sam+x_out_cam
        x_out=x*x_out
        x_out=self.act(x_out)
        x_out=x_out*x
        x_out=x_out+x

        return x_out

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output

class MultiScaleConvModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConvModule, self).__init__()
        middim=out_channels*4

        self.bnrelu1=BNPReLU(nIn=in_channels)

        # 1x1 convolutions
        self.conv1x1_1 = nn.Conv2d(in_channels, middim, kernel_size=1, dilation=1)

        # Convolutions with different kernel sizes and dilations
        self.conv1x3 = nn.Conv2d(middim, middim, kernel_size=(1, 3), dilation=1, padding=(0, 1),groups=middim)
        self.conv3x1 = nn.Conv2d(middim, middim, kernel_size=(3, 1), dilation=1, padding=(1, 0),groups=middim)
        self.conv3x3_1 = nn.Conv2d(middim, middim, kernel_size=3, dilation=3, padding=3,groups=middim)

        self.conv1x5 = nn.Conv2d(middim, middim, kernel_size=(1, 5), dilation=1, padding=(0, 2),groups=middim)
        self.conv5x1 = nn.Conv2d(middim, middim, kernel_size=(5, 1), dilation=1, padding=(2, 0),groups=middim)
        self.conv3x3_2 = nn.Conv2d(middim, middim, kernel_size=3, dilation=5, padding=5,groups=middim)

        self.conv1x7 = nn.Conv2d(middim, middim, kernel_size=(1, 7), dilation=1, padding=(0, 3),groups=middim)
        self.conv7x1 = nn.Conv2d(middim, middim, kernel_size=(7, 1), dilation=1, padding=(3, 0),groups=middim)
        self.conv3x3_3 = nn.Conv2d(middim, middim, kernel_size=3, dilation=7, padding=7,groups=middim)
        self.eca_1=eca_layer(middim)
        self.eca_2=eca_layer(middim)

        # Final 3x3 convolution
        self.conv3x3_final = nn.Conv2d(middim * 3, out_channels, kernel_size=3, dilation=1, padding=1,stride=1)

    def forward(self, x):
        # residual=self.conv1x1_0(x)
        # First path
        x=self.bnrelu1(x)

        x1 = self.conv1x1_1(x)
        # print("x1:", x1.shape)
        x_eca_1=self.eca_1(x1)
        # print("eca_1:", x_eca_1.shape)
        x_eca_2=self.eca_2(x1)
        # print("eca_2:", x_eca_2.shape)
        # print(x1.shape)
        # Second path

        x2 = self.conv1x3(x1)
        x2 = self.conv3x1(x2)
        x2 = self.conv3x3_1(x2)
        # print("x2:",x2.shape)
        # Third path

        x3 = self.conv1x5(x1)
        x3 = self.conv5x1(x3)
        x3 = self.conv3x3_2(x3)
        # print("x3:", x3.shape)
        # Fourth path

        x4 = self.conv1x7(x1)
        x4 = self.conv7x1(x4)
        x4 = self.conv3x3_3(x4)
        # print("x4:", x4.shape)
        x_branch1=x2+x_eca_1+x3
        # print("branch1:", x_branch1.shape)
        x_branch2=x3+x_eca_2+x4
        # print("branch_2:", x_branch2.shape)
        # Concatenate paths
        out = torch.cat([x_branch2, x_branch1,x1], dim=1)

        # Final 3x3 convolution
        out = self.conv3x3_final(out)

        return out


class AvgPoolingChannel(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=2, stride=2)


class MaxPoolingChannel(nn.Module):
    def forward(self, x):
        return F.max_pool2d(x, kernel_size=2, stride=2)


class SEAttention(nn.Module):
    def __init__(self, channel=3, reduction=3):
        super().__init__()
        # 池化层，将每一个通道的宽和高都变为 1 (平均值)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            # 先降低维
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            # 再升维
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # y是权重
        return x * y.expand_as(x)


# Quad Stream Efficient Mining Embedding
class QSEME(nn.Module):
    def __init__(self, out_c):
        super().__init__()

        self.out_c = out_c
        self.se = SEAttention(channel=out_c)
        self.maxpool = MaxPoolingChannel()
        self.avgpool = AvgPoolingChannel()
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
        )

        self.wtconv1 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
        )

        self.wtconv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
        )

        self.wtconv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU()
        )
        # self.out_conv=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=1)
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=self.out_c, kernel_size=1),
            nn.BatchNorm2d(num_features=self.out_c),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.init_conv(x)  # BCHW

        x1, x2, x3, x4 = x.chunk(4, dim=1)
        # branch1_maxpool
        chaneel_1_max_pool = self.maxpool(x1)
        desired_size = (x1.size(2), x1.size(3))
        channel_1_max_pool_out = F.interpolate(chaneel_1_max_pool, size=desired_size, mode='bilinear',
                                               align_corners=False)

        # branch2_avgpool
        channel_2_avg_pool = self.avgpool(x2)
        desired_size = (x2.size(2), x2.size(3))
        channel_2_avg_pool_out = F.interpolate(channel_2_avg_pool, size=desired_size, mode='bilinear',
                                               align_corners=False)

        # branch3_Wtconv
        channel_3_1 = self.wtconv1(x3)
        channel_3_2 = self.wtconv2(channel_3_1)
        channel_3_3_out = self.wtconv3(channel_3_2)

        # branch4_residual
        channel_4 = x4

        output = torch.cat((channel_1_max_pool_out, channel_2_avg_pool_out, channel_3_3_out, channel_4), dim=1)
        output = self.out_conv(output)
        output = self.se(output)
        return output

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.dual_att = DualAttentionModule(in_channels=self.d_inner)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        z = z.permute(0, 3, 1, 2).contiguous()  # B C H W
        z = self.dual_att(z)  # B C H W
        z = z.permute(0, 2, 3, 1).contiguous()
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out
class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.hidden_dim = hidden_dim

    def forward(self, input: torch.Tensor):
        x_mamba = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x_mamba
# AttentionGate
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi
    
class OCTAMambaBlock(nn.Module):
    def __init__(self, in_c, out_c, ):
        super().__init__()
        self.in_c = in_c
        self.conv = MultiScaleConvModule(in_channels=in_c, out_channels=out_c)
        self.ln = nn.LayerNorm(out_c)
        self.act = nn.GELU()
        self.block = VSSBlock(hidden_dim=out_c)
        self.scale = nn.Parameter(torch.ones(1))
        self.residual_conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, padding="same")

    def forward(self, x):
        skip = x
        skip = self.residual_conv(skip)
        x = self.conv(x)

        x = x.permute(0, 2, 3, 1)  # B H W C
        x = self.block(x)
        x = x.permute(0, 3, 1, 2)  # B C H W

        x = x.permute(0, 2, 3, 1)  # B H W C
        x = self.act(self.ln(x))
        x = x.permute(0, 3, 1, 2)  # B C H W
        return x + skip * self.scale


class EncoderBlock(nn.Module):
    """Encoding then downsampling"""

    def __init__(self, in_c, out_c):
        super().__init__()

        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.GELU()
        self.octamamba = OCTAMambaBlock(in_c, out_c)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.octamamba(x)
        skip = self.act(self.bn(x))
        x = self.down(skip)
        return x, skip

class DecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.attGate = Attention_block(F_g=in_c, F_l=skip_c, F_int=skip_c // 2)

        self.bn2 = nn.BatchNorm2d(in_c + skip_c, out_c)
        self.octamamba = OCTAMambaBlock(in_c + skip_c, out_c)
        self.act = nn.ReLU()

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.attGate(x, skip)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.bn2(x))
        x = self.octamamba(x)
        return x


class OCTAMamba(nn.Module):
    def __init__(self):
        super().__init__()
        self.qseme = QSEME(out_c=16)

        """Encoder"""
        self.e1 = EncoderBlock(16, 32)
        self.e2 = EncoderBlock(32, 64)
        self.e3 = EncoderBlock(64, 128)

        """Decoder"""
        self.d3 = DecoderBlock(128, 128, 64)
        self.d2 = DecoderBlock(64, 64, 32)
        self.d1 = DecoderBlock(32, 32, 16)

        """Final"""
        self.conv_out = nn.Conv2d(16, 2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Pre-Component"""
        x = self.qseme(x)

        """Encoder"""
        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)

        """Decoder"""
        x = self.d3(x, skip3)
        x = self.d2(x, skip2)
        x = self.d1(x, skip1)

        """Final"""
        x = self.conv_out(x)
        x = self.sigmoid(x)
        return x
###########################################

class myencoderBlock(nn.Module):
    def __init__(self,filters):
        super().__init__()
        self.mlp=FRFN(filters)
        self.norm = LayerNorm(filters)

    def forward(self, x):
        x=x+(self.mlp(self.norm(x)))
        return x

class mydoubleconv(nn.Module):
    def __init__(self,filters):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(filters, eps=0.0001, momentum = 0.95),
            nn.PReLU(),
            nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters, eps=0.0001, momentum = 0.95),
            )
        self.act=nn.PReLU()

    def forward(self, x):
        x=x+self.conv(x)
        return self.act(x)
class Illumination_Estimator(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):  #__init__部分是内部属性，而forward的输入才是外部输入
        super(Illumination_Estimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w
        
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w
        
        mean_c = img.mean(dim=1).unsqueeze(1)
        # stx()
        input = torch.cat([img,mean_c], dim=1)

        x_1 = self.conv1(input)
        fea = self.depth_conv(x_1)
        lum = self.conv2(fea)
        return fea, lum
        
##############################################

class MultiScaleATTention(nn.Module):
    def __init__(self, in_channels,num_heads,bias=False):
        super(MultiScaleATTention, self).__init__()
        middim=in_channels

        self.bnrelu1=BNPReLU(nIn=in_channels)

        # 1x1 convolutions
        self.conv1x1_1 = nn.Conv2d(in_channels, middim, kernel_size=1, dilation=1)

        # Convolutions with different kernel sizes and dilations
        self.conv1x3 = nn.Conv2d(middim, middim, kernel_size=(1, 3), dilation=1, padding=(0, 1),groups=middim)
        self.conv3x1 = nn.Conv2d(middim, middim, kernel_size=(3, 1), dilation=1, padding=(1, 0),groups=middim)
        self.conv3x3_1 = nn.Conv2d(middim, middim, kernel_size=3, dilation=3, padding=3,groups=middim)

        self.conv1x5 = nn.Conv2d(middim, middim, kernel_size=(1, 5), dilation=1, padding=(0, 2),groups=middim)
        self.conv5x1 = nn.Conv2d(middim, middim, kernel_size=(5, 1), dilation=1, padding=(2, 0),groups=middim)
        self.conv3x3_2 = nn.Conv2d(middim, middim, kernel_size=3, dilation=5, padding=5,groups=middim)

        self.conv1x7 = nn.Conv2d(middim, middim, kernel_size=(1, 7), dilation=1, padding=(0, 3),groups=middim)
        self.conv7x1 = nn.Conv2d(middim, middim, kernel_size=(7, 1), dilation=1, padding=(3, 0),groups=middim)
        self.conv3x3_3 = nn.Conv2d(middim, middim, kernel_size=3, dilation=7, padding=7,groups=middim)
        # Final 3x3 convolution
        self.conv3x3_final = nn.Conv2d(middim * 3, in_channels, kernel_size=3, dilation=1, padding=1,stride=1)
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=bias)
        self.kv = nn.Conv2d(in_channels, in_channels*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(in_channels*2, in_channels*2, kernel_size=3, stride=1, padding=1, groups=in_channels*2, bias=bias)
        self.project_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        # residual=self.conv1x1_0(x)
        # First path
        b, c, h, w = x.shape
        x=self.bnrelu1(x)
        x1 = self.conv1x1_1(x)

        x2 = self.conv1x3(x1)
        x2 = self.conv3x1(x2)
        x2 = self.conv3x3_1(x2)
        # print("x2:",x2.shape)
        # Third path
        x3 = self.conv1x5(x1)
        x3 = self.conv5x1(x3)
        x3 = self.conv3x3_2(x3)
        # print("x3:", x3.shape)
        # Fourth path
        x4 = self.conv1x7(x1)
        x4 = self.conv7x1(x4)
        x4 = self.conv3x3_3(x4)

        #y=torch.stack([x1,x2,x3,x4], dim=-1)

        q = self.q_dwconv(self.q(x))
        kv1 = self.kv_dwconv(self.kv(x1))
        k1, v1 = kv1.chunk(2, dim=1)
        kv2 = self.kv_dwconv(self.kv(x2))
        k2, v2 = kv2.chunk(2, dim=1)
        kv3 = self.kv_dwconv(self.kv(x3))
        k3, v3 = kv3.chunk(2, dim=1)
        kv4 = self.kv_dwconv(self.kv(x4))
        k4, v4 = kv4.chunk(2, dim=1)
        k=torch.stack([k1,k2,k3,k4],dim=1)
        v=torch.stack([v1,v2,v3,v4],dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b q (head c)  h w -> b head (c q) (h w)', head=self.num_heads)
        v = rearrange(v, 'b q (head c) h w -> b head (c q) (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature 
        attn = torch.nn.functional.softmax(attn,dim=-1)

        out= (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)+self.conv3x3_final(torch.cat(x2,x2,x4),dim=1)
        return out




class stupidLight(nn.Module):
    def __init__(self, dim):  #__init__部分是内部属性，而forward的输入才是外部输入
        super().__init__()

        self.lum_pool8 = nn.MaxPool2d(8)
        self.lum_mhsa8 = FRFN(dim)
        self.lum_up8 = nn.Upsample(scale_factor=8, mode='nearest')
        self.lum_pool4 = nn.MaxPool2d(4)
        self.lum_mhsa4 = FRFN(dim)
        self.lum_up4 = nn.Upsample(scale_factor=4, mode='nearest')
        self.act = nn.LeakyReLU(inplace=True)
    def forward(self, lum):
        lum_8 = self.lum_pool8(lum)
        lum_8 = self.lum_mhsa8(lum_8)
        lum_8 = self.lum_up8(lum_8)+lum
        lum_4 = self.lum_pool4(lum)
        lum_4 = self.lum_mhsa4(lum_4)
        lum_4 = self.lum_up4(lum_4)+lum
        lum = lum + self.act(lum_4)+self.act(lum_8)
        return lum

################################

class IG_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, illu_fea_trans):
        """
        x_in: [b,h,w,c]         # input_feature
        illu_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        illu_attn = illu_fea_trans # illu_fea: b,c,h,w -> b,h,w,c
        q, k, v, illu_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                 (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2)))
        v = v * illu_attn
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(
            0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out





class IGAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                FeedForward(dim=dim)
            ]))

    def forward(self, x, illu_fea):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out


class SPDConv(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = HuaConv(inc * 4, ouc, kernel_size=3)
        

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], self.d)
        x = self.conv(x)
        return x



class HuaConv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, ch_in, ch_out, kernel_size=1, stride=1, padding=None, groups=1, dilation=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, stride, self.autopad(kernel_size, padding, dilation), groups=groups, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(ch_out)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.up =nn.UpsamplingBilinear2d(scale_factor=2)
    def autopad(self,kernel_size, padding=None, dilation=1):
        if padding is not None:
            return padding
        return (kernel_size - 1) // 2 * dilation

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.up(self.act(self.bn(self.conv(x))))
    



class RetinexFormer_Single_Stage(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, level=2, num_blocks=[1, 1, 1]):
        super(RetinexFormer_Single_Stage, self).__init__()
        self.estimator = Illumination_Estimator(n_feat)
        self.denoiser = Denoiser(in_dim=in_channels,out_dim=out_channels,dim=n_feat,level=level,num_blocks=num_blocks)  #### 将 Denoiser 改为 img2img
    
    def forward(self, img):
        # img:        b,c=3,h,w
        
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        illu_fea, illu_map = self.estimator(img)
        input_img = img * illu_map + img
        output_img = self.denoiser(input_img,illu_fea)

        return output_img




class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        return self.bottleneckBlock(x)


class DetailNode(nn.Module):
    def __init__(self,dim=32):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=dim, oup=dim, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=dim, oup=dim, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=dim, oup=dim, expand_ratio=2)
        self.shffleconv = nn.Conv2d(
            dim*2, dim*2, kernel_size=1, stride=1, padding=0, bias=True
        )

    def separateFeature(self, x):
        z1, z2 = x[:, : x.shape[1] // 2], x[:, x.shape[1] // 2 : x.shape[1]]
        return z1, z2

    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2

class DetailFeatureExtractor(nn.Module):
    def __init__(self, num_layers=1,dim=32):
        super(DetailFeatureExtractor, self).__init__()
        INNmodules = [DetailNode(dim=dim) for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)

    def forward(self, x):
        z1, z2 = x[:, : x.shape[1] // 2], x[:, x.shape[1] // 2 : x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)

class LHSB(nn.Module):
    def __init__(self,
                 dim,  
                 attn_drop=0.,
                 proj_drop=0.,
                 n_levels=4,):
        
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels

        # Spatial Weighting
        self.vdown0=NormDownsample(dim, dim)
        self.vdown1=NormDownsample(dim, dim)
        self.vdown2=NormDownsample(dim, dim)

        self.cdown0=NormDownsample(dim, dim)
        self.cdown1=NormDownsample(dim, dim)
        self.cdown2=NormDownsample(dim, dim)
        
        self.attn=CAB(dim,4, bias=False)
        # # Feature Aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)
        
        # Activation
        self.act = nn.GELU() 

    def forward(self, x,y):
        h, w = x.size()[-2:]

        out = []
        downsampled_feat = []
        x1=self.vdown0(x)
        x2=self.vdown1(x1)
        x3=self.vdown2(x2)
        y1=self.cdown0(y)
        y2=self.cdown1(y1)
        y3=self.cdown2(y2)
        for i in reversed(range(self.n_levels)):
            s = self.mfr[i](downsampled_feat[i])
            s_upsample = F.interpolate(s, size=(s.shape[2]*2, s.shape[3]*2), mode='nearest')
            
            if i > 0:
                downsampled_feat[i-1] = downsampled_feat[i-1] + s_upsample
                
            s_original_shape = F.interpolate(s, size=(h, w), mode='nearest')
            out.append(s_original_shape)

        out = self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * x
        return out

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x
    

class se_block(nn.Module):
    def __init__(self, channels, ratio=16):
        super(se_block, self).__init__()
        # 空间信息进行压缩
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 经过两次全连接层，学习不同通道的重要性
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // ratio, False),
            nn.ReLU(),
            nn.Linear(channels // ratio, channels, False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()  # 取出batch size和通道数

        # b,c,w,h->b,c,1,1->b,c 压缩与通道信息学习
        avg = self.avgpool(x).view(b, c)

        # b,c->b,c->b,c,1,1 激励操作
        y = self.fc(avg).view(b, c, 1, 1)

        return x * y.expand_as(x)
    
class DenoisingCNN(nn.Module):
    def __init__(self, channles):
        super(DenoisingCNN, self).__init__()

        # 输入预处理
        self.input_preprocess = nn.Sequential(
            nn.Conv2d(4, channles // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=False),
            DepthwiseSeparableConv2d(channles // 2, channles, kernel_size=3, padding=1,stride=1),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(channles, channles, kernel_size=1, padding=0),
            nn.LeakyReLU(inplace=False),
        )

        # 卷积层堆叠
        self.conv_layers = nn.Sequential(
            nn.Conv2d(channles, channles, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(channles, channles, kernel_size=5, padding=2),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(channles, channles, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=False),
        )

        # 反卷积层或上采样
        self.output_layer = nn.Sequential(
            nn.Conv2d(channles, channles, kernel_size=1, padding=0),
            nn.LeakyReLU(inplace=False),
            DepthwiseSeparableConv2d(channles, channles // 2, kernel_size=3, padding=1,stride=1),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(channles // 2, 3, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=False)
        )
        self.se_block = se_block(channels=channles)
        self.out_conv = DepthwiseSeparableConv2d(4, 3, 1, 1, 0)

    def forward(self, x):
        x_bright, _ = torch.max(x, dim=1, keepdim=True)  # CPU
        x_in = torch.cat((x, x_bright), 1)
        # 前向传播
        T = self.input_preprocess(x_in)
        # print(T.shape)
        S = self.conv_layers(T)
        # print(S.shape)
        T = T + S

        T = self.se_block(T)

        T = self.output_layer(T)
        final_min, _ = torch.min(T, dim=1, keepdim=True)  # CPU
        final = torch.cat((T, final_min), 1)
        final = self.out_conv(final)

        T = final
        return T






    

#########################ExpoMamba#########################
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)

        return x
class kan(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5, version=4):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        

        grid_size=5
        spline_order=3
        scale_noise=0.1
        scale_base=1.0
        scale_spline=1.0
        base_activation=torch.nn.SiLU
        grid_eps=0.02
        grid_range=[-1, 1]

        self.fc1 = KANLinear(
                    in_features,
                    hidden_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
        
        # self.fc2 = nn.Linear(hidden_features, out_features)
        self.fc2 = KANLinear(
                    hidden_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )

        self.fc3 = KANLinear(
                    hidden_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )   

        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(hidden_features)
        self.dwconv_3 = DW_bn_relu(hidden_features)


    
        self.drop = nn.Dropout(drop)

        self.shift_size = shift_size
        self.pad = shift_size // 2

        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    

    def forward(self, x):
        # pdb.set_trace()
        B,C,H,W=x.shape
        x=x.permute(0,2,3,1).view(B,-1,C)
        B, N, C = x.shape


        x = self.fc1(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_1(x, H, W)

        x = self.fc2(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_2(x, H, W)

        x = self.fc3(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_3(x, H, W)
    
        x=x.permute(0,2,1).view(B,C,H,W)
        return x
######################
class mymodelycbcr(nn.Module):
    def __init__(self, filters=32,
                 channels=[32, 32, 64, 128],
                 heads=[1, 4, 8, 16],
                 norm=False
        ):
        super().__init__()
        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads
        self.colorEncoder =nn.Sequential(
            nn.Conv2d(3, ch1, 3, stride=1, padding=1,bias=False),
            myencoderBlock(filters),
            myencoderBlock(filters),
            mydoubleconv(filters)
        )
        self.LightEncoder=nn.Sequential(
            nn.Conv2d(1, ch1, 3, stride=1, padding=1,bias=False),
            myencoderBlock(filters),
            myencoderBlock(filters),
            mydoubleconv(filters)
        )    

        self.estimator = Illumination_Estimator(filters)
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm = norm)
        
        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm = norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm = norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm = norm)

        self.IE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm = norm)
        
        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        
        self.HV_LCA1 = MyI_LCA2(ch2, head2)
        self.HV_LCA2 = MyI_LCA2(ch3, head3)
        self.HV_LCA3 = MyI_LCA2(ch4, head4)
        self.HV_LCA4 = MyI_LCA2(ch4, head4)
        self.HV_LCA5 = MyI_LCA2(ch3, head3)
        self.HV_LCA6 = MyI_LCA2(ch2, head2)
        
        self.kan1=kan(ch4,ch4)
        self.kan2=kan(ch4,ch4)
        self.I_LCA1 = MyI_LCA2(ch2, head2)
        self.I_LCA2 = MyI_LCA2(ch3, head3)
        self.I_LCA3 = MyI_LCA2(ch4, head4)
        self.I_LCA4 = MyI_LCA2(ch4, head4)
        self.I_LCA5 = MyI_LCA2(ch3, head3)
        self.I_LCA6 = MyI_LCA2(ch2, head2)
        self.fuse2=DetailFeatureExtractor(num_layers=1,dim=filters)
        #self.fuse1=DetailFeatureExtractor(num_layers=1,dim=ch2)
        #self.fuse0=DetailFeatureExtractor(num_layers=1,dim=ch3)
        self.v2 = nn.Conv2d(filters*2,filters,3,1,1)
        #self.v1 = nn.Conv2d(ch2*2,filters,3,1,1)
        #self.v0 = nn.Conv2d(ch3*2,filters,3,1,1)
        self.c = nn.Conv2d(filters,3,3,1,1)
        self.sigmoid=Swish()
        self.apply(self._init_weights)

    def _rgb_to_ycbcr(self, image):
        r, g, b = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]
    
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.28886 * g + 0.436 * b + 0.5
        v = 0.615 * r - 0.51499 * g - 0.10001 * b + 0.5
        
        yuv = torch.stack((y, u, v), dim=1)
        return yuv
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, input):
        _, illu_map = self.estimator(input)
        input_img = input * illu_map +input

        ycbcr=self._rgb_to_ycbcr(input_img)
        y, _, _ = torch.split(ycbcr, 1, dim=1)
        lum=self.LightEncoder(y)
        fea=self.colorEncoder(ycbcr)


        i_enc1 = self.IE_block1(lum)
        hv_1 = self.HVE_block1(fea)


        i_jump0 = lum
        hv_jump0 = fea
        
        i_enc2 = self.I_LCA1(i_enc1, hv_1)
        hv_2 = self.HV_LCA1(hv_1, i_enc1)

        v_jump1 = i_enc2
        hv_jump1 = hv_2
        i_enc2 = self.IE_block2(i_enc2)
        hv_2 = self.HVE_block2(hv_2)

        i_enc3 = self.I_LCA2(i_enc2, hv_2)
        hv_3 = self.HV_LCA2(hv_2, i_enc2)

        v_jump2 = i_enc3
        hv_jump2 = hv_3
        i_enc3 = self.IE_block3(i_enc2)
        hv_3 = self.HVE_block3(hv_2)

        i_enc4 = self.I_LCA3(i_enc3, hv_3)
        hv_4 = self.HV_LCA3(hv_3, i_enc3)

        i_enc4=self.kan1(i_enc4)
        i_enc4=self.kan2(i_enc4)
        hv_4=self.kan1(hv_4)
        hv_4=self.kan2(hv_4)

        i_dec4 = self.I_LCA4(i_enc4,hv_4)
        hv_4 = self.HV_LCA4(hv_4, i_enc4)
        
        hv_3 = self.HVD_block3(hv_4, hv_jump2)
        i_dec3 = self.ID_block3(i_dec4, v_jump2)

        i_dec2 = self.I_LCA5(i_dec3, hv_3)
        hv_2 = self.HV_LCA5(hv_3, i_dec3)
        
        hv_2 = self.HVD_block2(hv_2, hv_jump1)
        i_dec2 = self.ID_block2(i_dec3, v_jump1)
        
        i_dec1 = self.I_LCA6(i_dec2, hv_2)
        hv_1 = self.HV_LCA6(hv_2, i_dec2)
        
        i_dec1 = self.ID_block1(i_dec1, i_jump0)
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        #fuse0=self.c(self.v0(self.fuse0(torch.cat([i_dec3,hv_3], dim=1))))
        #fuse1=self.c(self.v1(self.fuse1(torch.cat([i_dec2,hv_2], dim=1))))
        fuse2=self.c(self.sigmoid(self.v2(self.fuse2(torch.cat([i_dec1,hv_1], dim=1)))))
        return fuse2#,fuse1,fuse0


class mymodel(nn.Module):
    def __init__(self, filters=64,
                 channels=[32, 32, 64, 128],
                 heads=[1, 4, 8, 16],
                 norm=False
        ):
        super().__init__()
        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads
        self.pixeldown=nn.PixelUnshuffle(4)
        self.pixeldownconv=nn.Conv2d(48,filters,kernel_size=3, bias=True)

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, input):
        pass
class net(nn.Module):
    def __init__(self, filters=32):
        super().__init__()
        self.Unet=mymodelycbcr()

    def forward(self, inputs):
        out_unet  = self.Unet(inputs)
        final = out_unet + inputs
        return final
##########################################################################
n