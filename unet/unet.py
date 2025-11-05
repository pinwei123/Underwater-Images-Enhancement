import torch
import torch.nn as nn
import torch.nn.functional as F
from CBAM import *
from TDN_network import *

class DoubleConv(nn.Module):
    """做兩次捲積提取特徵"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        # 定義兩層卷積
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.cbam = CBAM(in_channels)
        self.cbam_2 = CBAM(out_channels)
        # 如果輸入和輸出通道數不同，則需要一個1x1卷積來匹配通道數
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()  # 保持通道數相同的捷徑連接

    def forward(self, x):
        # 原始輸入
        identity = self.shortcut(x)
        x = self.cbam(x)
        # 殘差分支
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        # 將原始輸入加回來
        out += identity
        out = self.relu(out)  # 激活後的輸出
        out = self.cbam_2(out)
        return out

class Down(nn.Module):
    """下採樣"""

    def __init__(self, in_channels, out_channels, num_heads):
        super().__init__()
        self.maxpooling = nn.MaxPool2d(2)
        self.res = ResidualBlock(in_channels, out_channels)
        self.FAM = FAM(in_channels)
        self.in_channels = in_channels
        self.tdn = TransformerBlock(dim=in_channels, num_heads=num_heads, ffn_expansion_factor=2, bias=True, LayerNorm_type='WithBias')
    def forward(self, x, y):
        x = self.maxpooling(x)
        if ( self.in_channels != 256 ):
          x = self.FAM(x, y)
        x = self.tdn(x)
        x = self.res(x)
        return x


class Up(nn.Module):
    """上採樣"""
    def __init__(self, in_channels, out_channels, cat_channels, num_heads):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.res = ResidualBlock(cat_channels, out_channels)
        self.tdn = TransformerBlock(dim=cat_channels, num_heads=num_heads, ffn_expansion_factor=2, bias=True, LayerNorm_type='WithBias')
        self.in_channels = in_channels
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.tdn(x)
        x = self.res(x)
        return x

class OutConv(nn.Module):
    """結束前用1x1的kernel調整channels數量"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

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
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane-3, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inc = ResidualBlock(in_channels, 64)
        self.down1 = Down(64, 128, 1)
        self.down2 = Down(128, 256, 2)
        self.down3 = Down(256, 512, 4)
        #self.down4 = Down(512, 1024)
        #self.up1 = Up(1024, 512, 768)
        self.up2 = Up(512, 256, 448, 4)
        self.up3 = Up(256, 128, 320, 2)
        self.up4 = Up(128, 64, 256, 1)
        self.outc = OutConv(64, out_channels)
        self.maxpool_2 = nn.MaxPool2d(2)
        self.maxpool_4 = nn.MaxPool2d(4)
        self.maxpool_8 = nn.MaxPool2d(8)
        self.conv_1 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv_2 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv_3 = nn.Conv2d(256, 64, kernel_size=1)
        self.conv_4 = nn.Conv2d(512, 64, kernel_size=1)
        base_channel = 64
        self.FAM1 = FAM(base_channel )
        self.SCM1 = SCM(base_channel )
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)
        self.outconv1 = nn.Conv2d(256, 3, kernel_size=3, padding=1)
        self.outconv2 = nn.Conv2d(128, 3, kernel_size=3, padding=1)
       
        
    def forward(self, x):
        x_origianl = x
        y_origianl = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        z_origianl = F.interpolate(x, scale_factor=0.25, mode='bilinear')
        y = self.SCM1(y_origianl)
        z = self.SCM2(z_origianl)

        
        x1 = self.inc(x)
        x2 = self.down1(x1, y)
        x3 = self.down2(x2, z)
        x4 = self.down3(x3, x3)   #沒要做


        x1_part = self.conv_1(self.maxpool_4(x1))
        x2_part = self.conv_2(self.maxpool_2(x2))
        x3_part = self.conv_3(x3) 
        x_cat = torch.cat([x3_part, x2_part, x1_part], dim=1)
        
        x = self.up2(x4, x_cat)
        output1 = self.outconv1(x) + z_origianl

        x1_part = self.conv_1(self.maxpool_2(x1))
        x2_part = self.conv_2(x2)
        x3_part = self.conv_3(nn.functional.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True))
        x_cat = torch.cat([x3_part, x2_part, x1_part], dim=1)
    
        x = self.up3(x, x_cat)
        output2 = self.outconv2(x) + y_origianl 

        x1_part = self.conv_1(x1)
        x2_part = self.conv_2(nn.functional.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True))
        x3_part = self.conv_3(nn.functional.interpolate(x3, scale_factor=4, mode='bilinear', align_corners=True))
        x_cat = torch.cat([x3_part, x2_part, x1_part], dim=1)
        
        x = self.up4(x, x_cat)
        output3 = self.outc(x) + x_origianl
        
        return output3, output2, output1


