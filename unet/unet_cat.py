import torch
import torch.nn as nn
import torch.nn.functional as F
from CBAM import *
from SE import *
from restormer import *

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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpooling = nn.MaxPool2d(2)
        self.res = ResidualBlock(in_channels, out_channels)
        self.FAM = FAM(in_channels)
        self.in_channels = in_channels
    def forward(self, x, y):
        x = self.maxpooling(x)
        if ( self.in_channels != 256 ):
          x = self.FAM(x, y)
        x = self.res(x)
        return x


class Up(nn.Module):
    """上採樣"""
    def __init__(self, in_channels, out_channels, cat_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.res = ResidualBlock(cat_channels, out_channels)
        #self.tdn = TransformerBlock(dim=cat_channels, num_heads=4, ffn_expansion_factor=2.0, bias=True, LayerNorm_type='WithBias')
        self.in_channels = in_channels
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        #x = self.tdn(x)
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

class Downsample(nn.Module):
    def __init__(self, n_feat, heads, num_blocks):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2),
                                  *[TransformerBlock(dim=n_feat*2, num_heads=heads, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias') for i in range(num_blocks)])
        
    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat, heads, num_blocks):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2),
                                  *[TransformerBlock(dim=n_feat//2, num_heads=heads, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias') for i in range(num_blocks)])

    def forward(self, x):
        return self.body(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inc = ResidualBlock(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        #self.down4 = Down(512, 1024)
        #self.up1 = Up(1024, 512, 768)
        self.up2 = Up(512, 256, 400)
        self.up3 = Up(256, 128, 272)
        self.up4 = Up(128, 64, 208)
        self.outc = OutConv(64, out_channels)
        self.maxpool_2 = nn.MaxPool2d(2)
        self.maxpool_4 = nn.MaxPool2d(4)
        self.maxpool_8 = nn.MaxPool2d(8)
        self.conv_1 = nn.Conv2d(64, 48, kernel_size=1)
        self.conv_2 = nn.Conv2d(128, 48, kernel_size=1)
        self.conv_3 = nn.Conv2d(256, 48, kernel_size=1)
        self.conv_4 = nn.Conv2d(512, 64, kernel_size=1)
        base_channel = 64
        self.FAM1 = FAM(base_channel )
        self.SCM1 = SCM(base_channel )
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)
        self.outconv1 = nn.Conv2d(256, 3, kernel_size=3, padding=1)
        self.outconv2 = nn.Conv2d(128, 3, kernel_size=3, padding=1)
        self.conv_cat_3 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=1))
        self.conv_cat_2 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1))
        self.conv_cat_1 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1))
        self.conv_up_1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1))
        self.conv_up_2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1))
        self.conv_up_3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1))
        self.down1_t = Downsample(n_feat=64, heads=1, num_blocks=1)
        self.down2_t = Downsample(n_feat=128, heads=2, num_blocks=2)
        self.down3_t = Downsample(n_feat=256, heads=4, num_blocks=4)

        self.up1_t = Upsample(n_feat=512, heads=4, num_blocks=4)
        self.up2_t = Upsample(n_feat=256, heads=2, num_blocks=2)
        self.up3_t = Upsample(n_feat=128, heads=1, num_blocks=1)
        
    def forward(self, x):
        x_origianl = x
        y_origianl = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        z_origianl = F.interpolate(x, scale_factor=0.25, mode='bilinear')
        y = self.SCM1(y_origianl)
        z = self.SCM2(z_origianl)
     
        x1 = self.inc(x)
        x2 = self.down1(x1, y)
        x2_t = self.down1_t(x1)
        x2 = self.conv_cat_1(torch.cat([x2, x2_t], dim=1))
        x3 = self.down2(x2, z)
        x3_t = self.down2_t(x2)
        x3 = self.conv_cat_2(torch.cat([x3, x3_t], dim=1))
        x4 = self.down3(x3, x3)   #沒要做
        x4_t = self.down3_t(x3)
        x4 = self.conv_cat_3(torch.cat([x4, x4_t], dim=1))

        x1_part = self.conv_1(self.maxpool_4(x1))
        x2_part = self.conv_2(self.maxpool_2(x2))
        x3_part = self.conv_3(x3) 
        x_cat = torch.cat([x3_part, x2_part, x1_part], dim=1)
        
        x5 = self.up2(x4, x_cat)
        x5_t = self.up1_t(x4)
        x5 = self.conv_up_1(torch.cat([x5, x5_t], dim=1))
        output1 = self.outconv1(x5) + z_origianl

        x1_part = self.conv_1(self.maxpool_2(x1))
        x2_part = self.conv_2(x2)
        x3_part = self.conv_3(nn.functional.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True))
        x_cat = torch.cat([x3_part, x2_part, x1_part], dim=1)
    
        x6 = self.up3(x5, x_cat)
        x6_t = self.up2_t(x5)
        x6 = self.conv_up_2(torch.cat([x6, x6_t], dim=1))
        output2 = self.outconv2(x6) + y_origianl 

        x1_part = self.conv_1(x1)
        x2_part = self.conv_2(nn.functional.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True))
        x3_part = self.conv_3(nn.functional.interpolate(x3, scale_factor=4, mode='bilinear', align_corners=True))
        x_cat = torch.cat([x3_part, x2_part, x1_part], dim=1)
        
        x7 = self.up4(x6, x_cat)
        x7_t = self.up3_t(x6)
        x = self.conv_up_3(torch.cat([x7, x7_t], dim=1))
        output3 = self.outc(x) + x_origianl
        
        return output3, output2, output1


