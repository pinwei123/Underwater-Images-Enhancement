import torch
import torch.nn as nn

# 通道注意力模塊
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        
        # 全局平均池化和最大池化
        avg_pool = self.fc(self.global_avg_pool(x).view(b, c)).view(b, c, 1, 1)
        max_pool = self.fc(self.global_max_pool(x).view(b, c)).view(b, c, 1, 1)
        
        # 加和後經過 Sigmoid 激活
        attention = self.sigmoid(avg_pool + max_pool)
        return x * attention

# 空間注意力模塊
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2  # 保證卷積不改變特徵圖大小
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 在通道維度進行平均池化和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        pool_out = torch.cat([avg_out, max_out], dim=1)
        
        # 通過卷積和激活
        attention = self.sigmoid(self.conv(pool_out))
        return x * attention

# CBAM 模塊
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # 先進行通道注意力，再進行空間注意力
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x