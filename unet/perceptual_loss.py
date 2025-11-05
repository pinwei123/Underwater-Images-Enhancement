import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class PerceptualLoss(nn.Module):
    def __init__(self, device='cuda', pretrained=True):
        super(PerceptualLoss, self).__init__()
        # 使用預訓練的 VGG16 網絡，取它的卷積層部分
        vgg = models.vgg16(pretrained=pretrained).features.to(device)
        
        # 我們選擇 VGG 的某些卷積層來提取特徵
        self.layers = nn.Sequential(
            vgg[0],  # Conv1_1
            vgg[1],  # Relu1_1
            vgg[2],  # Conv1_2
            vgg[3],  # Relu1_2
            vgg[4],  # MaxPool1
            vgg[5],  # Conv2_1
            vgg[6],  # Relu2_1
            vgg[7],  # Conv2_2
            vgg[8],  # Relu2_2
            vgg[9],  # MaxPool2
            vgg[10], # Conv3_1
        ).to(device)  # 把層移到 GPU
        
        # 設置為不訓練模式，保持權重不更新
        for param in self.layers.parameters():
            param.requires_grad = False
        self.device = device

    def forward(self, output, target):
        # 確保 output 和 target 也在 GPU 上
        output = output.to(self.device)
        target = target.to(self.device)
        
        # 從重建圖像和真實圖像中提取 VGG 特徵
        output_features = self.layers(output)
        target_features = self.layers(target)
        
        # 計算兩者的 L2 損失
        loss = F.mse_loss(output_features, target_features)
        return loss
