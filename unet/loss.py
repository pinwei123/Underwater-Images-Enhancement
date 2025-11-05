import math
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F


def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')
    
class AFFTLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(AFFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.criterion = torch.nn.L1Loss(reduction=reduction)

    def forward(self, pred, target, weight=None, **kwargs):
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)
        pred_fft = torch.abs(pred_fft)
        target_fft = torch.abs(target_fft)
        return self.loss_weight * self.criterion(pred_fft, target_fft)
        