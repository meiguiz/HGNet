import torch
from torch import nn
import torch.nn.functional as F
class MFModel(nn.Module):
    def __init__(self):
        super(MFModel, self).__init__()
        self.h_avg = nn.AdaptiveAvgPool2d((20, 1))
        self.w_avg = nn.AdaptiveAvgPool2d((1, 20))
        self.F_1 = nn.Sequential(nn.Conv2d(128, 128, (3, 1), 1, 1))
        self.F_2 = nn.Sequential(nn.Conv2d(128, 128, (1, 3), 1, 1))


    def forward(self, x):
        b, c, _, _ = x.size()
        x_1 = self.h_avg(x)
        x_2 = self.F_1(x_1)
        x_3 = F.interpolate(x_2, size=x.size()[2:], mode='bilinear', align_corners=True)
        x_4 = self.w_avg(x)
        x_5 = self.F_1(x_4)
        x_6 = F.interpolate(x_5, size=x.size()[2:], mode='bilinear', align_corners=True)
        return x_3, x_6


