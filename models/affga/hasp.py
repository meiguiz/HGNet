import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.affga.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=128, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = xa
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo

class _HASPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_HASPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class HASP(nn.Module):
    def __init__(self, BatchNorm):
        """
        :param backbone: resnet
        :param output_stride: 16
        :param BatchNorm:
        """
        super(HASP, self).__init__()

        # HASP
        #self.hasp1_1 = _HASPModule(256, 128, 1, padding=0, dilation=1, BatchNorm=BatchNorm)
        self.hasp1_2 = _HASPModule(64, 64, 3, padding=6, dilation=6, BatchNorm=BatchNorm)
        self.hasp2_1 = _HASPModule(64, 64, 3, padding=12, dilation=12, BatchNorm=BatchNorm)
        self.hasp2_2 = _HASPModule(64, 64, 3, padding=18, dilation=18, BatchNorm=BatchNorm)
        self.aff = AFF(channels=64)
        self.aff_all = AFF(channels=256)
        self.conv_small = nn.Sequential(nn.Conv2d(192, 128, 1, bias=False),
                                        BatchNorm(128),
                                        nn.ReLU(),
                                        nn.Dropout(0.5))

        self.conv_big = nn.Sequential(nn.Conv2d(192, 128, 1, bias=False),
                                      BatchNorm(128),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))

        self.conv_all = nn.Sequential(nn.Conv2d(256, 128, 1, bias=False),
                                      BatchNorm(128),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))

        self._init_weight()

    def forward(self, x):
        x1_1 = x
        x1_2 = self.hasp1_2(x)
        x2_1 = self.hasp2_1(x)
        x2_2 = self.hasp2_2(x)

        x_small = torch.cat((x1_1, x1_2), dim=1)
        x_big = torch.cat((x2_1, x2_2), dim=1)
        #x_all = torch.cat((x1_1, x1_2, x2_1, x2_2),dim=1)

        x_small = self.aff(x1_1,x1_2)
        x_big = self.aff(x2_1,x2_2)
        x_all = self.aff(x_small, x_big)
        #x_all = torch.cat((x_small,x_big), dim=1)
        #x_all = self.conv_all(x_all)
        #x_all = x

        return x_all

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_hasp( BatchNorm):
    return HASP(BatchNorm)
