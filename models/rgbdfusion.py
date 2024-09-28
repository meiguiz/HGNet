import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import thop

from torch.nn import init
from thop import profile



class GraspModel(nn.Module):
    """
    An abstract model for grasp network in a common format.
    """

    def __init__(self):
        super(GraspModel, self).__init__()

    def forward(self, xd, xr):
        raise NotImplementedError()

    def compute_loss(self, xdc, xrc, yc):
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xdc, xrc)

        p_loss = F.smooth_l1_loss(pos_pred, y_pos)
        cos_loss = F.smooth_l1_loss(cos_pred, y_cos)
        sin_loss = F.smooth_l1_loss(sin_pred, y_sin)
        width_loss = F.smooth_l1_loss(width_pred, y_width)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }

    def predict(self, xdc, xrc):
        pos_pred, cos_pred, sin_pred, width_pred = self(xdc, xrc)
        return {
            'pos': pos_pred,
            'cos': cos_pred,
            'sin': sin_pred,
            'width': width_pred
        }


def Conv1(input_channel, output_channel, kernel_size):
    return nn.Sequential(
        nn.Conv2d(input_channel, output_channel, kernel_size, stride=2, padding=0),
        nn.BatchNorm2d(output_channel),
        nn.ReLU(inplace=True)
    )


def Conv2(input_channel, output_channel, kernel_size):
    return nn.Sequential(
        nn.Conv2d(input_channel, output_channel, kernel_size, stride=1, padding=kernel_size//2),
        nn.BatchNorm2d(output_channel),
        nn.ReLU(inplace=True)
    )


def Conv3(input_channel, output_channel, kernel_size1, kernel_size2):
    return nn.Sequential(
        nn.Conv2d(input_channel, output_channel, kernel_size=(kernel_size1, kernel_size2),
                  stride=1, padding=(kernel_size1//2, kernel_size2//2)),
        nn.BatchNorm2d(output_channel),
        nn.ReLU(inplace=True)
    )


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class MFF(nn.Module):

    def __init__(self, in_channels, rate, reduction):
        super(MFF, self).__init__()
        self.layer1 = nn.BatchNorm2d(in_channels)
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=rate, dilation=rate),
            nn.BatchNorm2d(in_channels),
        )
        self.se = SEAttention(in_channels, reduction=reduction)
        self.rescale = nn.Conv2d(in_channels*2, in_channels, 1, 1, 0)

    def forward(self, xd, xr):
        xd = self.se(self.layer1(xd)+self.layer2(xd)+self.layer3(xd))
        xr = self.se(self.layer1(xr)+self.layer2(xr)+self.layer3(xr))
        out = self.rescale(torch.cat((xd, xr), dim=1))

        return out


class MFA1(nn.Module):
    def __init__(self, in_channels):
        super(MFA1, self).__init__()
        self.block1 = nn.Sequential(
            Conv2(in_channels, 32, 1),
        )

        self.block2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            Conv2(in_channels, 64, 1),
        )

        self.block3 = nn.Sequential(
            Conv2(in_channels, 128, 1),
            Conv3(128, 96, 1, 3),
            Conv3(96, 64, 3, 1)
        )

        self.block4 = nn.Sequential(
            Conv2(in_channels, 128, 1),
            Conv3(128, 96, 1, 7),
            Conv3(96, 96, 7, 1)
        )

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x)
        x3 = self.block3(x)
        x4 = self.block4(x)
        x5 = torch.cat([x1, x2, x3, x4], dim=1)
        return x5


class MFA2(nn.Module):
    def __init__(self, in_channels):
        super(MFA2, self).__init__()
        self.block1 = nn.Sequential(
            Conv2(in_channels, 16, 1),
        )

        self.block2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            Conv2(in_channels, 32, 1),
        )

        self.block3 = nn.Sequential(
            Conv2(in_channels, 64, 1),
            Conv3(64, 48, 1, 3),
            Conv3(48, 32, 3, 1)
        )

        self.block4 = nn.Sequential(
            Conv2(in_channels, 64, 1),
            Conv3(64, 48, 1, 7),
            Conv3(48, 48, 7, 1)
        )

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x)
        x3 = self.block3(x)
        x4 = self.block4(x)
        x5 = torch.cat([x1, x2, x3, x4], dim=1)
        return x5


class MFA3(nn.Module):
    def __init__(self, in_channels):
        super(MFA3, self).__init__()
        self.block1 = nn.Sequential(
            Conv2(in_channels, 8, 1),
        )

        self.block2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            Conv2(in_channels, 16, 1),
        )

        self.block3 = nn.Sequential(
            Conv2(in_channels, 32, 1),
            Conv3(32, 24, 1, 3),
            Conv3(24, 16, 3, 1)
        )

        self.block4 = nn.Sequential(
            Conv2(in_channels, 32, 1),
            Conv3(32, 24, 1, 7),
            Conv3(24, 24, 7, 1)
        )


    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x)
        x3 = self.block3(x)
        x4 = self.block4(x)
        x5 = torch.cat([x1, x2, x3, x4], dim=1)
        return x5


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):

    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = in_channels
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))
        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class SEAttention(nn.Module):

    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Fusion(GraspModel):

    def __init__(self, input_channels1, input_channels2, channel_size=32, output_channels=1, dropout=False, prob=0.0):
        super(Fusion, self).__init__()
        # Depth Encoder
        self.d1 = nn.Sequential(
         nn.Conv2d(input_channels1, channel_size, kernel_size=3, stride=1, padding=1),
         nn.BatchNorm2d(channel_size),
         )
        self.d2 = nn.Sequential(
         nn.Conv2d(channel_size, channel_size * 2, kernel_size=3, stride=2, padding=3, dilation=3),
         nn.BatchNorm2d(channel_size * 2),
        )
        self.d3 = nn.Sequential(
         nn.Conv2d(channel_size * 2, channel_size * 4, kernel_size=3, stride=2, padding=5, dilation=5),
         nn.BatchNorm2d(channel_size * 4),
        )
        self.d4 = nn.Sequential(
         nn.Conv2d(channel_size * 4, channel_size * 8, kernel_size=3, stride=2, padding=7, dilation=7),
         nn.BatchNorm2d(channel_size * 8),
        )
        # RGB Encoder
        self.r1 = nn.Sequential(
         nn.Conv2d(input_channels2, channel_size, kernel_size=3, stride=1, padding=1),
         nn.BatchNorm2d(channel_size),
        )
        self.r2 = nn.Sequential(
         nn.Conv2d(channel_size, channel_size * 2, kernel_size=3, stride=2, padding=3, dilation=3),
         nn.BatchNorm2d(channel_size * 2),
        )
        self.r3 = nn.Sequential(
         nn.Conv2d(channel_size * 2, channel_size * 4, kernel_size=3, stride=2, padding=5, dilation=5),
         nn.BatchNorm2d(channel_size * 4),
        )
        self.r4 = nn.Sequential(
         nn.Conv2d(channel_size * 4, channel_size * 8, kernel_size=3, stride=2, padding=7, dilation=7),
         nn.BatchNorm2d(channel_size * 8),
        )
        #  Fusion
        self.mff1 = MFF(32, 1, 1)
        self.mff2 = MFF(64, 3, 2)
        self.mff3 = MFF(128, 5, 4)
        self.mff4 = MFF(256, 7, 8)
        #  Decoder
        self.rgbaspp = ASPP(256, [6, 12, 18])
        self.depthaspp = ASPP(256, [6, 12, 18])
        self.rescale = nn.Conv2d(512, 256, 1, 1, 0)

        self.mfa1 = MFA1(channel_size * 8)
        self.mfa2 = MFA2(channel_size * 4)
        self.mfa3 = MFA3(channel_size * 2)

        self.conv1 = nn.ConvTranspose2d(channel_size * 8, channel_size * 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(channel_size * 4)

        self.conv2 = nn.ConvTranspose2d(channel_size * 4, channel_size * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(channel_size * 2)

        self.conv3 = nn.ConvTranspose2d(channel_size * 2, channel_size, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(channel_size)

        self.conv4 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=4, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(channel_size)

        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_in):
        # print(xd.shape)
        # print(xr.shape)
        xr = x_in[:, :3, :, :]
        xd = x_in[:, 3:4, :, :]
        xd1 = F.relu(self.d1(xd))
        xr1 = F.relu(self.r1(xr))
        xf1 = self.mff1(xd1, xr1)

        xd2 = F.relu(self.d2(xf1))
        xr2 = F.relu(self.r2(xr1))
        xf2 = self.mff2(xd2, xr2)

        xd3 = F.relu(self.d3(xf2))
        xr3 = F.relu(self.r3(xr2))
        xf3 = self.mff3(xd3, xr3)

        xd4 = F.relu(self.d4(xf3))
        xr4 = F.relu(self.r4(xr3))
        xf4 = self.mff4(xd4, xr4)

        xrgbaspp = F.relu(self.rgbaspp(xr4))
        xdepthaspp = F.relu(self.depthaspp(xd4))
        xconcat = self.rescale(torch.cat((xrgbaspp, xdepthaspp), dim=1))
        xdecoder1 = F.relu(self.mfa1(torch.add(xconcat, xf4)))

        xdecoder1 = F.relu(self.bn1(self.conv1(xdecoder1)))

        xdecoder2 = F.relu(self.mfa2(torch.add(xdecoder1, xf3)))
        xdecoder2 = F.relu(self.bn2(self.conv2(xdecoder2)))

        xdecoder3 = F.relu(self.mfa3(torch.add(xdecoder2, xf2)))
        xdecoder3 = F.relu(self.bn3(self.conv3(xdecoder3)))

        out = F.relu(self.bn4(self.conv4(xdecoder3)))

        if self.dropout:
            pos_output = self.sigmoid(self.pos_output(self.dropout_pos(out)))
            cos_output = self.tanh(self.cos_output(self.dropout_cos(out)))
            sin_output = self.tanh(self.sin_output(self.dropout_sin(out)))
            width_output = self.width_output(self.dropout_wid(out))
        else:
            pos_output = self.sigmoid(self.pos_output(out))
            cos_output = self.tanh(self.cos_output(out))
            sin_output = self.tanh(self.sin_output(out))
            width_output = self.width_output(out)

        return pos_output, cos_output, sin_output, width_output




