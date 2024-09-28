# 童凌
import torch.nn as nn
import torch
import torch.nn.functional as F
from models.affga.decoder import build_decoder
from models.affga.hasp import build_hasp
from models.tranformer_grasp import lvt
from models.tranformer_grasp_depth import lvt_depth
from models.mf import MFModel
from models.affga.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        #output = self.sigmoid(output)
        output = self.softmax(output)
        return output
class SEModel(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEModel, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class Exchange(nn.Module):
    def __init__(self):
        super(Exchange, self).__init__()

    def forward(self, x, bn, bn_threshold):
        bn1, bn2 = bn[0].weight.abs(), bn[1].weight.abs()
        x1, x2 = torch.zeros_like(x[0]), torch.zeros_like(x[1])
        x1[:, bn1 >= bn_threshold] = x[0][:, bn1 >= bn_threshold]
        x1[:, bn1 < bn_threshold] = x[1][:, bn1 < bn_threshold]
        x2[:, bn2 >= bn_threshold] = x[1][:, bn2 >= bn_threshold]
        x2[:, bn2 < bn_threshold] = x[0][:, bn2 < bn_threshold]
        return [x1, x2]
class BatchNorm2dParallel(nn.Module):
    def __init__(self, num_features, num_parallel):
        super(BatchNorm2dParallel, self).__init__()
        for i in range(num_parallel):
            setattr(self, 'bn_' + str(i), nn.BatchNorm2d(num_features))

    def forward(self, x_parallel):
        return [getattr(self, 'bn_' + str(i))(x) for i, x in enumerate(x_parallel)]
class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]
def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return ModuleParallel(nn.Conv2d(in_planes, out_planes, kernel_size=3,
                                    stride=stride, padding=1, bias=bias))


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return ModuleParallel(nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                    stride=stride, padding=0, bias=bias))
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, num_parallel, bn_threshold, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = BatchNorm2dParallel(planes, num_parallel)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = BatchNorm2dParallel(planes, num_parallel)
        self.conv3 = conv1x1(planes, planes * 4)
        self.bn3 = BatchNorm2dParallel(planes * 4, num_parallel)
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.num_parallel = num_parallel
        self.downsample = downsample
        self.stride = stride

        self.exchange = Exchange()
        self.bn_threshold = bn_threshold
        self.bn2_list = []
        for module in self.bn2.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.bn2_list.append(module)

    def forward(self, x):
        residual = x
        out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if len(x) > 1:
            out = self.exchange(out, self.bn2_list, self.bn_threshold)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = [out[l] for l in range(self.num_parallel)]
        out = self.relu(out)

        return out
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



class HGNet(nn.Module):


    def __init__(self, angle_cls,  backbone='resnet',  num_classes=21,
                 freeze_bn=False, size=300):
        super(HGNet, self).__init__()
        BatchNorm = nn.BatchNorm2d
        self.backbone_rgb = lvt(with_cls_head=False)
        self.backbone_depth = lvt_depth(with_cls_head=False)
        self.weight = SpatialAttention()
        self.channel = SEModel(channel=128)

        self.hasp = build_hasp(BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm, size, angle_cls=angle_cls)          # 解码器
        self.freeze_bn = freeze_bn
        self.fusion = MFModel()
        self.fusion_high = nn.Sequential(
            nn.Conv2d(256, 128, 1, 1, 0),

        )
        self.fusion_low = nn.Sequential(
            nn.Conv2d(128, 64, 1, 1, 0))

        self.block = Bottleneck(inplanes=3, planes=16, num_parallel=2, bn_threshold=2e-2)

        # HASP
        # self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=9, stride=1, padding=4)
        # self.bn1 = nn.BatchNorm2d(32)
        #
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        # self.bn2 = nn.BatchNorm2d(64)
        #
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        # self.bn3 = nn.BatchNorm2d(128)
        #
        # self.res1 = ResidualBlock(128, 128)
        # self.res2 = ResidualBlock(128, 128)
        # self.res3 = ResidualBlock(128, 128)
        # self.res4 = ResidualBlock(128, 128)
        # self.res5 = ResidualBlock(128, 128)
        #
        # self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=1)
        # self.bn4 = nn.BatchNorm2d(64)
        #
        # self.conv5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=2, output_padding=1)
        # self.bn5 = nn.BatchNorm2d(32)
        #
        # self.conv6 = nn.ConvTranspose2d(32, 32, kernel_size=9, stride=1, padding=4)
        #
        # self.pos_output = nn.Conv2d(32, 1, kernel_size=2)
        # self.cos_output = nn.Conv2d(32, 1, kernel_size=2)
        # self.sin_output = nn.Conv2d(32, 1, kernel_size=2)
        # self.width_output = nn.Conv2d(32, 1, kernel_size=2)
        #
        # self.dropout1 = nn.Dropout(p=prob)
        #
        # for m in self.modules():
        #     if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        #         nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_in):
        # feat_1, x = self.backbone(x_in)
        # x_all = self.hasp(x)
        # x_edge = self.dexinet(x_in[:, :3, :, :])
        # x_edge = x_edge[6]
        #x_edge =  F.interpolate(x_edge, size=(80,80), mode='bilinear', align_corners=True)

        # angle_pred = F.interpolate(angle_pred, size=input.size()[2:], mode='bilinear', align_corners=True)  # 上采样（双线性插值）
        # width_pred = F.interpolate(width_pred, size=input.size()[2:], mode='bilinear', align_corners=True)  # 上采样（双线性插值）
        rgb_x = x_in[:, :3, :, :]
        depth_x = x_in[:, 3:6, :, :]
        rgb_x = self.backbone_rgb(rgb_x)
        depth_x = self.backbone_depth(depth_x)
        rgb_high = rgb_x[2]
        rgb_low = rgb_x[0]

        rgb_high_1, rgb_high_2 = self.fusion(rgb_high)
        depth_high = depth_x[2]
        depth_low = depth_x[0]

        depth_high_1, depth_high_2 = self.fusion(depth_high)
        high_f_1 = rgb_high_1 + depth_high_1
        high_f_2 = rgb_high_2 + depth_high_2
        high_f_3 = rgb_high_1 * depth_high_1
        high_f_4 = rgb_high_2 * depth_high_2
        high_f_W = torch.cat((high_f_1,high_f_3),dim=1)
        high_f_H = torch.cat((high_f_2,high_f_4),dim=1)

        high_f_W = self.fusion_high(high_f_W)
        high_f_H = self.fusion_high(high_f_H)
        high_f_w = self.weight(high_f_W)
        high_f_h = self.weight(high_f_H)

        # high_f_ww = self.weight(high_f_3)
        # high_f_hh = self.weight(high_f_4)
        high_rgb = torch.matmul((1-high_f_w)*rgb_high_1, (1-high_f_h)*rgb_high_2) + rgb_high
        high_depth = torch.matmul((high_f_w)*depth_high_1, (high_f_h)*depth_high_2) + depth_high
        high_f = high_rgb + high_depth

        # high_f = torch.cat((high_f_1, high_f_2, high_f_3, high_f_4),dim=1)
        # high_f = self.fusion_high(high_f)

        # high_f = torch.cat((rgb_high,depth_high), dim=1)

        low_f = torch.cat((rgb_low, depth_low), dim=1)
        low_f = self.fusion_low(low_f)
        low_f = self.hasp(low_f)
        # high_f = self.fusion_high(high_f)
        # low_f = self.fusion_low(low_f)
        # low = [x_in[:, :3, :, :], x_in[:, 3:6, :, :]]
        # out = self.block(low)
        # rgb_low = out[0]
        # depth_low = out[1]
        # low_f = torch.cat((rgb_low,depth_low),dim=1)
        pos_output, cos_output, sin_output, width_output = self.decoder(high_f,low_f)
        pos_output = F.interpolate(pos_output, size=x_in.size()[2:], mode='bilinear', align_corners=True)  # 上采样（双线性插值）
        cos_output = F.interpolate(cos_output, size=x_in.size()[2:], mode='bilinear', align_corners=True)  # 上采样（双线性插值）
        sin_output = F.interpolate(sin_output, size=x_in.size()[2:], mode='bilinear', align_corners=True)  # 上采样（双线性插值）
        width_output = F.interpolate(width_output, size=x_in.size()[2:], mode='bilinear', align_corners=True)  # 上采
# class DenseAttenGraspNet(nn.Module):
#
#     def __init__(self, input_channels=1, dropout=False, prob=0.0, bottleneck=True):
#         super(DenseAttenGraspNet, self).__init__()
#
#         if bottleneck == True:
#             block = BottleneckBlock
#         else:
#             block = BasicBlock
#
#         self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=9, stride=3, padding=3)
#         self.bn1 = nn.BatchNorm2d(32)
#
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
#         self.bn2 = nn.BatchNorm2d(64)
#
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
#
#         nb_layers = 16  # 16     layers can be much more   now the best is 16
#         input_channels = 128
#         growth_rate = 24  # 12  now the best is 24
#         self.block = DenseBlock(block, nb_layers=nb_layers, input_channels=input_channels, growth_rate=growth_rate, dropRate=prob)
#
#         self.change_channel = nn.Conv2d(input_channels + nb_layers * growth_rate, 128, kernel_size=1)
#
#         self.channel_attention1 = ChannelAttention(in_planes=128)
#         self.spatial_attention1 = SpatialAttention()
#
#         # self.gam_attention1 = GAM_Attention(128, 128)
#
#         self.attention1 = Self_Attn(128)
#
#         self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.bn4 = nn.BatchNorm2d(64)
#         self.attention2 = Self_Attn(64)
#
#         self.channel_attention2 = ChannelAttention(in_planes=64)
#         self.spatial_attention2 = SpatialAttention()
#
#         # self.gam_attention2 = GAM_Attention(64, 64)
#
#
#         self.conv5 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1)
#         self.bn5 = nn.BatchNorm2d(32)
#         self.attention3 = Self_Attn(32)
#
#         self.channel_attention3 = ChannelAttention(in_planes=32)
#         self.spatial_attention3 = SpatialAttention()
#
#         # self.gam_attention3 = GAM_Attention(32, 32)
#
#
#         self.conv6 = nn.ConvTranspose2d(32, 32, kernel_size=9, stride=3, padding=3, output_padding=1)
#
#         self.pos_output = nn.Conv2d(32, 1, kernel_size=2)
#         self.cos_output = nn.Conv2d(32, 1, kernel_size=2)
#         self.sin_output = nn.Conv2d(32, 1, kernel_size=2)
#         self.width_output = nn.Conv2d(32, 1, kernel_size=2)
#
#         self.dropout1 = nn.Dropout(p=prob)
#
#         for m in self.modules():
#             if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
#                 nn.init.xavier_uniform_(m.weight, gain=1)
#
#     def forward(self, x_in):
#         x = F.relu(self.bn1(self.conv1(x_in)))
#         query_x3 = x
#         #100 * 100
#         x = F.relu(self.bn2(self.conv2(x)))
#         query_x2 = x
#         #50 * 50
#         x = F.relu(self.bn3(self.conv3(x)))
#         query_x1 = x
#         #25 * 25
#         x = F.relu(self.block(x))
#         x = F.relu(self.change_channel(x))
#         # attention_x1 = self.attention1(x)
#         channel_x = F.relu(self.channel_attention1(x))
#         spatial_x = F.relu(self.spatial_attention1(query_x1))
#         x = torch.add(channel_x, spatial_x)
#         # x = F.relu(self.gam_attention1(x))
#         #25 * 25
#         # x = F.relu(self.bn4(self.conv4(attention_x1)))
#         x = F.relu(self.bn4(self.conv4(x)))
#         channel_x = F.relu(self.channel_attention2(x))
#         spatial_x = F.relu(self.spatial_attention2(query_x2))
#         x = torch.add(channel_x, spatial_x)
#         # x = F.relu(self.gam_attention2(x))
#
#         # attention_x2 = F.relu(self.attention2(x))
#         # attention_x2 = self.attention2(x)
#         #50 * 50
#         x = F.relu(self.bn5(self.conv5(x)))
#         # attention_x3 = F.relu(self.attention3(x, query_x3))
#         # attention_x3 = self.attention3(x)
#         channel_x = F.relu(self.channel_attention3(x))
#         spatial_x = F.relu(self.spatial_attention3(query_x3))
#         x = torch.add(channel_x, spatial_x)
#         # x = F.relu(self.gam_attention3(x))
#
#         #100 * 100
#         x = self.conv6(x)
#
#         pos_output = self.pos_output(self.dropout1(x))
#         cos_output = self.cos_output(self.dropout1(x))
#         sin_output = self.sin_output(self.dropout1(x))
#         width_output = self.width_output(self.dropout1(x))

        return pos_output, cos_output, sin_output, width_output

    def compute_loss(self, xc, yc):
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)

        p_loss = F.mse_loss(pos_pred, y_pos)
        cos_loss = F.mse_loss(cos_pred, y_cos)
        sin_loss = F.mse_loss(sin_pred, y_sin)
        width_loss = F.mse_loss(width_pred, y_width)
        # p_loss = F.smooth_l1_loss(pos_pred, y_pos)
        # cos_loss = F.smooth_l1_loss(cos_pred, y_cos)
        # sin_loss = F.smooth_l1_loss(sin_pred, y_sin)
        # width_loss = F.smooth_l1_loss(width_pred, y_width)

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

class BasicBlock(nn.Module):
    def __init__(self, input_channels, output_channels, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)

class BottleneckBlock(nn.Module):
    def __init__(self, input_channels, output_channels, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = output_channels * 4
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channels, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, output_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    def __init__(self, block, nb_layers=8, input_channels=128, growth_rate=16, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, input_channels, growth_rate, nb_layers, dropRate)
    def _make_layer(self, block, input_channels, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(input_channels+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.channel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x_input):
        """
            inputs :
                x_input : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x_input.size()
        proj_query = self.query_conv(x_input).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x_input).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        # attention = self.sigmoid(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x_input).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        # out = out + x_input
        # out = self.gamma * out + x_input
        out = self.gamma * out
        # return out, attention
        return out



class Attention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, key_in_dim, query_in_dim):
        super(Attention, self).__init__()
        self.key_channel_in = key_in_dim
        self.query_channel_in = query_in_dim

        self.query_conv = nn.Conv2d(in_channels=query_in_dim, out_channels=query_in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=key_in_dim, out_channels=key_in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=key_in_dim, out_channels=key_in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x_input, x_query):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x_input.size()
        proj_query = self.query_conv(x_query).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x_input).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x_input).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x_input
        # return out, attention
        return out

class SpatialAttention(nn.Module):

    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        temp_x = torch.cat([avg_out, max_out], dim=1)
        temp_x = self.conv1(temp_x)
        attention = self.sigmoid(temp_x)
        x = attention * x
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        attention = self.sigmoid(out)
        out = attention * x
        return out

class GAM_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, rate=4):
        super(GAM_Attention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)

        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att

        return out