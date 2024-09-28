import torch
from torch import nn
from typing import Dict
from collections import OrderedDict
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torch.nn import functional as F
from models import resnet
import numpy as np
import cv2
from einops import rearrange




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

class IntermediateLayerGetter(nn.ModuleDict):
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():



            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out
# def show_feature_map(
#         feature_map):  # feature_map=torch.Size([1, 64, 55, 55]),feature_map[0].shape=torch.Size([64, 55, 55])
#     # feature_map[2].shape     out of bounds
#     feature_map = feature_map.detach().cpu().numpy().squeeze(0)  # 压缩成torch.Size([64, 55, 55])
#     feature_map_num = feature_map.shape[0]
#
#     for index in range(feature_map_num):  # 通过遍历的方式，将64个通道的tensor拿出
#
#         feature = feature_map[index]
#         feature = np.asarray(feature * 255, dtype=np.uint8)
#         feature = cv2.resize(feature, (480, 480), interpolation=cv2.INTER_NEAREST)
#
#         feature = cv2.applyColorMap(feature, cv2.COLORMAP_HSV)  # 变成伪彩图
#         cv2.imwrite('/home/meiguiz/下载/AFFGA-Net-main (1)/demo/feature_map/channel_{}.png'.format(str(index)), feature)
class BackboneWithFPN(nn.Module):

    def __init__(self, backbones,return_layers, in_channels_list, out_channels, extra_blocks=None):
        super(BackboneWithFPN, self).__init__()



        self.body_1 = IntermediateLayerGetter(backbones["rgb"],return_layers=return_layers)
        self.body_2 = IntermediateLayerGetter(backbones["depth"], return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )
        self.confidence_map_estimator_1 = nn.Sequential(
            nn.Conv2d(4, 1, (1, 3), 1, 1),
            nn.Conv2d(1, 1, (3, 1), 1, 0),
            nn.Conv2d(1, 1, 1, 1, 0)
        )
        self.confidence_map_estimator_2 = nn.Sequential(
            nn.Conv2d(4, 1, (1, 5), 1, 2),
            nn.Conv2d(1, 1, (5, 1), 1, 0),
            nn.Conv2d(1, 1, 1, 1, 0),
            #nn.Conv2d(1, 1, 1, 1, 0),
            #nn.Conv2d(1, 1, 1, 1, 0)
        )
        self.confidence_map_estimator_3 = nn.Sequential(
    nn.Conv2d(512, 256, 1, 1, 0),

            #nn.Conv2d(1, 1, 1, 1, 0),
            #nn.Conv2d(1, 1, 1, 1, 0)
        )
        self.out_channels = out_channels
        self.se = SEModel(channel=512, reduction=8)
        self.fea = nn.Conv2d(6, 3, 1, 1, 0)

    def forward(self, x):
        rgb_x = x[:, :3, :, :]
        depth_x = x[:, 3:6, :, :]

        # feature_map_num = depth_x.shape[0]
        # for index in range(feature_map_num):  # 通过遍历的方式，将64个通道的tensor拿出
        #
        #     feature = depth_x[index]
        #     feature = feature.detach().cpu().numpy()
        #     feature = np.asarray(feature*255, dtype=np.uint8)
        #     #feature = cv2.resize(feature, (320, 320), interpolation=cv2.INTER_NEAREST)
        #
        #     feature = cv2.applyColorMap(feature, cv2.COLORMAP_HOT)  # 变成伪彩图
        #     cv2.imwrite('/media/manman/HIKVISION/a/channel_{}.png'.format(str(index)), feature)

        confidence_map_1 = self.confidence_map_estimator_1(x[:, :4, :, :])
        confidence_map_2 = self.confidence_map_estimator_2(x[:, :4, :, :])
        #confidence_map = torch.cat((confidence_map_1, confidence_map_2), dim=1)
        #confidence_map_3 = self.confidence_map_estimator_3(confidence_map)
        _, _, H, W = x.shape
        confidence_maps_1 = F.interpolate(confidence_map_1,
                                                size=(int(H / (2 ** 2)), int(W / (2 ** 2))),
                                                mode='bilinear', align_corners=True)
        confidence_maps_2 = F.interpolate(confidence_map_2,
                                        size=(int(H / (2 ** 2)), int(W / (2 ** 2))),
                                        mode='bilinear', align_corners=True)
        # depth_feature_1 = depth_x * confidence_map_1
        # depth_feature_2 = depth_x * confidence_map_2
        # depth_feature = torch.cat((depth_feature_1,depth_feature_2),dim=1)
        # feature = torch.cat((rgb_x, depth_x), dim=1)
        # depth_feature = self.fea(depth_feature)
        # show_feature_map(depth_x)
        #show_feature_map(depth_feature_2)
        rgb_x = self.body_1(rgb_x)
        rgb_x = self.fpn(rgb_x)
        depth_x = self.body_2(depth_x)

        feat_1 = rgb_x[str(0)]
        rgb_x = rgb_x[str(2)]
        feat_2 = depth_x[str(0)]


        feat_21 = feat_2 * confidence_maps_1
        feat_22 = feat_2 *confidence_maps_2
        feat_2 = torch.cat((feat_21,feat_22),dim=1)

        feat_2 = self.confidence_map_estimator_3(feat_2)
        feat_1 = torch.cat((feat_1,feat_2),dim=1)
        feat_1 = self.se(feat_1)

        return feat_1, rgb_x






def get_backbone_with_fpn( backbone_name, trainable_layers, pretrained_backbone=True,extra_blocks=None, returned_layers=None):
    backbones = dict.fromkeys(["rgb","depth"])

    backbones["rgb"] = resnet.__dict__[backbone_name](pretrained=True)
    backbones["depth"] = resnet.__dict__[backbone_name](pretrained=True)
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    # freeze layers single if pretrained backbone is used
    for k in backbones:
        if backbones[k] is None:
            continue
        for name, parameter in backbones[k].named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)
        in_channels_stage2 = backbones[k].inplanes // 8
    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()
    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    assert min(returned_layers) > 0 and max(returned_layers) < 5
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256

    return BackboneWithFPN(backbones, return_layers, in_channels_list, out_channels,
                           extra_blocks=extra_blocks)


if __name__ == "__main__":
    import torch
    model = get_backbone_with_fpn(backbone_name='resnet50',
                        trainable_layers=3, pretrained_backbone=True,extra_blocks=None, returned_layers=None)
    input = torch.rand(4, 6, 512, 384)
    output,xs= model(input)
    # print(output[str(0)].shape)
    # print(output[str(1)].shape)
    # print(output[str(2)].shape)
    # print(output[str(3)].shape)
    print(output.shape)
    print(xs.shape)
    #print(feat.shape)
