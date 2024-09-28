import torch
from torch import nn
from typing import Dict
from collections import OrderedDict
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torch.nn import functional as F
from models import resnet


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

class BackboneWithFPN(nn.Module):

    def __init__(self, backbones,return_layers, in_channels_list, out_channels, extra_blocks=None):
        super(BackboneWithFPN, self).__init__()



        self.body = IntermediateLayerGetter(backbones["rgb"],return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )
        self.out_channels = out_channels
    def forward(self, x):


        x = self.body(x)
        x = self.fpn(x)
        feat_1 = x[str(0)]
        x = x[str(3)]

        return feat_1, x


def get_backbone_with_fpn( backbone_name, trainable_layers, pretrained_backbone=True,extra_blocks=None, returned_layers=None):
    backbones = dict.fromkeys(["rgb","depth"])

    backbones["rgb"] = resnet.__dict__[backbone_name](pretrained=True)
    #backbones["depth"] = resnet.__dict__[backbone_name](BatchNorm=nn.BatchNorm2d, output_stride=16, device='cuda',pretrained=False)
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
    input = torch.rand(1, 3, 320, 320)
    output,xs= model(input)
    # print(output[str(0)].shape)
    # print(output[str(1)].shape)
    # print(output[str(2)].shape)
    # print(output[str(3)].shape)
    print(output.shape)
    print(xs.shape)
