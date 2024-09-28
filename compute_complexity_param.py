from models.dense_attention import DenseAttenGraspNet
from ptflops import get_model_complexity_info
from models.grconvnet import GRConvNet


# # model = DenseAttenGraspNet(input_channels=1)
#
# flops, params = get_model_complexity_info(model, (1, 300, 300), as_strings=True)
# print("FLOPS", flops)
# print("params", params)