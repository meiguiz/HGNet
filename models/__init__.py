def get_network(network_name):
    network_name = network_name.lower()
    if network_name == 'dense-grasp':
        from .dense_grasp import DenseGraspNet
        return DenseGraspNet
    elif network_name == 'hgnet-grasp':
        from .dense_attention import HGNet
        return HGNet
    elif network_name == 'ggcnn':
        from .ggcnn import GGCNN
        return GGCNN
    elif network_name == 'grconvnet':
        from .grconvnet import GenerativeResnet
        return GenerativeResnet
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
