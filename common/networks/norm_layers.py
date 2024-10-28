import torch

from common.definitions import NormLayer


def get_2d_layer(norm_layer, num_features):
    if norm_layer == NormLayer.NONE:
        return torch.nn.Identity()
    elif norm_layer == NormLayer.BATCH_NORM:
        return torch.nn.BatchNorm2d(num_features, affine=True)  # num_features
    elif norm_layer == NormLayer.INSTANCE_NORM:
        return torch.nn.InstanceNorm2d(num_features, affine=True)
    elif norm_layer == NormLayer.INSTANCE_NORM_F:
        return torch.nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)

    return None


def get_1d_layer(norm_layer, num_features):
    if norm_layer == NormLayer.NONE:
        return torch.nn.Identity()
    elif norm_layer == NormLayer.BATCH_NORM:
        return torch.nn.BatchNorm1d(num_features, affine=True)  # num_features
    elif norm_layer == NormLayer.INSTANCE_NORM:
        return torch.nn.InstanceNorm1d(num_features, affine=True)
    elif norm_layer == NormLayer.INSTANCE_NORM_F:
        return torch.nn.InstanceNorm1d(num_features, affine=False, track_running_stats=False)

    return None
