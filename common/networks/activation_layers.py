import torch

from common.definitions import ActFunc


def get_layer(act_func):
    if act_func == ActFunc.IDENTITY:
        return torch.nn.Identity()
    elif act_func == ActFunc.RELU:
        return torch.nn.ReLU()
    elif act_func == ActFunc.LEAKY_RELU:
        return torch.nn.LeakyReLU(negative_slope=0.3)
    elif act_func == ActFunc.SIGMOID:
        return torch.nn.Sigmoid()
    elif act_func == ActFunc.TANH:
        return torch.nn.Tanh()
    elif act_func == ActFunc.MISH:
        return torch.nn.Mish()

    return None
