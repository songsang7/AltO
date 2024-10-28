import torch

from common.definitions import OptimizerName


def get_optimizer(model_params, optimizer_name: str, init_lr: float, l2_regular):
    if OptimizerName.SGD == optimizer_name:
        return torch.optim.SGD(params=model_params, lr=init_lr, nesterov=False, weight_decay=l2_regular)
    elif OptimizerName.NAG == optimizer_name:
        return torch.optim.SGD(params=model_params, lr=init_lr, nesterov=True, momentum=0.9, weight_decay=l2_regular)
    elif OptimizerName.ADAM == optimizer_name:
        return torch.optim.Adam(params=model_params, lr=init_lr, weight_decay=l2_regular)
    elif OptimizerName.RADAM == optimizer_name:
        return torch.optim.RAdam(params=model_params, lr=init_lr, weight_decay=l2_regular)
    elif OptimizerName.ADAMW == optimizer_name:
        return torch.optim.AdamW(params=model_params, lr=init_lr, weight_decay=l2_regular)

    return None
