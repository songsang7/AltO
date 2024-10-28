import math

import torch
from torch.optim.lr_scheduler import _LRScheduler

from common.definitions import LrParams, LrPolicy


def get_lr_scheduler(optimizer, max_epochs, steps_per_epoch, lr_param: dict):  # -> torch.optim.lr_scheduler._LRScheduler:
    lr_policy = lr_param[LrParams.LR_POLICY]
    result = None
    if lr_policy == LrPolicy.STEP_DECAY:
        epoch_counts = lr_param[LrParams.DECAY_PERIOD]
        step_decay_ratio = lr_param[LrParams.DECAY_RATE]
        result = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=(epoch_counts * steps_per_epoch), gamma=step_decay_ratio)
    elif lr_policy == LrPolicy.COS_ANNEAL:
        result = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=(max_epochs * steps_per_epoch), eta_min=0.0)
    elif lr_policy == LrPolicy.COS_RESTART:
        cos_restart_init_period = lr_param[LrParams.INIT_PERIOD]
        cos_restart_period_multiply = lr_param[LrParams.PERIOD_MULT_FACTOR]
        result = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=cos_restart_init_period, T_mult=cos_restart_period_multiply, eta_min=0.0)
    elif lr_policy == LrPolicy.ONE_CYCLE:
        init_lr = optimizer.defaults.get("lr")
        result = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=init_lr, total_steps=(max_epochs * steps_per_epoch), pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    use_warm_up = lr_param.get("use_warm_up")
    if use_warm_up:
        warm_up_end = lr_param["warm_up_end"] * steps_per_epoch
        result = WarmupLR(result, 0, warm_up_end)

    return result


class WarmupLR(_LRScheduler):
    def __init__(self, scheduler, init_lr=1e-3, num_warmup=1, warmup_strategy='linear'):
        if warmup_strategy not in ['linear', 'cos', 'constant']:
            raise ValueError("Expect warmup_strategy to be one of ['linear', 'cos', 'constant'] but got {}".format(warmup_strategy))
        self._scheduler = scheduler
        self._init_lr = init_lr
        self._num_warmup = num_warmup
        self._step_count = 0
        # Define the strategy to warm up learning rate
        self._warmup_strategy = warmup_strategy
        if warmup_strategy == 'cos':
            self._warmup_func = self._warmup_cos
        elif warmup_strategy == 'linear':
            self._warmup_func = self._warmup_linear
        else:
            self._warmup_func = self._warmup_const
        # save initial learning rate of each param group
        # only useful when each param groups having different learning rate
        self._format_param()

    def __getattr__(self, name):
        return getattr(self._scheduler, name)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        wrapper_state_dict = {key: value for key, value in self.__dict__.items() if (key != 'optimizer' and key != '_scheduler')}
        wrapped_state_dict = {key: value for key, value in self._scheduler.__dict__.items() if key != 'optimizer'}
        return {'wrapped': wrapped_state_dict, 'wrapper': wrapper_state_dict}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict['wrapper'])
        self._scheduler.__dict__.update(state_dict['wrapped'])

    def _format_param(self):
        # learning rate of each param group will increase
        # from the min_lr to initial_lr
        for group in self._scheduler.optimizer.param_groups:
            group['warmup_max_lr'] = group['lr']
            group['warmup_initial_lr'] = min(self._init_lr, group['lr'])

    def _warmup_cos(self, start, end, pct):
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    def _warmup_const(self, start, end, pct):
        return start if pct < 0.9999 else end

    def _warmup_linear(self, start, end, pct):
        return (end - start) * pct + start

    def get_lr(self):
        lrs = []
        step_num = self._step_count
        # warm up learning rate
        if step_num <= self._num_warmup:
            for group in self._scheduler.optimizer.param_groups:
                computed_lr = self._warmup_func(group['warmup_initial_lr'],
                                                group['warmup_max_lr'],
                                                step_num / self._num_warmup)
                lrs.append(computed_lr)
        else:
            lrs = self._scheduler.get_lr()
        return lrs

    def get_last_lr(self):
        return self.get_lr()

    def step(self, *args):
        if self._step_count <= self._num_warmup:
            values = self.get_lr()
            for param_group, lr in zip(self._scheduler.optimizer.param_groups, values):
                param_group['lr'] = lr
            self._step_count += 1
        else:
            self._scheduler.step(*args)