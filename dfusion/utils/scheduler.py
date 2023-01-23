import numpy as np
from torch.optim.lr_scheduler import LambdaLR


class WarmupScheduler(LambdaLR):
    def __init__(self, optimizer, warmup):
        self.warmup = warmup
        super().__init__(optimizer, self.warmup_lr)

    def warmup_lr(self, step):
        return min(step, self.warmup) / self.warmup


def LinearWarmup(optimizer, warmup_steps: int, total_steps: int, f_min: float):
    def fn(ep):
        if ep < warmup_steps:
            u = f_min + (1 - f_min) * ep / warmup_steps
        else:
            u = f_min + (1 - f_min) * (1 - ep / total_steps)
        return min(1, max(f_min, u))

    return LambdaLR(optimizer, lr_lambda=fn)
