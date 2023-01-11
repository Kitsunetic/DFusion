import numpy as np
from torch.optim.lr_scheduler import LambdaLR


class WarmupScheduler(LambdaLR):
    def __init__(self, optimizer, warmup):
        self.warmup = warmup
        super().__init__(optimizer, self.warmup_lr)

    def warmup_lr(self, step):
        return min(step, self.warmup) / self.warmup
