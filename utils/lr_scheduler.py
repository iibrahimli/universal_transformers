"""
Custom LR scheduler with warmup
"""

import torch


class CustomLRScheduler:
    """Implements LR schedule described in [Vaswani et al. 2017]"""

    def __init__(
        self,
        optim: torch.optim.Optimizer,
        d_model: int,
        warmup_steps: int = 5000,
        lr_mul: float = 2.0,
    ):
        self.optim = optim
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.lr_mul = lr_mul
        self.i_step = 0

    def step(self) -> float:
        self.i_step += 1
        lr = (
            self.lr_mul
            * self.d_model**-0.5
            * min(self.i_step**-0.5, self.i_step * self.warmup_steps**-1.5)
        )
        for param_group in self.optim.param_groups:
            param_group["lr"] = lr
        return lr
