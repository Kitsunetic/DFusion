import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from dfusion import DDIMSampler, DDPMSampler, DDPMTrainer, make_beta_schedule
from dfusion.models.kitsunetic import UNet
from dfusion.utils.common import infinite_dataloader
from dfusion.utils.scheduler import WarmupScheduler

N_STEPS = 400000
SAMPLE_PER_STEPS = 10000
N_SAMPLES = 256


def sample(model: nn.Module, sampler: nn.Module) -> Tensor:
    samples = sampler(model, (N_SAMPLES, 3, 32, 32))
    samples = samples / 2 + 0.5  # [-1, 1] -> [0, 1]
    return samples


def main():
    result_dir = Path("results/cifar10_ddpm_unconditional2")
    sample_dir = result_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    model = UNet(
        dims=2,
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=[2],
        dropout=0.1,
        channel_mult=[1, 2, 2, 2],
        num_groups=32,
        num_heads=1,
    ).cuda()
    optim = Adam(model.parameters(), lr=0.0002, weight_decay=0.0)
    sched = WarmupScheduler(optim, 5000)

    betas = make_beta_schedule("linear", 1000)
    trainer = DDPMTrainer(betas, loss_type="l2", model_mean_type="eps", model_var_type="fixed_large").cuda()
    # sampler = DDPMSampler(betas, model_mean_type="eps", model_var_type="fixed_large", clip_denoised=True).cuda()
    sampler = DDIMSampler(
        betas,
        ddim_s=20,
        ddim_eta=0.0,
        model_mean_type="eps",
        model_var_type="fixed_large",
        clip_denoised=True,
    ).cuda()

    ds_train = CIFAR10("data/cifar10", train=True, transform=ToTensor(), download=True)
    dl_kwargs = dict(batch_size=256, num_workers=2, pin_memory=True, persistent_workers=True, drop_last=True)
    dl_train = infinite_dataloader(DataLoader(ds_train, shuffle=True, **dl_kwargs), n_steps=N_STEPS)

    for step, (im, label) in tqdm(enumerate(dl_train, 1), total=N_STEPS, ncols=100):
        im: Tensor = im.cuda(non_blocking=True) * 2 - 1  # [0, 1] -> [-1, 1]
        # label: Tensor = label.cuda(non_blocking=True)

        optim.zero_grad()
        losses = trainer(model, im)
        losses["loss"].mean().backward()
        optim.step()
        sched.step()

        if step % SAMPLE_PER_STEPS == 0:
            model.eval()
            with th.no_grad():
                # save sample
                samples = sample(model, sampler)
                save_image(samples, sample_dir / f"{step:06d}.png", nrow=int(math.sqrt(N_SAMPLES)))

                # save checkpoint
                th.save(model.state_dict(), result_dir / "best.pth")
            model.train()


if __name__ == "__main__":
    main()
