import math
from copy import deepcopy
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch as th
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid, save_image
from tqdm import tqdm, trange

from dfusion import DDIMSampler, DDPMSampler, DDPMTrainer, make_beta_schedule
from dfusion.models.kitsunetic import UNet
from dfusion.utils.common import infinite_dataloader
from dfusion.utils.scheduler import WarmupScheduler
from dfusion.utils.score.both import get_inception_and_fid_score

th.set_grad_enabled(False)

BATCH_SIZE = 256
N_SAMPLES = 50000


@th.no_grad()
def sample(model: nn.Module, sampler: nn.Module, batch_size: int) -> Tensor:
    samples = sampler(model, (batch_size, 3, 32, 32))
    samples = samples / 2 + 0.5  # [-1, 1] -> [0, 1]
    return samples


def main():
    result_dir = Path("results/cifar10_ddpm_unconditional2")
    output_dir = result_dir / "output" / "ddim" / "step130k_s20_no_clip_denoised"
    output_dir.mkdir(parents=True, exist_ok=True)

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
    )
    model = model.cuda().eval()
    ckpt = th.load(result_dir / "best.pth", map_location="cpu")
    model.load_state_dict(ckpt)

    betas = make_beta_schedule("linear", 1000)
    # sampler = DDPMSampler(betas, model_mean_type="eps", model_var_type="fixed_large", clip_denoised=True).cuda()
    sampler = DDIMSampler(
        betas,
        ddim_s=20,
        ddim_eta=0.0,
        model_mean_type="eps",
        model_var_type="fixed_large",
        clip_denoised=False,
    ).cuda()

    # check already generated samples
    output_files = []
    for i in range(N_SAMPLES):
        f = output_dir / f"{i:05d}.png"
        if not f.exists():
            output_files.append(f)

    # ignore already generated samples
    with tqdm(total=len(output_files), ncols=100) as pbar:
        for i in range(0, pbar.total, BATCH_SIZE):
            b = min(pbar.total - i, BATCH_SIZE)

            samples = sample(model, sampler, b)
            samples = samples.mul_(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", th.uint8).numpy()
            for f, x in zip(output_files[i : i + b], samples):
                imageio.imwrite(f, x)

            pbar.update(b)

    # calculate FID
    images_total = [output_dir / f"{i:05d}.png" for i in range(N_SAMPLES)]
    images_total = th.stack([th.from_numpy(imageio.imread(f)) for f in images_total])
    images_total = images_total.permute(0, 3, 1, 2).float().div_(255)  # [0, 1]
    assert len(images_total) == N_SAMPLES

    (IS, IS_std), FID = get_inception_and_fid_score(
        images_total,
        fid_cache="results/stats/cifar10.train.npz",
        use_torch=False,
        verbose=True,
    )
    print("IS:", IS)
    print("IS_std:", IS_std)
    print("FID:", FID)


if __name__ == "__main__":
    main()
