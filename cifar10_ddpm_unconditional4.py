import argparse
import math
import os
import random
from copy import deepcopy
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch as th
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch import Tensor
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from dfusion import DDIMSampler, DDPMSampler, DDPMTrainer, make_beta_schedule
from dfusion.dfusion.diffusion2 import GaussianDiffusionSampler, GaussianDiffusionTrainer
from dfusion.models.kitsunetic import UNet
from dfusion.models.kitsunetic.unet2 import UNet as UNet2
from dfusion.utils.common import infinite_dataloader
from dfusion.utils.ema import ema
from dfusion.utils.scheduler import WarmupScheduler
from dfusion.utils.score.both import get_inception_and_fid_score


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--gpus", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_samples", type=int, default=256)
    parser.add_argument("--n_samples_eval", type=int, default=50000)
    parser.add_argument("--n_steps", type=int, default=400000)
    parser.add_argument("--samples_per_steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--warmup", type=int, default=5000)
    args = parser.parse_args()
    return args


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


class AverageMeter(object):
    def __init__(self):
        self.sum = 0
        self.cnt = 0
        self.avg = 0

    def update(self, val, n=1):
        if n > 0:
            self.sum += val * n
            self.cnt += n
            self.avg = self.sum / self.cnt

    def get(self):
        return self.avg

    def __call__(self):
        return self.avg


def find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def train(args, model: nn.Module, model_ema: nn.Module):
    optim = Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = WarmupScheduler(optim, args.warmup)

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
    # trainer = GaussianDiffusionTrainer(model, 1e-4, 2e-2, 1000).cuda()
    # sampler = GaussianDiffusionSampler(model, 1e-4, 2e-2, 1000).cuda()

    if args.rankzero:
        ds_train = CIFAR10("data/cifar10", train=True, download=True)
    ds_train = CIFAR10("data/cifar10", train=True, transform=Compose([RandomHorizontalFlip(), ToTensor()]))
    dl_kwargs = dict(batch_size=args.batch_size, num_workers=2, pin_memory=True, persistent_workers=True, drop_last=True)
    dl_train = infinite_dataloader(DataLoader(ds_train, shuffle=True, **dl_kwargs), n_steps=args.n_steps)

    o = AverageMeter()
    with tqdm(total=args.n_steps, ncols=100, disable=not args.rankzero) as pbar:
        for step, (im, label) in enumerate(dl_train, 1):
            im: Tensor = im.cuda(non_blocking=True) * 2 - 1  # [0, 1] -> [-1, 1]
            # label: Tensor = label.cuda(non_blocking=True)

            optim.zero_grad()
            losses = trainer(model, im)
            loss = losses["loss"].mean()
            # loss = trainer(im).mean()
            loss.backward()
            nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

            if args.ddp:
                ema(model.module, model_ema, 0.9999)
            else:
                ema(model, model_ema, 0.9999)

            o.update(loss.item(), n=im.size(0))
            pbar.set_postfix_str(f"loss: {o():.4f}", refresh=False)
            pbar.update()

            if step % args.samples_per_steps == 0:
                model.eval()
                with th.no_grad():
                    # save sample
                    if args.ddp:
                        n = args.n_samples
                        m = math.ceil(n / dist.get_world_size())
                        samples = sampler(model.module, (m, 3, 32, 32)) / 2 + 0.5  # [-1, 1] -> [0, 1]
                        # samples = sampler((m, 3, 32, 32)) / 2 + 0.5
                        samples_lst = [th.empty_like(samples) for _ in range(args.world_size)]
                        dist.all_gather(samples_lst, samples)
                        samples = th.cat(samples_lst)[:n]
                    else:
                        samples = sampler(model, (args.n_samples, 3, 32, 32)) / 2 + 0.5  # [-1, 1] -> [0, 1]

                    if args.rankzero:
                        save_image(samples, args.sample_dir / f"{step:06d}.png", nrow=int(math.sqrt(args.n_samples)))

                        # save checkpoint
                        state_dict = {
                            "model": model.module.state_dict() if args.ddp else model.state_dict(),
                            "model_ema": model_ema.state_dict(),
                        }
                        th.save(state_dict, args.result_dir / "best.pth")

                model.train()

    if args.ddp:
        dist.barrier()


@th.no_grad()
def eval(args, model: nn.Module, model_ema: nn.Module):
    ckpt = th.load(args.result_dir / "best.pth", map_location="cpu")
    if args.ddp:
        model.module.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt["model"])
    model_ema.load_state_dict(ckpt["model_ema"])

    # ckpt = th.load("results/DDPM_CIFAR10_EPS/ckpt.pt", map_location="cpu")
    # model.module.load_state_dict(ckpt["net_model"])

    model.eval()
    model_ema.eval()

    if args.ema:
        model = model_ema
    else:
        model = model.module

    betas = make_beta_schedule("linear", 1000)
    sampler = DDPMSampler(betas, model_mean_type="eps", model_var_type="fixed_large", clip_denoised=True).cuda()
    # sampler = DDIMSampler(
    #     betas,
    #     ddim_s=20,
    #     ddim_eta=0.0,
    #     model_mean_type="eps",
    #     model_var_type="fixed_large",
    #     clip_denoised=True,
    # ).cuda()
    # sampler = GaussianDiffusionSampler(model, 1e-4, 2e-2, 1000).cuda()

    # generate images
    n = args.n_samples_eval
    m = math.ceil(n / args.world_size)
    batch_size = args.batch_size

    ims = []
    with tqdm(total=n, ncols=100, disable=not args.rankzero) as pbar:
        for i in range(0, m, batch_size):
            b = min(m - i, batch_size)
            x: Tensor = sampler(model, (b, 3, 32, 32))
            # x: Tensor = sampler((b, 3, 32, 32))
            x = x.div_(2).add_(0.5).clamp_(0, 1)  # [-1, 1] -> [0, 1]

            if args.ddp:
                xs = [th.empty_like(x) for _ in range(args.world_size)]
                dist.all_gather(xs, x)
                x = th.cat(xs)
            if args.rankzero:
                ims.append(x.cpu())

            pbar.update(min(pbar.total - pbar.n, b * args.world_size))

    # calculate FID
    if args.rankzero:
        ims = th.cat(ims)
        (IS, IS_std), FID = get_inception_and_fid_score(
            ims,
            fid_cache="results/stats/cifar10.train.npz",
            use_torch=False,
            verbose=True,
        )
        print("IS:", IS)
        print("IS_std:", IS_std)
        print("FID:", FID)

    if args.ddp:
        dist.barrier()


def main_worker(rank: int, args: argparse.Namespace):
    if args.ddp:
        dist.init_process_group(backend="nccl", init_method=args.dist_url, world_size=args.world_size, rank=rank)

    args.rank = rank
    args.rankzero = rank == 0
    args.gpu = args.gpus[rank]
    th.cuda.set_device(args.gpu)
    seed_everything(args.rank)

    if args.ddp:
        print(f"main_worker with rank:{rank} (gpu:{args.gpu}) is loaded", th.__version__)
    else:
        print(f"main_worker with gpu:{args.gpu} in main thread is loaded", th.__version__)

    args.result_dir = Path(args.result_dir)
    args.sample_dir = args.result_dir / "samples"
    args.output_dir = args.result_dir / "outputs"
    if args.rankzero:
        args.sample_dir.mkdir(parents=True, exist_ok=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)

    # model = UNet(
    #     dims=2,
    #     in_channels=3,
    #     model_channels=128,
    #     out_channels=3,
    #     num_res_blocks=2,
    #     attention_resolutions=[2],
    #     dropout=0.1,
    #     channel_mult=[1, 2, 2, 2],
    #     num_groups=32,
    #     num_heads=8,
    #     use_scale_shift_norm=False,
    # ).cuda()
    model = UNet2(1000, 128, [1, 2, 2, 2], [1], 2, 0.1).cuda()
    model_ema: nn.Module = deepcopy(model)
    if args.ddp:
        model = DDP(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_ema.load_state_dict(model.module.state_dict())
    else:
        model_ema.load_state_dict(model.state_dict())
    model_ema.eval().requires_grad_(False)

    if not args.eval:
        train(args, model, model_ema)
    else:
        eval(args, model, model_ema)


def main():
    args = get_args()

    args.gpus = list(map(int, args.gpus.split(",")))
    args.world_size = len(args.gpus)
    args.ddp = args.world_size > 1

    if args.ddp:
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}"
        pc = mp.spawn(main_worker, nprocs=args.world_size, args=(args,), join=False)
        pids = " ".join(map(str, pc.pids()))
        print("\33[101mProcess Ids:", pids, "\33[0m")
        try:
            pc.join()
        except KeyboardInterrupt:
            print("\33[101mkill %s\33[0m" % pids)
            os.system("kill %s" % pids)
    else:
        main_worker(0, args)


if __name__ == "__main__":
    main()
