import math
from functools import partial
from inspect import isfunction
from typing import Callable, Sequence, Union

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from einops import reduce
from torch import Size, Tensor, nn


def unsqueeze_as(x, y) -> th.Tensor:
    if isinstance(y, th.Tensor):
        d = y.dim()
    else:
        d = len(y)
    return x.view(list(x.shape) + [1] * (d - x.dim()))


def identity(*args):
    if len(args) == 0:
        return None
    elif len(args) == 1:
        return args[0]
    else:
        return args


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "quad":
        betas = np.linspace(linear_start**0.5, linear_end**0.5, n_timestep, dtype=np.float64) ** 2
    elif schedule == "linear":
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    elif schedule == "warmup10":
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.1)
    elif schedule == "warmup50":
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.5)
    elif schedule == "const":
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(n_timestep, 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        x = np.linspace(0, n_timestep, n_timestep + 1, dtype=np.float64)
        alphas_cumprod = np.cos(((x / n_timestep) + cosine_s) / (1 + cosine_s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, 0, 0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# def make_beta_schedule(schedule, n_timestep):
#     if schedule == "linear":
#         scale = 1000 / n_timestep
#         # return np.linspace(scale * 0.0001, scale * 0.02, n_timestep, dtype=np.float64)
#         return np.linspace(0.0001, 0.02, n_timestep, dtype=np.float64)
#     elif schedule == "cosine":
#         betas = []
#         alpha_bar = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
#         for i in range(n_timestep):
#             t1 = i / n_timestep
#             t2 = (i + 1) / n_timestep
#             betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
#         return np.array(betas, dtype=np.float64)
#     else:
#         raise NotImplementedError(schedule)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor) for x in (logvar1, logvar2)]

    return 0.5 * (-1.0 + logvar2 - logvar1 + th.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * th.exp(-logvar2))


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.
    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


class DiagonalGaussianDistribution:
    def __init__(self, parameters: Tensor, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = th.chunk(parameters, 2, dim=1)
        self.logvar = th.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = th.exp(0.5 * self.logvar)
        self.var = th.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = th.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * th.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return self.parameters.new_zeros(1)
        else:
            if other is None:
                return 0.5 * (th.pow(self.mean, 2) + self.var - 1.0 - self.logvar).mean()
            else:
                x = th.pow(self.mean - other.mean, 2) / other.var + self.var / other.var - 1.0 - self.logvar + other.logvar
                return 0.5 * x.mean()

    def nll(self, sample: Tensor):
        if self.deterministic:
            return self.parameters.new_zeros(1)
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * (logtwopi + self.logvar + th.pow(sample - self.mean, 2) / self.var).mean()

    def mode(self):
        return self.mean
