"""
based on https://github.com/openai/point-e/blob/main/point_e/diffusion/k_diffusion.py
"""

from scipy.interpolate import interp1d

from dfusion.dfusion.diffusion_base import *


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def append_zero(x: np.ndarray):
    return np.concatenate([x, np.zeros((1,), dtype=x.dtype)])


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = np.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas)


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


def get_ancestral_step(sigma_from, sigma_to):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


class KarrasSampler(DiffusionBase):
    """
    input:
    - betas:
    - ddim_eta: 0.0 - deterministic, 1.0 - stochastic
    - model_mean_type: (eps | x_start | x_prev)
    - model_var_type: (fixed_small | fixed_large | leraned | learned_range)
    """

    def __init__(
        self,
        betas: np.ndarray,
        n_steps: int,
        sigma_min=0.002,
        sigma_max=80,  # higher for highres?
        rho=7.0,
        sampler="heun",  # heun|dpm|ancestral
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        model_mean_type="eps",
        model_var_type="fixed_small",
        clip_denoised=True,
    ):
        assert sampler in "heun|dpm|ancestral".split("|")
        super().__init__(betas, model_mean_type, model_var_type, clip_denoised)

        self.n_steps = n_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sampler = sampler
        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

        # Karras sampler paramters
        with self.register_diffusion_parameters():
            self.sigmas = get_sigmas_karras(n_steps, sigma_min, sigma_max, rho)

            self.gammas = np.zeros_like(self.sigmas[:-1])
            self.gammas[s_tmin <= self.sigmas[:-1]] = min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1)

            sigmas_alpha_cumprod = 1.0 / (self.sigmas[:-1] ** 2 + 1)
            alpha_cumprod_to_t = interp1d(self.alphas_cumprod, np.arange(0, self.num_timesteps))
            c1 = sigmas_alpha_cumprod > self.alphas_cumprod[0]
            c2 = sigmas_alpha_cumprod <= self.alphas_cumprod[-1]
            c3 = ~(c1 | c2)
            self.karras_t_sigmas = np.zeros_like(sigmas_alpha_cumprod)
            self.karras_t_sigmas[c2] = self.num_timesteps - 1
            self.karras_t_sigmas[c3] = alpha_cumprod_to_t(sigmas_alpha_cumprod[c3])

            self.sigmas_hat = self.sigmas[:-1] * (self.gammas + 1)
            sigmas_hat_alpha_cumprod = 1.0 / (self.sigmas_hat**2 + 1)
            c1 = sigmas_hat_alpha_cumprod > self.alphas_cumprod[0]
            c2 = sigmas_hat_alpha_cumprod <= self.alphas_cumprod[-1]
            c3 = ~(c1 | c2)
            self.karras_t_sigmas_hat = np.zeros_like(sigmas_hat_alpha_cumprod)
            self.karras_t_sigmas_hat[c2] = self.num_timesteps - 1
            self.karras_t_sigmas_hat[c3] = alpha_cumprod_to_t(sigmas_hat_alpha_cumprod[c3])

            self.sigmas_mid = ((self.sigmas_hat ** (1 / 3) + self.sigmas[1:] ** (1 / 3)) / 2) ** 3
            sigmas_mid_alpha_cumprod = 1.0 / (self.sigmas_mid**2 + 1)
            c1 = sigmas_mid_alpha_cumprod > self.alphas_cumprod[0]
            c2 = sigmas_mid_alpha_cumprod <= self.alphas_cumprod[-1]
            c3 = ~(c1 | c2)
            self.karras_t_sigmas_mid = np.zeros_like(sigmas_mid_alpha_cumprod)
            self.karras_t_sigmas_mid[c2] = self.num_timesteps - 1
            self.karras_t_sigmas_mid[c3] = alpha_cumprod_to_t(sigmas_mid_alpha_cumprod[c3])

    def sample_heun(self, denoise_fn: Callable[[Tensor, Tensor], Tensor], shape: Size):
        """
        Implements Algorithm 2 (Heun steps) from Karras et al. (2022).

        ### yield:
        - x: current denoised sample (x_t)
        - x_0_pred: prediction of x_0 of current step
        """
        b = shape[0]
        x = th.randn(shape, device=self.device) * self.sigma_max
        indices = range(len(self.sigmas) - 1)

        for i in indices:
            gamma = self.gammas[i]
            sigma_hat = self.sigmas_hat[i]
            eps = th.randn_like(x) * self.s_noise
            if gamma > 0:
                x = x + eps * (sigma_hat**2 - self.sigmas[i] ** 2) ** 0.5

            t = th.full((b,), self.karras_t_sigmas_hat[i], device=self.device, dtype=th.long)
            c_in = unsqueeze_as(1.0 / (sigma_hat**2 + 1) ** 0.5, x)
            denoised = self.p_mean_variance(denoise_fn, x * c_in, t)["pred_x_start"]
            d = to_d(x, sigma_hat, denoised)
            yield x, denoised

            dt = self.sigmas[i + 1] - sigma_hat
            if self.sigmas[i + 1] == 0:
                # Euler method
                x = x + d * dt
            else:
                # Heun's method
                x_2 = x + d * dt
                t_2 = th.full((b,), self.karras_t_sigmas[i + 1], device=self.device, dtype=th.long)
                c_in2 = unsqueeze_as(1.0 / (self.sigmas[i + 1] ** 2 + 1) ** 0.5, x_2)
                denoised_2 = self.p_mean_variance(denoise_fn, x_2 * c_in2, t_2)["pred_x_start"]
                d_2 = to_d(x_2, self.sigmas[i + 1], denoised_2)
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt
        yield x, denoised

    def sample_dpm(self, denoise_fn: Callable[[Tensor, Tensor], Tensor], shape: Size):
        """
        A sampler inspired by DPM-Solver-2 and Algorithm 2 from Karras et al. (2022).

        ### yield:
        - x: current denoised sample (x_t)
        - x_0_pred: prediction of x_0 of current step
        """
        b = shape[0]
        x = th.randn(shape, device=self.device)
        indices = range(len(self.sigmas) - 1)

        for i in indices:
            gamma = self.gammas[i]
            eps = th.randn_like(x) * self.s_noise
            sigma_hat = self.sigmas_hat[i]
            if gamma > 0:
                x = x + eps * (sigma_hat**2 - self.sigmas[i] ** 2) ** 0.5

            t = th.full((b,), self.karras_t_sigmas_hat[i], device=self.device, dtype=th.long)
            c_in = unsqueeze_as(1.0 / (sigma_hat**2 + 1) ** 0.5, x)
            denoised = self.p_mean_variance(denoise_fn, x * c_in, t)["pred_x_start"]
            d = to_d(x, sigma_hat, denoised)
            yield x, denoised

            # Midpoint method, where the midpoint is chosen according to a rho=3 Karras schedule
            sigma_mid = self.sigmas_mid[i]
            dt_1 = sigma_mid - sigma_hat
            dt_2 = self.sigmas[i + 1] - sigma_hat

            x_2 = x + d * dt_1
            t_2 = th.full((b,), self.karras_t_sigmas_mid[i], device=self.device, dtype=th.long)
            c_in2 = unsqueeze_as(1.0 / (sigma_mid**2 + 1) ** 0.5, x)
            denoised_2 = self.p_mean_variance(denoise_fn, x_2 * c_in2, t_2)["pred_x_start"]
            d_2 = to_d(x_2, sigma_mid, denoised_2)
            x = x + d_2 * dt_2
        yield x, denoised

    def sample_euler_ancestral(self, denoise_fn: Callable[[Tensor, Tensor], Tensor], shape: Size):
        """
        Ancestral sampling with Euler method steps.

        ### yield:
        - x: current denoised sample (x_t)
        - x_0_pred: prediction of x_0 of current step
        """
        b = shape[0]
        x = th.randn(shape, device=self.device)
        indices = range(len(self.sigmas) - 1)

        for i in indices:
            t = th.full((b,), self.karras_t_sigmas[i], device=self.device, dtype=th.long)
            c_in = unsqueeze_as(1.0 / (self.sigmas[i] ** 2 + 1) ** 0.5, x)
            denoised = self.p_mean_variance(denoise_fn, x * c_in, t)["pred_x_start"]

            sigma_down, sigma_up = get_ancestral_step(self.sigmas[i], self.sigmas[i + 1])
            # yield {"x": x, "i": i, "sigma": self.sigmas[i], "sigma_hat": self.sigmas[i], "pred_xstart": denoised}
            yield x, denoised
            d = to_d(x, self.sigmas[i], denoised)
            # Euler method
            dt = sigma_down - self.sigmas[i]
            x = x + d * dt
            x = x + th.randn_like(x) * sigma_up
        # yield {"x": x, "pred_xstart": x}
        yield x, x

    def forward(self, denoise_fn: Callable[[Tensor, Tensor], Tensor], shape: Size, num_samples=1):
        sampler_fn = {
            "heun": self.sample_heun,
            "dpm": self.sample_dpm,
            "ancestral": self.sample_euler_ancestral,
        }[self.sampler]

        # variables for sampling intermediations too
        do_sampling = num_samples > 1
        sample_indices = np.linspace(0, self.n_steps, num_samples, dtype=np.int64).tolist()
        sample_lst = []

        for i, (x_t, pred_xstart) in enumerate(sampler_fn(denoise_fn, shape)):
            if do_sampling and i in sample_indices:
                sample_lst.append(x_t)

        if not do_sampling:
            return pred_xstart
        else:
            sample_lst.append(pred_xstart)
            return th.stack(sample_lst)


def __test__():
    betas = make_beta_schedule("linear", 1000)
    model = lambda x, t: th.cat([x, x], dim=1)  # any model that doubles the channel (only when learnt model variance)
    shape = (2, 3, 4, 5, 6, 7, 8, 9)  # arbitrary shape

    diffusion_sampler = KarrasSampler(betas, sampler="heun", model_var_type="learned_range", n_steps=200)
    out = diffusion_sampler(model, shape, num_samples=5)
    print(out.shape)
    # torch.Size([5, 2, 3, 4, 5, 6, 7, 8, 9])

    diffusion_sampler = KarrasSampler(betas, sampler="dpm", model_var_type="learned_range", n_steps=200)
    out = diffusion_sampler(model, shape, num_samples=5)
    print(out.shape)
    # torch.Size([5, 2, 3, 4, 5, 6, 7, 8, 9])

    diffusion_sampler = KarrasSampler(betas, sampler="ancestral", model_var_type="learned_range", n_steps=200)
    out = diffusion_sampler(model, shape, num_samples=5)
    print(out.shape)
    # torch.Size([5, 2, 3, 4, 5, 6, 7, 8, 9])


if __name__ == "__main__":
    __test__()
