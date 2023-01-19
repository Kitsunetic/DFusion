from dfusion.dfusion.diffusion_base import *


class DDIMSampler(DiffusionBase):
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
        ddim_s: int,
        ddim_eta: float,
        model_mean_type="eps",
        model_var_type="fixed_small",
        clip_denoised=True,
    ):
        super().__init__(betas, model_mean_type, model_var_type, clip_denoised)

        self.ddim_eta = ddim_eta
        self.ddim_s = ddim_s

        # DDIM parameter
        c = self.num_timesteps // self.ddim_s
        self.ddim_timesteps = ddim_timesteps = np.asarray(list(range(0, self.num_timesteps, c))) + 1

        with self.register_diffusion_parameters():
            # paramers that should be calculated in ddim timestep
            self.ddim_alphas_cumprod = self.alphas_cumprod[ddim_timesteps]
            self.ddim_alphas_cumprod_prev = np.asarray(
                [self.ddim_alphas_cumprod[0]] + self.alphas_cumprod[ddim_timesteps[:-1]].tolist()
            )
            self.ddim_sqrt_alphas_cumprod_prev = np.sqrt(self.ddim_alphas_cumprod_prev)
            self.ddim_sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.ddim_alphas_cumprod)
            self.ddim_sigma = self.ddim_eta * np.sqrt(
                (1 - self.ddim_alphas_cumprod_prev)
                / (1 - self.ddim_alphas_cumprod)
                * (1 - self.ddim_alphas_cumprod / self.ddim_alphas_cumprod_prev)
            )
            self.ddim_dir = np.sqrt(1 - self.ddim_alphas_cumprod_prev - self.ddim_sigma**2)

    def ddim_sample(self, denoise_fn, x_t, t, s):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(denoise_fn, x_t, t)
        eps = default(out.get("pred_eps", None), lambda: self._predict_eps_from_xstart(x_t, t, out["pred_x_start"]))

        # ddim_alphas_cumprod = unsqueeze_as(self.ddim_alphas_cumprod[s], x_t)
        # ddim_alphas_cumprod_prev = unsqueeze_as(self.ddim_alphas_cumprod_prev[s], x_t)
        ddim_sqrt_alphas_cumprod_prev = unsqueeze_as(self.ddim_sqrt_alphas_cumprod_prev[s], x_t)
        # ddim_sqrt_one_minus_alphas_cumprod = unsqueeze_as(self.ddim_sqrt_one_minus_alphas_cumprod[s], x_t)
        ddim_sigma = unsqueeze_as(self.ddim_sigma[s], x_t)
        ddim_dir = unsqueeze_as(self.ddim_dir[s], x_t)

        mean_pred = out["pred_x_start"] * ddim_sqrt_alphas_cumprod_prev + ddim_dir * eps
        nonzero_mask = unsqueeze_as(t != 0, x_t)
        noise = th.randn_like(x_t)
        sample = mean_pred + nonzero_mask * ddim_sigma * noise
        return sample

    def ddim_reverse_sample(self, denoise_fn, x_t, t):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        if self.ddim_eta != 0.0:
            print(f"Warn! Reverse ODE only for deterministic path, but `ddim_eta={self.ddim_eta}`")

        sqrt_recip_alphas_cumprod = unsqueeze_as(self.sqrt_recip_alphas_cumprod[t], x_t)
        sqrt_recipm1_alphas_cumprod = unsqueeze_as(self.sqrt_recipm1_alphas_cumprod[t], x_t)
        sqrt_one_minus_alphas_cumprod_next = unsqueeze_as(self.sqrt_one_minus_alphas_cumprod_next[t], x_t)
        sqrt_alphas_cumprod_prev = unsqueeze_as(self.sqrt_alphas_cumprod_prev[t], x_t)

        out = self.p_mean_variance(denoise_fn, x_t, t)
        if self.model_mean_type == "x_prev":
            out["pred_x_start"] = self._predict_xstart_from_xprev(x_t, t, out["model_mean"])
        eps = (sqrt_recip_alphas_cumprod * x_t - out["pred_x_start"]) / sqrt_recipm1_alphas_cumprod

        sample = out["pred_x_start"] * sqrt_alphas_cumprod_prev + sqrt_one_minus_alphas_cumprod_next * eps
        # return dict(sample=sample, eps=eps, **out)
        return sample

    def sample_progressive(
        self,
        denoise_fn: Callable[[Tensor, Tensor], Tensor],
        shape: Union[Sequence, Size],
        noise: Tensor = None,
    ) -> Tensor:
        """
        return: Tensor (num_samples, *shape)
        """
        b = shape[0]
        sample = default(noise, lambda: th.randn(shape, device=self.device))
        indices = np.flip(self.ddim_timesteps)

        for i, step in enumerate(indices):
            s = th.full((b,), len(indices) - i - 1, dtype=th.long, device=self.device)
            t = th.full((b,), step, dtype=th.long, device=self.device)
            sample = self.ddim_sample(denoise_fn, sample, t, s)
            yield sample

    def forward(
        self,
        denoise_fn: Callable[[Tensor, Tensor], Tensor],
        shape: Union[Sequence, Size],
        noise: Tensor = None,
        num_samples=1,
    ) -> Tensor:
        """
        return: Tensor (num_samples, *shape)
        """
        # variables for sampling intermediations too
        do_sampling = num_samples > 1
        sample_indices = np.linspace(0, len(self.ddim_timesteps), num_samples, dtype=np.int64).tolist()
        sample_lst = []

        for i, sample in enumerate(self.sample_progressive(denoise_fn, shape, noise)):
            if do_sampling and i in sample_indices:
                sample_lst.append(sample)

        if not do_sampling:
            return sample
        else:
            sample_lst.append(sample)
            return th.stack(sample_lst)


def __test__():
    betas = make_beta_schedule("linear", 10)
    diffusion_sampler = DDIMSampler(betas, ddim_eta=0.0, model_var_type="learned_range")
    model = lambda x, t: th.cat([x, x], dim=1)  # any model that doubles the channel (only when learnt model variance)
    shape = (2, 3, 4, 5, 6, 7, 8, 9)  # arbitrary shape
    out = diffusion_sampler(model, shape, num_samples=5)
    print(out.shape)
    # torch.Size([5, 2, 3, 4, 5, 6, 7, 8, 9])
    out = diffusion_sampler(model, shape, num_samples=1)
    print(out.shape)
    # torch.Size([2, 3, 4, 5, 6, 7, 8, 9])


if __name__ == "__main__":
    __test__()
