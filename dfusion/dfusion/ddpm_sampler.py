from dfusion.dfusion.diffusion_base import *


class DDPMSampler(DiffusionBase):
    """
    input:
    - betas:
    - model_mean_type: (eps | x_start | x_prev)
    - model_var_type: (fixed_small | fixed_large | leraned | learned_range)
    """

    def __init__(
        self,
        betas: np.ndarray,
        model_mean_type="eps",
        model_var_type="fixed_small",
        clip_denoised=False,
    ):
        assert model_mean_type in "eps|x_start|x_prev".split("|")
        assert model_var_type in "fixed_small|fixed_large|leraned|learned_range".split("|")
        super().__init__()

        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.clip_denoised = clip_denoised
        self.num_timesteps = len(betas)

        # noise schedule caches - betas
        betas = betas.astype(np.float64)
        betas_log = np.log(betas)
        self.num_timesteps = len(betas)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)
        sqrt_alphas_cumprod_prev = np.sqrt(np.append(1.0, alphas_cumprod))

        # noise schedule caches - vlb calculation
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_log_variance_clipped = np.log(np.append(posterior_variance[1], posterior_variance[1:]))
        posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
        posterior_variance_large = np.append(posterior_variance[1], betas[1:])
        posterior_log_variance_clipped_large = np.log(np.append(posterior_variance[1], betas[1:]))

        reg = lambda name, x: self.register_buffer(name, th.from_numpy(x.astype(np.float32)))
        reg("betas", betas)
        reg("betas_log", betas_log)
        reg("alphas", alphas)
        reg("alphas_cumprod", alphas_cumprod)
        reg("alphas_cumprod_prev", alphas_cumprod_prev)
        reg("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        reg("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)
        reg("sqrt_recip_alphas_cumprod", sqrt_recip_alphas_cumprod)
        reg("sqrt_recipm1_alphas_cumprod", sqrt_recipm1_alphas_cumprod)
        reg("sqrt_alphas_cumprod_prev", sqrt_alphas_cumprod_prev)
        reg("posterior_variance", posterior_variance)
        reg("posterior_log_variance_clipped", posterior_log_variance_clipped)
        reg("posterior_mean_coef1", posterior_mean_coef1)
        reg("posterior_mean_coef2", posterior_mean_coef2)
        reg("posterior_variance_large", posterior_variance_large)
        reg("posterior_log_variance_clipped_large", posterior_log_variance_clipped_large)

    def p_sample(self, denoise_fn: Callable[[Tensor, Tensor], Tensor], x_t: Tensor, t: Tensor):
        out = self.p_mean_variance(denoise_fn, x_t, t, clip_denoised=self.clip_denoised)
        noise = th.randn_like(x_t)
        nonzero_mask = unsqueeze_as(t != 0, x_t)
        sample = out["model_mean"] + nonzero_mask * th.exp(0.5 * out["model_log_var"]) * noise
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
        indices = list(range(self.num_timesteps))[::-1]

        for i in indices:
            t = th.full((b,), i, dtype=th.long, device=self.device)
            sample = self.p_sample(denoise_fn, sample, t)
            yield sample

    def forward(
        self,
        denoise_fn: Callable[[Tensor, Tensor], Tensor],
        shape: Union[Sequence, Size],
        noise: Tensor = None,
        num_samples=1,
    ) -> Tensor:
        """
        return: Tensor (num_samples, *shape) or (*shape, )
        """
        sample = default(noise, lambda: th.randn(shape, device=self.device))

        # variables for filtering intermediations
        do_sampling = num_samples > 1
        sample_indices = np.linspace(0, self.num_timesteps, num_samples, dtype=np.int64).tolist()
        sample_lst = []

        for i, sample in enumerate(self.sample_progressive(denoise_fn, shape, sample)):
            if do_sampling and i in sample_indices:
                sample_lst.append(sample)

        if not do_sampling:
            return sample
        else:
            sample_lst.append(sample)
            return th.stack(sample_lst)


def __test__():
    betas = make_beta_schedule("linear", 10)
    diffusion_sampler = DDPMSampler(betas, model_var_type="learned_range")
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
