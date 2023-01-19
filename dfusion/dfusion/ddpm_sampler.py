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
        clip_denoised=True,
    ):
        super().__init__(betas, model_mean_type, model_var_type, clip_denoised)

    def p_sample(self, denoise_fn: Callable[[Tensor, Tensor], Tensor], x_t: Tensor, t: Tensor):
        out = self.p_mean_variance(denoise_fn, x_t, t)
        noise = th.randn_like(x_t)
        nonzero_mask = unsqueeze_as(t > 0, x_t)
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
