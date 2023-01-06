from dfusion.dfusion.diffusion_base import *


class DDPMTrainer(DiffusionBase):
    """
    input:
    - betas:
    - loss_type: (l2 | rescaled_l2 | l1 | rescaled_l1 | kl | rescaled_kl)
    - model_mean_type: (eps | x_start | x_prev)
    - model_var_type: (fixed_small | fixed_large | leraned | learned_range)
    - p2_loss_weight_gamma: (0.0 = not applied), 1.0 or 0.5 is good following paper
    - p2_loss_weight_k: 1.0 is good following paper
    """

    def __init__(
        self,
        betas: np.ndarray,
        loss_type="l2",
        model_mean_type="eps",
        model_var_type="fixed_small",
        p2_loss_weight_gamma=0.0,
        p2_loss_weight_k=1.0,
    ):
        assert loss_type in "l2|rescaled_l2|l1|rescaled_l1|kl|rescaled_kl".split("|")
        assert model_mean_type in "eps|x_start|x_prev".split("|")
        assert model_var_type in "fixed_small|fixed_large|leraned|learned_range".split("|")
        super().__init__()

        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.num_timesteps = len(betas)

        # noise schedule caches - betas
        betas = betas.astype(np.float64)
        betas_log = np.log(betas)
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

        # p2 loss weight, from https://arxiv.org/abs/2204.00227
        p2_loss_weight = (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma

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
        reg("p2_loss_weight", p2_loss_weight)

    def loss_fn(self, pred: Tensor, target: Tensor) -> Tensor:
        return (
            {
                "l2": partial(F.mse_loss, reduction="none"),
                "rescaled_l2": partial(F.mse_loss, reduction="none"),
                "l1": partial(F.l1_loss, reduction="none"),
                "rescaled_l1": partial(F.l1_loss, reduction="none"),
            }[self.loss_type](pred, target)
            .flatten(1)
            .mean(1)
        )

    def get_vlb(self, denoise_fn, x_start, x_t, t):
        true_mean, _, true_log_var = self.q_posterior_mean_variance(x_start, x_t, t)
        out = self.p_mean_variance(denoise_fn, x_t, t, clip_denoised=False)
        model_mean, model_log_var = out["model_mean"], out["model_log_var"]
        kl = normal_kl(true_mean, true_log_var, model_mean, model_log_var)
        kl = kl.flatten(1).mean(1)

        decoder_nll = -discretized_gaussian_log_likelihood(x_start, model_mean, 0.5 * model_log_var)
        decoder_nll = decoder_nll.flatten(1).mean(1)

        vlb = th.where(t == 0, decoder_nll, kl)
        return vlb

    def forward(self, denoise_fn: Callable[[Tensor, Tensor], Tensor], x_start: Tensor, t: Tensor = None, noise: Tensor = None):
        b = x_start.size(0)
        t = default(t, lambda: th.randint(0, self.num_timesteps, (b,), dtype=th.long, device=self.device))
        noise = default(noise, lambda: th.randn_like(x_start))
        x_t = self.q_sample(x_start, t, noise)

        losses = {}
        if self.loss_type in ("kl", "rescaled_kl"):
            # \mathcal L_\text{kl}
            losses["loss"] = self.get_vlb(denoise_fn, x_start, x_t, t)

        elif self.loss_type in ("l2", "rescaled_l2", "l1", "rescaled_l1"):
            # \mathcal L_\text{vlb}
            model_out = denoise_fn(x_t, t)
            if self.model_var_type in ("learned", "learned_range"):
                assert model_out.size(1) == 2 * x_t.size(1)
                model_out, tmp = th.chunk(model_out, chunks=2, dim=1)
                frozen_out = th.cat([model_out.detach(), tmp], dim=1)
                losses["vlb"] = self.get_vlb(lambda *_: frozen_out, x_start, x_t, t)
                if self.loss_type in ("rescaled_l1", "rescaled_l2"):
                    losses["vlb"] = losses["vlb"] * self.num_timesteps / 1000.0
            # note: vlb is not calculated when its fixed variance setting, but the fixed variances are used during sampling

            # \mathcal L_\text{simple}
            target = {
                "eps": lambda: noise,
                "x_start": lambda: x_start,
                "x_prev": lambda: self.q_posterior_mean_variance(x_start, x_t, t)[0],
            }[self.model_mean_type]()
            losses["recon"] = self.loss_fn(model_out, target)

            # \mathcal L = \mathcal L_\text{simple} * \lambda_\text{p2}
            losses["loss"] = losses["recon"] * self.p2_loss_weight[t]
            # \mathcal L_\text{hybrid}
            if "vlb" in losses:
                losses["loss"] = losses["loss"] + losses["vlb"]

        return losses


def __test__():
    betas = make_beta_schedule("linear", 1000)
    diffusion_trainer = DDPMTrainer(betas, model_var_type="learned_range", p2_loss_weight_gamma=1.0)
    model = lambda x, t: th.cat([x, x], dim=1)  # any model that doubles the channel (only when learnt model variance)
    data = th.rand(2, 3, 4, 5, 6, 7, 8, 9)  # arbitrary shape
    losses = diffusion_trainer(model, data)
    # {'vlb': tensor([5.2729e-06, 5.2618e-05]), 'eps': tensor([0.0007, 0.0084]), 'loss': tensor([0.0007, 0.0082])}


if __name__ == "__main__":
    __test__()
