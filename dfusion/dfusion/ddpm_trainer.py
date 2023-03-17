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
        clip_denoised=True,
    ):
        super().__init__(betas, model_mean_type, model_var_type, clip_denoised)

        assert loss_type in "l2|rescaled_l2|l1|rescaled_l1|kl|rescaled_kl".split("|")
        self.loss_type = loss_type
        self.do_p2 = p2_loss_weight_gamma > 0.0

        with self.register_diffusion_parameters():
            # p2 loss weight, from https://arxiv.org/abs/2204.00227
            self.p2_loss_weight = (p2_loss_weight_k + self.alphas_cumprod / (1 - self.alphas_cumprod)) ** -p2_loss_weight_gamma

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
        out = self.p_mean_variance(denoise_fn, x_t, t)
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
            if self.do_p2:
                losses["loss"] = losses["recon"] * self.p2_loss_weight[t]
            else:
                losses["loss"] = losses["recon"]

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
    print(losses)
    # {'vlb': tensor([5.2729e-06, 5.2618e-05]), 'eps': tensor([0.0007, 0.0084]), 'loss': tensor([0.0007, 0.0082])}


if __name__ == "__main__":
    __test__()
