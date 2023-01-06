from dfusion.dfusion.common import *


class DiffusionBase(nn.Module):
    @property
    def device(self):
        return self.betas.device

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            unsqueeze_as(self.sqrt_recip_alphas_cumprod[t], x_t) * x_t
            - unsqueeze_as(self.sqrt_recipm1_alphas_cumprod[t], x_t) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            unsqueeze_as(1.0 / self.posterior_mean_coef1[t], x_t.shape) * xprev
            - unsqueeze_as(self.posterior_mean_coef2[t] / self.posterior_mean_coef1[t], x_t.shape) * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (unsqueeze_as(self.sqrt_recip_alphas_cumprod[t], x_t.shape) * x_t - pred_xstart) / unsqueeze_as(
            self.sqrt_recipm1_alphas_cumprod[t], x_t.shape
        )

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: th.randn_like(x_start))
        return (
            unsqueeze_as(self.sqrt_alphas_cumprod[t], x_start) * x_start
            + unsqueeze_as(self.sqrt_one_minus_alphas_cumprod[t], x_start) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            unsqueeze_as(self.posterior_mean_coef1[t], x_t) * x_start + unsqueeze_as(self.posterior_mean_coef2[t], x_t) * x_t
        )
        posterior_variance = unsqueeze_as(self.posterior_variance[t], x_t)
        posterior_log_variance_clipped = unsqueeze_as(self.posterior_log_variance_clipped[t], x_t)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, denoise_fn: Callable, x_t: Tensor, t: Tensor, clip_denoised=False):
        model_out: Tensor = denoise_fn(x_t, t)
        out_dict = {"model_out": model_out}

        # get variance
        if self.model_var_type in ("learned", "learned_range"):
            assert model_out.size(1) == 2 * x_t.size(1)
            model_out, tmp = th.chunk(model_out, 2, dim=1)
            if self.model_var_type == "learned":
                model_log_var = tmp
                # model_var = model_log_var.exp()
            else:
                min_log = unsqueeze_as(self.posterior_log_variance_clipped[t], x_t)
                max_log = unsqueeze_as(self.betas_log[t], x_t)
                frac = (tmp + 1) * 0.5
                model_log_var = frac * max_log + (1 - frac) * min_log
                # model_var = model_log_var.exp()
        elif self.model_var_type == "fixed_large":
            # model_var = unsqueeze_as(self.posterior_variance_large[t], x_t)
            model_log_var = unsqueeze_as(self.posterior_log_variance_clipped_large[t], x_t)
        elif self.model_var_type == "fixed_small":
            # model_var = unsqueeze_as(self.posterior_variance[t], x_t)
            model_log_var = unsqueeze_as(self.posterior_log_variance_clipped[t], x_t)
        else:
            raise NotImplementedError
        # out_dict["model_var"] = model_var
        out_dict["model_log_var"] = model_log_var

        clip = (lambda x: x.clamp_(-1.0, 1.0)) if clip_denoised else (lambda x: x)

        # get mean
        if self.model_mean_type == "x_prev":
            model_mean = model_out
            pred_x_start = clip(self._predict_xstart_from_xprev(x_t, t, model_mean))
        elif self.model_mean_type == "x_start":
            pred_x_start = clip(model_out)
            model_mean, _, _ = self.q_posterior_mean_variance(pred_x_start, x_t, t)
        elif self.model_mean_type == "eps":
            out_dict["pred_eps"] = model_out
            pred_x_start = clip(self._predict_xstart_from_eps(x_t, t, model_out))
            model_mean, _, _ = self.q_posterior_mean_variance(pred_x_start, x_t, t)
        else:
            raise NotImplementedError

        out_dict["model_mean"] = model_mean  # \tilde x_{t-1}
        out_dict["pred_x_start"] = pred_x_start  # \tilde x_0

        # return model_mean, model_var, model_log_var
        return out_dict
