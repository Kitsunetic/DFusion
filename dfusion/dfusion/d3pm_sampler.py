"""
referred to https://github.com/google-research/google-research/blob/master/d3pm/images/diffusion_categorical.py
"""
from dfusion.dfusion.d3pm_utils import *


class D3PMSampler(nn.Module):
    """
    input:
    - betas:
    - model_pred_type: (x_start | x_prev)
    - model_output_type: (logits | logistic_pars)
    """

    def __init__(
        self,
        betas: np.ndarray,
        num_channels: int,
        model_pred_type="x_start",
        model_output_type="logits",
        transition_mat_type="uniform",
        transition_bands=None,
    ):
        assert model_pred_type in "x_start | x_prev".split(" | ")
        assert model_output_type in "logits | logistic_pars".split(" | ")
        super().__init__()

        self.num_channels = num_channels
        self.model_pred_type = model_pred_type
        self.model_output_type = model_output_type
        self.transition_mat_type = transition_mat_type
        self.num_timesteps = len(betas)

        # noise schedule caches - betas
        betas = betas.astype(np.float64)

        ## D3PM transition matrices
        if self.transition_mat_type == "uniform":
            q_one_step_mats = [get_transition_mat(betas, t, num_channels, transition_bands) for t in range(self.num_timesteps)]
        elif self.transition_mat_type == "gaussian":
            q_one_step_mats = [
                get_gaussian_transition_mat(betas, t, num_channels, transition_bands) for t in range(self.num_timesteps)
            ]
        elif self.transition_mat_type == "absorbing":
            q_one_step_mats = [
                get_absorbing_transition_mat(betas, t, num_channels, transition_bands) for t in range(self.num_timesteps)
            ]
        q_one_step_mats = np.stack(q_one_step_mats)  # t num_channels num_channels

        q_mat_t = q_one_step_mats[0]
        q_mats = [q_mat_t]
        for t in range(1, self.num_timesteps):
            q_mat_t = np.tensordot(q_mat_t, q_one_step_mats[t], axes=[[1], [0]])  # num_channels num_channels
            q_mats.append(q_mat_t)
        q_mats = np.stack(q_mats)  # t num_channels num_channels
        assert q_mats.shape == (self.num_timesteps, num_channels, num_channels), q_mats.shape

        transpose_q_onestep_mats = np.transpose(q_one_step_mats, axes=(0, 2, 1))

        reg = lambda name, x: self.register_buffer(name, th.from_numpy(x.astype(np.float32)))
        reg("betas", betas)
        reg("q_one_step_mats", q_one_step_mats)
        reg("q_mats", q_mats)
        reg("transpose_q_onestep_mats", transpose_q_onestep_mats)

    @property
    def device(self):
        return self.betas.device

    def _at(self, a, t, x):
        """Extract coefficients at specified timesteps t and conditioning data x.

        Args:
          a: np.ndarray: plain NumPy float64 array of constants indexed by time.
          t: jnp.ndarray: Jax array of time indices, shape = (batch_size,).
          x: jnp.ndarray: jax array of shape (bs, ...) of int32 or int64 type.
            (Noisy) data. Should not be of one hot representation, but have integer
            values representing the class values.

        Returns:
          a[t, x]: jnp.ndarray: Jax array.
        """
        return a[unsqueeze_as(t, x), x]

        # a = jnp.asarray(a, dtype=self.jax_dtype)
        # t_broadcast = jnp.expand_dims(t, tuple(range(1, x.ndim)))

        # # x.shape = (bs, height, width, channels)
        # # t_broadcast_shape = (bs, 1, 1, 1)
        # # a.shape = (num_timesteps, num_pixel_vals, num_pixel_vals)
        # # out.shape = (bs, height, width, channels, num_pixel_vals)
        # # out[i, j, k, l, m] = a[t[i, j, k, l], x[i, j, k, l], m]
        # return a[t_broadcast, x]

    def _at_onehot(self, a, t, x):
        """Extract coefficients at specified timesteps t and conditioning data x.

        Args:
          a: np.ndarray: plain NumPy float64 array of constants indexed by time.
          t: jnp.ndarray: Jax array of time indices, shape = (bs,).
          x: jnp.ndarray: jax array, shape (bs, ..., num_pixel_vals), float32 type.
            (Noisy) data. Should be of one-hot-type representation.

        Returns:
          out: jnp.ndarray: Jax array. output of dot(x, a[t], axis=[[-1], [1]]).
            shape = (bs, ..., num_pixel_vals)
        """
        # x: B ... N
        # a: T N N
        return th.matmul(x, a[t, None, None, ...])

        # a = jnp.asarray(a, dtype=self.jax_dtype)

        # # x.shape = (bs, height, width, channels, num_pixel_vals)
        # # a[t]shape = (bs, num_pixel_vals, num_pixel_vals)
        # # out.shape = (bs, height, width, channels, num_pixel_vals)
        # return jnp.matmul(x, a[t, None, None, Ellipsis], precision=jax.lax.Precision.HIGHEST)

    def q_posterior_logits(self, x_start: Tensor, x_t: Tensor, t: Tensor, x_start_logits: bool):
        """Compute logits of q(x_{t-1} | x_t, x_start)."""

        if x_start_logits:
            assert x_start.shape == x_t.shape + (self.num_channels,), (x_start.shape, x_t.shape)
        else:
            assert x_start.shape == x_t.shape, (x_start.shape, x_t.shape)

        fact1 = self._at(self.transpose_q_onestep_mats, t, x_t)
        if x_start_logits:
            fact2 = self._at_onehot(self.q_mats, t - 1, th.softmax(x_start, dim=-1))
            tzero_logits = x_start
        else:
            fact2 = self._at(self.q_mats, t - 1, x_start)
            tzero_logits = th.log(F.one_hot(x_start, num_classes=self.num_channels) + 1e-6)

        # At t=0 we need the logits of q(x_{-1}|x_0, x_start)
        # where x_{-1} == x_start. This should be equal the log of x_0.
        out = th.log(fact1 + 1e-6) + th.log(fact2 + 1e-6)
        t_broadcast = unsqueeze_as(t, out)
        return th.where(t_broadcast == 0, tzero_logits, out)

    def p_logits(self, denoise_fn, x, t):
        """Compute logits of p(x_{t-1} | x_t)."""
        assert t.shape == (x.shape[0],)
        model_output = denoise_fn(x, t)

        if self.model_output_type == "logits":
            model_logits = model_output
        elif self.model_output_type == "logistic_pars":
            # Get logits out of discretized logistic distribution.
            loc, log_scale = model_output
            model_logits = self._get_logits_from_logistic_pars(loc, log_scale)
        else:
            raise NotImplementedError(self.model_output_type)

        if self.model_pred_type == "x_start":
            # Predict the logits of p(x_{t-1}|x_t) by parameterizing this distribution
            # as ~ sum_{pred_x_start} q(x_{t-1}, x_t |pred_x_start)p(pred_x_start|x_t)
            pred_x_start_logits = model_logits  # b ... c

            t_broadcast = unsqueeze_as(t, model_logits)
            other = self.q_posterior_logits(pred_x_start_logits, x, t, x_start_logits=True)
            model_logits = th.where(t_broadcast == 0, pred_x_start_logits, other)
        elif self.model_pred_type == "x_prev":
            # Use the logits out of the model directly as the logits for
            # p(x_{t-1}|x_t). model_logits are already set correctly.
            # NOTE: the pred_x_start_logits in this case makes no sense.
            # For Gaussian DDPM diffusion the model predicts the mean of
            # p(x_{t-1}}|x_t), and uses inserts this as the eq for the mean of
            # q(x_{t-1}}|x_t, x_0) to compute the predicted x_0/x_start.
            # The equivalent for the categorical case is nontrivial.
            pred_x_start_logits = model_logits
        else:
            # note: IDK why, this `raise` was in `x_prev` case
            raise NotImplementedError(self.model_pred_type)

        assert model_logits.shape == pred_x_start_logits.shape == x.shape + (self.num_channels,)
        return model_logits, pred_x_start_logits

    def make_x_init(self, shape):
        if self.transition_mat_type in ("gaussian", "uniform"):
            x = th.randint(0, self.num_channels, shape, device=self.device)
        elif self.transition_mat_type == "absorbing":
            x = th.full(shape, self.num_channels // 2, dtype=th.long, device=self.device)
        else:
            raise ValueError(
                f"transition_mat_type must be 'gaussian', 'uniform', 'absorbing' " f", but is {self.transition_mat_type}"
            )
        return x

    def p_sample(self, denoise_fn: Callable[[Tensor, Tensor], Tensor], x_t: Tensor, t: Tensor, noise: Tensor = None):
        noise = default(noise, lambda: th.rand((*x_t.shape, self.num_channels), device=self.device))
        model_logits, pred_x_start_logits = self.p_logits(denoise_fn, x_t, t)
        assert noise.shape == model_logits.shape, (noise.shape, model_logits.shape, x_t.shape)

        nonzero_mask = unsqueeze_as((t != 0), noise)
        noise.clamp_(th.finfo(noise.dtype).tiny, 1.0)
        gumbel_noise = -th.log(-th.log(noise))
        sample = th.argmax(model_logits + nonzero_mask * gumbel_noise, dim=-1)

        assert sample.shape == x_t.shape
        assert pred_x_start_logits.shape == model_logits.shape
        return sample, pred_x_start_logits

    def sample_progressive(self, denoise_fn: Callable[[Tensor, Tensor], Tensor], shape: Sequence, noise: Tensor = None):
        b = shape[0]
        sample = default(noise, lambda: self.make_x_init(shape))
        indices = list(range(self.num_timesteps))[::-1]

        for i in indices:
            t = th.full((b,), i, device=self.device)
            sample, pred_x_start_logit = self.p_sample(denoise_fn, sample, t)
            yield sample

    def forward(self, denoise_fn: Callable[[Tensor, Tensor], Tensor], shape: Sequence, noise: Tensor = None, num_samples=1):
        # variables for filtering intermediations
        do_sampling = num_samples > 1
        sample_indices = np.linspace(0, self.num_timesteps, num_samples, dtype=np.int64).tolist()
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
    betas = make_beta_schedule("linear", 50)
    diffusion_trainer = D3PMSampler(
        betas,
        num_channels=256,
        model_pred_type="x_start",
        model_output_type="logits",
        transition_mat_type="uniform",
        transition_bands=None,
    )
    # model = lambda x, t: th.cat([x, x], dim=1)  # any model that doubles the channel (only when learnt model variance)
    model = lambda x, t: th.rand(2, 3, 32, 32, 256)
    out = diffusion_trainer(model, (2, 3, 32, 32))
    print(out.shape)
    # torch.Size([2, 3, 32, 32])


if __name__ == "__main__":
    __test__()
