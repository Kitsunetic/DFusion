"""
referred to https://github.com/google-research/google-research/blob/master/d3pm/images/diffusion_categorical.py
"""
from dfusion.dfusion.d3pm_utils import *


class D3PMTrainer(nn.Module):
    """
    input:
    - betas:
    - loss_type: (kl | hybrid | cross_entropy_x_start)
    - model_pred_type: (x_start | x_prev)
    - model_output_type: (logits | logistic_pars)
    """

    def __init__(
        self,
        betas: np.ndarray,
        num_channels: int,
        loss_type="kl",
        model_pred_type="x_start",
        model_output_type="logits",
        transition_mat_type="uniform",
        transition_bands=None,
        hybrid_coeff=1.0,
        focal_loss_gamma=0.0,
    ):
        assert loss_type in "kl | hybrid | cross_entropy_x_start".split(" | ")
        assert model_pred_type in "x_start | x_prev".split(" | ")
        assert model_output_type in "logits | logistic_pars".split(" | ")
        super().__init__()

        self.num_channels = num_channels
        self.loss_type = loss_type
        self.model_pred_type = model_pred_type
        self.model_output_type = model_output_type
        self.transition_mat_type = transition_mat_type
        self.hybrid_coeff = hybrid_coeff
        self.focal_loss_gamma = focal_loss_gamma
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

    def q_probs(self, x_start, t):
        """Compute probabilities of q(x_t | x_start).

        Args:
          x_start: jnp.ndarray: jax array of shape (bs, ...) of int32 or int64 type.
             Should not be of one hot representation, but have integer values
             representing the class values.
          t: jnp.ndarray: jax array of shape (bs,).

        Returns:
          probs: jnp.ndarray: jax array, shape (bs, x_start.shape[1:],
                                                num_pixel_vals).
        """
        return self._at(self.q_mats, t, x_start)

    def q_sample(self, x_start, t, noise):
        """Sample from q(x_t | x_start) (i.e. add noise to the data).

        Args:
          x_start: jnp.array: original clean data, in integer form (not onehot).
            shape = (bs, ...).
          t: :jnp.array: timestep of the diffusion process, shape (bs,).
          noise: jnp.ndarray: uniform noise on [0, 1) used to sample noisy data.
            Should be of shape (*x_start.shape, num_pixel_vals).

        Returns:
          sample: jnp.ndarray: same shape as x_start. noisy data.
        """
        assert noise.shape == (*x_start.shape, self.num_channels)
        logits = th.log(self.q_probs(x_start, t) + 1e-6)

        # To avoid numerical issues clip the noise to a minimum value
        noise = th.clamp(noise, th.finfo(noise.dtype).tiny, 1.0)
        gumbel_noise = -th.log(-th.log(noise))
        return th.argmax(logits + gumbel_noise, dim=-1)

    def _get_logits_from_logistic_pars(self, loc: Tensor, log_scale: Tensor):
        """Computes logits for an underlying logistic distribution."""

        loc = loc.unsqueeze(dim=-1)
        log_scale = log_scale.unsqueeze(dim=-1)

        # Shift log_scale such that if it's zero the probs have a scale
        # that is not too wide and not too narrow either.
        inv_scale = th.exp(-(log_scale - 2.0))

        bin_width = 2.0 / (self.num_channels - 1.0)
        bin_centers = th.linspace(-1.0, 1.0, self.num_channels)

        bin_centers = unsqueeze_as(bin_centers, loc)
        bin_centers = bin_centers - loc

        log_cdf_min = F.logsigmoid(inv_scale * (bin_centers - 0.5 * bin_width))
        log_cdf_plus = F.logsigmoid(inv_scale * (bin_centers + 0.5 * bin_width))

        logits = log_min_exp(log_cdf_plus, log_cdf_min, 1e-6)

        # Normalization:
        # # Option 1:
        # # Assign cdf over range (-\inf, x + 0.5] to pmf for pixel with
        # # value x = 0.
        # logits = logits.at[..., 0].set(log_cdf_plus[..., 0])
        # # Assign cdf over range (x - 0.5, \inf) to pmf for pixel with
        # # value x = 255.
        # log_one_minus_cdf_min = - jax.nn.softplus(
        #     inv_scale * (bin_centers - 0.5 * bin_width))
        # logits = logits.at[..., -1].set(log_one_minus_cdf_min[..., -1])
        # # Option 2:
        # # Alternatively normalize by reweighting all terms. This avoids
        # # sharp peaks at 0 and 255.
        # since we are outputting logits here, we don't need to do anything.
        # they will be normalized by softmax anyway.

        return logits

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
        # fact1 = fact1.type(fact2.dtype) # TODO half problem in here, fact1 is float32

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
            pred_x_start_logits = model_logits

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

    def vb_terms_bpd(self, denoise_fn, x_start, x_t, t):
        """Calculate specified terms of the variational bound.

        Args:
          denoise_fn: the denoising network
          x_start: original clean data
          x_t: noisy data
          t: timestep of the noisy data (and the corresponding term of the bound
            to return)

        Returns:
          a pair `(kl, pred_start_logits)`, where `kl` are the requested bound terms
          (specified by `t`), and `pred_x_start_logits` is logits of
          the denoised image.
        """
        true_logits = self.q_posterior_logits(x_start, x_t, t, x_start_logits=False)
        model_logits, pred_x_start_logits = self.p_logits(denoise_fn, x=x_t, t=t)

        kl = categorical_kl_logits(logits1=true_logits, logits2=model_logits)
        assert kl.shape == x_start.shape
        kl = kl.flatten(1).mean(1) / math.log(2.0)

        decoder_nll = -categorical_log_likelihood(x_start, model_logits, gamma=self.focal_loss_gamma)
        assert decoder_nll.shape == x_start.shape
        decoder_nll = decoder_nll.flatten(1).mean(1) / math.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_start) || p(x_{t-1}|x_t))
        assert kl.shape == decoder_nll.shape == t.shape == (x_start.shape[0],)
        return th.where(t == 0, decoder_nll, kl), pred_x_start_logits

    def prior_bpd(self, x_start):
        """KL(q(x_{T-1}|x_start)|| U(x_{T-1}|0, num_pixel_vals-1))."""
        q_probs = self.q_probs(x_start=x_start, t=th.full((x_start.shape[0],), self.num_timesteps - 1))

        if self.transition_mat_type in ["gaussian", "uniform"]:
            # Stationary distribution is a uniform distribution over all pixel values.
            prior_probs = th.ones_like(q_probs) / self.num_channels

        elif self.transition_mat_type == "absorbing":
            # Stationary distribution is a kronecker delta distribution
            # with all its mass on the absorbing state.
            # Absorbing state is located at rgb values (128, 128, 128)
            absorbing_int = th.full(shape=q_probs.shape[:-1], fill_value=self.num_channels // 2, dtype=th.long)
            prior_probs = F.one_hot(absorbing_int, num_classes=self.num_channels, dim=-1, dtype=self.betas.dtype)
        else:
            raise ValueError(
                f"transition_mat_type must be 'gaussian', 'uniform', 'absorbing' " f", but is {self.transition_mat_type}"
            )

        assert prior_probs.shape == q_probs.shape

        kl_prior = categorical_kl_probs(q_probs, prior_probs)
        assert kl_prior.shape == x_start.shape
        return kl_prior.flatten(1).mean(1) / math.log(2.0)

    def cross_entropy_x_start(self, x_start, pred_x_start_logits):
        """Calculate crossentropy between x_start and predicted x_start.

        Args:
          x_start: original clean data
          pred_x_start_logits: predicted_logits

        Returns:
          ce: cross entropy.
        """

        ce = -categorical_log_likelihood(x_start, pred_x_start_logits, gamma=self.focal_loss_gamma)
        assert ce.shape == x_start.shape
        ce = ce.flatten(1).mean(1) / math.log(2.0)

        assert ce.shape == (x_start.shape[0],)

        return ce

    def forward(self, denoise_fn: Callable[[Tensor, Tensor], Tensor], x_start: Tensor, return_x_start: bool = False):
        # Add noise to data
        # noise_rng, time_rng = jax.random.split(rng)
        # noise = jax.random.uniform(noise_rng, shape=x_start.shape + (self.num_pixel_vals,))
        # t = jax.random.randint(time_rng, shape=(x_start.shape[0],), minval=0, maxval=self.num_timesteps, dtype=jnp.int32)
        b = x_start.size(0)
        noise = th.rand((*x_start.shape, self.num_channels), device=self.device)
        t = th.randint(0, self.num_timesteps, (b,), device=self.device)

        # t starts at zero. so x_0 is the first noisy datapoint, not the datapoint
        # itself.
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Calculate the loss
        losses = {}
        if self.loss_type == "kl":
            # Optimizes the variational bound L_vb.
            losses["vb"], pred_x_start_logits = self.vb_terms_bpd(denoise_fn=denoise_fn, x_start=x_start, x_t=x_t, t=t)
            losses["loss"] = losses["vb"]

        elif self.loss_type == "cross_entropy_x_start":
            # Optimizes - sum_x_start x_start log pred_x_start.
            _, pred_x_start_logits = self.p_logits(denoise_fn, x=x_t, t=t)
            losses["loss"] = losses["ce"] = self.cross_entropy_x_start(x_start=x_start, pred_x_start_logits=pred_x_start_logits)

        elif self.loss_type == "hybrid":
            # Optimizes L_vb - lambda * sum_x_start x_start log pred_x_start.
            losses["vb"], pred_x_start_logits = self.vb_terms_bpd(denoise_fn=denoise_fn, x_start=x_start, x_t=x_t, t=t)
            losses["ce"] = self.cross_entropy_x_start(x_start=x_start, pred_x_start_logits=pred_x_start_logits)
            losses["loss"] = losses["vb"] + self.hybrid_coeff * losses["ce"]

        else:
            raise NotImplementedError(self.loss_type)

        if return_x_start:
            return losses, pred_x_start_logits
        else:
            return losses


def __test__():
    betas = make_beta_schedule("linear", 1000)
    diffusion_trainer = D3PMTrainer(
        betas,
        num_channels=256,
        loss_type="hybrid",
        model_pred_type="x_start",
        model_output_type="logits",
        transition_mat_type="uniform",
        transition_bands=None,
        hybrid_coeff=1.0,
        focal_loss_gamma=2.0,
    )
    # model = lambda x, t: th.cat([x, x], dim=1)  # any model that doubles the channel (only when learnt model variance)
    model = lambda x, t: th.rand(2, 3, 32, 32, 256)
    data = th.randint(0, 256, (2, 3, 32, 32))  # arbitrary shape
    # data = th.rand(2, 3, 32, 32, 256)
    losses, pred_x_start = diffusion_trainer(model, data, return_x_start=True)
    print(losses)
    # {'vlb': tensor([5.2729e-06, 5.2618e-05]), 'eps': tensor([0.0007, 0.0084]), 'loss': tensor([0.0007, 0.0082])}
    print(pred_x_start.shape)


if __name__ == "__main__":
    __test__()
