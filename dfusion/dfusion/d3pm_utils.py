import scipy.special
from dfusion.dfusion.diffusion_base import *


def log_min_exp(a, b, epsilon=1.0e-6):
    """Computes the log(exp(a) - exp(b)) (b<a) in a numerically stable fashion."""
    y = a + th.log1p(-th.exp(b - a) + epsilon)
    return y


def sample_categorical(logits, uniform_noise):
    """Samples from a categorical distribution.

    Args:
      logits: logits that determine categorical distributions. Shape should be
        broadcastable under addition with noise shape, and of the form (...,
        num_classes).
      uniform_noise: uniform noise in range [0, 1). Shape: (..., num_classes).

    Returns:
      samples: samples.shape == noise.shape, with samples.shape[-1] equal to
        num_classes.
    """
    # For numerical precision clip the noise to a minimum value
    uniform_noise = th.clamp(uniform_noise, th.finfo(uniform_noise.dtype).tiny, 1.0)
    gumbel_noise = -th.log(-th.log(uniform_noise))
    sample = th.argmax(logits + gumbel_noise, dim=-1)
    return F.one_hot(sample, num_classes=logits.size(-1))


def categorical_kl_logits(logits1, logits2, eps=1.0e-6):
    """KL divergence between categorical distributions.

    Distributions parameterized by logits.

    Args:
      logits1: logits of the first distribution. Last dim is class dim.
      logits2: logits of the second distribution. Last dim is class dim.
      eps: float small number to avoid numerical issues.

    Returns:
      KL(C(logits1) || C(logits2)): shape: logits1.shape[:-1]
    """
    out = th.softmax(logits1 + eps, dim=-1) * (F.log_softmax(logits1 + eps, dim=-1) - F.log_softmax(logits2 + eps, dim=-1))
    return th.sum(out, dim=-1)


def categorical_kl_probs(probs1, probs2, eps=1.0e-6):
    """KL divergence between categorical distributions.

    Distributions parameterized by logits.

    Args:
      probs1: probs of the first distribution. Last dim is class dim.
      probs2: probs of the second distribution. Last dim is class dim.
      eps: float small number to avoid numerical issues.

    Returns:
      KL(C(probs) || C(logits2)): shape: logits1.shape[:-1]
    """
    out = probs1 * (th.log(probs1 + eps) - th.log(probs2 + eps))
    return th.sum(out, dim=-1)


def categorical_log_likelihood(x, logits):
    """Log likelihood of a discretized Gaussian specialized for image data.

    Assumes data `x` consists of integers [0, num_classes-1].

    Args:
      x: where to evaluate the distribution. shape = (bs, ...), dtype=int32/int64
      logits: logits, shape = (bs, ..., num_classes)

    Returns:
      log likelihoods
    """
    log_probs = F.log_softmax(logits, dim=-1)
    x_onehot = F.one_hot(x, logits.shape[-1])
    return th.sum(log_probs * x_onehot, dim=-1)


def _get_full_transition_mat(betas: np.ndarray, t: int, num_channels: int):
    """Computes transition matrix for q(x_t|x_{t-1}).

    Contrary to the band diagonal version, this method constructs a transition
    matrix with uniform probability to all other states.

    Args:
        t: timestep. integer scalar.

    Returns:
        Q_t: transition matrix. shape = (num_pixel_vals, num_pixel_vals).
    """
    beta_t = betas[t]
    mat = np.full((num_channels, num_channels), beta_t / float(num_channels), dtype=np.float64)
    diag_indices = np.diag_indices_from(mat)
    diag_val = 1.0 - beta_t * (num_channels - 1.0) / num_channels
    mat[diag_indices] = diag_val
    return mat


def get_transition_mat(betas: np.ndarray, t: int, num_channels: int, transition_bands: int):
    r"""Computes transition matrix for q(x_t|x_{t-1}).

    This method constructs a transition
    matrix Q with
    Q_{ij} = beta_t / num_pixel_vals       if |i-j| <= self.transition_bands
                1 - \sum_{l \neq i} Q_{il}    if i==j.
                0                             else.

    Args:
        t: timestep. integer scalar (or numpy array?)

    Returns:
        Q_t: transition matrix. shape = (num_pixel_vals, num_pixel_vals).
    """
    if transition_bands is None:
        return _get_full_transition_mat(betas, t, num_channels)
    # Assumes num_off_diags < num_pixel_vals
    beta_t = betas[t]

    mat = np.zeros((num_channels, num_channels), dtype=np.float64)
    off_diag = np.full(shape=(num_channels - 1,), fill_value=beta_t / float(num_channels), dtype=np.float64)
    for k in range(1, transition_bands + 1):
        mat += np.diag(off_diag, k=k)
        mat += np.diag(off_diag, k=-k)
        off_diag = off_diag[:-1]

    # Add diagonal values such that rows sum to one.
    diag = 1.0 - mat.sum(1)
    mat += np.diag(diag, k=0)
    return mat


def get_gaussian_transition_mat(betas: np.ndarray, t: int, num_channels: int, transition_bands: int):
    r"""Computes transition matrix for q(x_t|x_{t-1}).

    This method constructs a transition matrix Q with
    decaying entries as a function of how far off diagonal the entry is.
    Normalization option 1:
    Q_{ij} =  ~ softmax(-val^2/beta_t)   if |i-j| <= self.transition_bands
                1 - \sum_{l \neq i} Q_{il}  if i==j.
                0                          else.

    Normalization option 2:
    tilde{Q}_{ij} =  softmax(-val^2/beta_t)   if |i-j| <= self.transition_bands
                        0                        else.

    Q_{ij} =  tilde{Q}_{ij} / sum_l{tilde{Q}_{lj}}

    Args:
        t: timestep. integer scalar (or numpy array?)

    Returns:
        Q_t: transition matrix. shape = (num_pixel_vals, num_pixel_vals).
    """
    transition_bands = transition_bands if transition_bands else num_channels - 1

    beta_t = betas[t]

    mat = np.zeros((num_channels, num_channels), dtype=np.float64)

    # Make the values correspond to a similar type of gaussian as in the
    # gaussian diffusion case for continuous state spaces.
    values = np.linspace(start=0.0, stop=255.0, num=num_channels, endpoint=True, dtype=np.float64)
    values = values * 2.0 / (num_channels - 1.0)
    values = values[: transition_bands + 1]
    values = -values * values / beta_t

    values = np.concatenate([values[:0:-1], values], axis=0)
    values = scipy.special.softmax(values, axis=0)
    values = values[transition_bands:]
    for k in range(1, transition_bands + 1):
        off_diag = np.full(shape=(num_channels - k,), fill_value=values[k], dtype=np.float64)

        mat += np.diag(off_diag, k=k)
        mat += np.diag(off_diag, k=-k)

    # Add diagonal values such that rows and columns sum to one.
    # Technically only the ROWS need to sum to one
    # NOTE: this normalization leads to a doubly stochastic matrix,
    # which is necessary if we want to have a uniform stationary distribution.
    diag = 1.0 - mat.sum(1)
    mat += np.diag(diag, k=0)

    return mat


def get_absorbing_transition_mat(betas: np.ndarray, t: int, num_channels: int, transition_bands: int = None):
    """Computes transition matrix for q(x_t|x_{t-1}).

    Has an absorbing state for pixelvalues num_channels//2.

    Args:
        t: timestep. integer scalar.

    Returns:
        Q_t: transition matrix. shape = (num_pixel_vals, num_pixel_vals).
    """
    beta_t = betas[t]

    diag = np.full(shape=(num_channels,), fill_value=1.0 - beta_t, dtype=np.float64)
    mat = np.diag(diag, k=0)
    # Add beta_t to the num_pixel_vals/2-th column for the absorbing state.
    mat[:, num_channels // 2] += beta_t

    return mat
