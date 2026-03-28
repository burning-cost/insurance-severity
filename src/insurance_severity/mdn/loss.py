"""
MDN loss functions and mixture distribution utilities.

All functions operate in log-space using the log-sum-exp trick to avoid
numerical underflow when evaluating mixture likelihoods. This is essential:
summing K small probabilities directly leads to catastrophic cancellation
or underflow for even moderate K.

The target ``y`` passed to the loss should be the **raw severity** (positive
reals). The log transformation is applied internally: the mixture components
are Gaussian in log-space, producing a lognormal mixture on the original scale.

References
----------
Bishop, C.M. (1994). 'Mixture Density Networks.'
    Technical Report NCRG/94/004, Aston University.
Delong, L., Lindholm, M., Wüthrich, M.V. (2021). 'Gamma Mixture Density
    Networks and their application to modelling insurance claim amounts.'
    Insurance: Mathematics and Economics.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


_LOG_SQRT_2PI = 0.5 * torch.log(torch.tensor(2.0 * 3.14159265358979323846))
_LOG_SQRT_2PI_VALUE = 0.9189385332046727  # float constant


def _log_normal_pdf(
    log_y: torch.Tensor,      # (batch,)
    mu: torch.Tensor,          # (batch, K)
    sigma: torch.Tensor,       # (batch, K)
    sigma_floor: float = 1e-4,
) -> torch.Tensor:
    """
    Log-PDF of Normal(mu, sigma^2) evaluated at log_y.

    Parameters
    ----------
    log_y : (batch,) — log-transformed observed severities.
    mu : (batch, K) — component means in log-space.
    sigma : (batch, K) — component standard deviations (positive).
    sigma_floor : float
        Lower bound on sigma to prevent division-by-zero / infinite NLL.

    Returns
    -------
    log_pdf : torch.Tensor, shape (batch, K)
        Log-density of each component at the observation.
    """
    sigma = sigma.clamp(min=sigma_floor)
    z = (log_y.unsqueeze(1) - mu) / sigma  # (batch, K)
    log_pdf = -0.5 * z ** 2 - torch.log(sigma) - _LOG_SQRT_2PI_VALUE
    return log_pdf


def mdn_nll_loss(
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    y: torch.Tensor,
    sigma_floor: float = 1e-4,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Negative log-likelihood of a lognormal mixture, computed via log-sum-exp.

    The mixture operates on log(y): each component is Normal(μ_k, σ_k²) in
    log-space. The NLL for observation i is:

        -log Σ_k π_k · φ(log(y_i); μ_k, σ_k)  - log(y_i)

    The ``-log(y_i)`` term arises from the Jacobian of the log transformation
    (change-of-variables from log-scale density to original-scale density).
    Including it makes NLL comparable across datasets and allows proper
    calibration — omitting it biases the loss but does not affect parameter
    estimation (since log(y_i) does not depend on network parameters).

    The log-sum-exp formulation avoids numerical issues:
        log Σ_k exp(log π_k + log φ_k)
    is computed via torch's built-in logsumexp for numerical stability.

    Parameters
    ----------
    pi : torch.Tensor, shape (batch, K)
        Mixing weights (sum to 1 along dim=1).
    mu : torch.Tensor, shape (batch, K)
        Component means in log-space.
    sigma : torch.Tensor, shape (batch, K)
        Component standard deviations (positive).
    y : torch.Tensor, shape (batch,)
        Observed severities (raw, positive — log transform applied internally).
    sigma_floor : float
        Minimum sigma to prevent infinite NLL from collapsed components.
    weights : torch.Tensor, shape (batch,), optional
        Observation-level weights (e.g. exposure). If None, equal weights.

    Returns
    -------
    torch.Tensor — scalar mean NLL.
    """
    eps = 1e-8
    log_y = torch.log(y.clamp(min=eps))

    log_pi = torch.log(pi.clamp(min=eps))              # (batch, K)
    log_phi = _log_normal_pdf(log_y, mu, sigma, sigma_floor)  # (batch, K)

    # log mixture density in log-space: log Σ π_k φ_k = logsumexp(log π_k + log φ_k)
    log_mix = torch.logsumexp(log_pi + log_phi, dim=1)  # (batch,)

    # Jacobian: density on original scale = density on log scale / y
    # => log p_y = log p_{log y} - log y
    log_lik = log_mix - log_y  # (batch,)

    nll = -log_lik  # (batch,)

    if weights is not None:
        nll = nll * weights / weights.sum() * len(nll)

    return nll.mean()


def mdn_log_prob(
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    y: torch.Tensor,
    sigma_floor: float = 1e-4,
) -> torch.Tensor:
    """
    Log-probability (original-scale density) for each observation.

    Same computation as the NLL loss but returns per-observation log-probs
    rather than a scalar mean. Useful for scoring and calibration testing.

    Parameters
    ----------
    pi : (batch, K) mixing weights.
    mu : (batch, K) component means in log-space.
    sigma : (batch, K) component standard deviations.
    y : (batch,) observed severities (positive).
    sigma_floor : float

    Returns
    -------
    log_prob : torch.Tensor, shape (batch,)
    """
    eps = 1e-8
    log_y = torch.log(y.clamp(min=eps))
    log_pi = torch.log(pi.clamp(min=eps))
    log_phi = _log_normal_pdf(log_y, mu, sigma, sigma_floor)
    log_mix = torch.logsumexp(log_pi + log_phi, dim=1)
    return log_mix - log_y


def mixture_mean(
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """
    Expected value of the lognormal mixture on the original scale.

    For a lognormal component with log-space parameters (μ_k, σ_k):
        E[Y_k] = exp(μ_k + σ_k²/2)

    The mixture mean is:
        E[Y] = Σ_k π_k · exp(μ_k + σ_k²/2)

    Parameters
    ----------
    pi : (batch, K) or (K,)
    mu : (batch, K) or (K,)
    sigma : (batch, K) or (K,)

    Returns
    -------
    torch.Tensor — shape (batch,) or scalar.
    """
    component_means = torch.exp(mu + 0.5 * sigma ** 2)  # (batch, K)
    return (pi * component_means).sum(dim=-1)            # (batch,)


def mixture_quantile(
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    q: float,
    n_grid: int = 1000,
    y_lo: float | None = None,
    y_hi: float | None = None,
) -> torch.Tensor:
    """
    Quantile of the lognormal mixture via CDF inversion on a log-space grid.

    Uses a fixed grid from ``y_lo`` to ``y_hi`` (defaults to exp(mu_min - 5σ)
    to exp(mu_max + 5σ)), evaluates the mixture CDF at each grid point, and
    interpolates.

    This is approximate (grid-based) but accurate to within a few basis points
    for reasonable grid sizes.

    Parameters
    ----------
    pi : (batch, K) mixing weights.
    mu : (batch, K) component means in log-space.
    sigma : (batch, K) component standard deviations.
    q : float
        Quantile level in (0, 1).
    n_grid : int
        Number of grid points for CDF inversion. Default: 1000.
    y_lo : float, optional
        Lower bound of the grid (original scale). Defaults to a safe lower
        bound based on component parameters.
    y_hi : float, optional
        Upper bound of the grid (original scale).

    Returns
    -------
    torch.Tensor, shape (batch,) or scalar.
    """
    device = pi.device
    squeeze = pi.dim() == 1
    if squeeze:
        pi = pi.unsqueeze(0)
        mu = mu.unsqueeze(0)
        sigma = sigma.unsqueeze(0)

    batch = pi.shape[0]

    # Grid bounds from component parameters
    if y_lo is None:
        lo_log = float((mu - 5.0 * sigma).min().item())
        y_lo = max(float(torch.exp(torch.tensor(lo_log)).item()), 1e-6)
    if y_hi is None:
        hi_log = float((mu + 5.0 * sigma).max().item())
        y_hi = float(torch.exp(torch.tensor(hi_log)).item())

    # Evaluate CDF on log-space grid (finer resolution where it matters)
    log_grid = torch.linspace(
        float(torch.log(torch.tensor(y_lo)).item()),
        float(torch.log(torch.tensor(y_hi)).item()),
        n_grid,
        device=device,
    )  # (n_grid,)

    # For each observation and each grid point: CDF = Σ_k π_k Φ((log_y - μ_k)/σ_k)
    # log_grid: (n_grid,), mu: (batch, K) -> broadcast
    # z: (batch, n_grid, K)
    log_grid_bc = log_grid.view(1, n_grid, 1)       # (1, n_grid, 1)
    mu_bc = mu.unsqueeze(1)                           # (batch, 1, K)
    sigma_bc = sigma.unsqueeze(1).clamp(min=1e-4)    # (batch, 1, K)
    z = (log_grid_bc - mu_bc) / sigma_bc             # (batch, n_grid, K)

    from torch.distributions import Normal
    std_normal = Normal(0.0, 1.0)
    cdf_components = std_normal.cdf(z)               # (batch, n_grid, K)

    pi_bc = pi.unsqueeze(1)                          # (batch, 1, K)
    mix_cdf = (pi_bc * cdf_components).sum(dim=2)    # (batch, n_grid)

    # Find first grid index where CDF >= q (per observation)
    q_tensor = torch.tensor(q, device=device)
    idx = (mix_cdf >= q_tensor).float().argmax(dim=1).clamp(0, n_grid - 1)  # (batch,)
    result = torch.exp(log_grid[idx])               # (batch,)

    if squeeze:
        result = result.squeeze(0)
    return result


def mixture_cdf(
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    y_grid: torch.Tensor,
) -> torch.Tensor:
    """
    CDF of the lognormal mixture at each point in ``y_grid``.

    Parameters
    ----------
    pi : (batch, K) or (K,) mixing weights.
    mu : (batch, K) or (K,) component means in log-space.
    sigma : (batch, K) or (K,) component standard deviations.
    y_grid : (m,) grid of positive y values (original scale).

    Returns
    -------
    torch.Tensor, shape (batch, m) or (m,).
    """
    device = pi.device
    squeeze = pi.dim() == 1
    if squeeze:
        pi = pi.unsqueeze(0)
        mu = mu.unsqueeze(0)
        sigma = sigma.unsqueeze(0)

    log_grid = torch.log(y_grid.clamp(min=1e-8)).to(device)  # (m,)

    # z: (batch, m, K)
    log_grid_bc = log_grid.view(1, -1, 1)
    mu_bc = mu.unsqueeze(1)
    sigma_bc = sigma.unsqueeze(1).clamp(min=1e-4)
    z = (log_grid_bc - mu_bc) / sigma_bc

    from torch.distributions import Normal
    std_normal = Normal(0.0, 1.0)
    cdf_components = std_normal.cdf(z)   # (batch, m, K)

    pi_bc = pi.unsqueeze(1)
    mix_cdf = (pi_bc * cdf_components).sum(dim=2)  # (batch, m)

    if squeeze:
        mix_cdf = mix_cdf.squeeze(0)
    return mix_cdf
