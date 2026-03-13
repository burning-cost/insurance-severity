"""
Loss functions for DRN training.

Primary loss: JBCE (Joint Binary Cross-Entropy).
Optional regularisation: KL divergence, mean penalty, total variation, discrete variation.

The JBCE loss is the key methodological contribution of Avanzi et al. (2024).
It evaluates the model by treating each cutpoint as a binary classification:
does the observation fall at or below c_k? This is more stable than NLL for
histogram-based models, because it evaluates the CDF (well-defined everywhere)
rather than the PDF (undefined at bin boundaries).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def jbce_loss(
    cdf_at_cutpoints: torch.Tensor,
    y_indicators: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Joint Binary Cross-Entropy loss.

    L_JBCE = (1/nK) * sum_i sum_k BCE(F_DRN(c_k | x_i), 1(y_i <= c_k))

    Parameters
    ----------
    cdf_at_cutpoints : torch.Tensor, shape (n, K)
        Model-predicted CDF at each of the K interior cutpoints.
        (Excluding c_0 which has CDF=baseline_cdf_c0, and c_K which has CDF~1.)
        Values must be in (0, 1).
    y_indicators : torch.Tensor, shape (n, K)
        Binary indicators: 1 if y_i <= c_k, else 0.
        Pre-computed from training y and cutpoints.
    weights : torch.Tensor, shape (n,), optional
        Observation weights (e.g. exposure). If None, uniform weights.

    Returns
    -------
    torch.Tensor, scalar
        Mean JBCE loss.

    Notes
    -----
    BCE is computed with logits for numerical stability:
        BCE(p, t) = -t*log(p) - (1-t)*log(1-p)
    We clamp p to (eps, 1-eps) to avoid log(0).
    """
    eps = 1e-7
    p = torch.clamp(cdf_at_cutpoints, eps, 1.0 - eps)
    t = y_indicators.float()

    # BCE per (observation, cutpoint) pair
    bce = -t * torch.log(p) - (1.0 - t) * torch.log(1.0 - p)  # (n, K)

    if weights is not None:
        # Weight each observation; normalise by sum of weights
        w = weights.unsqueeze(1)  # (n, 1)
        loss = (bce * w).sum() / (weights.sum() * bce.shape[1])
    else:
        loss = bce.mean()

    return loss


def drn_regularisation(
    drn_pmf: torch.Tensor,
    baseline_probs: torch.Tensor,
    kl_alpha: float = 0.0,
    mean_alpha: float = 0.0,
    tv_alpha: float = 0.0,
    dv_alpha: float = 0.0,
    kl_direction: str = "forwards",
    bin_midpoints: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    DRN regularisation penalty.

    Combines four optional terms:
    1. KL divergence (kl_alpha): penalises deviation from baseline distribution
    2. Mean alpha (mean_alpha): forces DRN mean close to baseline mean
    3. Total variation (tv_alpha): penalises jagged density
    4. Discrete variation (dv_alpha): penalises non-smooth density curvature

    Parameters
    ----------
    drn_pmf : torch.Tensor, shape (n, K)
        DRN bin probabilities (after softmax).
    baseline_probs : torch.Tensor, shape (n, K)
        Baseline bin probabilities.
    kl_alpha : float
        Weight for KL divergence term.
    mean_alpha : float
        Weight for mean consistency term.
    tv_alpha : float
        Weight for total variation term.
    dv_alpha : float
        Weight for discrete variation (roughness) term.
    kl_direction : str
        'forwards' = KL(baseline || DRN), 'reverse' = KL(DRN || baseline).
    bin_midpoints : torch.Tensor, shape (K,), optional
        Required if mean_alpha > 0. Midpoints of histogram bins.

    Returns
    -------
    torch.Tensor, scalar
        Total regularisation penalty.
    """
    reg = torch.tensor(0.0, device=drn_pmf.device, dtype=drn_pmf.dtype)

    # Numerical stability
    eps = 1e-8
    p = drn_pmf + eps
    b = baseline_probs + eps

    # 1. KL divergence
    if kl_alpha > 0:
        if kl_direction == "forwards":
            # KL(baseline || DRN) = sum_k b_k * log(b_k / p_k) = -sum_k b_k * log(a_k)
            kl = (b * (torch.log(b) - torch.log(p))).sum(dim=1).mean()
        else:
            # KL(DRN || baseline) = sum_k p_k * log(p_k / b_k)
            kl = (p * (torch.log(p) - torch.log(b))).sum(dim=1).mean()
        reg = reg + kl_alpha * kl

    # 2. Mean consistency: ||E_DRN[Y] - E_baseline[Y]||^2
    if mean_alpha > 0 and bin_midpoints is not None:
        m = bin_midpoints  # (K,)
        mean_drn = (drn_pmf * m.unsqueeze(0)).sum(dim=1)       # (n,)
        mean_base = (baseline_probs * m.unsqueeze(0)).sum(dim=1) # (n,)
        mean_penalty = ((mean_drn - mean_base) ** 2).mean()
        reg = reg + mean_alpha * mean_penalty

    # 3. Total variation: sum_k |p_k - p_{k-1}|
    if tv_alpha > 0:
        tv = (drn_pmf[:, 1:] - drn_pmf[:, :-1]).abs().sum(dim=1).mean()
        reg = reg + tv_alpha * tv

    # 4. Discrete variation (roughness): sum_k |p_{k+1} - 2*p_k + p_{k-1}|
    if dv_alpha > 0:
        second_diff = drn_pmf[:, 2:] - 2.0 * drn_pmf[:, 1:-1] + drn_pmf[:, :-2]
        dv = second_diff.abs().sum(dim=1).mean()
        reg = reg + dv_alpha * dv

    return reg


def nll_loss(
    drn_pmf: torch.Tensor,
    bin_indices: torch.Tensor,
    bin_widths: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Negative log-likelihood loss for histogram distribution.

    L_NLL = -mean_i log(p_k(x_i) / bin_width_k)

    Less numerically stable than JBCE when bins have varying widths.
    Provided as an alternative for research comparison.

    Parameters
    ----------
    drn_pmf : torch.Tensor, shape (n, K)
        DRN bin probabilities.
    bin_indices : torch.Tensor, shape (n,)
        Index of the bin containing y_i for each observation.
    bin_widths : torch.Tensor, shape (K,)
        Width of each bin.
    weights : torch.Tensor, shape (n,), optional

    Returns
    -------
    torch.Tensor, scalar
    """
    n = drn_pmf.shape[0]
    # Gather the probability for the bin containing each y_i
    bin_idx = bin_indices.long()
    p_k = drn_pmf[torch.arange(n), bin_idx]     # (n,)
    w_k = bin_widths[bin_idx]                    # (n,)

    eps = 1e-8
    log_density = torch.log(p_k + eps) - torch.log(w_k + eps)

    if weights is not None:
        loss = -(log_density * weights).sum() / weights.sum()
    else:
        loss = -log_density.mean()
    return loss
