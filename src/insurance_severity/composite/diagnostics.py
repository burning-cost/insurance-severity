"""
Diagnostic tools for composite severity models.

These functions are deliberately decoupled from the model classes so they
can be used with any fitted model object, including custom subclasses.

Functions
---------
quantile_residuals     — randomized quantile residuals (Dunn & Smyth 1996)
mean_excess_plot       — empirical mean excess plot with threshold guidance
density_overlay_plot   — fitted density overlaid on empirical histogram
qq_plot                — Q-Q plot against fitted composite quantiles
ilf_comparison_plot    — compare computed ILF to market benchmarks
"""

from __future__ import annotations

from typing import Optional, Union
import warnings

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Quantile residuals
# ---------------------------------------------------------------------------


def quantile_residuals(model, y: np.ndarray) -> np.ndarray:
    """
    Compute randomized quantile residuals.

    For a continuous distribution Y with CDF F, the transform U = F(Y)
    is uniform on [0,1], and Phi^{-1}(U) is standard normal.

    For a well-specified model, residuals should be iid N(0,1).
    Systematic patterns indicate misspecification:
    - S-curve in QQ plot: wrong tail shape
    - Bimodality: two unmodeled populations
    - Heteroskedasticity by covariate: missing interaction

    Parameters
    ----------
    model : fitted CompositeSeverityModel
    y : array-like

    Returns
    -------
    residuals : ndarray of shape (n,)
    """
    y = np.asarray(y, dtype=float)
    p = model.cdf(y)
    p = np.clip(p, 1e-10, 1.0 - 1e-10)
    return stats.norm.ppf(p)


# ---------------------------------------------------------------------------
# Mean excess plot
# ---------------------------------------------------------------------------


def mean_excess_plot(y: np.ndarray, ax=None, max_quantile: float = 0.98):
    """
    Empirical mean excess loss plot.

    The mean excess function e(u) = E[X - u | X > u] is linear in u
    for GPD and exponential tails (with different slopes). A plot of
    empirical e(u) against u helps identify the tail behavior and
    choose a threshold:

    - GPD (xi > 0): e(u) is linearly increasing
    - Exponential: e(u) is roughly constant
    - GPD (xi < 0): e(u) is linearly decreasing (bounded support)

    The breakpoint where behavior changes from non-GPD to GPD is a
    natural threshold candidate.

    Parameters
    ----------
    y : array-like
    ax : matplotlib Axes, optional
    max_quantile : float
        Upper quantile cutoff to avoid instability in the upper tail.

    Returns
    -------
    ax : matplotlib Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    y = np.sort(np.asarray(y, dtype=float))
    cutoff = np.quantile(y, max_quantile)
    y_plot = y[y < cutoff]

    thresholds = np.quantile(y_plot, np.linspace(0.05, 0.95, 100))
    thresholds = np.unique(thresholds)
    mex = []
    counts = []

    for u in thresholds:
        excesses = y[y > u] - u
        if len(excesses) > 5:
            mex.append(np.mean(excesses))
            counts.append(len(excesses))
        else:
            mex.append(np.nan)
            counts.append(0)

    mex = np.array(mex)
    counts = np.array(counts)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))

    mask = ~np.isnan(mex)
    ax.plot(thresholds[mask], mex[mask], "b-", lw=1.5)
    ax.fill_between(
        thresholds[mask],
        mex[mask] - 1.96 * mex[mask] / np.sqrt(counts[mask]),
        mex[mask] + 1.96 * mex[mask] / np.sqrt(counts[mask]),
        alpha=0.2,
        color="blue",
        label="95% CI",
    )
    ax.set_xlabel("Threshold u")
    ax.set_ylabel("Mean excess e(u) = E[X-u | X>u]")
    ax.set_title("Mean Excess Plot — threshold guidance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


# ---------------------------------------------------------------------------
# Density overlay plot
# ---------------------------------------------------------------------------


def density_overlay_plot(
    model,
    y: np.ndarray,
    ax=None,
    log_scale: bool = True,
    n_points: int = 500,
    title: Optional[str] = None,
):
    """
    Overlay fitted composite density on empirical histogram.

    Parameters
    ----------
    model : fitted CompositeSeverityModel
    y : array-like
    ax : matplotlib Axes, optional
    log_scale : bool
        Use log scale on y-axis for better tail visualization.
    n_points : int
    title : str, optional

    Returns
    -------
    ax : matplotlib Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    y = np.asarray(y, dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    # Histogram
    ax.hist(y, bins=60, density=True, alpha=0.4, color="steelblue", label="Data")

    # Fitted density
    x_min = np.percentile(y, 0.5)
    x_max = np.percentile(y, 99.5)
    x_plot = np.linspace(x_min, x_max, n_points)
    dens = model.pdf(x_plot)

    ax.plot(x_plot, dens, "r-", lw=2, label="Fitted composite")

    # Threshold line
    t = model.threshold_
    ax.axvline(t, ls="--", color="k", alpha=0.7, label=f"Threshold = {t:,.0f}")

    # Body and tail densities separately
    body_dens = np.zeros(n_points)
    tail_dens = np.zeros(n_points)
    mask_body = x_plot <= t
    mask_tail = x_plot > t
    if np.any(mask_body):
        body_dens[mask_body] = model.pi_ * np.exp(
            model._body.logpdf(x_plot[mask_body], t)
        )
    if np.any(mask_tail):
        tail_dens[mask_tail] = (1 - model.pi_) * np.exp(
            model._tail.logpdf(x_plot[mask_tail], t)
        )

    ax.plot(x_plot, body_dens, "g--", lw=1, alpha=0.7, label="Body component")
    ax.plot(x_plot, tail_dens, "m--", lw=1, alpha=0.7, label="Tail component")

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Claim amount")
    ax.set_ylabel("Density")
    ax.set_title(title or f"Fitted {type(model).__name__}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    return ax


# ---------------------------------------------------------------------------
# QQ plot
# ---------------------------------------------------------------------------


def qq_plot(
    model,
    y: np.ndarray,
    ax=None,
    title: Optional[str] = None,
    n_quantiles: int = 200,
):
    """
    Q-Q plot of data quantiles against fitted model quantiles.

    Points near the 45-degree line indicate good fit. Systematic
    curvature identifies tail misspecification.

    Parameters
    ----------
    model : fitted CompositeSeverityModel
    y : array-like
    ax : matplotlib Axes, optional
    title : str, optional
    n_quantiles : int

    Returns
    -------
    ax : matplotlib Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    y = np.sort(np.asarray(y, dtype=float))
    n = len(y)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    # Plotting positions (Hazen / midpoint)
    probs = (np.arange(1, n + 1) - 0.5) / n
    # Subsample for large n
    if n > n_quantiles:
        idx = np.round(np.linspace(0, n - 1, n_quantiles)).astype(int)
        y_plot = y[idx]
        probs_plot = probs[idx]
    else:
        y_plot = y
        probs_plot = probs

    theoretical = model.ppf(probs_plot)

    ax.scatter(theoretical, y_plot, s=10, alpha=0.6, color="steelblue")

    # Reference line
    lims = [min(theoretical.min(), y_plot.min()), max(theoretical.max(), y_plot.max())]
    ax.plot(lims, lims, "r--", lw=1.5, label="y = x")

    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Empirical quantiles")
    ax.set_title(title or f"Q-Q Plot: {type(model).__name__}")
    ax.legend()
    ax.grid(True, alpha=0.2)
    return ax


# ---------------------------------------------------------------------------
# Quantile residual QQ plot
# ---------------------------------------------------------------------------


def residual_qq_plot(model, y: np.ndarray, ax=None):
    """
    Normal Q-Q plot of quantile residuals.

    Under correct specification, residuals ~ N(0,1).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    resid = quantile_residuals(model, y)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    stats.probplot(resid, plot=ax)
    ax.set_title(f"Quantile Residuals Q-Q: {type(model).__name__}")
    return ax


# ---------------------------------------------------------------------------
# ILF plot
# ---------------------------------------------------------------------------


def ilf_comparison_plot(
    model,
    limits: list,
    basic_limit: float,
    market_ilf: Optional[np.ndarray] = None,
    ax=None,
):
    """
    Plot computed ILF schedule with optional market benchmark.

    Parameters
    ----------
    model : fitted CompositeSeverityModel
    limits : list of float
    basic_limit : float
    market_ilf : array-like, optional
        Market benchmark ILF values (same length as limits).
    ax : matplotlib Axes, optional

    Returns
    -------
    ax : matplotlib Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    limits = sorted(limits)
    ilf_values = [model.ilf(l, basic_limit) for l in limits]

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(limits, ilf_values, "b-o", lw=2, label="Model ILF", markersize=6)

    if market_ilf is not None:
        market_ilf = np.asarray(market_ilf)
        ax.plot(limits, market_ilf, "r--s", lw=1.5, label="Market benchmark", markersize=6)

    ax.set_xlabel("Policy limit")
    ax.set_ylabel("ILF")
    ax.set_title(f"Increased Limit Factors: {type(model).__name__}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(1.0, ls=":", color="gray")
    return ax
