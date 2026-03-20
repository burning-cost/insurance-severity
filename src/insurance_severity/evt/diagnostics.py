"""
EVT diagnostic plots for threshold selection and tail assessment.

These are the primary visual tools for deciding:
1. Whether the data has a heavy tail (Pareto QQ plot)
2. Where the tail begins (mean excess plot, threshold stability plot)
3. What the tail index is (Hill plot)

All functions follow a consistent API: they accept data and return a matplotlib
Axes object, which the caller can further customise.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def hill_plot(
    x: np.ndarray,
    delta: Optional[np.ndarray] = None,
    k_min: int = 10,
    k_max: Optional[int] = None,
    ax=None,
) -> object:
    """
    Hill plot: xi estimate vs k.

    If delta provided, uses censored Hill (simple method).
    Otherwise uses standard Hill estimator.

    Parameters
    ----------
    x : array of positive claim amounts
    delta : censoring indicators (1=uncensored, 0=censored), optional
    k_min : minimum k for plot
    k_max : maximum k (default n//5)
    ax : matplotlib Axes or None

    Returns
    -------
    ax : matplotlib Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install insurance-severity[plotting]"
        )

    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        raise ValueError("All x values must be strictly positive")

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    if delta is not None:
        from insurance_severity.evt.censored_hill import CensoredHillEstimator
        delta = np.asarray(delta, dtype=float)
        est = CensoredHillEstimator(method="simple", k_min=k_min, k_max=k_max)
        est.fit(x, delta)
        est.hill_plot(ax=ax)
        ax.set_title("Censored Hill Plot")
    else:
        # Standard Hill estimator
        n = len(x)
        k_max_val = k_max if k_max is not None else max(n // 5, k_min + 1)
        k_max_val = min(k_max_val, n - 1)
        k_grid = np.arange(k_min, k_max_val + 1)

        # Sort descending
        x_sorted = np.sort(x)[::-1]

        xi_k = np.zeros(len(k_grid))
        for idx, k in enumerate(k_grid):
            # Hill estimator: (1/k) sum_{j=1}^{k} log(X_{(n-j+1)} / X_{(n-k)})
            log_ratios = np.log(x_sorted[:k] / x_sorted[k])
            xi_k[idx] = float(np.mean(log_ratios))

        ax.plot(k_grid, xi_k, "b-", lw=1.5, label=r"$\hat{\xi}_k$ (Hill)")
        ax.axhline(0, ls=":", color="gray", alpha=0.5)
        ax.set_xlabel("k (number of upper order statistics)")
        ax.set_ylabel(r"$\hat{\xi}$ (EVI estimate)")
        ax.set_title("Hill Plot")
        ax.legend()
        ax.grid(True, alpha=0.3)

    return ax


def mean_excess_censored(
    z: np.ndarray,
    delta: np.ndarray,
    ax=None,
) -> object:
    """
    KM-adjusted mean excess function.

    e_hat^KM(u) = integral_{u}^{inf} (1 - F_hat^KM(x)) dx / (1 - F_hat^KM(u))

    Useful for threshold selection with censored data. A linear increasing
    mean excess function is characteristic of a heavy Pareto tail.

    Parameters
    ----------
    z : array of observed values
    delta : censoring indicators (1=uncensored, 0=censored)
    ax : matplotlib Axes or None

    Returns
    -------
    ax : matplotlib Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install insurance-severity[plotting]"
        )

    z = np.asarray(z, dtype=float)
    delta = np.asarray(delta, dtype=float)

    from insurance_severity.evt.censored_hill import CensoredHillEstimator
    est = CensoredHillEstimator()
    z_unique, km_sf = est._kaplan_meier(z, delta)

    # Compute KM CDF: F_hat = 1 - km_sf
    # Mean excess at u: integral from u to inf of (1-F) dx / (1-F(u))
    # Discrete approximation using trapezoid

    n_pts = min(len(z_unique), 100)
    # Use a subset of quantile points for the x-axis
    q_vals = np.linspace(0.05, 0.90, n_pts)
    u_vals = np.quantile(z, q_vals)
    me_vals = np.zeros(n_pts)

    for i, u in enumerate(u_vals):
        # Find km_sf at u
        idx_u = np.searchsorted(z_unique, u, side='right') - 1
        idx_u = max(0, min(idx_u, len(km_sf) - 1))
        sf_u = km_sf[idx_u]
        if sf_u <= 0:
            me_vals[i] = np.nan
            continue

        # Integrate sf from u onwards using trapezoidal rule
        mask = z_unique >= u
        z_tail = z_unique[mask]
        sf_tail = km_sf[mask]

        if len(z_tail) < 2:
            me_vals[i] = np.nan
            continue

        integral = float(np.trapz(sf_tail, z_tail))
        me_vals[i] = integral / sf_u

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    valid = np.isfinite(me_vals)
    ax.plot(u_vals[valid], me_vals[valid], "b-o", markersize=3, lw=1.5)
    ax.set_xlabel("Threshold u")
    ax.set_ylabel("Mean excess e(u)")
    ax.set_title("KM-Adjusted Mean Excess Plot")
    ax.grid(True, alpha=0.3)

    return ax


def threshold_stability_plot(
    x: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    limits: Optional[np.ndarray] = None,
    n_bootstrap: int = 99,
    ax=None,
) -> object:
    """
    xi estimate vs threshold with bootstrap CI bands.

    Uses TruncatedGPD if limits provided, standard GPD otherwise.

    Parameters
    ----------
    x : claim amounts
    thresholds : array of thresholds to test (default: 10th–90th percentile grid)
    limits : policy limits array (optional, same length as x)
    n_bootstrap : number of bootstrap samples for CI bands
    ax : matplotlib Axes or None

    Returns
    -------
    ax : matplotlib Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install insurance-severity[plotting]"
        )

    x = np.asarray(x, dtype=float)

    if thresholds is None:
        thresholds = np.quantile(x, np.linspace(0.50, 0.90, 20))
        thresholds = np.unique(thresholds)

    from insurance_severity.evt.truncated_gpd import TruncatedGPD
    import warnings

    xi_vals = []
    xi_lower = []
    xi_upper = []
    valid_thresholds = []

    rng = np.random.default_rng(42)

    for u in thresholds:
        x_above = x[x > u]
        if len(x_above) < 20:
            continue

        lim_above = None
        if limits is not None:
            limits_arr = np.asarray(limits, dtype=float)
            lim_above = limits_arr[x > u]
            if np.any(lim_above <= u):
                lim_above = lim_above[lim_above > u]
                x_above = x_above[limits_arr[x > u] > u]
            if len(x_above) < 20:
                continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = TruncatedGPD()
                _, info = model.fit_mle(x_above, u, limits=lim_above)
                xi_hat = model._xi
        except Exception:
            continue

        # Bootstrap CI
        xi_boot = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, len(x_above), size=len(x_above))
            x_b = x_above[idx]
            lim_b = lim_above[idx] if lim_above is not None else None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m_b = TruncatedGPD()
                    m_b.fit_mle(x_b, u, limits=lim_b)
                    xi_boot.append(m_b._xi)
            except Exception:
                pass

        xi_vals.append(xi_hat)
        valid_thresholds.append(u)
        if len(xi_boot) > 10:
            xi_lower.append(float(np.percentile(xi_boot, 2.5)))
            xi_upper.append(float(np.percentile(xi_boot, 97.5)))
        else:
            xi_lower.append(xi_hat - 0.1)
            xi_upper.append(xi_hat + 0.1)

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    if len(valid_thresholds) > 0:
        t_arr = np.array(valid_thresholds)
        xi_arr = np.array(xi_vals)
        ax.plot(t_arr, xi_arr, "b-o", markersize=4, lw=1.5, label=r"$\hat{\xi}$(threshold)")
        ax.fill_between(t_arr, xi_lower, xi_upper, alpha=0.2, color="blue", label="95% bootstrap CI")

    ax.axhline(0, ls=":", color="gray", alpha=0.5)
    ax.set_xlabel("Threshold u")
    ax.set_ylabel(r"$\hat{\xi}$")
    title = "Threshold Stability Plot"
    if limits is not None:
        title += " (TruncatedGPD with limits)"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def pareto_qq_plot(
    x: np.ndarray,
    delta: Optional[np.ndarray] = None,
    k: Optional[int] = None,
    ax=None,
) -> object:
    """
    Pareto QQ plot for threshold selection.

    If delta provided, uses KM adjustment for censored data.
    A straight line indicates Pareto-type tail.

    Parameters
    ----------
    x : positive claim amounts
    delta : censoring indicators (1=uncensored, 0=censored), optional
    k : number of upper order statistics to use (default: top 50%)
    ax : matplotlib Axes or None

    Returns
    -------
    ax : matplotlib Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install insurance-severity[plotting]"
        )

    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        raise ValueError("All x values must be strictly positive")

    n = len(x)
    if k is None:
        k = n // 2

    # Sort descending
    order = np.argsort(x)[::-1]
    x_sorted = x[order]

    if delta is not None:
        delta_sorted = np.asarray(delta, dtype=float)[order]
        # KM-adjusted Pareto QQ plot
        from insurance_severity.evt.censored_hill import CensoredHillEstimator
        est = CensoredHillEstimator()
        z_unique, km_sf = est._kaplan_meier(x, delta)

        # Theoretical quantiles from Pareto: -log(S_hat(x_j))
        # Empirical: log(x_j)
        log_x_vals = []
        neg_log_sf_vals = []

        for j in range(min(k, n)):
            xj = x_sorted[j]
            idx = np.searchsorted(z_unique, xj, side='right') - 1
            idx = max(0, min(idx, len(km_sf) - 1))
            sf_val = km_sf[idx]
            if sf_val <= 0:
                continue
            log_x_vals.append(np.log(xj))
            neg_log_sf_vals.append(-np.log(sf_val))

        log_x_arr = np.array(log_x_vals)
        neg_log_sf_arr = np.array(neg_log_sf_vals)
        xlabel = "-log(1 - F_KM(x)) [Pareto quantile]"
        ylabel = "log(x) [empirical]"
        title = "Pareto QQ Plot (KM-adjusted for censoring)"
    else:
        # Standard Pareto QQ: plot log(x_{(j)}) vs -log(j/n)
        # Use top-k
        log_x_arr = np.log(x_sorted[:k])
        j_arr = np.arange(1, k + 1)
        neg_log_sf_arr = -np.log(j_arr / (n + 1.0))
        xlabel = "-log(j/(n+1)) [Pareto quantile]"
        ylabel = "log(x_{(j)}) [empirical]"
        title = "Pareto QQ Plot"

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(neg_log_sf_arr, log_x_arr, s=10, alpha=0.6, color="steelblue", label="Data")

    # Fit reference line
    if len(neg_log_sf_arr) > 2:
        coeffs = np.polyfit(neg_log_sf_arr, log_x_arr, 1)
        x_ref = np.linspace(neg_log_sf_arr.min(), neg_log_sf_arr.max(), 100)
        ax.plot(x_ref, np.polyval(coeffs, x_ref), "r-", lw=1.5,
                label=f"Reference line (slope={coeffs[0]:.3f} = 1/alpha_hat)")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax
