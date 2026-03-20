"""
CensoredHillEstimator: Hill estimator corrected for IBNR right-censoring.

Open/developing claims are right-censored at their current development value.
Standard Hill treats them as fully observed, producing downward-biased xi.
This estimator corrects via the proportion of uncensored observations in the
top-k order statistics (simple method) or Kaplan-Meier reweighting (KM-Worms).

References:
- Einmahl, Fils-Villetard & Guillou (2008), Bernoulli 14(1): 207-227.
- Beirlant, Guillou, Dierckx & Fils-Villetard (2007), Extremes 10: 151-174.
- Albrecher, Beirlant & Teugels (2025), arXiv:2511.22272.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np


class CensoredHillEstimator:
    """
    Hill estimator for the extreme value index corrected for right-censoring.

    Implements:
    1. Simple censored Hill (Einmahl et al. 2008): H_k / p_k where p_k is
       proportion uncensored among top-k.
    2. Worms-Kaplan-Meier (KM-weighted) estimator.

    Parameters
    ----------
    method : {'simple', 'km_worms'}, default 'simple'
    k_min : int, default 10 — minimum k for Hill plot
    k_max : int or None — maximum k (default n//5)

    Attributes (after fit)
    ----------------------
    xi_hat_ : float — optimal xi estimate
    k_opt_ : int — optimal k selected
    k_grid_ : ndarray — k values used
    xi_k_ : ndarray — xi estimates at each k
    ci_lower_ : ndarray — bootstrap lower CI at each k (None until bootstrap_ci called)
    ci_upper_ : ndarray — bootstrap upper CI at each k
    """

    def __init__(
        self,
        method: str = "simple",
        k_min: int = 10,
        k_max: Optional[int] = None,
    ):
        valid_methods = {"simple", "km_worms"}
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got {method!r}")
        self.method = method
        self.k_min = k_min
        self.k_max = k_max

        # Fitted attributes
        self.xi_hat_: Optional[float] = None
        self.k_opt_: Optional[int] = None
        self.k_grid_: Optional[np.ndarray] = None
        self.xi_k_: Optional[np.ndarray] = None
        self.ci_lower_: Optional[np.ndarray] = None
        self.ci_upper_: Optional[np.ndarray] = None

        # Store for bootstrap
        self._z_fit: Optional[np.ndarray] = None
        self._delta_fit: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public fit
    # ------------------------------------------------------------------

    def fit(
        self,
        z: np.ndarray,
        delta: np.ndarray,
        k: Optional[int] = None,
    ) -> "CensoredHillEstimator":
        """
        Fit the censored Hill estimator.

        Parameters
        ----------
        z : array of observed values (claim amounts or capped values)
        delta : array of censoring indicators (1=uncensored, 0=censored)
        k : int or None
            If provided, use this k directly. Otherwise select via stability.

        Returns
        -------
        self
        """
        z = np.asarray(z, dtype=float)
        delta = np.asarray(delta, dtype=float)
        if len(z) != len(delta):
            raise ValueError("z and delta must have the same length")
        if np.any(z <= 0):
            raise ValueError("All z values must be strictly positive")

        n = len(z)
        k_max = self.k_max if self.k_max is not None else max(n // 5, self.k_min + 1)
        k_max = min(k_max, n - 1)

        # Sort descending
        order = np.argsort(z)[::-1]
        z_sorted = z[order]
        delta_sorted = delta[order]

        self._z_fit = z.copy()
        self._delta_fit = delta.copy()

        # Compute KM if needed
        km_sf: Optional[np.ndarray] = None
        if self.method == "km_worms":
            km_sf = self._kaplan_meier(z, delta)

        # Compute xi_k for each k in grid
        k_grid = np.arange(self.k_min, k_max + 1)
        xi_k = np.full(len(k_grid), np.nan)

        for idx, ki in enumerate(k_grid):
            xi_k[idx] = self._compute_xi_k(
                ki, z_sorted, delta_sorted, km_sf, z
            )

        self.k_grid_ = k_grid
        self.xi_k_ = xi_k

        # Select k
        if k is not None:
            self.k_opt_ = int(k)
            if k < self.k_min or k > k_max:
                warnings.warn(
                    f"Specified k={k} is outside [k_min={self.k_min}, k_max={k_max}]",
                    UserWarning,
                    stacklevel=2,
                )
            # Find closest k in grid for xi_hat
            idx_k = np.searchsorted(k_grid, k)
            idx_k = min(idx_k, len(k_grid) - 1)
            self.xi_hat_ = float(xi_k[idx_k]) if np.isfinite(xi_k[idx_k]) else float(np.nanmean(xi_k))
        else:
            self.k_opt_ = int(self._select_k_stability(xi_k))
            k_idx = np.searchsorted(k_grid, self.k_opt_)
            k_idx = min(k_idx, len(xi_k) - 1)
            self.xi_hat_ = float(xi_k[k_idx]) if np.isfinite(xi_k[k_idx]) else float(np.nanmean(xi_k))

        # Warn if proportion uncensored among top-k is very low
        top_k_delta = delta_sorted[:self.k_opt_]
        p_uncensored = float(np.mean(top_k_delta))
        if p_uncensored < 0.4:
            warnings.warn(
                f"Only {p_uncensored:.1%} of top-{self.k_opt_} observations are uncensored. "
                "Censoring correction may be unreliable. Consider a lower threshold or "
                "more complete data.",
                UserWarning,
                stacklevel=2,
            )

        return self

    # ------------------------------------------------------------------
    # Core estimators
    # ------------------------------------------------------------------

    def _compute_xi_k(
        self,
        k: int,
        z_sorted: np.ndarray,
        delta_sorted: np.ndarray,
        km_sf: Optional[np.ndarray],
        z_orig: Optional[np.ndarray],
    ) -> float:
        """Compute censored Hill estimate at a given k."""
        # z_sorted[0] >= z_sorted[1] >= ... (descending)
        # Z_{(n-k)} is the (k+1)-th largest, i.e., z_sorted[k]
        z_nk = z_sorted[k]  # Z_{(n-k)} threshold value
        if z_nk <= 0:
            return np.nan

        # Top-k exceedances: z_sorted[0], ..., z_sorted[k-1]
        log_ratios = np.log(z_sorted[:k] / z_nk)

        if self.method == "simple":
            # H_k^c = [mean of log ratios] / [proportion uncensored among top-k]
            p_k = float(np.mean(delta_sorted[:k]))
            if p_k <= 0:
                return np.nan
            hill_num = float(np.mean(log_ratios))
            xi = hill_num / p_k

        elif self.method == "km_worms":
            if km_sf is None or z_orig is None:
                return np.nan
            # Worms-KM: integral of log(z) against empirical tail measure / S_hat(z_{n-k})
            # Discrete approximation using order statistics
            # km_sf is evaluated at z_orig sorted values
            # We need KM values at z_sorted[:k] / z_sorted[k]
            # Approximate using the KM survival at each of the top-k points
            # normalized at z_nk
            z_unique, km_unique = km_sf

            def get_km(z_val: float) -> float:
                """Get KM survival at z_val via step function."""
                idx = np.searchsorted(z_unique, z_val, side='right') - 1
                idx = max(0, min(idx, len(km_unique) - 1))
                return float(km_unique[idx])

            km_nk = get_km(z_nk)
            if km_nk <= 0:
                return np.nan

            # Worms estimator: sum_j [S_hat(Z_{n-j+1}) / S_hat(Z_{n-k})] * log(Z_{n-j+1}/Z_{n-j})
            xi_sum = 0.0
            for j in range(k):
                z_j = z_sorted[j]
                z_j_next = z_sorted[j + 1] if j + 1 < len(z_sorted) else z_nk
                km_j = get_km(z_j)
                weight = km_j / km_nk
                xi_sum += weight * np.log(z_j / z_j_next)
            xi = xi_sum

        else:
            return np.nan

        return float(xi)

    def _apply_bias_reduction(self, xi: float, k: int, log_ratios: np.ndarray) -> float:
        """
        Second-order bias correction.

        bias_k = xi^2 * (mean_log_spacing / k) / (1 - 2*xi)

        Only applied when |bias_k| > 5% of |xi|.
        """
        if not np.isfinite(xi) or xi <= 0 or k < 10:
            return xi
        if xi >= 0.5:
            return xi  # correction denominator (1-2*xi) near zero or negative

        mean_log_spacing = float(np.mean(np.diff(log_ratios))) if len(log_ratios) > 1 else 0.0
        bias_k = xi**2 * (mean_log_spacing / k) / (1.0 - 2.0 * xi)

        if abs(bias_k) > 0.05 * abs(xi):
            xi_corrected = xi - bias_k
            # Don't over-correct
            if xi_corrected > 0:
                return float(xi_corrected)
        return float(xi)

    # ------------------------------------------------------------------
    # k selection
    # ------------------------------------------------------------------

    def _select_k_stability(self, xi_k: np.ndarray) -> int:
        """
        Select k where xi is most stable.

        Find k in [k_min, k_max] where the variance of xi over a rolling
        window is minimized. Returns the k grid value (not index) at the
        minimum variance.
        """
        window = max(10, len(xi_k) // 10)
        if len(xi_k) <= window:
            # Too few points: return midpoint
            idx = len(xi_k) // 2
            return int(self.k_grid_[idx])

        # Rolling variance
        min_var = np.inf
        best_center_idx = window // 2

        for i in range(len(xi_k) - window + 1):
            segment = xi_k[i:i + window]
            if not np.all(np.isfinite(segment)):
                continue
            var = float(np.var(segment))
            if var < min_var:
                min_var = var
                best_center_idx = i + window // 2

        return int(self.k_grid_[best_center_idx])

    # ------------------------------------------------------------------
    # Kaplan-Meier
    # ------------------------------------------------------------------

    def _kaplan_meier(
        self, z: np.ndarray, delta: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute KM survival function at each unique z value.

        Returns (z_unique, km_survival) sorted ascending.
        """
        n = len(z)
        order = np.argsort(z)
        z_s = z[order]
        d_s = delta[order]

        # Standard KM: at each unique time, update survival
        unique_z = []
        km_sf = []
        sf = 1.0

        i = 0
        while i < n:
            # Find all tied values
            z_i = z_s[i]
            j = i
            while j < n and z_s[j] == z_i:
                j += 1
            # Between i and j: n_at_risk = n - i, events = sum of delta in [i,j)
            n_at_risk = n - i
            events = int(np.sum(d_s[i:j]))
            if events > 0:
                sf *= (1.0 - events / n_at_risk)
            unique_z.append(z_i)
            km_sf.append(sf)
            i = j

        return np.array(unique_z), np.array(km_sf)

    # ------------------------------------------------------------------
    # Bootstrap CI
    # ------------------------------------------------------------------

    def bootstrap_ci(
        self,
        z: np.ndarray,
        delta: np.ndarray,
        k: int,
        n_bootstrap: int = 499,
        alpha: float = 0.05,
    ) -> tuple[float, float]:
        """
        Percentile bootstrap CI for xi at fixed k.

        Parameters
        ----------
        z : observations
        delta : censoring indicators
        k : fixed k value
        n_bootstrap : number of bootstrap samples
        alpha : CI level

        Returns
        -------
        (lower, upper) confidence interval
        """
        n = len(z)
        rng = np.random.default_rng(42)
        xi_boot = np.zeros(n_bootstrap)

        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            z_b = z[idx]
            delta_b = delta[idx]
            try:
                est = CensoredHillEstimator(
                    method=self.method, k_min=self.k_min, k_max=self.k_max
                )
                est.fit(z_b, delta_b, k=k)
                xi_boot[b] = est.xi_hat_ if est.xi_hat_ is not None else np.nan
            except Exception:
                xi_boot[b] = np.nan

        xi_boot = xi_boot[np.isfinite(xi_boot)]
        if len(xi_boot) == 0:
            return (0.0, 1.0)

        lower = float(np.percentile(xi_boot, 100.0 * alpha / 2.0))
        upper = float(np.percentile(xi_boot, 100.0 * (1.0 - alpha / 2.0)))
        return lower, upper

    # ------------------------------------------------------------------
    # Hill plot
    # ------------------------------------------------------------------

    def hill_plot(
        self,
        ax=None,
        n_bootstrap: int = 199,
        ci_alpha: float = 0.05,
    ):
        """
        Plot xi_hat vs k with bootstrap confidence bands.

        Vertical line at k_opt_ (if selected via stability).

        Requires matplotlib.
        """
        if self.k_grid_ is None or self.xi_k_ is None:
            raise RuntimeError("Call fit() before hill_plot()")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install with: pip install insurance-severity[plotting]"
            )

        if ax is None:
            _, ax = plt.subplots(figsize=(9, 5))

        ax.plot(self.k_grid_, self.xi_k_, "b-", lw=1.5, label=r"$\hat{\xi}_k$")

        if self.ci_lower_ is not None and self.ci_upper_ is not None:
            ax.fill_between(
                self.k_grid_, self.ci_lower_, self.ci_upper_,
                alpha=0.2, color="blue", label=f"{int((1-ci_alpha)*100)}% bootstrap CI"
            )

        if self.k_opt_ is not None:
            ax.axvline(self.k_opt_, ls="--", color="red", alpha=0.7,
                       label=f"k_opt = {self.k_opt_}")
        if self.xi_hat_ is not None:
            ax.axhline(self.xi_hat_, ls=":", color="orange", alpha=0.8,
                       label=f"xi_hat = {self.xi_hat_:.3f}")

        ax.set_xlabel("k (number of upper order statistics)")
        ax.set_ylabel(r"$\hat{\xi}$ (EVI estimate)")
        ax.set_title(f"Censored Hill Plot ({self.method} method)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax
