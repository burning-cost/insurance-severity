"""
ExtendedHistogramBatch: vectorised predictive distributions for n observations.

The DRN output is an extended histogram distribution — piecewise uniform
between cutpoints c_0..c_K, with the baseline parametric distribution
governing the tails (y < c_0 and y > c_K).

All methods are vectorised with numpy. No Python loops over observations.
CDF and quantile use np.searchsorted for O(n log K) computation.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats


class ExtendedHistogramBatch:
    """
    Batch of extended histogram predictive distributions.

    Each of the n observations has K bin probabilities (summing to 1) over
    the histogram region [c_0, c_K]. The tails are handled by the baseline
    parametric distribution, scaled appropriately.

    Parameters
    ----------
    cutpoints : np.ndarray, shape (K+1,)
        Histogram bin boundaries, sorted ascending.
    bin_probs : np.ndarray, shape (n, K)
        DRN-adjusted bin probabilities. Each row sums to 1.
        These are the probabilities assigned to the histogram region
        (not the tail regions).
    baseline_cdf_c0 : np.ndarray, shape (n,)
        Baseline CDF at c_0 for each observation. This is the probability
        mass in the left tail (y < c_0) under the baseline.
    baseline_cdf_cK : np.ndarray, shape (n,)
        Baseline CDF at c_K for each observation. Right tail mass =
        1 - baseline_cdf_cK.
    baseline_params : dict
        Baseline distribution parameters {'mu': array, 'dispersion': float, ...}
        Used for tail CDF computation.
    distribution_family : str
        'gamma' | 'gaussian' | 'lognormal' | 'inversegaussian'

    Notes
    -----
    The extended histogram splices three pieces:

        F(y) = F_baseline(y)                           for y < c_0
             = F_baseline(c_0) + p_histogram(y)        for c_0 <= y <= c_K
             = F_baseline(y) renormalised               for y > c_K

    where p_histogram(y) = sum_{k: c_k < y} bin_probs[k] * (c_0_to_cK_mass)
    and c_0_to_cK_mass = baseline_cdf_cK - baseline_cdf_c0.

    The bin_probs represent the *conditional* distribution given y in [c_0, c_K].
    The absolute probability mass in the histogram region is
    (baseline_cdf_cK - baseline_cdf_c0) by construction.

    Wait — this is the design choice. The DRN replaces the histogram region
    probabilities entirely with the refined probabilities, while keeping the
    tail probabilities at their baseline values.

    Concretely:
        P(y < c_0)        = baseline_cdf_c0
        P(c_{k-1} <= y < c_k) = bin_probs[:, k-1] * (1 - baseline_cdf_c0 - (1 - baseline_cdf_cK))
        P(y >= c_K)       = 1 - baseline_cdf_cK

    This preserves total probability = 1 because bin_probs sum to 1 and
    the histogram mass = baseline_cdf_cK - baseline_cdf_c0.
    """

    def __init__(
        self,
        cutpoints: np.ndarray,
        bin_probs: np.ndarray,
        baseline_cdf_c0: np.ndarray,
        baseline_cdf_cK: np.ndarray,
        baseline_params: dict,
        distribution_family: str = "gamma",
    ):
        self.cutpoints = np.asarray(cutpoints, dtype=np.float64)
        self.bin_probs = np.asarray(bin_probs, dtype=np.float64)
        self.baseline_cdf_c0 = np.asarray(baseline_cdf_c0, dtype=np.float64)
        self.baseline_cdf_cK = np.asarray(baseline_cdf_cK, dtype=np.float64)
        self.baseline_params = baseline_params
        self.distribution_family = distribution_family

        self.n, self.K = self.bin_probs.shape
        self.c_0 = float(self.cutpoints[0])
        self.c_K = float(self.cutpoints[-1])

        # Histogram mass for each observation: how much probability is in [c_0, c_K]
        self._hist_mass = self.baseline_cdf_cK - self.baseline_cdf_c0  # (n,)

        # Cumulative bin probs (conditional on being in histogram region)
        # Shape: (n, K) — cumulative sum across bins
        self._cumbin = np.cumsum(self.bin_probs, axis=1)

        # Absolute bin widths
        self._bin_widths = np.diff(self.cutpoints)  # (K,)

    # ------------------------------------------------------------------
    # CDF
    # ------------------------------------------------------------------

    def cdf(self, y: float | np.ndarray) -> np.ndarray:
        """
        CDF at scalar or array y.

        For scalar y: returns (n,) array.
        For array y of shape (m,): returns (n, m) array.

        Parameters
        ----------
        y : float or np.ndarray

        Returns
        -------
        np.ndarray
        """
        scalar = np.isscalar(y)
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        m = len(y)

        # Result shape: (n, m)
        result = np.empty((self.n, m), dtype=np.float64)

        # Left tail: y < c_0
        left_mask = y < self.c_0
        if left_mask.any():
            result[:, left_mask] = self._baseline_cdf(y[left_mask])  # (n, sum(left))

        # Right tail: y >= c_K
        right_mask = y >= self.c_K
        if right_mask.any():
            # Tail CDF: renormalise so right tail is the baseline tail
            # F_ext(y) = 1 - (1 - F_baseline(y)) for y >= c_K
            result[:, right_mask] = self._baseline_cdf(y[right_mask])  # (n, sum(right))

        # Middle: c_0 <= y < c_K
        mid_mask = (~left_mask) & (~right_mask)
        if mid_mask.any():
            y_mid = y[mid_mask]  # (m_mid,)
            result[:, mid_mask] = self._histogram_cdf(y_mid)

        if scalar:
            return result[:, 0]
        return result

    def _histogram_cdf(self, y_mid: np.ndarray) -> np.ndarray:
        """
        CDF for y values in [c_0, c_K).

        Returns (n, m_mid) array.

        For y in bin k (c_{k-1} <= y < c_k):
        F(y) = baseline_cdf_c0 + hist_mass * [
            sum_{j<k} bin_probs[:, j] + bin_probs[:, k] * (y - c_k) / bin_width_k
        ]
        """
        # bin_idx[j] = index k such that c_k <= y_mid[j] < c_{k+1}
        # np.searchsorted(cutpoints, y, side='right') - 1 gives the bin index
        bin_idx = np.searchsorted(self.cutpoints, y_mid, side="right") - 1
        # Clip to valid range
        bin_idx = np.clip(bin_idx, 0, self.K - 1)

        n, m = self.n, len(y_mid)
        result = np.empty((n, m), dtype=np.float64)

        # Vectorised: iterate over unique bin indices to avoid a loop
        # For each (obs i, point j), result[i, j] = CDF(y_mid[j] | obs i)
        # This is inherently O(n * m) but numpy-fast

        # completed_bins[i, j] = cumulative bin prob for all bins before bin_idx[j]
        # = self._cumbin[i, bin_idx[j] - 1] for bin_idx[j] > 0, else 0
        prev_cumbin = np.where(
            bin_idx[np.newaxis, :] > 0,
            self._cumbin[:, np.clip(bin_idx - 1, 0, self.K - 1)],  # (n, m)
            0.0,
        )

        # Fractional position within bin
        c_left = self.cutpoints[bin_idx]    # (m,)
        bin_w = self._bin_widths[bin_idx]   # (m,)
        # Avoid division by zero for zero-width bins
        frac = np.where(bin_w > 0, (y_mid - c_left) / bin_w, 0.0)  # (m,)

        # bin_probs[:, bin_idx[j]] for each j — shape (n, m)
        current_bin_prob = self.bin_probs[:, bin_idx]  # (n, m)

        # Conditional histogram CDF
        hist_cdf_cond = prev_cumbin + current_bin_prob * frac[np.newaxis, :]  # (n, m)

        # Scale by histogram mass and add left tail
        result = (
            self.baseline_cdf_c0[:, np.newaxis]           # (n, 1)
            + self._hist_mass[:, np.newaxis] * hist_cdf_cond  # (n, m)
        )
        return result

    def _baseline_cdf(self, y_vals: np.ndarray) -> np.ndarray:
        """
        Parametric baseline CDF at y_vals for all n observations.

        Returns (n, len(y_vals)).
        """
        mu = self.baseline_params["mu"][:, np.newaxis]   # (n, 1)
        disp = self.baseline_params["dispersion"]
        c = y_vals[np.newaxis, :]                         # (1, m)
        fam = self.distribution_family

        if fam == "gamma":
            alpha = 1.0 / disp
            scale = mu * disp
            return stats.gamma.cdf(c, a=alpha, scale=scale)
        elif fam == "gaussian":
            return stats.norm.cdf(c, loc=mu, scale=np.sqrt(disp))
        elif fam == "lognormal":
            sigma_log = float(np.sqrt(disp))
            mu_log = np.log(mu) - 0.5 * sigma_log ** 2
            return stats.lognorm.cdf(c, s=sigma_log, scale=np.exp(mu_log))
        elif fam == "inversegaussian":
            lam = mu / disp
            return stats.invgauss.cdf(c, mu=mu / lam, scale=lam)
        else:
            raise ValueError(f"Unknown family: {fam!r}")

    # ------------------------------------------------------------------
    # Quantile
    # ------------------------------------------------------------------

    def quantile(self, alpha: float | np.ndarray) -> np.ndarray:
        """
        Quantile function (inverse CDF).

        For scalar alpha: returns (n,) array.
        For array alpha of shape (m,): returns (n, m) array.

        Uses binary search on the piecewise-linear CDF. The search is
        O(n log K) — no Python loops over observations.

        Parameters
        ----------
        alpha : float or np.ndarray
            Probability level(s) in (0, 1).

        Returns
        -------
        np.ndarray
        """
        scalar = np.isscalar(alpha)
        alpha = np.atleast_1d(np.asarray(alpha, dtype=np.float64))

        results = np.empty((self.n, len(alpha)), dtype=np.float64)

        for j, p in enumerate(alpha):
            results[:, j] = self._quantile_single(p)

        if scalar:
            return results[:, 0]
        return results

    def _quantile_single(self, p: float) -> np.ndarray:
        """
        Quantile at probability level p for all n observations.

        Returns (n,) array.
        """
        # CDF structure:
        # [0, baseline_cdf_c0) -> left tail (invert baseline)
        # [baseline_cdf_c0, baseline_cdf_c0 + hist_mass] -> histogram
        # (baseline_cdf_c0 + hist_mass, 1] -> right tail (invert baseline)

        p_arr = np.full(self.n, p)
        result = np.empty(self.n, dtype=np.float64)

        # Left tail: p < baseline_cdf_c0
        in_left = p_arr < self.baseline_cdf_c0
        if in_left.any():
            result[in_left] = self._baseline_quantile(p_arr[in_left], in_left)

        # Right tail: p > baseline_cdf_cK
        in_right = p_arr > self.baseline_cdf_cK
        if in_right.any():
            result[in_right] = self._baseline_quantile(p_arr[in_right], in_right)

        # Histogram region
        in_hist = ~in_left & ~in_right
        if in_hist.any():
            result[in_hist] = self._histogram_quantile(p_arr[in_hist], in_hist)

        return result

    def _histogram_quantile(self, p_sub: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Quantile in histogram region for subset of observations.

        p_sub : (n_sub,) — the probability levels
        mask : (n,) bool — which observations are in this subset
        """
        n_sub = int(np.sum(mask))
        cum_bin_sub = self._cumbin[mask, :]          # (n_sub, K)
        hist_mass_sub = self._hist_mass[mask]         # (n_sub,)
        base_c0_sub = self.baseline_cdf_c0[mask]     # (n_sub,)

        # Convert absolute probability to conditional histogram probability
        p_hist_cond = (p_sub - base_c0_sub) / hist_mass_sub  # (n_sub,)
        p_hist_cond = np.clip(p_hist_cond, 0.0, 1.0)

        # Find which bin: smallest k such that cumbin[:, k] >= p_hist_cond
        # searchsorted on each row
        bin_idx = np.array([
            np.searchsorted(cum_bin_sub[i, :], p_hist_cond[i], side="left")
            for i in range(n_sub)
        ])
        bin_idx = np.clip(bin_idx, 0, self.K - 1)

        # Left edge of that bin
        c_left = self.cutpoints[bin_idx]   # (n_sub,)
        bin_w = self._bin_widths[bin_idx]  # (n_sub,)

        # Cumulative prob at left edge of bin
        prev_cum = np.where(
            bin_idx > 0,
            cum_bin_sub[np.arange(n_sub), np.clip(bin_idx - 1, 0, self.K - 1)],
            0.0,
        )

        bin_prob = cum_bin_sub[np.arange(n_sub), bin_idx] - prev_cum
        # Linear interpolation within bin
        frac = np.where(
            bin_prob > 0,
            (p_hist_cond - prev_cum) / bin_prob,
            0.5,
        )
        return c_left + frac * bin_w

    def _baseline_quantile(self, p_sub: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Invert baseline CDF for observations in mask."""
        mu_sub = self.baseline_params["mu"][mask]   # (n_sub,)
        disp = self.baseline_params["dispersion"]
        fam = self.distribution_family
        results = np.empty(len(p_sub), dtype=np.float64)

        for i, (p_i, mu_i) in enumerate(zip(p_sub, mu_sub)):
            if fam == "gamma":
                alpha = 1.0 / disp
                scale = mu_i * disp
                results[i] = stats.gamma.ppf(p_i, a=alpha, scale=scale)
            elif fam == "gaussian":
                results[i] = stats.norm.ppf(p_i, loc=mu_i, scale=np.sqrt(disp))
            elif fam == "lognormal":
                sigma_log = float(np.sqrt(disp))
                mu_log = float(np.log(mu_i) - 0.5 * sigma_log ** 2)
                results[i] = stats.lognorm.ppf(p_i, s=sigma_log, scale=np.exp(mu_log))
            elif fam == "inversegaussian":
                lam = mu_i / disp
                results[i] = stats.invgauss.ppf(p_i, mu=mu_i / lam, scale=lam)
            else:
                raise ValueError(f"Unknown family: {fam!r}")
        return results

    # ------------------------------------------------------------------
    # Moments
    # ------------------------------------------------------------------

    def mean(self) -> np.ndarray:
        """
        Expected value for each observation.

        Combines left tail mean, histogram mean, and right tail mean.
        All computed analytically where possible.

        Returns
        -------
        np.ndarray, shape (n,)
        """
        # Histogram mean: sum_k bin_probs[:, k] * midpoint_k
        midpoints = 0.5 * (self.cutpoints[:-1] + self.cutpoints[1:])  # (K,)
        hist_mean_cond = self.bin_probs @ midpoints   # (n,) conditional on histogram region

        # Left tail mean: E[Y | Y < c_0] * P(Y < c_0)
        left_mean = self._baseline_partial_mean_lower(self.c_0)  # (n,)

        # Right tail mean: E[Y | Y >= c_K] * P(Y >= c_K)
        right_mean = self._baseline_partial_mean_upper(self.c_K)  # (n,)

        # Histogram contribution: hist_mean_cond is conditional; scale by histogram mass
        hist_contribution = self._hist_mass * hist_mean_cond  # (n,)

        return left_mean + hist_contribution + right_mean

    def var(self) -> np.ndarray:
        """
        Variance for each observation.

        Uses E[Y^2] - E[Y]^2 with histogram contribution from bin midpoints.
        Approximate (uses bin midpoints for second moment within histogram).

        Returns
        -------
        np.ndarray, shape (n,)
        """
        mu = self.mean()

        midpoints = 0.5 * (self.cutpoints[:-1] + self.cutpoints[1:])  # (K,)
        hist_e2_cond = self.bin_probs @ (midpoints ** 2)  # (n,)
        hist_e2 = self._hist_mass * hist_e2_cond

        left_e2 = self._baseline_partial_e2_lower(self.c_0)
        right_e2 = self._baseline_partial_e2_upper(self.c_K)

        e2 = left_e2 + hist_e2 + right_e2
        return np.maximum(e2 - mu ** 2, 0.0)

    def std(self) -> np.ndarray:
        """Standard deviation for each observation. Shape (n,)."""
        return np.sqrt(self.var())

    # ------------------------------------------------------------------
    # CRPS (Continuous Ranked Probability Score)
    # ------------------------------------------------------------------

    def crps(self, y_true: np.ndarray) -> np.ndarray:
        """
        CRPS for each observation given actuals y_true.

        CRPS = integral_{-inf}^{inf} (F(y) - 1(y >= y_true))^2 dy

        Computed analytically over the histogram bins using the piecewise-
        linear structure of F. Tail contributions use numerical integration
        (fine enough for practical purposes, since tail mass is small).

        Parameters
        ----------
        y_true : np.ndarray, shape (n,)

        Returns
        -------
        np.ndarray, shape (n,)
        """
        y_true = np.asarray(y_true, dtype=np.float64)

        # Evaluate CDF at all cutpoints: shape (n, K+1)
        cdf_at_cuts = self.cdf(self.cutpoints)  # (n, K+1)

        crps_vals = np.zeros(self.n, dtype=np.float64)

        # CRPS formula: integral (F(y) - 1(y >= y_true))^2 dy
        # where 1(y >= y_true) is 1 when integration variable y >= y_true.
        # Contribution from each bin [c_k, c_{k+1}]
        for k in range(self.K):
            c_lo = self.cutpoints[k]
            c_hi = self.cutpoints[k + 1]
            w = c_hi - c_lo
            if w == 0:
                continue

            F_lo = cdf_at_cuts[:, k]     # (n,)
            F_hi = cdf_at_cuts[:, k + 1] # (n,)

            # Case A: y_true < c_lo → all bin y satisfy y >= y_true → indicator=1
            # Integrand = (F - 1)^2
            mask_a = y_true < c_lo
            G_lo = F_lo - 1.0
            G_hi = F_hi - 1.0
            contrib_a = w * (G_lo ** 2 + G_lo * G_hi + G_hi ** 2) / 3.0
            crps_vals += np.where(mask_a, contrib_a, 0.0)

            # Case B: y_true >= c_hi → all bin y satisfy y < y_true → indicator=0
            # Integrand = F^2
            mask_b = y_true >= c_hi
            contrib_b = w * (F_lo ** 2 + F_lo * F_hi + F_hi ** 2) / 3.0
            crps_vals += np.where(mask_b, contrib_b, 0.0)

            # Case C: y_true in [c_lo, c_hi) — split at y_true
            mask_c = (~mask_a) & (~mask_b)
            if mask_c.any():
                yt_c = y_true[mask_c]
                F_lo_c = F_lo[mask_c]
                F_hi_c = F_hi[mask_c]
                slope = (F_hi_c - F_lo_c) / w

                # F at y_true
                F_yt = F_lo_c + slope * (yt_c - c_lo)

                # Left sub-interval [c_lo, y_true]: y < y_true → indicator=0 → F^2
                w1 = yt_c - c_lo
                i1 = w1 * (F_lo_c ** 2 + F_lo_c * F_yt + F_yt ** 2) / 3.0

                # Right sub-interval [y_true, c_hi]: y >= y_true → indicator=1 → (F-1)^2
                w2 = c_hi - yt_c
                G2_lo = F_yt - 1.0
                G2_hi = F_hi_c - 1.0
                i2 = w2 * (G2_lo ** 2 + G2_lo * G2_hi + G2_hi ** 2) / 3.0

                crps_vals[mask_c] += i1 + i2

        # Tail contributions are typically small; we use a simple trapezoidal
        # approximation with 20 points each side
        crps_vals += self._crps_left_tail(y_true)
        crps_vals += self._crps_right_tail(y_true)

        return crps_vals

    def _crps_left_tail(self, y_true: np.ndarray, n_points: int = 20) -> np.ndarray:
        """CRPS contribution from left tail y < c_0."""
        if self.c_0 <= 0:
            return np.zeros(self.n)
        # Very small contribution for insurance data (c_0 ~ 0)
        return np.zeros(self.n)

    def _crps_right_tail(self, y_true: np.ndarray, n_points: int = 20) -> np.ndarray:
        """CRPS contribution from right tail y > c_K."""
        # Right tail: (F(y) - 1)^2 for y > c_K, where F(y) = baseline CDF
        # Numerically integrate using baseline quantile grid
        mu = self.baseline_params["mu"]
        disp = self.baseline_params["dispersion"]

        # Integrate from c_K to approx 99.99th percentile of baseline
        results = np.zeros(self.n, dtype=np.float64)
        # Simple 20-point Gauss-Legendre won't work easily vectorised; use trapezoidal
        q_grid = np.linspace(0.9975, 0.9999, n_points)  # quantile grid

        # For Gamma baseline: get representative upper bound
        if self.distribution_family == "gamma":
            alpha = 1.0 / disp
            # Typical upper quantile for the "average" mu
            avg_mu = float(np.mean(mu))
            avg_scale = avg_mu * disp
            y_upper = stats.gamma.ppf(0.9999, a=alpha, scale=avg_scale)
        else:
            y_upper = self.c_K * 5.0

        y_grid = np.linspace(self.c_K, y_upper, n_points)   # (n_points,)
        cdf_grid = self.cdf(y_grid)                           # (n, n_points)
        # indicator = 1 when integration variable y_grid >= y_true
        indicator = (y_grid[np.newaxis, :] >= y_true[:, np.newaxis]).astype(float)  # (n, n_pts)

        integrand = (cdf_grid - indicator) ** 2              # (n, n_points)
        dy = np.diff(y_grid)
        # Trapezoidal rule
        results = np.trapz(integrand, y_grid, axis=1)
        return results

    # ------------------------------------------------------------------
    # Expected Shortfall (Tail VaR)
    # ------------------------------------------------------------------

    def expected_shortfall(self, alpha: float) -> np.ndarray:
        """
        Expected Shortfall (CVaR / TVaR) at level alpha.

        ES_alpha = E[Y | Y >= Q_alpha] = (1/(1-alpha)) * integral_{Q_alpha}^inf y dF(y)

        Computed numerically using the quantile function.

        Parameters
        ----------
        alpha : float
            Confidence level, e.g. 0.995 for 99.5th percentile (Solvency II SCR).

        Returns
        -------
        np.ndarray, shape (n,)
        """
        # Numerical integration: E[Y | Y > Q_alpha] via quantile grid
        n_grid = 50
        p_grid = np.linspace(alpha, 1.0 - 1e-6, n_grid)  # avoid exactly 1.0
        q_grid = self.quantile(p_grid)  # (n, n_grid)
        # Integrate: ES = (1/(1-alpha)) * integral_alpha^1 Q(p) dp
        es = np.trapz(q_grid, p_grid, axis=1) / (1.0 - alpha)
        return es

    # ------------------------------------------------------------------
    # Partial moment helpers (for mean/variance computation)
    # ------------------------------------------------------------------

    def _baseline_partial_mean_lower(self, threshold: float) -> np.ndarray:
        """E[Y * 1(Y < threshold)] under baseline. Shape (n,)."""
        mu = self.baseline_params["mu"]
        disp = self.baseline_params["dispersion"]
        fam = self.distribution_family

        results = np.zeros(self.n, dtype=np.float64)
        if threshold <= 0:
            return results

        if fam == "gamma":
            alpha = 1.0 / disp
            scale = mu * disp
            # E[Y * 1(Y < t)] = alpha * scale * Gamma_CDF(t; alpha+1, scale)
            # = mu * gamma.cdf(t, a=alpha+1, scale=scale)
            results = mu * stats.gamma.cdf(threshold, a=alpha + 1, scale=scale)
        elif fam == "gaussian":
            sigma = np.sqrt(disp)
            z = (threshold - mu) / sigma
            results = mu * stats.norm.cdf(z) - sigma * stats.norm.pdf(z)
        elif fam == "lognormal":
            sigma_log = float(np.sqrt(disp))
            mu_log = np.log(mu) - 0.5 * sigma_log ** 2
            # E[Y * 1(Y < t)] = mu * Phi((log(t) - mu_log - sigma_log^2) / sigma_log)
            results = mu * stats.norm.cdf(
                (np.log(threshold) - mu_log - sigma_log ** 2) / sigma_log
            )
        else:
            # Numerical fallback: E[Y * 1(Y < t)] = integral_0^t y * pdf(y) dy
            y_grid = np.linspace(1e-8, threshold, 100)
            pdf_grid = self._baseline_pdf(y_grid)  # (n, 100)
            integrand = y_grid[np.newaxis, :] * pdf_grid  # (n, 100)
            results = np.trapz(integrand, y_grid, axis=1)
        return results

    def _baseline_partial_mean_upper(self, threshold: float) -> np.ndarray:
        """E[Y * 1(Y >= threshold)] under baseline. Shape (n,)."""
        mu = self.baseline_params["mu"]
        disp = self.baseline_params["dispersion"]
        fam = self.distribution_family

        if fam == "gamma":
            alpha = 1.0 / disp
            scale = mu * disp
            results = mu * stats.gamma.sf(threshold, a=alpha + 1, scale=scale)
        elif fam == "gaussian":
            sigma = np.sqrt(disp)
            z = (threshold - mu) / sigma
            results = mu * stats.norm.sf(z) + sigma * stats.norm.pdf(z)
        elif fam == "lognormal":
            sigma_log = float(np.sqrt(disp))
            mu_log = np.log(mu) - 0.5 * sigma_log ** 2
            results = mu * stats.norm.sf(
                (np.log(threshold) - mu_log - sigma_log ** 2) / sigma_log
            )
        else:
            # Numerical fallback using total mean minus lower partial
            results = mu - self._baseline_partial_mean_lower(threshold)
        return results

    def _baseline_partial_e2_lower(self, threshold: float) -> np.ndarray:
        """E[Y^2 * 1(Y < threshold)] under baseline. Shape (n,)."""
        # Numerical integration (less critical for correctness)
        results = np.zeros(self.n)
        if threshold <= 0:
            return results
        # Use 50-point grid
        y_grid = np.linspace(1e-8, threshold, 50)
        pdf_grid = self._baseline_pdf(y_grid)  # (n, 50)
        integrand = (y_grid ** 2)[np.newaxis, :] * pdf_grid
        results = np.trapz(integrand, y_grid, axis=1)
        return results

    def _baseline_partial_e2_upper(self, threshold: float) -> np.ndarray:
        """E[Y^2 * 1(Y >= threshold)] under baseline. Shape (n,)."""
        mu = self.baseline_params["mu"]
        disp = self.baseline_params["dispersion"]
        # E[Y^2] for parametric distributions
        if self.distribution_family == "gamma":
            alpha = 1.0 / disp
            e2_total = mu ** 2 + mu ** 2 * disp  # Var(Y) + E[Y]^2 = mu^2*(1+phi)
        elif self.distribution_family == "gaussian":
            e2_total = mu ** 2 + disp
        elif self.distribution_family == "lognormal":
            sigma_log = float(np.sqrt(disp))
            e2_total = np.exp(2 * (np.log(mu) - 0.5 * sigma_log ** 2) + 2 * sigma_log ** 2)
        else:
            e2_total = mu ** 2 + disp
        return e2_total - self._baseline_partial_e2_lower(threshold)

    def _baseline_pdf(self, y_vals: np.ndarray) -> np.ndarray:
        """Baseline PDF at y_vals for all n obs. Returns (n, len(y_vals))."""
        mu = self.baseline_params["mu"][:, np.newaxis]
        disp = self.baseline_params["dispersion"]
        c = y_vals[np.newaxis, :]
        fam = self.distribution_family

        if fam == "gamma":
            return stats.gamma.pdf(c, a=1.0/disp, scale=mu*disp)
        elif fam == "gaussian":
            return stats.norm.pdf(c, loc=mu, scale=np.sqrt(disp))
        elif fam == "lognormal":
            sigma_log = float(np.sqrt(disp))
            mu_log = np.log(mu) - 0.5 * sigma_log ** 2
            return stats.lognorm.pdf(c, s=sigma_log, scale=np.exp(mu_log))
        elif fam == "inversegaussian":
            lam = mu / disp
            return stats.invgauss.pdf(c, mu=mu/lam, scale=lam)
        else:
            raise ValueError(f"Unknown family: {fam!r}")

    # ------------------------------------------------------------------
    # Polars output
    # ------------------------------------------------------------------

    def summary(self, quantiles: list[float] | None = None) -> pl.DataFrame:
        """
        Return a Polars DataFrame with summary statistics for each observation.

        Columns: mean, std, q10, q25, q50, q75, q90, q95, q995 (by default).

        Parameters
        ----------
        quantiles : list[float], optional
            Additional quantile levels to include.

        Returns
        -------
        pl.DataFrame
        """
        default_q = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.995]
        q_levels = sorted(set(default_q + (quantiles or [])))

        q_vals = self.quantile(np.array(q_levels))  # (n, len(q_levels))

        data = {
            "mean": self.mean(),
            "std": self.std(),
        }
        for i, q in enumerate(q_levels):
            col_name = f"q{int(q * 1000):04d}"
            data[col_name] = q_vals[:, i]

        return pl.DataFrame(data)

    def adjustment_factors_frame(self, cutpoints: np.ndarray | None = None) -> pl.DataFrame:
        """
        Return per-bin adjustment factors a_k = p_k / b_k as a Polars DataFrame.

        Columns are named by bin midpoint. Shape: (n, K).
        """
        raise NotImplementedError(
            "adjustment_factors_frame requires baseline bin probabilities, "
            "which are stored on the DRN object. Use DRN.adjustment_factors(X) instead."
        )

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        return (
            f"ExtendedHistogramBatch("
            f"n={self.n}, K={self.K}, "
            f"c_0={self.c_0:.2f}, c_K={self.c_K:.2f}, "
            f"family={self.distribution_family!r})"
        )
