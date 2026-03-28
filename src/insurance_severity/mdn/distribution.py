"""
MDNMixture: a fitted lognormal mixture distribution for a batch of observations.

Returned by MDN.predict_distribution(). Provides the same high-level interface
as other distribution objects in insurance-severity: mean(), quantile(), cdf(),
pdf(), and pit_samples() for calibration testing.

All numpy-based (no torch dependency at this level).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


class MDNMixture:
    """
    Lognormal mixture distribution for a batch of observations.

    Wraps the three arrays of mixture parameters (weights, log-space means,
    log-space standard deviations) for n observations with K components, and
    provides analytical/numerical methods for computing distributional
    quantities.

    Parameters
    ----------
    pi : np.ndarray, shape (n, K)
        Mixing weights. Rows sum to 1.
    mu : np.ndarray, shape (n, K)
        Component means in log-space.
    sigma : np.ndarray, shape (n, K)
        Component standard deviations in log-space (positive).

    Notes
    -----
    The component distribution is Lognormal(μ_k, σ_k²). The density on the
    original scale y > 0 is:

        f(y | x) = Σ_k π_k(x) · (1 / (y σ_k √(2π))) · exp(-½((log y - μ_k)/σ_k)²)

    The mean of component k is exp(μ_k + σ_k²/2).
    """

    def __init__(
        self,
        pi: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
    ):
        self.pi = np.asarray(pi, dtype=np.float64)
        self.mu = np.asarray(mu, dtype=np.float64)
        self.sigma = np.asarray(sigma, dtype=np.float64).clip(min=1e-8)

        if self.pi.ndim == 1:
            self.pi = self.pi[np.newaxis, :]
            self.mu = self.mu[np.newaxis, :]
            self.sigma = self.sigma[np.newaxis, :]

        self.n, self.K = self.pi.shape

    # ------------------------------------------------------------------
    # Core distributional methods
    # ------------------------------------------------------------------

    def mean(self) -> np.ndarray:
        """
        Expected value on the original scale.

        E[Y] = Σ_k π_k · exp(μ_k + σ_k²/2)

        Returns
        -------
        np.ndarray, shape (n,)
        """
        component_means = np.exp(self.mu + 0.5 * self.sigma ** 2)  # (n, K)
        return (self.pi * component_means).sum(axis=1)              # (n,)

    def variance(self) -> np.ndarray:
        """
        Variance on the original scale.

        Var[Y] = E[Y²] - E[Y]²

        For a lognormal component: E[Y_k²] = exp(2μ_k + 2σ_k²)
        So E[Y²] = Σ_k π_k · exp(2μ_k + 2σ_k²) (by law of total expectation
        for the second moment, noting cross terms depend on mixture structure).

        Returns
        -------
        np.ndarray, shape (n,)
        """
        # E[Y²] = E_mixture[Y²] = Σ_k π_k E_k[Y²]
        # For lognormal: E_k[Y²] = exp(2μ_k + 2σ_k²)
        e_y2 = (self.pi * np.exp(2 * self.mu + 2 * self.sigma ** 2)).sum(axis=1)
        e_y = self.mean()
        return e_y2 - e_y ** 2

    def std(self) -> np.ndarray:
        """Standard deviation on the original scale. Shape (n,)."""
        return np.sqrt(np.clip(self.variance(), 0, None))

    def cdf(self, y_grid: np.ndarray) -> np.ndarray:
        """
        CDF evaluated at each point in ``y_grid`` for all observations.

        Parameters
        ----------
        y_grid : np.ndarray, shape (m,)
            Grid of positive values on the original scale.

        Returns
        -------
        np.ndarray, shape (n, m)
        """
        y_grid = np.asarray(y_grid, dtype=np.float64)
        log_grid = np.log(np.clip(y_grid, 1e-8, None))  # (m,)

        # z[i, j, k] = (log_grid[j] - mu[i, k]) / sigma[i, k]
        log_grid_bc = log_grid[np.newaxis, :, np.newaxis]     # (1, m, 1)
        mu_bc = self.mu[:, np.newaxis, :]                     # (n, 1, K)
        sigma_bc = self.sigma[:, np.newaxis, :]               # (n, 1, K)
        z = (log_grid_bc - mu_bc) / sigma_bc                  # (n, m, K)

        cdf_k = norm.cdf(z)                                   # (n, m, K)
        pi_bc = self.pi[:, np.newaxis, :]                     # (n, 1, K)
        return (pi_bc * cdf_k).sum(axis=2)                    # (n, m)

    def pdf(self, y_grid: np.ndarray) -> np.ndarray:
        """
        PDF on the original scale at each point in ``y_grid``.

        Parameters
        ----------
        y_grid : np.ndarray, shape (m,)

        Returns
        -------
        np.ndarray, shape (n, m)
        """
        y_grid = np.asarray(y_grid, dtype=np.float64)
        log_grid = np.log(np.clip(y_grid, 1e-8, None))  # (m,)

        log_grid_bc = log_grid[np.newaxis, :, np.newaxis]
        mu_bc = self.mu[:, np.newaxis, :]
        sigma_bc = self.sigma[:, np.newaxis, :]
        z = (log_grid_bc - mu_bc) / sigma_bc             # (n, m, K)

        # Log-space normal PDF: φ(z)/σ/y
        log_phi = -0.5 * z ** 2 - np.log(sigma_bc) - 0.5 * np.log(2 * np.pi)
        log_pdf_k = log_phi - log_grid_bc                # (n, m, K), divide by y

        # Mixture: Σ_k π_k · f_k(y)
        pi_bc = self.pi[:, np.newaxis, :]
        pdf_k = pi_bc * np.exp(log_pdf_k)
        return pdf_k.sum(axis=2)                         # (n, m)

    def log_prob(self, y: np.ndarray) -> np.ndarray:
        """
        Log-density at each observed value (one value per observation).

        Useful for NLL evaluation on held-out data.

        Parameters
        ----------
        y : np.ndarray, shape (n,)
            One observed severity per observation.

        Returns
        -------
        np.ndarray, shape (n,)
        """
        y = np.asarray(y, dtype=np.float64)
        log_y = np.log(np.clip(y, 1e-8, None))          # (n,)

        log_y_bc = log_y[:, np.newaxis]                  # (n, 1)
        z = (log_y_bc - self.mu) / self.sigma            # (n, K)

        log_phi = -0.5 * z ** 2 - np.log(self.sigma) - 0.5 * np.log(2 * np.pi)
        log_pi = np.log(np.clip(self.pi, 1e-8, None))

        # logsumexp: log Σ_k exp(log π_k + log φ_k)
        log_mix = _logsumexp(log_pi + log_phi, axis=1)   # (n,)
        return log_mix - log_y                           # Jacobian

    def quantile(self, q: float | np.ndarray, n_grid: int = 2000) -> np.ndarray:
        """
        Quantile(s) via CDF inversion on a log-space grid.

        Parameters
        ----------
        q : float or np.ndarray
            Quantile level(s) in (0, 1).
        n_grid : int
            Grid resolution for CDF inversion. 2000 is accurate to ~0.1%.

        Returns
        -------
        np.ndarray
            Shape (n,) for scalar q, or (n, len(q)) for array q.
        """
        scalar = np.isscalar(q)
        q_arr = np.atleast_1d(np.asarray(q, dtype=np.float64))

        # Log-space grid spanning all components' ranges
        lo = float((self.mu - 5 * self.sigma).min())
        hi = float((self.mu + 5 * self.sigma).max())
        log_grid = np.linspace(lo, hi, n_grid)
        y_grid = np.exp(log_grid)

        # CDF at all grid points: (n, n_grid)
        cdf_grid = self.cdf(y_grid)

        results = np.empty((self.n, len(q_arr)))
        for j, qj in enumerate(q_arr):
            # First index where CDF >= q
            idx = (cdf_grid >= qj).argmax(axis=1)  # (n,)
            idx = np.clip(idx, 0, n_grid - 1)
            results[:, j] = y_grid[idx]

        if scalar:
            return results[:, 0]
        return results

    def pit_samples(self, y: np.ndarray) -> np.ndarray:
        """
        Probability integral transform (PIT) values.

        Returns F(y_i | x_i) for each observation. If the model is
        calibrated, these should be uniform on [0, 1].

        Parameters
        ----------
        y : np.ndarray, shape (n,)

        Returns
        -------
        np.ndarray, shape (n,)
        """
        y = np.asarray(y, dtype=np.float64)
        log_y = np.log(np.clip(y, 1e-8, None))

        z = (log_y[:, np.newaxis] - self.mu) / self.sigma  # (n, K)
        cdf_k = norm.cdf(z)
        return (self.pi * cdf_k).sum(axis=1)

    def ilf(
        self,
        limit: float,
        basic_limit: float,
        n_grid: int = 2000,
    ) -> np.ndarray:
        """
        Increased limits factor (ILF) for each observation.

        ILF(L, b) = E[min(Y, L)] / E[min(Y, b)]

        Computed numerically via the limited expected value:
        E[min(Y, L)] = ∫₀ᴸ (1 - F(y)) dy ≈ Σ_j (1 - F(y_j)) Δy_j

        Parameters
        ----------
        limit : float
            Policy limit to price to (numerator limit).
        basic_limit : float
            Basic limit (denominator).
        n_grid : int
            Grid points for numerical integration.

        Returns
        -------
        np.ndarray, shape (n,)
        """
        lev_L = self._lev(limit, n_grid)
        lev_b = self._lev(basic_limit, n_grid)
        return lev_L / np.clip(lev_b, 1e-8, None)

    def _lev(self, limit: float, n_grid: int = 2000) -> np.ndarray:
        """
        Limited expected value E[min(Y, L)].

        E[min(Y, L)] = ∫₀ᴸ (1 - F(y)) dy

        Computed via trapezoidal rule on a log-space grid up to ``limit``.

        Parameters
        ----------
        limit : float
            The limit L.
        n_grid : int

        Returns
        -------
        np.ndarray, shape (n,)
        """
        lo = float((self.mu - 5 * self.sigma).min())
        y_lo = max(np.exp(lo), 1e-2)
        y_grid = np.exp(np.linspace(np.log(y_lo), np.log(limit), n_grid))
        survival = 1.0 - self.cdf(y_grid)    # (n, n_grid)
        dy = np.diff(y_grid)
        # Trapezoidal: ∫ f dy ≈ Σ 0.5*(f[j] + f[j+1]) * dy[j]
        lev = (0.5 * (survival[:, :-1] + survival[:, 1:]) * dy[np.newaxis, :]).sum(axis=1)
        return lev

    def __repr__(self) -> str:
        return (
            f"MDNMixture(n={self.n}, K={self.K}, "
            f"mean=[{self.mean().min():.0f}, {self.mean().max():.0f}])"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    """Numerically stable logsumexp along an axis."""
    a_max = a.max(axis=axis, keepdims=True)
    out = np.log(np.exp(a - a_max).sum(axis=axis)) + a_max.squeeze(axis=axis)
    return out
