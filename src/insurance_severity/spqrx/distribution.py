"""
SPQRxDistribution: per-observation distribution object for SPQRx.

Wraps the fitted model parameters for a batch of observations and provides
the standard insurance-severity distribution API: quantile(), cdf(), pdf(),
mean(), and ilf().

All computation is numpy-based. The bGPD CDF and PDF are implemented in
log-space to avoid numerical issues near the bulk-tail transition.

References
----------
Majumder, S. & Richards, J. (2025). arXiv:2504.19994.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import quad

if TYPE_CHECKING:
    from insurance_severity.spqrx.spqrx import SPQRxSeverity


class SPQRxDistribution:
    """
    Blended GPD distribution for a batch of observations.

    Returned by SPQRxSeverity.predict_distribution(). Provides methods for
    quantile prediction, CDF/PDF evaluation, mean computation, and ILF.

    Parameters
    ----------
    model : SPQRxSeverity
        The fitted model (needed for bulk CDF evaluation).
    X : np.ndarray, shape (n, p)
        Feature matrix for the batch.
    xi : np.ndarray, shape (n,)
        Fitted GPD tail shape per observation.
    u_tilde : np.ndarray, shape (n,)
        Effective GPD threshold per observation (in original scale).
    sigma_tilde : np.ndarray, shape (n,)
        GPD scale per observation.
    a : np.ndarray, shape (n,)
        Lower blend boundary Q(pa | x) in original scale.
    b : np.ndarray, shape (n,)
        Upper blend boundary Q(pb | x) in original scale.
    """

    def __init__(
        self,
        model: "SPQRxSeverity",
        X: np.ndarray,
        xi: np.ndarray,
        u_tilde: np.ndarray,
        sigma_tilde: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
    ):
        self.model = model
        self.X = X
        self.xi = xi
        self.u_tilde = u_tilde
        self.sigma_tilde = sigma_tilde
        self.a = a
        self.b = b
        self.n = len(xi)
        self._pa = model.pa
        self._pb = model.pb

    # ------------------------------------------------------------------
    # Core distributional methods
    # ------------------------------------------------------------------

    def quantile(self, tau: float | np.ndarray) -> np.ndarray:
        """
        Quantile(s) for each observation.

        Parameters
        ----------
        tau : float or np.ndarray of shape (m,)
            Quantile level(s) in (0, 1).

        Returns
        -------
        np.ndarray, shape (n,) for scalar tau, or (n, m) for array tau.
        """
        scalar = np.isscalar(tau)
        taus = np.atleast_1d(np.asarray(tau, dtype=np.float64))
        results = np.column_stack([
            self.model.predict_quantile(self.X, float(t)) for t in taus
        ])  # (n, m)
        if scalar:
            return results[:, 0]
        return results

    def cdf(self, y_vals: np.ndarray) -> np.ndarray:
        """
        CDF evaluated at y_vals for each observation.

        Parameters
        ----------
        y_vals : np.ndarray, shape (m,) or (n, m)
            Values at which to evaluate the CDF.

        Returns
        -------
        np.ndarray, shape (n, m)
        """
        y_vals = np.asarray(y_vals, dtype=np.float64)
        if y_vals.ndim == 1:
            # Broadcast: evaluate same y grid for all observations
            results = np.column_stack([
                self.model.cdf(self.X, np.full(self.n, yj)) for yj in y_vals
            ])
            return results
        # y_vals is (n, m) — evaluate each row independently
        n, m = y_vals.shape
        out = np.zeros((n, m))
        for j in range(m):
            out[:, j] = self.model.cdf(self.X, y_vals[:, j])
        return out

    def pdf(self, y_vals: np.ndarray) -> np.ndarray:
        """
        PDF evaluated at y_vals for each observation.

        Parameters
        ----------
        y_vals : np.ndarray, shape (m,) or (n, m)

        Returns
        -------
        np.ndarray, shape (n, m)
        """
        y_vals = np.asarray(y_vals, dtype=np.float64)
        if y_vals.ndim == 1:
            results = np.column_stack([
                self.model.pdf(self.X, np.full(self.n, yj)) for yj in y_vals
            ])
            return results
        n, m = y_vals.shape
        out = np.zeros((n, m))
        for j in range(m):
            out[:, j] = self.model.pdf(self.X, y_vals[:, j])
        return out

    def mean(self, n_grid: int = 500) -> np.ndarray:
        """
        E[Y | x] via numerical integration of the survival function.

        E[Y] = integral_0^inf S(y) dy,  S(y) = 1 - F(y).

        Integration is truncated at Q(0.9999 | x) to avoid numerical issues
        with infinite support.

        Parameters
        ----------
        n_grid : int
            Number of quadrature points per observation.

        Returns
        -------
        np.ndarray, shape (n,)
        """
        means = np.empty(self.n)
        for i in range(self.n):
            # Upper bound: GPD quantile at 0.9999
            xi_i = float(self.xi[i])
            u_i = float(self.u_tilde[i])
            s_i = float(self.sigma_tilde[i])
            # Q_GP(0.9999) for bounding the integral
            y_upper = u_i + (s_i / xi_i) * (((1 - 0.9999) / (1 - self._pb)) ** (-xi_i) - 1)
            y_upper = max(y_upper, self.b[i] * 10)

            y_grid = np.exp(np.linspace(np.log(max(1e-2, self.a[i] / 100)), np.log(y_upper), n_grid))
            X_i = self.X[i:i+1]
            X_rep = np.repeat(X_i, len(y_grid), axis=0)
            cdf_vals = self.model.cdf(X_rep, y_grid)
            surv = 1.0 - cdf_vals
            dy = np.diff(y_grid)
            means[i] = (0.5 * (surv[:-1] + surv[1:]) * dy).sum()
        return means

    def ilf(
        self,
        limit: float,
        basic_limit: float,
        n_grid: int = 500,
    ) -> np.ndarray:
        """
        Increased limits factor for each observation.

        ILF(L, b) = E[min(Y, L)] / E[min(Y, b)]

        E[min(Y, L)] = integral_0^L S(y) dy.

        Parameters
        ----------
        limit : float
            Policy limit to price to.
        basic_limit : float
            Basic (reference) limit.
        n_grid : int

        Returns
        -------
        np.ndarray, shape (n,)
        """
        lev_l = self._lev(limit, n_grid)
        lev_b = self._lev(basic_limit, n_grid)
        return lev_l / np.clip(lev_b, 1e-8, None)

    def _lev(self, limit: float, n_grid: int = 500) -> np.ndarray:
        """Limited expected value E[min(Y, L)] = integral_0^L S(y) dy."""
        y_lo = 1e-2
        y_grid = np.exp(np.linspace(np.log(y_lo), np.log(limit), n_grid))
        out = np.zeros(self.n)
        for i in range(self.n):
            X_i = self.X[i:i+1]
            X_rep = np.repeat(X_i, len(y_grid), axis=0)
            cdf_vals = self.model.cdf(X_rep, y_grid)
            surv = 1.0 - cdf_vals
            dy = np.diff(y_grid)
            out[i] = (0.5 * (surv[:-1] + surv[1:]) * dy).sum()
        return out

    def __repr__(self) -> str:
        return (
            f"SPQRxDistribution(n={self.n}, "
            f"xi=[{self.xi.min():.3f}, {self.xi.max():.3f}])"
        )
