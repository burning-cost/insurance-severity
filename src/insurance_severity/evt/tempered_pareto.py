"""
WeibullTemperedPareto: tempered Pareto for physically bounded insurance tails.

Pure Pareto overpredicts the extreme tail for property and D&O losses because
physical upper bounds exist (total insured value, company balance sheet). The
Weibull-tempered Pareto adds an exponential tempering term that preserves the
Pareto character at moderate exceedances but thins the tail at extremes.

POT survival: P(X/t > r | X > t) = r^{-alpha} * exp(-lambda * (r^tau - 1))

Setting lambda=0 recovers pure Pareto (GPD with xi=1/alpha).

Reference: Albrecher, Beirlant & Teugels (2025), arXiv:2511.22272, §5.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy import optimize


class WeibullTemperedPareto:
    """
    Weibull-tempered Pareto for insurance tails with physical upper bound.

    POT survival: P(X/t > r | X > t) = r^{-alpha} * exp(-lambda * (r^tau - 1))

    Equivalent extreme value index: xi = 1/alpha (but tail thins faster at extreme).
    Set lambda=0 to recover pure Pareto (GPD with xi=1/alpha, sigma=t/alpha).

    Parameters
    ----------
    alpha : float, default 1.5 — Pareto index
    lambda_ : float, default 0.1 — tempering intensity
    tau : float, default 0.5 — Weibull shape

    Attributes (after fit)
    ----------------------
    alpha_ : float
    lambda_ : float
    tau_ : float
    loglik_ : float
    k_ : int — number of exceedances used
    threshold_ : float — POT threshold
    """

    def __init__(
        self,
        alpha: float = 1.5,
        lambda_: float = 0.1,
        tau: float = 0.5,
    ):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if lambda_ < 0:
            raise ValueError("lambda_ must be non-negative")
        if tau <= 0:
            raise ValueError("tau must be positive")
        self._alpha = float(alpha)
        self._lambda = float(lambda_)
        self._tau = float(tau)

        self.alpha_: Optional[float] = None
        self.lambda_: Optional[float] = None
        self.tau_: Optional[float] = None
        self.loglik_: Optional[float] = None
        self.k_: Optional[int] = None
        self.threshold_: Optional[float] = None

    # ------------------------------------------------------------------
    # Log-likelihood
    # ------------------------------------------------------------------

    def _log_sf_r(self, r: np.ndarray, alpha: float, lam: float, tau: float) -> np.ndarray:
        """Log survival P(R > r | X > t) for r >= 1."""
        return -alpha * np.log(r) - lam * (r**tau - 1.0)

    def _neg_loglik(
        self, params: np.ndarray, R: np.ndarray
    ) -> float:
        """
        Negative log-likelihood using relative excesses R_{j,k} = X_{(n-j+1)} / X_{(n-k)}.

        log L = -(1 + alpha) * sum log R - lambda * sum (R^tau - 1)
                + sum log(alpha + lambda * tau * R^tau)
        """
        log_alpha, log_lam, log_tau = params
        alpha = np.exp(log_alpha)
        lam = np.exp(log_lam)
        tau = np.exp(log_tau)

        log_R = np.log(R)
        R_tau = R ** tau

        term1 = -(1.0 + alpha) * np.sum(log_R)
        term2 = -lam * np.sum(R_tau - 1.0)

        denom = alpha + lam * tau * R_tau
        if np.any(denom <= 0):
            return 1e10
        term3 = np.sum(np.log(denom))

        ll = term1 + term2 + term3
        if not np.isfinite(ll):
            return 1e10
        return float(-ll)

    # ------------------------------------------------------------------
    # WLS initializer
    # ------------------------------------------------------------------

    def _wls_init(self, R: np.ndarray) -> tuple[float, float, float]:
        """
        WLS initialization for alpha, lambda, tau.

        Minimize: sum_j [log R_j - (1/alpha)*log((k+1)/(k-j+1))
                         - lambda * h_tau(R_j)]^2
        where h_tau(x) = (x^tau - 1) / tau.

        Simple grid search over tau, then closed-form alpha, lambda.
        """
        k = len(R)
        log_R = np.log(R)

        # Plotting positions: expected log survival
        j_arr = np.arange(1, k + 1)
        log_pp = np.log((k + 1.0) / (k - j_arr + 1.0))  # -log S_empirical

        best_val = np.inf
        best_params = (self._alpha, self._lambda, self._tau)

        tau_grid = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.5])

        for tau in tau_grid:
            h_tau = (R**tau - 1.0) / tau

            # With fixed tau: linear in (1/alpha, lambda)
            # log_R_j = (1/alpha) * log_pp_j + lambda * h_tau_j + eps_j
            # Design matrix
            A = np.column_stack([log_pp, h_tau])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, log_R, rcond=None)
                inv_alpha = coeffs[0]
                lam = coeffs[1]
                if inv_alpha <= 0:
                    continue
                alpha = 1.0 / inv_alpha
                if lam < 0:
                    lam = 0.01
                residuals = log_R - A @ coeffs
                val = float(np.sum(residuals**2))
                if val < best_val:
                    best_val = val
                    best_params = (alpha, lam, tau)
            except Exception:
                continue

        return best_params

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        x: np.ndarray,
        threshold: float,
        k: Optional[int] = None,
    ) -> "WeibullTemperedPareto":
        """
        Fit by MLE to exceedances over threshold.

        Parameters
        ----------
        x : claim amounts
        threshold : float — POT threshold
        k : int or None — number of order statistics to use (default: all above threshold)

        Returns
        -------
        self
        """
        x = np.asarray(x, dtype=float)
        exceedances = x[x > threshold]
        if len(exceedances) == 0:
            raise ValueError("No observations above threshold")

        exceedances_sorted = np.sort(exceedances)[::-1]  # descending

        if k is None:
            k = len(exceedances_sorted)
        else:
            k = min(k, len(exceedances_sorted))

        if k < 10:
            warnings.warn(
                f"Only {k} exceedances used. WTP estimates may be unreliable.",
                UserWarning,
                stacklevel=2,
            )

        # Threshold in the relative-excess sense: X_{(n-k)}
        t_eff = exceedances_sorted[k - 1]  # k-th largest

        # Relative excesses R_j = X_{(n-j+1)} / X_{(n-k)} for j=1,...,k-1
        # (exclude the threshold itself to avoid log(1) = 0 dominating)
        R = exceedances_sorted[:k - 1] / t_eff
        if len(R) == 0:
            R = np.array([exceedances_sorted[0] / t_eff])

        # WLS initialization
        alpha_init, lam_init, tau_init = self._wls_init(R)

        # Multiple starts
        rng = np.random.default_rng(42)
        starts = [
            [np.log(alpha_init), np.log(max(lam_init, 1e-4)), np.log(tau_init)],
            [np.log(2.0), np.log(0.1), np.log(0.5)],
            [np.log(1.5), np.log(0.5), np.log(0.8)],
            [np.log(alpha_init * 1.2), np.log(max(lam_init * 0.5, 1e-4)), np.log(tau_init)],
            [np.log(alpha_init * 0.8), np.log(max(lam_init * 2.0, 1e-4)), np.log(tau_init * 1.2)],
        ]

        best_val = np.inf
        best_result = None

        for x0 in starts:
            try:
                res = optimize.minimize(
                    self._neg_loglik,
                    x0=x0,
                    args=(R,),
                    method="L-BFGS-B",
                    bounds=[
                        (np.log(0.1), np.log(20.0)),   # alpha
                        (np.log(1e-6), np.log(10.0)),   # lambda
                        (np.log(0.05), np.log(5.0)),    # tau
                    ],
                    options={"maxiter": 2000, "ftol": 1e-12},
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_result = res
            except Exception:
                continue

        if best_result is None:
            raise RuntimeError("WeibullTemperedPareto MLE failed to converge")

        self.alpha_ = float(np.exp(best_result.x[0]))
        self.lambda_ = float(np.exp(best_result.x[1]))
        self.tau_ = float(np.exp(best_result.x[2]))
        self.loglik_ = float(-best_val)
        self.k_ = k
        self.threshold_ = float(threshold)

        return self

    # ------------------------------------------------------------------
    # Probability methods
    # ------------------------------------------------------------------

    def logsf(self, x: np.ndarray, threshold: float) -> np.ndarray:
        """Log survival P(X > x | X > threshold)."""
        if self.alpha_ is None:
            raise RuntimeError("Call fit() before logsf()")
        x = np.asarray(x, dtype=float)
        r = x / threshold
        # Must have r >= 1 (x >= threshold)
        r = np.maximum(r, 1.0)
        return self._log_sf_r(r, self.alpha_, self.lambda_, self.tau_)

    def ppf(self, q: np.ndarray, threshold: float) -> np.ndarray:
        """
        Quantile (numerical inversion via Brentq).

        Find x such that P(X <= x | X > threshold) = q,
        i.e., P(X > x | X > threshold) = 1 - q.
        """
        if self.alpha_ is None:
            raise RuntimeError("Call fit() before ppf()")
        from scipy.optimize import brentq

        q = np.asarray(q, dtype=float)
        result = np.zeros_like(q)

        alpha, lam, tau = self.alpha_, self.lambda_, self.tau_

        for i, qi in enumerate(q.flat):
            if qi <= 0.0:
                result.flat[i] = threshold
                continue
            if qi >= 1.0:
                result.flat[i] = np.inf
                continue

            target_log_sf = np.log1p(-qi)  # log(1 - q)

            def objective(r):
                return self._log_sf_r(np.array([r]), alpha, lam, tau)[0] - target_log_sf

            # Find upper bracket: increase until log_sf < target
            r_upper = 2.0
            for _ in range(50):
                val = objective(r_upper)
                if val < 0:
                    break
                r_upper *= 2.0
            else:
                result.flat[i] = threshold * r_upper
                continue

            try:
                r_star = brentq(objective, 1.0, r_upper, xtol=1e-8, maxiter=100)
                result.flat[i] = float(threshold * r_star)
            except Exception:
                result.flat[i] = threshold * r_upper

        return result

    def return_level(
        self, T_years: float, n_per_year: float, alpha_threshold: float
    ) -> float:
        """
        Return level via numerical quantile.

        The probability of exceeding x in a single year is:
        P(X > x) = alpha_threshold * P(X > x | X > threshold)

        For a T-year return level: P(X > x) = 1 / (T_years * n_per_year).
        """
        if self.alpha_ is None:
            raise RuntimeError("Call fit() before return_level()")
        target_annual_prob = 1.0 / (T_years * n_per_year)
        # P(X > x | X > threshold) = target / alpha_threshold
        target_sf = target_annual_prob / alpha_threshold
        if target_sf >= 1.0:
            return float(self.threshold_)
        q = 1.0 - target_sf
        return float(self.ppf(np.array([q]), self.threshold_)[0])

    def effective_xi(self, x: float) -> float:
        """
        Local effective tail index at value x.

        d log P(X > x) / d log x ~= -alpha - lambda*tau*(x/threshold)^tau

        This decreases as x increases (tempering effect).
        Returns ~1/alpha for x near threshold.
        """
        if self.alpha_ is None:
            raise RuntimeError("Call fit() before effective_xi()")
        r = x / self.threshold_
        return float(1.0 / (self.alpha_ + self.lambda_ * self.tau_ * r**self.tau_))
