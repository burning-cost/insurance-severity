"""
TruncatedGPD: GPD tail with MLE corrected for upper truncation by policy limits.

Standard GPD MLE on truncated samples produces upward-biased xi (the tail
appears heavier than it is). This class corrects for that bias by including
the normalisation constant log F(T - u | xi, sigma) in the log-likelihood,
where T is the policy limit and u is the exceedance threshold.

Reference: Albrecher, Beirlant & Teugels (2025), arXiv:2511.22272, §4.3.
"""

from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
from scipy import optimize, stats

from insurance_severity.composite.distributions import TailDistribution


class TruncatedGPD(TailDistribution):
    """
    GPD tail with MLE corrected for upper truncation by policy limits.

    Parameters
    ----------
    xi : float, default 0.2
        Shape parameter. xi > 0: heavy tail; xi = 0: exponential; xi < 0: bounded.
    sigma : float, default 1.0
        Scale parameter (> 0).

    Notes
    -----
    Standard GPDTail ignores truncation; this class corrects for it.
    When limits=None (no truncation), fit_mle degenerates to standard GPD MLE.
    Per-observation heterogeneous limits are supported via a limits array.
    """

    def __init__(self, xi: float = 0.2, sigma: float = 1.0):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self._xi = float(xi)
        self._sigma = float(sigma)
        # Stored after fit for hessian-based CI
        self._fit_x: Optional[np.ndarray] = None
        self._fit_threshold: Optional[float] = None
        self._fit_limits: Optional[np.ndarray] = None
        self._loglik: Optional[float] = None

    @property
    def params(self) -> np.ndarray:
        return np.array([self._xi, self._sigma])

    @params.setter
    def params(self, values: np.ndarray) -> None:
        self._xi = float(values[0])
        self._sigma = float(values[1])

    def mode_value(self) -> Optional[float]:
        """GPD mode is positive only for xi < -0.5."""
        if self._xi < -0.5:
            return self._sigma * (-1.0 / self._xi - 1.0)
        return None

    # ------------------------------------------------------------------
    # Core log-likelihood helpers
    # ------------------------------------------------------------------

    def _log_f_gpd(self, z: np.ndarray, xi: float, sigma: float) -> np.ndarray:
        """Log density of GPD(xi, sigma) at exceedances z >= 0."""
        if abs(xi) < 1e-6:
            return -np.log(sigma) - z / sigma
        eta = 1.0 + xi * z / sigma
        # Return -inf for out-of-support points
        valid = eta > 0
        if xi < 0:
            # Bounded support: z < -sigma/xi
            valid = valid & (z < -sigma / xi)
        out = np.full_like(z, -np.inf)
        out[valid] = -np.log(sigma) - (1.0 / xi + 1.0) * np.log(eta[valid])
        return out

    def _log_F_gpd(self, w: float, xi: float, sigma: float) -> float:
        """
        log CDF of GPD(xi, sigma) at w (exceedance above threshold).
        F(w) = 1 - (1 + xi*w/sigma)^{-1/xi} for xi != 0.
        """
        if w <= 0:
            return -np.inf
        if abs(xi) < 1e-6:
            # F(w) = 1 - exp(-w/sigma)
            return float(np.log1p(-np.exp(-w / sigma)))
        eta_T = 1.0 + xi * w / sigma
        if eta_T <= 0:
            return -np.inf
        if xi < 0 and w >= -sigma / xi:
            # F reaches 1.0 at upper bound
            return 0.0  # log(1) = 0
        sf_T = eta_T ** (-1.0 / xi)
        if sf_T >= 1.0:
            return -np.inf  # F(w) <= 0
        return float(np.log1p(-sf_T))

    def _neg_loglik(
        self,
        params: np.ndarray,
        z: np.ndarray,
        T_minus_u: Optional[np.ndarray],
    ) -> float:
        """
        Negative log-likelihood for TruncatedGPD.

        Parameters
        ----------
        params : [xi, log_sigma]
        z : exceedances (x - threshold), shape (n,)
        T_minus_u : T_i - threshold per obs, or None if no truncation
        """
        xi = params[0]
        sigma = np.exp(params[1])

        log_f = self._log_f_gpd(z, xi, sigma)
        if not np.all(np.isfinite(log_f)):
            return 1e10

        ll = np.sum(log_f)

        if T_minus_u is not None:
            # Per-observation truncation correction
            for ti in T_minus_u:
                lF = self._log_F_gpd(float(ti), xi, sigma)
                if not np.isfinite(lF):
                    return 1e10
                ll -= lF

        if not np.isfinite(ll):
            return 1e10
        return float(-ll)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit_mle(
        self,
        x: np.ndarray,
        threshold: float,
        limits: Optional[Union[float, np.ndarray]] = None,
        xi_fixed: Optional[float] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Fit by truncation-corrected MLE.

        Parameters
        ----------
        x : array of claim amounts (must have x > threshold)
        threshold : float — splice threshold u
        limits : float, array of shape (n,), or None
            Policy limits. If scalar: same limit for all observations.
            If array: per-observation limits (must have limits > threshold).
            If None: no truncation (standard GPD MLE).
        xi_fixed : float or None
            If provided, fix xi at this value and fit only sigma.

        Returns
        -------
        params : ndarray [xi, sigma]
        info : dict with keys: success, loglik, n_obs, n_truncated
        """
        x = np.asarray(x, dtype=float)
        if len(x) == 0:
            raise ValueError("No observations above threshold")
        n = len(x)

        if n < 60:
            warnings.warn(
                f"Only {n} observations above threshold. "
                "Tail parameter estimates may be unreliable.",
                UserWarning,
                stacklevel=2,
            )

        z = x - threshold  # exceedances

        # Build per-obs truncation array T_i - u
        T_minus_u: Optional[np.ndarray] = None
        n_truncated = 0
        if limits is not None:
            if np.isscalar(limits):
                T_arr = np.full(n, float(limits))
            else:
                T_arr = np.asarray(limits, dtype=float)
                if len(T_arr) != n:
                    raise ValueError(
                        f"limits array length {len(T_arr)} != n_obs {n}"
                    )
            if np.any(T_arr <= threshold):
                raise ValueError("All limits must be strictly above threshold")
            # Only apply correction where limit is binding
            # (T_i - threshold > max(z) for that obs => limit not binding)
            T_minus_u = T_arr - threshold
            n_truncated = int(np.sum(T_arr < np.max(x) * 1.1))

        # Method of moments initialization for sigma starting point
        m = np.mean(z)
        v = np.var(z)
        if v > 0:
            xi0_mom = float(np.clip(0.5 * (1.0 - m**2 / v), -0.9, 1.5))
        else:
            xi0_mom = 0.1
        sigma0 = max(float(m * (1.0 - xi0_mom)), 0.1 * m, 1.0)

        best_val = np.inf
        best_result = None

        bounds = [(-0.9, 2.0), (-10.0, 10.0)]

        if xi_fixed is not None:
            # Only optimize sigma
            def neg_loglik_sigma_only(params):
                full_params = np.array([xi_fixed, params[0]])
                return self._neg_loglik(full_params, z, T_minus_u)

            for log_s0 in [np.log(sigma0), np.log(sigma0 * 1.5), np.log(sigma0 * 0.7)]:
                try:
                    res = optimize.minimize(
                        neg_loglik_sigma_only,
                        x0=[log_s0],
                        method="L-BFGS-B",
                        bounds=[(-10.0, 10.0)],
                    )
                    if res.fun < best_val:
                        best_val = res.fun
                        best_result = res
                except Exception:
                    continue

            if best_result is None:
                raise RuntimeError("TruncatedGPD MLE failed to converge")
            xi_hat = xi_fixed
            sigma_hat = float(np.exp(best_result.x[0]))
        else:
            # --- Two-step optimization ---
            # Step 1: Profile search over xi grid.
            # For each xi in the grid, find the optimal sigma via 1D optimization
            # with multiple sigma starting points. This is robust to the flat ridge
            # in the joint (xi, sigma) landscape that traps naive 2D optimization.
            xi_grid = np.linspace(-0.4, 1.5, 20)
            profile_starts = []  # (xi, log_sigma, neg_ll)

            # Multiple sigma starting points to avoid 1D local minima
            sigma_starts = [
                np.log(sigma0),
                np.log(max(sigma0 * 0.5, 1.0)),
                np.log(sigma0 * 2.0),
                np.log(max(m * 0.3, 1.0)),
                np.log(max(m, 1.0)),
            ]

            for xi_candidate in xi_grid:
                best_sub = np.inf
                best_log_s = np.log(sigma0)
                for log_s0 in sigma_starts:
                    def neg_ll_1d(log_s, xi_c=xi_candidate):
                        return self._neg_loglik(np.array([xi_c, log_s[0]]), z, T_minus_u)
                    try:
                        res_1d = optimize.minimize(
                            neg_ll_1d,
                            x0=[log_s0],
                            method="L-BFGS-B",
                            bounds=[(-5.0, 15.0)],
                            options={"maxiter": 300},
                        )
                        if res_1d.fun < best_sub:
                            best_sub = res_1d.fun
                            best_log_s = res_1d.x[0]
                    except Exception:
                        pass
                if np.isfinite(best_sub):
                    profile_starts.append((xi_candidate, best_log_s, best_sub))

            if not profile_starts:
                # Fallback to direct multi-start
                profile_starts = [(xi0_mom, np.log(sigma0), 1e10)]

            # Sort by neg_ll
            profile_starts.sort(key=lambda t: t[2])

            # Step 2: Refine using Powell (derivative-free) starting from best profile points.
            # L-BFGS-B fails on the flat landscape typical of the truncated GPD likelihood.
            # Powell is derivative-free and handles flat landscapes robustly.
            for xi_c, log_s, _ in profile_starts[:5]:
                try:
                    # Use Powell with soft bounds via clipping in the objective
                    def neg_ll_bounded(p):
                        xi_p = float(np.clip(p[0], -0.89, 1.99))
                        log_s_p = float(np.clip(p[1], -9.0, 14.0))
                        return self._neg_loglik(np.array([xi_p, log_s_p]), z, T_minus_u)

                    res = optimize.minimize(
                        neg_ll_bounded,
                        x0=[xi_c, log_s],
                        method="Powell",
                        options={"maxiter": 5000, "ftol": 1e-12, "xtol": 1e-8},
                    )
                    # Clip final result to hard bounds
                    xi_p = float(np.clip(res.x[0], -0.89, 1.99))
                    log_s_p = float(np.clip(res.x[1], -9.0, 14.0))
                    neg_ll_final = self._neg_loglik(np.array([xi_p, log_s_p]), z, T_minus_u)
                    if neg_ll_final < best_val:
                        best_val = neg_ll_final
                        # Build a fake result object to store params
                        best_result = type("Result", (), {"x": np.array([xi_p, log_s_p]), "success": res.success, "fun": neg_ll_final})()
                except Exception:
                    continue

            # Fallback: use best profile result directly
            if best_result is None:
                xi_c, log_s, nll = profile_starts[0]
                best_result = type("Result", (), {"x": np.array([xi_c, log_s]), "success": True, "fun": nll})()
                best_val = nll

            xi_hat = float(best_result.x[0])
            sigma_hat = float(np.exp(best_result.x[1]))

        self._xi = xi_hat
        self._sigma = sigma_hat
        self._fit_x = x.copy()
        self._fit_threshold = threshold
        self._fit_limits = T_minus_u
        self._loglik = float(-best_val)

        return np.array([xi_hat, sigma_hat]), {
            "success": best_result.success,
            "loglik": self._loglik,
            "n_obs": n,
            "n_truncated": n_truncated,
        }

    # ------------------------------------------------------------------
    # Probability methods
    # ------------------------------------------------------------------

    def logpdf(
        self,
        x: np.ndarray,
        threshold: float,
        limits: Optional[Union[float, np.ndarray]] = None,
    ) -> np.ndarray:
        """Truncation-adjusted log density."""
        x = np.asarray(x, dtype=float)
        z = x - threshold
        xi, sigma = self._xi, self._sigma

        log_f = self._log_f_gpd(z, xi, sigma)

        if limits is not None:
            n = len(x)
            if np.isscalar(limits):
                T_arr = np.full(n, float(limits))
            else:
                T_arr = np.asarray(limits, dtype=float)
            T_minus_u = T_arr - threshold
            correction = np.array([self._log_F_gpd(float(ti), xi, sigma) for ti in T_minus_u])
            log_f = log_f - correction

        return log_f

    def logsf(self, x: np.ndarray, threshold: float) -> np.ndarray:
        """Log survival — same as GPDTail (no truncation adjustment needed for evaluation)."""
        x = np.asarray(x, dtype=float)
        z = x - threshold
        return stats.genpareto.logsf(z, c=self._xi, scale=self._sigma)

    def ppf(self, q: np.ndarray, threshold: float) -> np.ndarray:
        """Quantile function (survival scale, no truncation adjustment)."""
        q = np.asarray(q, dtype=float)
        return stats.genpareto.ppf(q, c=self._xi, scale=self._sigma) + threshold

    # ------------------------------------------------------------------
    # Return level
    # ------------------------------------------------------------------

    def return_level(
        self,
        T_years: float,
        n_per_year: float,
        threshold: float,
        alpha_threshold: float,
        ci_method: str = "delta",
        alpha_ci: float = 0.05,
    ) -> dict:
        """
        Return level: loss exceeded on average once in T_years.

        z_T = u + (sigma/xi) * [((T_years * n_per_year * (1 - alpha_threshold))^xi) - 1]

        Parameters
        ----------
        T_years : float — return period
        n_per_year : float — mean claims per year
        threshold : float — threshold u
        alpha_threshold : float — probability of exceeding threshold (= 1 - pi)
        ci_method : 'delta' or 'profile_likelihood'
        alpha_ci : float — CI level (default 0.05 for 95% CI)

        Returns
        -------
        dict with keys: estimate, lower, upper, method
        """
        xi, sigma = self._xi, self._sigma
        m = T_years * n_per_year * alpha_threshold

        if abs(xi) < 1e-6:
            z_T = sigma * np.log(m)
        else:
            z_T = (sigma / xi) * (m ** xi - 1.0)

        rl = threshold + z_T

        if ci_method == "delta":
            # Delta method: gradient of z_T w.r.t. (xi, sigma)
            if abs(xi) < 1e-6:
                dz_dxi = sigma * (0.5 * np.log(m) ** 2)
                dz_dsigma = np.log(m)
            else:
                dz_dxi = (sigma / xi**2) * (1.0 - m**xi * (1.0 - xi * np.log(m)))
                dz_dsigma = (m**xi - 1.0) / xi

            # Numerical Hessian for variance estimate
            se_xi, se_sigma = self._parameter_se(threshold)
            var_rl = (dz_dxi * se_xi) ** 2 + (dz_dsigma * se_sigma) ** 2
            from scipy.stats import norm as snorm
            z_crit = snorm.ppf(1.0 - alpha_ci / 2.0)
            se_rl = float(np.sqrt(max(var_rl, 0.0)))
            lower = float(rl - z_crit * se_rl)
            upper = float(rl + z_crit * se_rl)

        elif ci_method == "profile_likelihood":
            lower, upper = self._profile_likelihood_rl_ci(
                T_years, n_per_year, threshold, alpha_threshold, alpha_ci
            )
        else:
            raise ValueError(f"ci_method must be 'delta' or 'profile_likelihood', got {ci_method!r}")

        return {"estimate": float(rl), "lower": lower, "upper": upper, "method": ci_method}

    def _parameter_se(self, threshold: float) -> tuple[float, float]:
        """Approximate standard errors for xi and sigma via numerical Hessian."""
        if self._fit_x is None:
            return (0.05, self._sigma * 0.1)

        z = self._fit_x - threshold
        xi, sigma = self._xi, self._sigma

        def neg_ll(p):
            return self._neg_loglik(p, z, self._fit_limits)

        x0 = np.array([xi, np.log(sigma)])
        try:
            from scipy.optimize import approx_fprime
            eps = 1e-5
            n_p = 2
            H = np.zeros((n_p, n_p))
            grad0 = approx_fprime(x0, neg_ll, eps)
            for i in range(n_p):
                x_plus = x0.copy()
                x_plus[i] += eps
                grad_plus = approx_fprime(x_plus, neg_ll, eps)
                H[i] = (grad_plus - grad0) / eps
            H = 0.5 * (H + H.T)  # symmetrize
            try:
                cov = np.linalg.inv(H)
                se_xi = float(np.sqrt(max(cov[0, 0], 0.0)))
                # sigma is parameterized as log_sigma, so SE in sigma space
                se_log_sigma = float(np.sqrt(max(cov[1, 1], 0.0)))
                se_sigma = se_log_sigma * sigma
                return se_xi, se_sigma
            except np.linalg.LinAlgError:
                return (0.05, sigma * 0.1)
        except Exception:
            return (0.05, sigma * 0.1)

    def _profile_likelihood_rl_ci(
        self,
        T_years: float,
        n_per_year: float,
        threshold: float,
        alpha_threshold: float,
        alpha_ci: float,
        n_grid: int = 50,
    ) -> tuple[float, float]:
        """Profile likelihood CI for return level."""
        if self._fit_x is None:
            rl = self.return_level(T_years, n_per_year, threshold, alpha_threshold)["estimate"]
            return rl * 0.8, rl * 1.2

        from scipy.stats import chi2
        cutoff = self._loglik - 0.5 * chi2.ppf(1.0 - alpha_ci, df=1)

        # Grid over xi
        xi_grid = np.linspace(max(self._xi - 0.5, -0.89), min(self._xi + 0.5, 1.9), n_grid)
        rl_values = []

        for xi_val in xi_grid:
            # Profile: maximize over sigma with xi fixed
            z = self._fit_x - threshold
            def neg_ll_sigma(p, xi_fixed=xi_val):
                return self._neg_loglik(np.array([xi_fixed, p[0]]), z, self._fit_limits)

            res = optimize.minimize(
                neg_ll_sigma,
                x0=[np.log(self._sigma)],
                method="L-BFGS-B",
                bounds=[(-10.0, 10.0)],
            )
            profile_ll = -res.fun
            if profile_ll >= cutoff:
                sigma_val = float(np.exp(res.x[0]))
                m = T_years * n_per_year * alpha_threshold
                if abs(xi_val) < 1e-6:
                    z_T = sigma_val * np.log(m)
                else:
                    z_T = (sigma_val / xi_val) * (m**xi_val - 1.0)
                rl_values.append(threshold + z_T)

        if len(rl_values) == 0:
            rl = self.return_level(T_years, n_per_year, threshold, alpha_threshold)["estimate"]
            return rl * 0.8, rl * 1.2

        return float(min(rl_values)), float(max(rl_values))

    # ------------------------------------------------------------------
    # Profile likelihood CI for parameters
    # ------------------------------------------------------------------

    def profile_likelihood_ci(
        self,
        x: np.ndarray,
        threshold: float,
        limits: Optional[Union[float, np.ndarray]] = None,
        param: str = "xi",
        alpha: float = 0.05,
        n_grid: int = 50,
    ) -> tuple[float, float]:
        """
        Profile likelihood confidence interval for xi or sigma.

        Based on chi-squared cutoff: 2*(loglik_max - loglik_profile) <= chi^2_{1,alpha}.

        Parameters
        ----------
        x : observations (above threshold)
        threshold : splice threshold
        limits : policy limits (optional)
        param : 'xi' or 'sigma'
        alpha : CI level
        n_grid : grid points for profile search

        Returns
        -------
        (lower, upper) confidence interval
        """
        from scipy.stats import chi2

        x = np.asarray(x, dtype=float)
        z = x - threshold

        # Resolve limits
        if limits is not None:
            if np.isscalar(limits):
                T_minus_u = np.full(len(x), float(limits) - threshold)
            else:
                T_minus_u = np.asarray(limits, dtype=float) - threshold
        else:
            T_minus_u = None

        # Fit to get MLE if not already done
        if self._loglik is None or self._fit_x is None:
            self.fit_mle(x, threshold, limits)

        loglik_max = self._loglik
        cutoff = loglik_max - 0.5 * chi2.ppf(1.0 - alpha, df=1)

        xi_mle, sigma_mle = self._xi, self._sigma

        if param == "xi":
            xi_grid = np.linspace(max(xi_mle - 1.0, -0.89), min(xi_mle + 1.0, 1.9), n_grid)
            in_ci = []
            for xi_val in xi_grid:
                def neg_ll_s(p, xv=xi_val):
                    return self._neg_loglik(np.array([xv, p[0]]), z, T_minus_u)
                res = optimize.minimize(
                    neg_ll_s, x0=[np.log(sigma_mle)], method="L-BFGS-B",
                    bounds=[(-10.0, 10.0)]
                )
                if -res.fun >= cutoff:
                    in_ci.append(xi_val)
            if len(in_ci) == 0:
                return float(xi_mle - 0.1), float(xi_mle + 0.1)
            return float(min(in_ci)), float(max(in_ci))

        elif param == "sigma":
            sigma_grid = np.exp(np.linspace(
                np.log(sigma_mle * 0.3), np.log(sigma_mle * 3.0), n_grid
            ))
            in_ci = []
            for sigma_val in sigma_grid:
                def neg_ll_xi(p, sv=sigma_val):
                    return self._neg_loglik(np.array([p[0], np.log(sv)]), z, T_minus_u)
                res = optimize.minimize(
                    neg_ll_xi, x0=[xi_mle], method="L-BFGS-B",
                    bounds=[(-0.9, 2.0)]
                )
                if -res.fun >= cutoff:
                    in_ci.append(sigma_val)
            if len(in_ci) == 0:
                return float(sigma_mle * 0.8), float(sigma_mle * 1.2)
            return float(min(in_ci)), float(max(in_ci))

        else:
            raise ValueError(f"param must be 'xi' or 'sigma', got {param!r}")
