"""
EVT (Extreme Value Theory) classes for truncated and censored insurance claims.

Three classes address the most common data quality issues in large-loss modelling:

- TruncatedGPD: GPD fitted via truncated MLE, correcting for upper truncation
  at policy limits. Essential when you have heterogeneous per-policy limits.

- CensoredHillEstimator: Hill estimator corrected for IBNR right-censoring.
  Applies the Albrecher et al. (2025) correction to the standard Hill estimator.

- WeibullTemperedPareto: Survival function S(x) = x^{-alpha} * exp(-lambda * x^tau).
  For tails that are heavy but physically bounded (property, D&O, PI).

Reference: Albrecher, Beirlant & Teugels (2025), arXiv:2511.22272
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy import optimize, stats


# ---------------------------------------------------------------------------
# TruncatedGPD
# ---------------------------------------------------------------------------


class TruncatedGPD:
    """
    GPD fitted via truncated MLE, correcting for upper truncation at policy limits.

    When policy limits T_i cap losses, the observed exceedances are drawn from
    a GPD *truncated* above at (T_i - u).  Standard GPD MLE ignores this and
    underestimates xi.  This class adjusts the log-likelihood accordingly.

    Parameters
    ----------
    threshold : float
        POT threshold u.  Only claim amounts strictly above u are modelled.

    Notes
    -----
    The log-likelihood correction subtracts log(F_GPD(T_i - u | xi, sigma))
    for each observation where T_i < inf.  For uncapped losses pass
    limits=np.inf * np.ones(n).

    Optimizer: the truncated GPD surface is extremely flat in xi.  L-BFGS-B
    and SLSQP fail to find the true optimum.  We use a two-step approach:
    1. Profile over a grid of 20 xi values; for each, minimise in sigma via
       1-D L-BFGS-B with 5 sigma starting points.
    2. Powell (derivative-free) joint refinement from the top-5 profile results.
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = float(threshold)
        self._xi: Optional[float] = None
        self._sigma: Optional[float] = None
        self._hess_inv: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Core likelihood
    # ------------------------------------------------------------------

    def _log_lik(self, xi: float, sigma: float, y: np.ndarray, trunc: np.ndarray) -> float:
        """Truncation-corrected GPD log-likelihood.

        y      : exceedances (x - u), shape (n,)
        trunc  : truncation points (T_i - u), shape (n,).  inf means uncapped.
        """
        n = len(y)
        if sigma <= 0:
            return -np.inf

        # GPD log-likelihood term
        if abs(xi) < 1e-10:
            # Exponential limit
            if np.any(y < 0):
                return -np.inf
            ll = -n * np.log(sigma) - np.sum(y) / sigma
        else:
            z = 1.0 + xi * y / sigma
            zt = np.where(np.isfinite(trunc), 1.0 + xi * trunc / sigma, np.inf)

            # Support check: z > 0 required
            if np.any(z <= 0):
                return -np.inf

            # For xi < 0, upper bound = -sigma/xi; clip truncation points
            if xi < 0:
                ub = -sigma / xi
                zt = np.where(np.isfinite(trunc), np.minimum(zt, 0.9999 * (1 + xi * ub / sigma)), zt)
                if np.any(zt <= 0):
                    return -np.inf

            ll = -n * np.log(sigma) - (1.0 + 1.0 / xi) * np.sum(np.log(z))

            # Truncation correction: subtract log CDF(trunc) for capped obs
            capped = np.isfinite(trunc)
            if np.any(capped):
                zt_capped = zt[capped]
                if xi < 0:
                    # Some zt values may be beyond finite upper bound
                    zt_capped = np.clip(zt_capped, 1e-10, None)
                log_cdf_trunc = np.log(1.0 - zt_capped ** (-1.0 / xi))
                ll -= np.sum(log_cdf_trunc)

        return ll

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, exceedances: np.ndarray, limits: np.ndarray) -> "TruncatedGPD":
        """
        Fit truncated GPD via profile MLE.

        Parameters
        ----------
        exceedances : array-like, shape (n,)
            Claim amounts above threshold (i.e. x - u).
        limits : array-like, shape (n,)
            Per-observation policy limits T_i.  Use np.inf for uncapped.

        Returns
        -------
        self
        """
        y = np.asarray(exceedances, dtype=float)
        lims = np.asarray(limits, dtype=float)
        trunc = lims - self.threshold  # truncation points for exceedances

        if len(y) == 0:
            raise ValueError("exceedances is empty")
        if len(y) != len(lims):
            raise ValueError("exceedances and limits must have the same length")

        # Method-of-moments starting point
        mean = np.mean(y)
        var = np.var(y)
        if var > 0:
            xi0 = 0.5 * (mean**2 / var - 1.0)
            sigma0 = 0.5 * mean * (mean**2 / var + 1.0)
        else:
            xi0 = 0.1
            sigma0 = mean

        sigma0 = max(sigma0, 1.0)

        # Step 1: profile search over xi grid
        xi_grid = np.linspace(-0.4, 1.5, 20)
        sigma_starts = [sigma0, sigma0 * 0.5, sigma0 * 2.0, mean * 0.3, mean]
        sigma_starts = [max(s, 1.0) for s in sigma_starts]

        profile_results: list[tuple[float, float, float]] = []

        for xi in xi_grid:
            best_nll = np.inf
            best_sigma = sigma0
            for s0 in sigma_starts:
                def neg_ll_sigma(log_s: np.ndarray, _xi: float = xi) -> float:
                    s = np.exp(log_s[0])
                    return -self._log_lik(_xi, s, y, trunc)

                res = optimize.minimize(
                    neg_ll_sigma,
                    x0=[np.log(s0)],
                    method="L-BFGS-B",
                    options={"maxiter": 200, "ftol": 1e-12},
                )
                if res.fun < best_nll:
                    best_nll = res.fun
                    best_sigma = np.exp(res.x[0])

            profile_results.append((best_nll, xi, best_sigma))

        # Step 2: Powell refinement from top-5 profile points
        profile_results.sort(key=lambda t: t[0])
        best_nll_final = np.inf
        best_xi_final = xi0
        best_sigma_final = sigma0

        for nll0, xi_init, sigma_init in profile_results[:5]:
            def neg_ll_joint(params: np.ndarray) -> float:
                xi, log_s = params
                s = np.exp(log_s)
                return -self._log_lik(xi, s, y, trunc)

            res = optimize.minimize(
                neg_ll_joint,
                x0=[xi_init, np.log(sigma_init)],
                method="Powell",
                options={"maxiter": 5000, "ftol": 1e-12, "xtol": 1e-8},
            )
            if res.fun < best_nll_final:
                best_nll_final = res.fun
                best_xi_final = res.x[0]
                best_sigma_final = np.exp(res.x[1])

        self._xi = float(best_xi_final)
        self._sigma = float(best_sigma_final)

        # Numerical Hessian for standard errors
        try:
            def neg_ll_se(params: np.ndarray) -> float:
                xi, log_s = params
                s = np.exp(log_s)
                return -self._log_lik(xi, s, y, trunc)

            hess = optimize.approx_fprime(
                [self._xi, np.log(self._sigma)],
                lambda p: optimize.approx_fprime(p, neg_ll_se, 1e-5),
                1e-5,
            )
            self._hess_inv = np.linalg.pinv(np.array(hess).reshape(2, 2))
        except Exception:
            self._hess_inv = None

        return self

    # ------------------------------------------------------------------
    # Distribution methods
    # ------------------------------------------------------------------

    @property
    def xi(self) -> float:
        if self._xi is None:
            raise RuntimeError("Call fit() first")
        return self._xi

    @property
    def sigma(self) -> float:
        if self._sigma is None:
            raise RuntimeError("Call fit() first")
        return self._sigma

    @property
    def params(self) -> dict:
        return {"xi": self.xi, "sigma": self.sigma, "threshold": self.threshold}

    def _z(self, x: np.ndarray) -> np.ndarray:
        """Standardised exceedance z = 1 + xi*(x-u)/sigma."""
        return 1.0 + self.xi * (x - self.threshold) / self.sigma

    def sf(self, x: np.ndarray) -> np.ndarray:
        """Survival function of fitted GPD."""
        x = np.asarray(x, dtype=float)
        z = self._z(x)
        if abs(self.xi) < 1e-10:
            return np.exp(-(x - self.threshold) / self.sigma)
        return np.where(z > 0, z ** (-1.0 / self.xi), 0.0)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return 1.0 - self.sf(x)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        z = self._z(x)
        if abs(self.xi) < 1e-10:
            return np.where(x >= self.threshold, np.exp(-(x - self.threshold) / self.sigma) / self.sigma, 0.0)
        exponent = 1.0 + 1.0 / self.xi
        return np.where(z > 0, z ** (-exponent) / self.sigma, 0.0)

    def isf(self, q: np.ndarray) -> np.ndarray:
        """Inverse survival function (quantile above threshold)."""
        q = np.asarray(q, dtype=float)
        if abs(self.xi) < 1e-10:
            return self.threshold - self.sigma * np.log(q)
        return self.threshold + self.sigma * (q ** (-self.xi) - 1.0) / self.xi

    def summary(self) -> dict:
        """Return fitted parameters with standard errors."""
        result = {"xi": self.xi, "sigma": self.sigma, "threshold": self.threshold}
        if self._hess_inv is not None:
            se = np.sqrt(np.abs(np.diag(self._hess_inv)))
            # se is for (xi, log_sigma); delta method for sigma
            result["se_xi"] = float(se[0])
            result["se_sigma"] = float(se[1] * self.sigma)
        return result


# ---------------------------------------------------------------------------
# CensoredHillEstimator
# ---------------------------------------------------------------------------


class CensoredHillEstimator:
    """
    Hill estimator corrected for IBNR right-censoring.

    Standard Hill uses top-k order statistics and assumes all are fully
    developed claims.  IBNR claims are right-censored: we see a partial
    development amount but the final settlement will be larger.  The Albrecher
    et al. (2025) correction divides the Hill numerator by the empirical
    proportion of uncensored claims among the top-k.

    The corrected estimator is:

        H_k^(c) = [ (1/k) * sum_{j=1}^{k} log(X_{(n-j+1)} / X_{(n-k)}) ]
                  / [ (number of uncensored in top-k) / k ]

    Parameters
    ----------
    None required at construction time.

    References
    ----------
    Albrecher, Beirlant & Teugels (2025), arXiv:2511.22272, §3.2
    """

    def __init__(self) -> None:
        self._xi: Optional[float] = None
        self._k_opt: Optional[int] = None
        self._ci: Optional[tuple[float, float]] = None
        self._hill_vals: Optional[np.ndarray] = None
        self._k_vals: Optional[np.ndarray] = None

    @staticmethod
    def _hill_corrected(x_ord: np.ndarray, c_ord: np.ndarray, k: int) -> float:
        """
        Compute censoring-corrected Hill estimator at order k.

        x_ord : observations sorted descending
        c_ord : censoring indicator sorted descending (True = censored)
        k     : number of top-k order statistics (boundary is x_ord[k])

        Returns xi estimate or np.nan if undefined.
        """
        if k < 2 or k >= len(x_ord):
            return np.nan
        # Standard Hill numerator: (1/k) * sum_{j=0}^{k-1} log(x_ord[j] / x_ord[k])
        hill_num = (np.sum(np.log(x_ord[:k])) - k * np.log(x_ord[k])) / k
        # Correction: divide by fraction of uncensored in top-k (positions 0..k-1)
        uncensored_frac = np.sum(~c_ord[:k]) / k
        if uncensored_frac < 1e-10:
            return np.nan
        return hill_num / uncensored_frac

    def fit(
        self,
        claims: np.ndarray,
        censored: np.ndarray,
        n_bootstrap: int = 200,
        rng_seed: int = 0,
    ) -> "CensoredHillEstimator":
        """
        Fit the censoring-corrected Hill estimator.

        Parameters
        ----------
        claims : array-like, shape (n,)
            Claim amounts (fully developed or partially developed for IBNR).
        censored : array-like of bool, shape (n,)
            True = right-censored (IBNR).  False = fully settled.
        n_bootstrap : int
            Bootstrap replicates for confidence interval.
        rng_seed : int
            Random seed for reproducibility.

        Returns
        -------
        self
        """
        x = np.asarray(claims, dtype=float)
        c = np.asarray(censored, dtype=bool)

        if len(x) == 0:
            raise ValueError("claims is empty")
        if np.all(c):
            warnings.warn("All claims are censored; Hill estimator unreliable.", stacklevel=2)

        n = len(x)
        order = np.argsort(x)[::-1]  # descending
        x_ord = x[order]
        c_ord = c[order]  # censoring indicator in sorted order

        # Compute corrected Hill(k) for k = 2 ... n-1
        # Need at least one observation after the boundary: k < n
        k_vals = np.arange(2, n)
        hill_vals = np.array([
            self._hill_corrected(x_ord, c_ord, int(k))
            for k in k_vals
        ])

        # k selection: rolling variance minimisation over window of 10
        window = 10
        valid = ~np.isnan(hill_vals)
        if np.sum(valid) < window:
            # Fallback: pick median k among valid
            valid_k = k_vals[valid]
            k_opt = int(valid_k[len(valid_k) // 2]) if len(valid_k) > 0 else int(k_vals[0])
        else:
            valid_idx = np.where(valid)[0]
            rolling_var = np.array([
                np.var(hill_vals[max(0, i - window // 2): i + window // 2])
                for i in valid_idx
            ])
            best_i = valid_idx[np.argmin(rolling_var)]
            k_opt = int(k_vals[best_i])

        xi_hat = float(self._hill_corrected(x_ord, c_ord, k_opt))

        # Bootstrap CI at fixed k_opt
        rng = np.random.default_rng(rng_seed)
        boot_xi = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            xb = x[idx]
            cb = c[idx]
            ob = np.argsort(xb)[::-1]
            xb_ord = xb[ob]
            cb_ord = cb[ob]
            k = min(k_opt, len(xb) - 2)
            if k < 2:
                boot_xi[b] = xi_hat
                continue
            val = self._hill_corrected(xb_ord, cb_ord, k)
            boot_xi[b] = val if np.isfinite(val) else xi_hat

        ci = (float(np.percentile(boot_xi, 2.5)), float(np.percentile(boot_xi, 97.5)))

        self._xi = xi_hat
        self._k_opt = k_opt
        self._ci = ci
        self._hill_vals = hill_vals
        self._k_vals = k_vals

        return self

    @property
    def xi(self) -> float:
        if self._xi is None:
            raise RuntimeError("Call fit() first")
        return self._xi

    @property
    def k_opt(self) -> int:
        if self._k_opt is None:
            raise RuntimeError("Call fit() first")
        return self._k_opt

    @property
    def ci(self) -> tuple[float, float]:
        if self._ci is None:
            raise RuntimeError("Call fit() first")
        return self._ci

    def hill_plot(self, ax=None):
        """
        Plot Hill estimates vs k with CI band at k_opt.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            If None, creates a new figure.

        Returns
        -------
        ax
        """
        import matplotlib.pyplot as plt

        if self._hill_vals is None:
            raise RuntimeError("Call fit() first")

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        ax.plot(self._k_vals, self._hill_vals, lw=1.5, color="steelblue", label="Hill(k)")
        ax.axvline(self._k_opt, color="tomato", ls="--", lw=1.5, label=f"k_opt={self._k_opt}")
        if self._ci is not None:
            ax.axhline(self._ci[0], color="grey", ls=":", lw=1)
            ax.axhline(self._ci[1], color="grey", ls=":", lw=1, label="95% CI")
        ax.set_xlabel("k (number of order statistics)")
        ax.set_ylabel("Hill estimate (xi)")
        ax.set_title("Censoring-corrected Hill plot")
        ax.legend()
        return ax


# ---------------------------------------------------------------------------
# WeibullTemperedPareto
# ---------------------------------------------------------------------------


class WeibullTemperedPareto:
    """
    Weibull-tempered Pareto tail model.

    Survival function: S(x) = x^{-alpha} * exp(-lambda * x^tau)

    This model is parameterised for x > 0 (the full claim amount, not an
    exceedance shifted to zero).  Fit it on the actual claim values above your
    threshold, not on (claim - threshold).

    Pure Pareto is recovered as lambda -> 0.  This model suits lines where the
    tail is heavy but physically bounded (property, D&O, PI): the tail index
    alpha describes the Pareto-like core, and (lambda, tau) describe the
    exponential dampening.

    Parameters
    ----------
    threshold : float
        POT threshold.  Exceedances passed to fit() should be the raw claim
        values (x > threshold), not (x - threshold).
    k : int, optional
        Number of top order statistics to use.  Defaults to all exceedances.

    References
    ----------
    Albrecher, Beirlant & Teugels (2025), arXiv:2511.22272
    """

    def __init__(self, threshold: float, k: Optional[int] = None) -> None:
        self.threshold = float(threshold)
        self.k = k
        self._alpha: Optional[float] = None
        self._lam: Optional[float] = None
        self._tau: Optional[float] = None
        self._x_ref: Optional[float] = None  # X_{(n-k)} reference point

    # ------------------------------------------------------------------
    # Log-likelihood
    # ------------------------------------------------------------------

    def _log_lik(
        self,
        alpha: float,
        lam: float,
        tau: float,
        top_k: np.ndarray,
        x_ref: float,
    ) -> float:
        """
        POT log-likelihood for WTP on top-k observations.

        log L = sum_j log f(X_j) - k * log S(x_ref)

        where f(x) = (alpha/x + lambda*tau*x^{tau-1}) * x^{-alpha} * exp(-lam*x^tau)
        """
        if alpha <= 0 or lam < 0 or tau <= 0:
            return -np.inf
        if np.any(top_k <= 0) or x_ref <= 0:
            return -np.inf

        log_x = np.log(top_k)
        x_tau = top_k ** tau

        # log f(x) = log(alpha/x + lambda*tau*x^{tau-1}) - alpha*log(x) - lam*x^tau
        hazard = alpha / top_k + lam * tau * top_k ** (tau - 1.0)
        log_hazard = np.log(np.maximum(hazard, 1e-300))

        density_term = np.sum(log_hazard - alpha * log_x - lam * x_tau)

        # Normalisation: subtract k * log S(x_ref)
        log_S_ref = -alpha * np.log(x_ref) - lam * x_ref ** tau
        norm_term = len(top_k) * log_S_ref

        return density_term - norm_term

    def fit(self, exceedances: np.ndarray) -> "WeibullTemperedPareto":
        """
        Fit Weibull-tempered Pareto via POT MLE.

        Parameters
        ----------
        exceedances : array-like, shape (n,)
            Claim amounts above threshold (raw claim values, x > threshold,
            NOT x - threshold).  Must be strictly positive.

        Returns
        -------
        self
        """
        y = np.asarray(exceedances, dtype=float)
        if len(y) == 0:
            raise ValueError("exceedances is empty")
        if np.any(y <= 0):
            raise ValueError(
                "WeibullTemperedPareto requires strictly positive claim values. "
                f"Got {int(np.sum(y <= 0))} non-positive value(s). "
                "Filter out non-positive values before calling fit()."
            )

        y = np.sort(y)[::-1]  # descending

        k = self.k if self.k is not None else len(y)
        k = min(k, len(y))
        if k < 2:
            raise ValueError(f"Need at least 2 observations, got {k}")

        x_ref = float(y[k - 1])  # X_{(n-k)}: k-th largest = boundary
        top_k = y[:k]

        self._x_ref = x_ref

        # Initial Hill estimate for alpha
        hill_denom = np.sum(np.log(top_k) - np.log(x_ref))
        hill_alpha = k / hill_denom if hill_denom > 0 else 1.0
        hill_alpha = max(hill_alpha, 0.5)

        # Scale-adjusted initial lambda values
        x_scale = np.median(top_k)
        initial_points = [
            (hill_alpha, 1e-6 / x_scale**0.5, 0.5),
            (hill_alpha * 0.7, 1e-4 / x_scale**0.5, 0.3),
            (hill_alpha * 1.3, 1e-7 / x_scale**0.5, 0.8),
            (max(hill_alpha * 0.5, 0.1), 1e-3 / x_scale**0.5, 1.0),
        ]

        best_nll = np.inf
        best_params: np.ndarray = np.array([np.log(hill_alpha), np.log(1e-6), np.log(0.5)])

        for alpha0, lam0, tau0 in initial_points:
            def neg_ll(
                p: np.ndarray,
                _top_k: np.ndarray = top_k,
                _xr: float = x_ref,
            ) -> float:
                alpha = np.exp(p[0])
                lam = np.exp(p[1])
                tau = np.exp(p[2])
                ll = self._log_lik(alpha, lam, tau, _top_k, _xr)
                return -ll if np.isfinite(ll) else 1e15

            p0 = [np.log(alpha0), np.log(max(lam0, 1e-12)), np.log(tau0)]
            res = optimize.minimize(
                neg_ll,
                x0=p0,
                method="Powell",
                options={"maxiter": 10000, "ftol": 1e-12, "xtol": 1e-8},
            )
            if res.fun < best_nll:
                best_nll = res.fun
                best_params = res.x

        self._alpha = float(np.exp(best_params[0]))
        self._lam = float(np.exp(best_params[1]))
        self._tau = float(np.exp(best_params[2]))

        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def alpha(self) -> float:
        if self._alpha is None:
            raise RuntimeError("Call fit() first")
        return self._alpha

    @property
    def lam(self) -> float:
        if self._lam is None:
            raise RuntimeError("Call fit() first")
        return self._lam

    @property
    def tau(self) -> float:
        if self._tau is None:
            raise RuntimeError("Call fit() first")
        return self._tau

    @property
    def xi(self) -> float:
        """Tail index xi = 1/alpha, for cross-model comparison with TruncatedGPD."""
        return 1.0 / self.alpha

    # ------------------------------------------------------------------
    # Distribution methods
    # ------------------------------------------------------------------

    def sf(self, x: np.ndarray) -> np.ndarray:
        """Survival function S(x) = x^{-alpha} * exp(-lambda * x^tau)."""
        x = np.asarray(x, dtype=float)
        return np.where(x > 0, x ** (-self.alpha) * np.exp(-self.lam * x ** self.tau), 1.0)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return 1.0 - self.sf(x)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """f(x) = (alpha/x + lambda*tau*x^{tau-1}) * S(x)."""
        x = np.asarray(x, dtype=float)
        s = self.sf(x)
        return np.where(
            x > 0,
            (self.alpha / x + self.lam * self.tau * x ** (self.tau - 1.0)) * s,
            0.0,
        )

    def isf(self, q: np.ndarray) -> np.ndarray:
        """
        Inverse survival function via Brent root-finding.

        Uses default argument _t=target_i to avoid closure capture bug.
        """
        q = np.asarray(q, dtype=float)
        scalar = q.ndim == 0
        q = np.atleast_1d(q)
        result = np.empty_like(q)

        x_max = 1e12

        for i, qi in enumerate(q):
            target_i = float(qi)

            def _obj(x: float, _t: float = target_i) -> float:
                return float(np.atleast_1d(self.sf(np.array([x])))[0]) - _t

            try:
                # Check bracket
                f_lo = _obj(1e-10)
                f_hi = _obj(x_max)
                if f_lo * f_hi > 0:
                    result[i] = x_max if f_hi > 0 else 1e-10
                else:
                    result[i] = float(optimize.brentq(_obj, 1e-10, x_max, xtol=1e-6, rtol=1e-9))
            except Exception:
                result[i] = np.nan

        if scalar:
            return float(np.atleast_1d(result)[0])
        return result

    def summary(self) -> dict:
        return {
            "alpha": self.alpha,
            "lambda": self.lam,
            "tau": self.tau,
            "xi": self.xi,
            "threshold": self.threshold,
        }
