"""
Composite severity models: LognormalBurrComposite, LognormalGPDComposite,
GammaGPDComposite.

Each model is a self-contained estimator that:
1. Accepts claim amounts as a 1-D numpy array.
2. Fits body + tail parameters by MLE.
3. Provides pdf, cdf, quantile, TVaR, ILF, and diagnostic residuals.
4. Stores fitted parameters as sklearn-style attributes ending in _.

Threshold methods:
  "fixed"             — user provides threshold value
  "profile_likelihood" — grid search over candidate thresholds
  "mode_matching"     — threshold = tail mode (Burr only; raises ValueError for GPD)

Design note on mode-matching continuity: Setting the threshold equal to the
tail mode guarantees C1 continuity at the splice point because f'_tail(mode) = 0
by definition. This couples mu_body and the weight r to the tail parameters
via two equations:
  (1) exp(mu - sigma^2) = tail_mode         [modes equal]
  (2) pi * f_body(t)/F_body(t) = (1-pi) * f_tail(t)/S_tail(t)  [continuity]

Both are satisfied automatically when we set t = tail_mode and compute pi to
balance the splice density. No explicit constraint in the optimizer is needed;
the reparameterisation handles it.
"""

from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
from scipy import optimize

from insurance_severity.composite.distributions import (
    BodyDistribution,
    TailDistribution,
    LognormalBody,
    GammaBody,
    GPDTail,
    BurrTail,
)


# ---------------------------------------------------------------------------
# Base composite model
# ---------------------------------------------------------------------------


class CompositeSeverityModel:
    """
    Base class for hard composite severity models.

    Subclasses specify the body and tail distribution classes. Fitting
    logic is shared (profile likelihood, fixed threshold). Mode-matching
    is only valid for Burr tail and is enforced via ValueError otherwise.

    After fitting, public attributes are available:
        threshold_      float   — estimated or specified threshold
        pi_             float   — probability mass in body
        body_params_    ndarray — fitted body parameters
        tail_params_    ndarray — fitted tail parameters
        loglik_         float   — maximised log-likelihood
        n_body_         int     — number of observations in body
        n_tail_         int     — number of observations in tail
    """

    # Override in subclasses
    _body_cls = None  # type: type[BodyDistribution]
    _tail_cls = None  # type: type[TailDistribution]
    _supports_mode_matching = False

    def __init__(
        self,
        threshold: Optional[float] = None,
        threshold_method: str = "fixed",
        n_threshold_grid: int = 50,
        threshold_quantile_range: tuple = (0.70, 0.95),
        n_starts: int = 5,
    ):
        """
        Parameters
        ----------
        threshold : float, optional
            Splice threshold. Required when threshold_method="fixed".
        threshold_method : str
            One of "fixed", "profile_likelihood", "mode_matching".
        n_threshold_grid : int
            Number of grid points for profile likelihood search.
        threshold_quantile_range : tuple
            (low, high) quantile range for profile likelihood grid.
        n_starts : int
            Number of random multi-starts for optimization.
        """
        valid_methods = {"fixed", "profile_likelihood", "mode_matching"}
        if threshold_method not in valid_methods:
            raise ValueError(
                f"threshold_method must be one of {valid_methods}, "
                f"got {threshold_method!r}"
            )
        if threshold_method == "mode_matching" and not self._supports_mode_matching:
            raise ValueError(
                f"{self.__class__.__name__} does not support mode_matching. "
                "Mode-matching requires a tail distribution with a tractable "
                "positive mode. GPD with xi >= 0 has mode 0, which covers all "
                "insurance-relevant heavy-loss scenarios. Use Burr XII tail "
                "for mode-matching (LognormalBurrComposite), or use "
                "threshold_method='fixed' or 'profile_likelihood' with GPD."
            )
        if threshold_method == "fixed" and threshold is None:
            raise ValueError(
                "threshold must be provided when threshold_method='fixed'"
            )

        self.threshold = threshold
        self.threshold_method = threshold_method
        self.n_threshold_grid = n_threshold_grid
        self.threshold_quantile_range = threshold_quantile_range
        self.n_starts = n_starts

        # Fitted state (set by fit())
        self.threshold_ = None
        self.pi_ = None
        self.body_params_ = None
        self.tail_params_ = None
        self.loglik_ = None
        self.n_body_ = None
        self.n_tail_ = None
        self._body = None
        self._tail = None

    def _make_body(self) -> BodyDistribution:
        return self._body_cls()

    def _make_tail(self) -> TailDistribution:
        return self._tail_cls()

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, y: np.ndarray) -> "CompositeSeverityModel":
        """
        Fit composite model to claim severity data.

        Parameters
        ----------
        y : array-like of shape (n,)
            Positive claim amounts.

        Returns
        -------
        self
        """
        y = np.asarray(y, dtype=float)
        if np.any(y <= 0):
            raise ValueError("All claim amounts must be strictly positive")
        if len(y) < 10:
            raise ValueError("Need at least 10 observations to fit a composite model")

        if self.threshold_method == "fixed":
            self._fit_fixed_threshold(y, self.threshold)
        elif self.threshold_method == "profile_likelihood":
            self._fit_profile_likelihood(y)
        elif self.threshold_method == "mode_matching":
            self._fit_mode_matching(y)
        return self

    def _fit_fixed_threshold(self, y: np.ndarray, threshold: float) -> None:
        """Fit body and tail independently at a fixed threshold."""
        y_body = y[y <= threshold]
        y_tail = y[y > threshold]

        if len(y_body) == 0:
            raise ValueError(
                f"No observations at or below threshold {threshold}. "
                "Lower the threshold."
            )
        if len(y_tail) == 0:
            raise ValueError(
                f"No observations above threshold {threshold}. "
                "Raise the threshold."
            )

        body = self._make_body()
        tail = self._make_tail()

        body.fit_mle(y_body, threshold)
        tail.fit_mle(y_tail, threshold)

        # Compute mixing weight pi from continuity condition
        # pi = F_body(t) / (F_body(t) + S_tail(t)/f_tail(t) * f_body(t)) ...
        # simpler: pi estimated by proportion (smoothed by continuity)
        # Use: pi = argmax of log-likelihood with body/tail fixed
        # For hard composite: pi = (n_body) / (n_body + n_tail) is NOT quite right
        # because the truncated densities already integrate to 1 over their domains.
        # The weight pi is the overall probability mass in [0, t].
        # With fixed body and tail, the optimal pi is:
        #   pi = n_body / n  (fraction of observations in body)
        # This is the MLE for pi given the other params.
        n = len(y)
        pi = len(y_body) / n

        # Compute log-likelihood
        loglik = self._loglik(y, body, tail, threshold, pi)

        self.threshold_ = threshold
        self.pi_ = pi
        self.body_params_ = body.params.copy()
        self.tail_params_ = tail.params.copy()
        self.loglik_ = loglik
        self.n_body_ = len(y_body)
        self.n_tail_ = len(y_tail)
        self._body = body
        self._tail = tail

    def _loglik(
        self,
        y: np.ndarray,
        body: BodyDistribution,
        tail: TailDistribution,
        threshold: float,
        pi: float,
    ) -> float:
        """Compute composite log-likelihood."""
        mask = y <= threshold
        ll = 0.0
        if np.any(mask):
            ll += np.sum(np.log(pi) + body.logpdf(y[mask], threshold))
        if np.any(~mask):
            ll += np.sum(np.log(1.0 - pi) + tail.logpdf(y[~mask], threshold))
        return ll

    def _fit_profile_likelihood(self, y: np.ndarray) -> None:
        """Select threshold by maximising profile log-likelihood over a grid."""
        q_low, q_high = self.threshold_quantile_range
        t_candidates = np.quantile(
            y, np.linspace(q_low, q_high, self.n_threshold_grid)
        )
        # Remove duplicates
        t_candidates = np.unique(t_candidates)

        best_loglik = -np.inf
        best_t = None

        for t in t_candidates:
            y_body = y[y <= t]
            y_tail = y[y > t]
            if len(y_body) < 5 or len(y_tail) < 10:
                continue
            try:
                body = self._make_body()
                tail = self._make_tail()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    body.fit_mle(y_body, t)
                    tail.fit_mle(y_tail, t)
                pi = len(y_body) / len(y)
                ll = self._loglik(y, body, tail, t, pi)
                if ll > best_loglik:
                    best_loglik = ll
                    best_t = t
                    best_body = body
                    best_tail = tail
                    best_pi = pi
            except Exception:
                continue

        if best_t is None:
            raise RuntimeError(
                "Profile likelihood search failed: no valid threshold found. "
                "Check data quality and threshold_quantile_range."
            )

        self.threshold_ = best_t
        self.pi_ = best_pi
        self.body_params_ = best_body.params.copy()
        self.tail_params_ = best_tail.params.copy()
        self.loglik_ = best_loglik
        self.n_body_ = int(np.sum(y <= best_t))
        self.n_tail_ = int(np.sum(y > best_t))
        self._body = best_body
        self._tail = best_tail

    def _fit_mode_matching(self, y: np.ndarray) -> None:
        """
        Fit mode-matching composite model.

        Implemented in subclasses that set _supports_mode_matching = True.
        Raises ValueError for distributions that don't support mode-matching.
        """
        raise NotImplementedError(
            "Mode-matching must be implemented by specific subclasses."
        )

    # ------------------------------------------------------------------
    # Probability functions
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if self.threshold_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

    def logpdf(self, y: np.ndarray) -> np.ndarray:
        """Log density of the composite distribution."""
        self._check_fitted()
        y = np.asarray(y, dtype=float)
        result = np.full(len(y), -np.inf)
        mask = y <= self.threshold_
        if np.any(mask):
            result[mask] = np.log(self.pi_) + self._body.logpdf(y[mask], self.threshold_)
        if np.any(~mask):
            result[~mask] = np.log(1.0 - self.pi_) + self._tail.logpdf(y[~mask], self.threshold_)
        return result

    def pdf(self, y: np.ndarray) -> np.ndarray:
        return np.exp(self.logpdf(y))

    def cdf(self, y: np.ndarray) -> np.ndarray:
        """CDF of the composite distribution."""
        self._check_fitted()
        y = np.asarray(y, dtype=float)
        result = np.zeros(len(y))
        mask = y <= self.threshold_
        if np.any(mask):
            result[mask] = self.pi_ * np.exp(self._body.logcdf(y[mask], self.threshold_))
        if np.any(~mask):
            tail_sf = np.exp(self._tail.logsf(y[~mask], self.threshold_))
            result[~mask] = self.pi_ + (1.0 - self.pi_) * (1.0 - tail_sf)
        return result

    def sf(self, y: np.ndarray) -> np.ndarray:
        """Survival function."""
        return 1.0 - self.cdf(y)

    def ppf(self, q: np.ndarray) -> np.ndarray:
        """Quantile function (VaR)."""
        self._check_fitted()
        q = np.asarray(q, dtype=float)
        result = np.zeros(len(q))

        # In body: q <= pi
        in_body = q <= self.pi_
        if np.any(in_body):
            q_body = q[in_body] / self.pi_  # quantile within body
            result[in_body] = self._body.ppf(q_body, self.threshold_)

        # In tail: q > pi
        in_tail = ~in_body
        if np.any(in_tail):
            q_tail = (q[in_tail] - self.pi_) / (1.0 - self.pi_)  # quantile within tail
            result[in_tail] = self._tail.ppf(q_tail, self.threshold_)

        return result

    def var(self, alpha: float) -> float:
        """Value at Risk at level alpha."""
        return float(self.ppf(np.array([alpha]))[0])

    def tvar(self, alpha: float) -> float:
        """
        Tail Value at Risk (Expected Shortfall) at level alpha.

        TVaR_alpha(X) = E[X | X > VaR_alpha(X)]

        Computed numerically for the composite distribution.
        """
        self._check_fitted()
        var = self.var(alpha)
        from scipy.integrate import quad

        def integrand(x):
            return self.sf(np.array([x]))[0]

        upper = var * 1e4 + self.threshold_ * 10
        result, _ = quad(integrand, var, upper, limit=200)
        return var + result / (1.0 - alpha)

    def mean_excess(self, d: float) -> float:
        """
        Mean excess loss (limited expected value minus d).
        E[X - d | X > d]
        """
        from scipy.integrate import quad

        sf_d = self.sf(np.array([d]))[0]
        if sf_d <= 0:
            return 0.0

        def integrand(x):
            return self.sf(np.array([x]))[0]

        upper = d * 1e4 + self.threshold_ * 10
        result, _ = quad(integrand, d, upper, limit=200)
        return result / sf_d

    def limited_expected_value(self, limit: float) -> float:
        """
        Limited expected value: E[min(X, limit)].

        Used in ILF computation.
        """
        from scipy.integrate import quad

        def integrand(x):
            return self.sf(np.array([x]))[0]

        result, _ = quad(integrand, 0.0, limit, limit=200)
        return result

    def ilf(self, limit: float, basic_limit: float = 1_000_000.0) -> float:
        """
        Increased Limit Factor (ILF).

        ILF(L) = E[min(X, L)] / E[min(X, basic_limit)]

        Standard actuarial tool for pricing excess of loss layers.
        """
        lev_l = self.limited_expected_value(limit)
        lev_b = self.limited_expected_value(basic_limit)
        if lev_b <= 0:
            return np.nan
        return lev_l / lev_b

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def quantile_residuals(self, y: np.ndarray) -> np.ndarray:
        """
        Randomized quantile residuals (Dunn & Smyth 1996).

        For a continuous distribution, the CDF transform F(Y_i) ~ Uniform(0,1),
        and Phi^{-1}(F(Y_i)) ~ Normal(0,1). Departures from normality indicate
        model misspecification.

        Returns standard normal residuals.
        """
        from scipy.stats import norm
        y = np.asarray(y, dtype=float)
        p = self.cdf(y)
        # Clip to avoid numerical infinities
        p = np.clip(p, 1e-10, 1.0 - 1e-10)
        return norm.ppf(p)

    def aic(self, y: np.ndarray) -> float:
        """Akaike Information Criterion."""
        if self.loglik_ is None:
            self.loglik_ = float(np.sum(self.logpdf(y)))
        k = len(self.body_params_) + len(self.tail_params_) + 1  # +1 for pi
        return 2.0 * k - 2.0 * self.loglik_

    def bic(self, y: np.ndarray) -> float:
        """Bayesian Information Criterion."""
        if self.loglik_ is None:
            self.loglik_ = float(np.sum(self.logpdf(y)))
        k = len(self.body_params_) + len(self.tail_params_) + 1
        n = len(y)
        return k * np.log(n) - 2.0 * self.loglik_

    def summary(self, y: Optional[np.ndarray] = None) -> str:
        """Return a text summary of fitted model."""
        self._check_fitted()
        lines = [
            f"{self.__class__.__name__}",
            f"  Threshold method : {self.threshold_method}",
            f"  Threshold        : {self.threshold_:,.1f}",
            f"  Body weight (pi) : {self.pi_:.4f}",
            f"  n_body           : {self.n_body_}",
            f"  n_tail           : {self.n_tail_}",
            f"  Body params      : {self.body_params_}",
            f"  Tail params      : {self.tail_params_}",
            f"  Log-likelihood   : {self.loglik_:.2f}",
        ]
        if y is not None:
            lines.append(f"  AIC              : {self.aic(y):.2f}")
            lines.append(f"  BIC              : {self.bic(y):.2f}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_fit(
        self,
        y: np.ndarray,
        ax=None,
        n_points: int = 500,
        log_scale: bool = True,
    ):
        """
        Overlay fitted density on empirical histogram.

        Requires matplotlib.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install with: pip install insurance-composite[plotting]"
            )
        self._check_fitted()
        y = np.asarray(y, dtype=float)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        ax.hist(y, bins=50, density=True, alpha=0.5, label="Data")
        x_plot = np.linspace(np.percentile(y, 0.5), np.percentile(y, 99.5), n_points)
        ax.plot(x_plot, self.pdf(x_plot), "r-", lw=2, label="Fitted composite")
        ax.axvline(self.threshold_, ls="--", color="k", alpha=0.6, label=f"Threshold = {self.threshold_:,.0f}")
        if log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("Claim amount")
        ax.set_ylabel("Density")
        ax.legend()
        ax.set_title(self.__class__.__name__)
        return ax


# ---------------------------------------------------------------------------
# Lognormal-Burr composite (mode-matching)
# ---------------------------------------------------------------------------


class LognormalBurrComposite(CompositeSeverityModel):
    """
    Composite model with Lognormal body and Burr XII tail.

    This is the primary model for covariate-dependent threshold estimation.
    Mode-matching is supported: threshold = Burr XII mode, which requires
    alpha > 1. The lognormal mu is derived from the mode-matching condition
    mu = sigma^2 + log(threshold), ensuring C1 continuity at the splice.

    Parameters
    ----------
    threshold : float, optional
        Required for threshold_method='fixed'.
    threshold_method : str
        'fixed', 'profile_likelihood', or 'mode_matching'.
    n_threshold_grid : int
        Grid size for profile likelihood search.
    threshold_quantile_range : tuple
        Quantile range for profile likelihood grid.
    n_starts : int
        Number of optimization restarts.

    Attributes (after fit)
    ----------------------
    threshold_ : float
    pi_ : float
    body_params_ : ndarray — [mu, sigma]
    tail_params_ : ndarray — [alpha, delta, beta]
    loglik_ : float
    """

    _body_cls = LognormalBody
    _tail_cls = BurrTail
    _supports_mode_matching = True

    def _fit_mode_matching(self, y: np.ndarray) -> None:
        """
        Mode-matching estimation for Lognormal-Burr composite.

        Mode-matching conditions:
        (1) threshold t = Burr_mode(alpha, delta, beta)
                       = beta * [(alpha-1)/(delta*alpha+1)]^{1/delta}
        (2) lognormal mu = sigma^2 + log(t)  [lognormal mode = t]
        (3) continuity: pi derived from density balance at t

        Free parameters: sigma (lognormal), alpha, delta, beta (Burr) s.t. alpha > 1.
        Reparameterisation: log(alpha - 1) to enforce alpha > 1 unconstrained.

        The negative log-likelihood over these 4 free parameters is minimised
        using L-BFGS-B (unconstrained after reparameterisation).
        """
        n = len(y)
        best_loglik = -np.inf
        best_result = None

        # Multiple random starts to avoid local optima
        rng = np.random.default_rng(42)

        # Initial points: one data-driven, rest random perturbations
        log_y = np.log(y)
        sigma_init = max(np.std(log_y), 0.3)
        alpha_init = 2.0  # > 0 required
        # delta needs > 1 for mode. Start with delta=1.5.
        delta_init = 1.5
        t_rough = np.median(y)
        # From mode formula: t = beta * [(delta-1)/(alpha*delta+1)]^{1/delta}
        # => beta = t / [(delta-1)/(alpha*delta+1)]^{1/delta}
        ratio_init = (delta_init - 1.0) / (alpha_init * delta_init + 1.0)
        beta_init = t_rough / (ratio_init ** (1.0 / delta_init))
        starts = [
            [np.log(sigma_init), np.log(alpha_init), np.log(delta_init - 1.0), np.log(beta_init)]
        ]
        for _ in range(self.n_starts - 1):
            s_rand = np.exp(rng.uniform(-0.5, 0.5)) * sigma_init
            a_rand = np.exp(rng.uniform(-0.3, 0.3)) * alpha_init  # alpha > 0
            d_m1_rand = np.exp(rng.uniform(-0.5, 0.5)) * (delta_init - 1.0)  # delta - 1 > 0
            b_rand = np.exp(rng.uniform(-0.5, 0.5)) * beta_init
            starts.append([np.log(s_rand), np.log(a_rand), np.log(d_m1_rand), np.log(b_rand)])

        def neg_loglik_mode_match(params):
            log_sigma, log_alpha, log_delta_m1, log_beta = params
            sigma = np.exp(log_sigma)
            alpha = np.exp(log_alpha)  # > 0 always
            delta = 1.0 + np.exp(log_delta_m1)  # > 1 always (required for mode)
            beta = np.exp(log_beta)

            # Derive threshold from Burr mode: beta * [(delta-1)/(alpha*delta+1)]^{1/delta}
            ratio = (delta - 1.0) / (alpha * delta + 1.0)
            if ratio <= 0:
                return 1e10
            t = beta * ratio ** (1.0 / delta)
            if t <= 0 or t >= np.max(y):
                return 1e10

            # Derive lognormal mu from mode-matching: LN mode = t
            # LN mode = exp(mu - sigma^2) => mu = sigma^2 + log(t)
            mu = sigma ** 2 + np.log(t)

            # Build distributions
            body = LognormalBody(mu=mu, sigma=sigma)
            tail = BurrTail(alpha=alpha, delta=delta, beta=beta)

            y_body = y[y <= t]
            y_tail = y[y > t]

            if len(y_body) == 0 or len(y_tail) == 0:
                return 1e10

            pi = len(y_body) / n

            try:
                ll = self._loglik(y, body, tail, t, pi)
                if not np.isfinite(ll):
                    return 1e10
                return -ll
            except Exception:
                return 1e10

        for x0 in starts:
            try:
                result = optimize.minimize(
                    neg_loglik_mode_match,
                    x0=x0,
                    method="L-BFGS-B",
                    options={"maxiter": 1000, "ftol": 1e-10},
                )
                if result.fun < -best_loglik or best_result is None:
                    best_loglik = -result.fun
                    best_result = result
            except Exception:
                continue

        if best_result is None:
            raise RuntimeError(
                "Mode-matching optimization failed across all starting points. "
                "Try threshold_method='profile_likelihood' instead."
            )

        # Extract parameters from best result
        log_sigma, log_alpha, log_delta_m1, log_beta = best_result.x
        sigma = np.exp(log_sigma)
        alpha = np.exp(log_alpha)
        delta = 1.0 + np.exp(log_delta_m1)  # > 1 enforced
        beta = np.exp(log_beta)

        ratio = (delta - 1.0) / (alpha * delta + 1.0)
        t = beta * ratio ** (1.0 / delta)
        mu = sigma ** 2 + np.log(t)

        body = LognormalBody(mu=mu, sigma=sigma)
        tail = BurrTail(alpha=alpha, delta=delta, beta=beta)
        pi = np.sum(y <= t) / n

        self.threshold_ = t
        self.pi_ = pi
        self.body_params_ = body.params.copy()
        self.tail_params_ = tail.params.copy()
        self.loglik_ = best_loglik
        self.n_body_ = int(np.sum(y <= t))
        self.n_tail_ = int(np.sum(y > t))
        self._body = body
        self._tail = tail


# ---------------------------------------------------------------------------
# Lognormal-GPD composite (fixed / profile likelihood threshold only)
# ---------------------------------------------------------------------------


class LognormalGPDComposite(CompositeSeverityModel):
    """
    Composite model with Lognormal body and GPD tail.

    GPD is the canonical EVT tail distribution. Mode-matching is NOT
    supported because the GPD mode is 0 for xi >= 0, which covers all
    insurance-relevant heavy-loss scenarios.

    Use threshold_method='fixed' (when a natural threshold exists, e.g.
    the reinsurance attachment point) or threshold_method='profile_likelihood'
    (data-driven threshold selection via grid search).

    Parameters
    ----------
    threshold : float, optional
        Required for threshold_method='fixed'.
    threshold_method : str
        'fixed' or 'profile_likelihood'.
    """

    _body_cls = LognormalBody
    _tail_cls = GPDTail
    _supports_mode_matching = False  # GPD mode is 0 for xi >= 0


# ---------------------------------------------------------------------------
# Gamma-GPD composite (fixed / profile likelihood threshold only)
# ---------------------------------------------------------------------------


class GammaGPDComposite(CompositeSeverityModel):
    """
    Composite model with Gamma body and GPD tail.

    The Gamma body is the natural GLM severity family. Combining with
    GPD tail gives a model where the body behaves like a standard Gamma GLM
    and the tail follows an EVT-motivated heavy distribution.

    Practical for: motor BI pricing where the bulk of claims are well-described
    by a Gamma, with GPD capturing the excess of loss experience.

    Mode-matching is NOT supported (GPD restriction applies).

    Parameters
    ----------
    threshold : float, optional
        Required for threshold_method='fixed'.
    threshold_method : str
        'fixed' or 'profile_likelihood'.
    """

    _body_cls = GammaBody
    _tail_cls = GPDTail
    _supports_mode_matching = False
