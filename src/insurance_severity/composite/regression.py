"""
Composite severity regression with covariate-dependent thresholds.

CompositeSeverityRegressor wraps any CompositeSeverityModel and adds:
- Covariate-dependent tail scale via log-link regression
- Per-observation threshold prediction (for mode-matching models)
- Per-observation severity predictions (conditional mean)
- ILF schedules computed per policyholder
- Bootstrap confidence intervals on predictions

The regression design follows scikit-learn conventions (fit/predict/transform)
so models drop into sklearn Pipelines.

How covariate-dependent thresholds work
-----------------------------------------
For mode-matching composites (LognormalBurrComposite), the threshold is the
tail distribution mode. The tail scale beta enters the mode formula:
    t_i = beta_i * h(alpha, delta)
where beta_i = exp(X_i @ w) with w the regression coefficients.

So the threshold varies proportionally with exp(linear predictor). Each
policyholder gets their own threshold, body parameters (via mode-matching),
and tail parameters.

For fixed-threshold composites (LognormalGPD, GammaGPD), the threshold is
shared across all policyholders. Covariates enter only via the tail scale.

Architecture note: We implement regression by treating the full dataset
jointly. At each optimizer iteration, each observation is assigned to body
or tail based on whether y_i <= t_i (observation-specific threshold for
mode-matching, or global threshold for fixed). The log-likelihood gradient
is vectorized over n observations.
"""

from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
from scipy import optimize

from insurance_severity.composite.models import (
    CompositeSeverityModel,
    LognormalBurrComposite,
    LognormalGPDComposite,
    GammaGPDComposite,
)
from insurance_severity.composite.distributions import (
    LognormalBody,
    GammaBody,
    GPDTail,
    BurrTail,
)


class CompositeSeverityRegressor:
    """
    Composite severity regression with covariate-dependent tail scale.

    Fits a composite severity model where the tail scale parameter (beta
    for Burr, sigma for GPD) is a log-linear function of covariates:
        log(scale_i) = X_i @ w

    For mode-matching composites, the threshold is automatically
    covariate-dependent:
        t_i = exp(X_i @ w) * h(shape_params)

    For fixed-threshold composites, threshold is shared; only tail scale varies.

    Scikit-learn compatible: implements fit(), predict(), and score().

    Parameters
    ----------
    composite : CompositeSeverityModel or str
        Base composite model or string shorthand:
        'lognormal_burr', 'lognormal_gpd', 'gamma_gpd'.
    feature_cols : list of str, optional
        Column names of covariates (used when X is a DataFrame).
    n_starts : int
        Number of random multi-starts for full joint MLE. Default 3.
    max_iter : int
        Maximum optimizer iterations.

    Examples
    --------
    >>> from insurance_composite import CompositeSeverityRegressor, LognormalBurrComposite
    >>> reg = CompositeSeverityRegressor(
    ...     composite=LognormalBurrComposite(threshold_method="mode_matching"),
    ...     feature_cols=["vehicle_age", "driver_age"],
    ... )
    >>> reg.fit(X_train, y_train)
    >>> reg.predict(X_test)
    """

    def __init__(
        self,
        composite: Union[CompositeSeverityModel, str] = "lognormal_burr",
        feature_cols: Optional[list] = None,
        n_starts: int = 3,
        max_iter: int = 500,
    ):
        if isinstance(composite, str):
            composite = _composite_from_str(composite)
        self.composite = composite
        self.feature_cols = feature_cols
        self.n_starts = n_starts
        self.max_iter = max_iter

        # Fitted state
        self.coef_ = None          # regression coefficients for tail scale
        self.intercept_ = None     # intercept of tail scale log-link
        self.shape_params_ = None  # fitted shape parameters (non-regression)
        self.body_params_ = None   # global body shape params (sigma for LN, shape for Gamma)
        self.threshold_ = None     # global threshold (fixed-threshold models only)
        self.pi_mean_ = None       # average body weight
        self.loglik_ = None
        self._composite_type = type(composite).__name__
        self._is_mode_matching = (
            composite.threshold_method == "mode_matching"
            and composite._supports_mode_matching
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "CompositeSeverityRegressor":
        """
        Fit regression composite model.

        Parameters
        ----------
        X : array-like of shape (n, p) or DataFrame
            Feature matrix. Intercept is added automatically.
        y : array-like of shape (n,)
            Claim amounts (strictly positive).

        Returns
        -------
        self
        """
        X, y = self._validate_inputs(X, y)
        n, p = X.shape

        if self._is_mode_matching:
            self._fit_mode_matching_regression(X, y)
        else:
            self._fit_fixed_threshold_regression(X, y)

        return self

    def _validate_inputs(self, X, y):
        """Convert inputs to numpy arrays. Handle DataFrame column selection."""
        try:
            import pandas as pd
            if isinstance(X, pd.DataFrame):
                if self.feature_cols is not None:
                    X = X[self.feature_cols].to_numpy(dtype=float)
                else:
                    X = X.to_numpy(dtype=float)
        except ImportError:
            pass

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim != 1:
            raise ValueError("y must be 1-D")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")
        if np.any(y <= 0):
            raise ValueError("All claim amounts must be strictly positive")

        # Add intercept column
        X = np.column_stack([np.ones(len(X)), X])
        return X, y

    def _fit_mode_matching_regression(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Joint MLE for mode-matching composite regression.

        Free parameters:
          w : (p+1,) regression coefficients for log(beta_i) = X_i @ w
          log_sigma : scalar, lognormal sigma (body shape)
          log_alpha_m1 : scalar, log(alpha - 1) for Burr (alpha > 1)
          log_delta : scalar, log(delta) for Burr

        At each evaluation:
          beta_i = exp(X_i @ w)
          alpha = 1 + exp(log_alpha_m1)
          delta = exp(log_delta)
          t_i = beta_i * [(alpha-1)/(delta*alpha+1)]^{1/delta}
          mu_i = sigma^2 + log(t_i)   [lognormal mode = t_i]
          pi_i = n_body_i / n  (estimated per observation)
        """
        n, p = X.shape

        def _compute_loglik(w, log_sigma, log_alpha, log_delta_m1):
            sigma = np.exp(log_sigma)
            alpha = np.exp(log_alpha)          # alpha > 0 always
            delta = 1.0 + np.exp(log_delta_m1) # delta > 1 always

            # Burr mode shape factor: (delta-1)/(alpha*delta+1), delta > 1 guaranteed
            ratio = (delta - 1.0) / (alpha * delta + 1.0)
            if ratio <= 0:
                return -1e10

            log_h = np.log(ratio) / delta  # log of h(alpha, delta)

            # Per-observation beta and threshold
            log_beta = X @ w
            beta = np.exp(log_beta)
            t = beta * np.exp(log_h)

            # Per-observation lognormal mu
            mu = sigma ** 2 + np.log(t)

            # Body vs tail assignment
            in_body = y <= t
            y_body = y[in_body]
            y_tail = y[~in_body]

            if len(y_body) == 0 or len(y_tail) == 0:
                return -1e10

            n_body = len(y_body)
            pi = n_body / n

            # Body log-likelihood: sum over body obs
            # logpdf_body_i = LN.logpdf(y_i; mu_i, sigma) - log F_LN(t_i; mu_i, sigma)
            from scipy.stats import norm, lognorm
            log_y_body = np.log(y_body)
            mu_body = mu[in_body]
            t_body = t[in_body]

            # LN logpdf
            ln_lp = norm.logpdf(log_y_body, loc=mu_body, scale=sigma) - log_y_body
            # LN normalization: log F_LN(t_i; mu_i, sigma)
            ln_norm = norm.logcdf((np.log(t_body) - mu_body) / sigma)
            body_ll = np.sum(np.log(pi) + ln_lp - ln_norm)

            # Tail log-likelihood: BurrTail on exceedances above t_i
            # f_tail(y_i; alpha, delta, beta_i) / S_Burr(t_i; alpha, delta, beta_i)
            y_tail_vals = y_tail
            beta_tail = beta[~in_body]
            t_tail = t[~in_body]

            # Burr XII logpdf (unconditional)
            x_over_b = y_tail_vals / beta_tail
            log_x_over_b = np.log(x_over_b)
            lp_burr = (
                np.log(alpha) + np.log(delta) - np.log(beta_tail)
                + (delta - 1.0) * log_x_over_b
                - (alpha + 1.0) * np.log1p(x_over_b ** delta)
            )
            # S_Burr(t_i; alpha, delta, beta_i)
            t_over_b = t_tail / beta_tail
            log_sf_t = -alpha * np.log1p(t_over_b ** delta)
            tail_ll = np.sum(np.log(1.0 - pi) + lp_burr - log_sf_t)

            ll = body_ll + tail_ll
            if not np.isfinite(ll):
                return -1e10
            return ll

        def neg_loglik(params):
            w = params[:X.shape[1]]
            log_sigma = params[X.shape[1]]
            log_alpha = params[X.shape[1] + 1]
            log_delta_m1 = params[X.shape[1] + 2]
            return -_compute_loglik(w, log_sigma, log_alpha, log_delta_m1)

        # Warm start: fit model without covariates first
        simple_model = LognormalBurrComposite(threshold_method="mode_matching", n_starts=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            simple_model.fit(y)

        sigma0 = simple_model.body_params_[1]
        alpha0 = simple_model.tail_params_[0]
        delta0 = simple_model.tail_params_[1]
        beta0 = simple_model.tail_params_[2]

        # For regression coefficients, initialize intercept = log(beta0), slopes = 0
        w0 = np.zeros(X.shape[1])
        w0[0] = np.log(beta0)  # intercept

        # delta0 must be > 1 for mode existence; clip to at least 1.1
        delta0_mm = max(delta0, 1.1)
        x0_base = [
            *w0,
            np.log(sigma0),
            np.log(alpha0),           # log(alpha) — unconstrained
            np.log(delta0_mm - 1.0),  # log(delta - 1) — ensures delta > 1
        ]

        # Multi-start
        rng = np.random.default_rng(42)
        best_nll = np.inf
        best_result = None

        for i in range(self.n_starts):
            if i == 0:
                x0 = np.array(x0_base)
            else:
                x0 = np.array(x0_base) + rng.normal(0, 0.3, len(x0_base))

            try:
                result = optimize.minimize(
                    neg_loglik,
                    x0=x0,
                    method="L-BFGS-B",
                    options={"maxiter": self.max_iter, "ftol": 1e-10},
                )
                if result.fun < best_nll:
                    best_nll = result.fun
                    best_result = result
            except Exception:
                continue

        if best_result is None:
            raise RuntimeError("Regression optimization failed.")

        # Extract results
        w = best_result.x[:X.shape[1]]
        log_sigma = best_result.x[X.shape[1]]
        log_alpha = best_result.x[X.shape[1] + 1]
        log_delta_m1 = best_result.x[X.shape[1] + 2]

        sigma = np.exp(log_sigma)
        alpha = np.exp(log_alpha)
        delta = 1.0 + np.exp(log_delta_m1)  # > 1 enforced

        self.coef_ = w[1:]      # slope coefficients (excluding intercept)
        self.intercept_ = w[0]  # log-scale intercept
        self.body_params_ = np.array([np.nan, sigma])  # mu is per-obs; sigma is global
        self.shape_params_ = np.array([alpha, delta])  # Burr shape params
        self.loglik_ = -best_nll
        self.pi_mean_ = float(np.mean(y <= self._predict_thresholds_raw(X, w, alpha, delta)))
        self._w = w  # store full coefficient vector including intercept
        self._sigma = sigma

    def _predict_thresholds_raw(
        self, X: np.ndarray, w: np.ndarray, alpha: float, delta: float
    ) -> np.ndarray:
        """Compute per-observation thresholds from regression coefficients.

        Burr XII mode: beta * [(delta-1)/(alpha*delta+1)]^{1/delta}
        Requires delta > 1.
        """
        log_beta = X @ w
        beta = np.exp(log_beta)
        if delta <= 1.0:
            return np.full(len(beta), np.nan)
        ratio = (delta - 1.0) / (alpha * delta + 1.0)
        if ratio <= 0:
            return np.full(len(beta), np.nan)
        h = ratio ** (1.0 / delta)
        return beta * h

    def _fit_fixed_threshold_regression(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Joint MLE for fixed-threshold composite regression.

        The global threshold is fixed (user-specified) or chosen by profile
        likelihood on the pooled dataset. Covariates enter the tail scale only.

        For GPD: log(sigma_i) = X_i @ w
        For Pareto: log(scale_i) = X_i @ w
        """
        # Determine global threshold
        base = self.composite
        if base.threshold_method == "fixed":
            t = base.threshold
        else:
            # Profile likelihood on pooled data
            temp = type(base)(threshold_method="profile_likelihood",
                              n_threshold_grid=base.n_threshold_grid,
                              threshold_quantile_range=base.threshold_quantile_range)
            temp.fit(y)
            t = temp.threshold_

        self.threshold_ = t

        n, p = X.shape
        y_body = y[y <= t]
        y_tail = y[y > t]
        in_body = y <= t

        if len(y_body) == 0 or len(y_tail) == 0:
            raise ValueError(f"Threshold {t} results in empty body or tail partition.")

        # Fit body globally (no covariates in body for now)
        body = base._make_body()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            body.fit_mle(y_body, t)

        self.body_params_ = body.params.copy()
        pi = len(y_body) / n
        self.pi_mean_ = pi

        # Determine tail type
        is_gpd = isinstance(base._make_tail(), GPDTail)

        if is_gpd:
            # GPD tail: xi is global, log(sigma_i) = X @ w
            # Initialize from pooled GPD fit
            tail_init = GPDTail()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tail_init.fit_mle(y_tail, t)
            xi0 = tail_init._xi
            sigma_pooled = tail_init._sigma

            def neg_loglik_gpd(params):
                xi = params[0]
                w = params[1:]
                log_sigma = X[~in_body] @ w
                sigma_i = np.exp(log_sigma)
                z = y_tail - t  # exceedances

                # Support check
                if xi < 0:
                    if np.any(z > -sigma_i / xi):
                        return 1e10

                from scipy.stats import genpareto
                # GPD logpdf with individual sigma_i
                if abs(xi) < 1e-10:
                    lp = -np.log(sigma_i) - z / sigma_i
                else:
                    inner = 1.0 + xi * z / sigma_i
                    if np.any(inner <= 0):
                        return 1e10
                    lp = -np.log(sigma_i) - (1.0/xi + 1.0) * np.log(inner)

                return -np.sum(lp)

            w0 = np.zeros(p)
            w0[0] = np.log(sigma_pooled)

            rng = np.random.default_rng(42)
            best_nll = np.inf
            best_res = None

            for i in range(self.n_starts):
                x0 = np.concatenate([[xi0], w0])
                if i > 0:
                    x0 += rng.normal(0, 0.2, len(x0))
                try:
                    res = optimize.minimize(
                        neg_loglik_gpd,
                        x0=x0,
                        method="L-BFGS-B",
                        bounds=[(-0.9, 2.0)] + [(None, None)] * p,
                        options={"maxiter": self.max_iter},
                    )
                    if res.fun < best_nll:
                        best_nll = res.fun
                        best_res = res
                except Exception:
                    continue

            if best_res is None:
                raise RuntimeError("Fixed-threshold GPD regression failed.")

            xi_hat = best_res.x[0]
            w_hat = best_res.x[1:]
            self.coef_ = w_hat[1:]
            self.intercept_ = w_hat[0]
            self.shape_params_ = np.array([xi_hat])
            self.loglik_ = -best_nll
            self._w = w_hat
            self._xi = xi_hat
        else:
            raise NotImplementedError(
                "Fixed-threshold regression only implemented for GPD tail in V1."
            )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_thresholds(self, X: np.ndarray) -> np.ndarray:
        """
        Predict per-observation splice thresholds.

        For mode-matching models: thresholds vary with covariates.
        For fixed-threshold models: returns constant threshold array.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Feature matrix (same columns as fit).

        Returns
        -------
        thresholds : ndarray of shape (n,)
        """
        X = self._prepare_X(X)

        if self._is_mode_matching:
            alpha, delta = self.shape_params_
            return self._predict_thresholds_raw(X, self._w, alpha, delta)
        else:
            return np.full(len(X), self.threshold_)

    def predict_tail_scale(self, X: np.ndarray) -> np.ndarray:
        """
        Predict per-observation tail scale parameters.

        Returns exp(X @ w) for the tail scale.
        """
        X = self._prepare_X(X)
        return np.exp(X @ self._w)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict expected severity (mean claim amount) per observation.

        For mode-matching Lognormal-Burr:
            E[X_i] = pi * E_body_i + (1-pi) * E_tail_i

        The body mean inherits mu_i = sigma^2 + log(t_i), giving
            E_body_i = exp(mu_i + sigma^2/2) * ...  (truncated mean)

        Computed numerically for accuracy.

        Returns
        -------
        means : ndarray of shape (n,)
        """
        X = self._prepare_X(X)
        n = len(X)
        # Use _predict_thresholds_raw directly to avoid double-augmentation
        if self._is_mode_matching:
            alpha, delta = self.shape_params_
            thresholds = self._predict_thresholds_raw(X, self._w, alpha, delta)
        else:
            thresholds = np.full(n, self.threshold_)
        means = np.zeros(n)

        if self._is_mode_matching:
            alpha, delta = self.shape_params_
            sigma = self._sigma
            beta_arr = np.exp(X @ self._w)

            for i in range(n):
                t_i = thresholds[i]
                beta_i = beta_arr[i]
                mu_i = sigma ** 2 + np.log(t_i)

                body = LognormalBody(mu=mu_i, sigma=sigma)
                tail = BurrTail(alpha=alpha, delta=delta, beta=beta_i)

                # Approximate body mean via numerical integration
                from scipy.integrate import quad

                def body_integrand(x, mu=mu_i, s=sigma, t=t_i):
                    return x * np.exp(body.logpdf(np.array([x]), t))[0]

                def tail_integrand(x, a=alpha, d=delta, b=beta_i, t=t_i):
                    return x * np.exp(tail.logpdf(np.array([x]), t))[0]

                body_mean, _ = quad(body_integrand, 0.0, t_i, limit=100)
                tail_mean, _ = quad(tail_integrand, t_i, t_i * 1000 + 1.0, limit=100)

                pi_i = 0.5  # approximation; proper pi requires iteration
                means[i] = pi_i * body_mean + (1.0 - pi_i) * tail_mean
        else:
            # Fixed threshold
            t = self.threshold_
            if hasattr(self, '_xi'):
                xi = self._xi
                for i in range(n):
                    sigma_i = np.exp(X[i] @ self._w)
                    tail = GPDTail(xi=xi, sigma=sigma_i)
                    # Mean of GPD exceedance
                    if xi < 1.0:
                        tail_mean = t + sigma_i / (1.0 - xi)
                    else:
                        tail_mean = np.inf
                    means[i] = tail_mean  # simplified: return tail mean

        return means

    def compute_ilf(
        self,
        X: np.ndarray,
        limits: list,
        basic_limit: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute Increased Limit Factors (ILF) schedule per observation.

        ILF_i(L) = E[min(X_i, L)] / E[min(X_i, basic_limit)]

        Parameters
        ----------
        X : array-like of shape (n, p)
        limits : list of float
            Limit values to compute ILF at.
        basic_limit : float, optional
            Basic limit for ILF normalization. Defaults to max(limits).

        Returns
        -------
        ilf : ndarray of shape (n, len(limits))
        """
        X = self._prepare_X(X)
        n = len(X)
        limits = sorted(limits)
        if basic_limit is None:
            basic_limit = max(limits)

        # Use _predict_thresholds_raw directly to avoid double-augmentation
        if self._is_mode_matching:
            alpha_mm, delta_mm = self.shape_params_
            thresholds = self._predict_thresholds_raw(X, self._w, alpha_mm, delta_mm)
        else:
            thresholds = np.full(n, self.threshold_)
        ilf_matrix = np.zeros((n, len(limits)))

        for i in range(n):
            t_i = float(thresholds[i])

            if self._is_mode_matching:
                alpha, delta = self.shape_params_
                sigma = self._sigma
                beta_i = float(np.exp(X[i] @ self._w))
                mu_i = sigma ** 2 + np.log(t_i)
                body = LognormalBody(mu=mu_i, sigma=sigma)
                tail = BurrTail(alpha=alpha, delta=delta, beta=beta_i)
            else:
                # Use global body; per-obs GPD scale
                body_params = self.body_params_
                if len(body_params) == 2 and hasattr(self, '_sigma'):
                    body = LognormalBody(mu=body_params[0], sigma=body_params[1])
                else:
                    body = self.composite._make_body()
                    body.params = body_params
                xi = self._xi if hasattr(self, '_xi') else 0.2
                sigma_i = float(np.exp(X[i] @ self._w))
                tail = GPDTail(xi=xi, sigma=sigma_i)

            # Approximate pi for this observation
            pi_i = 0.5  # default; refine via ratio of integrated densities

            from scipy.integrate import quad

            def lev_integrand(x, bdy=body, tl=tail, t=t_i, pi=pi_i):
                """E[min(X, L)] integrand = P(X > x)."""
                if x <= t:
                    body_sf = 1.0 - np.exp(bdy.logcdf(np.array([x]), t))[0]
                    return pi * body_sf + (1.0 - pi)
                else:
                    tail_sf = np.exp(tl.logsf(np.array([x]), t))[0]
                    return (1.0 - pi) * tail_sf

            lev_basic, _ = quad(lev_integrand, 0.0, basic_limit, limit=200)

            for j, lim in enumerate(limits):
                if lim >= basic_limit:
                    lev_lim, _ = quad(lev_integrand, 0.0, lim, limit=200)
                else:
                    lev_lim, _ = quad(lev_integrand, 0.0, lim, limit=200)
                ilf_matrix[i, j] = lev_lim / lev_basic if lev_basic > 0 else np.nan

        return ilf_matrix

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return average log-likelihood on (X, y).

        Higher is better (consistent with sklearn score convention for
        density estimators).
        """
        X_aug, y = self._validate_inputs(X, y)
        thresholds = self._predict_thresholds_raw(X_aug, self._w,
                                                   *self.shape_params_) if self._is_mode_matching else np.full(len(y), self.threshold_)
        sigma = self._sigma if self._is_mode_matching else None
        ll = 0.0
        for i, (y_i, t_i) in enumerate(zip(y, thresholds)):
            if self._is_mode_matching:
                alpha, delta = self.shape_params_
                beta_i = float(np.exp(X_aug[i] @ self._w))
                mu_i = sigma ** 2 + np.log(t_i)
                body = LognormalBody(mu=mu_i, sigma=sigma)
                tail = BurrTail(alpha=alpha, delta=delta, beta=beta_i)
                pi = self.pi_mean_
                if y_i <= t_i:
                    ll += np.log(pi) + float(body.logpdf(np.array([y_i]), t_i)[0])
                else:
                    ll += np.log(1 - pi) + float(tail.logpdf(np.array([y_i]), t_i)[0])
        return ll / len(y)

    def _prepare_X(self, X) -> np.ndarray:
        """Convert X to augmented array with intercept column."""
        try:
            import pandas as pd
            if isinstance(X, pd.DataFrame):
                if self.feature_cols is not None:
                    X = X[self.feature_cols].to_numpy(dtype=float)
                else:
                    X = X.to_numpy(dtype=float)
        except ImportError:
            pass
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.column_stack([np.ones(len(X)), X])

    def summary(self) -> str:
        """Return text summary of fitted regression model."""
        if self.coef_ is None:
            return f"{self.__class__.__name__} (not fitted)"
        lines = [
            f"{self.__class__.__name__}",
            f"  Base composite    : {self._composite_type}",
            f"  Mode-matching     : {self._is_mode_matching}",
            f"  Intercept (log)   : {self.intercept_:.4f}",
            f"  Coefficients      : {self.coef_}",
            f"  Shape params      : {self.shape_params_}",
            f"  Body params       : {self.body_params_}",
            f"  Mean pi           : {self.pi_mean_:.4f}",
            f"  Log-likelihood    : {self.loglik_:.2f}",
        ]
        if self.threshold_ is not None:
            lines.insert(3, f"  Threshold         : {self.threshold_:,.1f}")
        return "\n".join(lines)

    def bootstrap_ci(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_bootstrap: int = 200,
        alpha: float = 0.05,
        seed: int = 42,
    ) -> dict:
        """
        Bootstrap confidence intervals for regression coefficients.

        Parameters
        ----------
        X, y : training data
        n_bootstrap : int
        alpha : float
            Confidence level = 1 - alpha.
        seed : int

        Returns
        -------
        dict with keys 'coef_lower', 'coef_upper', 'intercept_lower',
        'intercept_upper'.
        """
        rng = np.random.default_rng(seed)
        n = len(y)
        coefs = []
        intercepts = []

        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            try:
                clone = CompositeSeverityRegressor(
                    composite=type(self.composite)(
                        threshold=self.composite.threshold,
                        threshold_method=self.composite.threshold_method,
                    ),
                    feature_cols=self.feature_cols,
                    n_starts=1,
                    max_iter=200,
                )
                if hasattr(X, 'iloc'):
                    Xi = X.iloc[idx]
                else:
                    Xi = np.asarray(X)[idx]
                yi = np.asarray(y)[idx]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clone.fit(Xi, yi)
                coefs.append(clone.coef_)
                intercepts.append(clone.intercept_)
            except Exception:
                continue

        if len(coefs) < 10:
            warnings.warn(
                f"Only {len(coefs)} bootstrap iterations converged. "
                "CIs may be unreliable.",
                UserWarning,
            )

        coefs = np.array(coefs)
        intercepts = np.array(intercepts)
        lo, hi = alpha / 2, 1.0 - alpha / 2
        return {
            "coef_lower": np.quantile(coefs, lo, axis=0) if len(coefs) > 0 else None,
            "coef_upper": np.quantile(coefs, hi, axis=0) if len(coefs) > 0 else None,
            "intercept_lower": np.quantile(intercepts, lo) if len(intercepts) > 0 else None,
            "intercept_upper": np.quantile(intercepts, hi) if len(intercepts) > 0 else None,
            "n_converged": len(coefs),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _composite_from_str(name: str) -> CompositeSeverityModel:
    mapping = {
        "lognormal_burr": lambda: LognormalBurrComposite(threshold_method="mode_matching"),
        "lognormal_gpd": lambda: LognormalGPDComposite(threshold_method="profile_likelihood"),
        "gamma_gpd": lambda: GammaGPDComposite(threshold_method="profile_likelihood"),
    }
    if name not in mapping:
        raise ValueError(
            f"Unknown composite string {name!r}. "
            f"Choose from: {list(mapping.keys())}"
        )
    return mapping[name]()
