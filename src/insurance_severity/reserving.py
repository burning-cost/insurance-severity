"""
IBNR claim reserving via the AIPW doubly-robust estimator.

Calcetero, Badescu, Lin (2025) reframe IBNR reserving as a population sampling
problem. Each claim is a Bernoulli trial: reported by valuation time τ with
inclusion probability π_i(τ) = P(U_i ≤ τ − T_i | x_i), where U_i is the
reporting delay.

The AIPW estimator is:

    L̂_IBNR = Σ_{IBNR} Ŷ_i  +  Σ_{reported} [(1−π̂_i)/π̂_i × (Y_i − Ŷ_i)]

First term: micro-level model predictions for unreported claims.
Second term: IPW-weighted residuals that correct the micro model for sampling
             bias (larger claims tend to be reported faster, so reported claims
             are a biased sample of the full population).

Double robustness (Proposition 2 of the paper): the AIPW estimate is unbiased
if EITHER the inclusion probabilities OR the severity model is correctly
specified. You only need one of the two components to be right.

Special cases:
- Chain-ladder: set π̂_i = 1/f_k (inverse CL development factor). No severity
  model needed. Recovers standard CL reserves exactly.
- Micro-level model: when the weighted balance property holds, the augmentation
  term vanishes and the micro model IS the AIPW estimator.
- Credibility: BF, Cape-Cod, Bühlmann-Straub are all AIPW with specific π and
  expert opinion as the assisting model.

References
----------
Calcetero, Badescu, Lin (2025). IBNR claim reserving as a population sampling
problem. arXiv:2502.15598.

Calcetero-Vanegas, Badescu, Lin (2024). Reserving based on an IPW estimator.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
from scipy import optimize, stats


# ---------------------------------------------------------------------------
# Weibull inclusion (reporting delay) model
# ---------------------------------------------------------------------------


class WeibullInclusionModel:
    """
    Parametric reporting delay model using the Weibull distribution.

    Models P(U ≤ τ − T | x) where U is the reporting delay. The scale
    parameter may depend on covariates via a log-linear link; the shape
    parameter is constant.

    The key likelihood complication: we only observe delays for reported
    claims, i.e. U_i ≤ τ − T_i. The data are **right-truncated**. Standard
    Weibull MLE ignores this and produces biased estimates. The corrected
    log-likelihood is::

        ℓ(θ) = Σ_i [log f(u_i | x_i) − log F(τ_i − T_i | x_i)]

    where τ_i − T_i is the maximum observable delay for claim i.

    Parameters
    ----------
    fit_covariates : bool
        If True, fit a log-linear model for the scale parameter using the
        supplied covariates. If False, fit a single scale parameter for all
        claims.

    Attributes
    ----------
    shape_ : float
        Fitted Weibull shape parameter (k > 0).
    intercept_ : float
        Log-scale intercept (log λ₀).
    coef_ : np.ndarray or None
        Covariate coefficients on the log scale. None if no covariates used.
    """

    def __init__(self, fit_covariates: bool = True) -> None:
        self.fit_covariates = fit_covariates
        self.shape_: Optional[float] = None
        self.intercept_: Optional[float] = None
        self.coef_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        delay: np.ndarray,
        covariates: Optional[np.ndarray] = None,
        truncation_times: Optional[np.ndarray] = None,
    ) -> "WeibullInclusionModel":
        """
        Fit Weibull model to right-truncated reporting delay data.

        Parameters
        ----------
        delay : np.ndarray, shape (n,)
            Observed reporting delays U_i (all ≥ 0). Only reported claims
            are present in this array.
        covariates : np.ndarray, shape (n, p), optional
            Covariate matrix for the scale parameter. If None, a single
            scale is fitted.
        truncation_times : np.ndarray, shape (n,)
            τ_i − T_i, the maximum observable delay for each claim. Must be
            ≥ delay_i everywhere. If None, defaults to delay (no truncation
            correction — only appropriate when all claims are observed).

        Returns
        -------
        self
        """
        delay = np.asarray(delay, dtype=float)
        n = len(delay)

        if truncation_times is None:
            truncation_times = delay.copy()
        else:
            truncation_times = np.asarray(truncation_times, dtype=float)

        if np.any(delay > truncation_times + 1e-10):
            raise ValueError(
                "Some delay values exceed truncation_times. Delays must be "
                "≤ τ − T for each claim."
            )

        use_covs = (
            self.fit_covariates
            and covariates is not None
            and np.ndarray is not None
        )
        if use_covs:
            X = np.asarray(covariates, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            p = X.shape[1]
        else:
            X = None
            p = 0

        # -------------------------------------------------------------------
        # Truncation-corrected Weibull log-likelihood
        # log f(u | λ, k) = log k − log λ + (k-1)(log u − log λ) − (u/λ)^k
        # log F(t | λ, k) = log(1 − exp(−(t/λ)^k))
        # For numerical stability use log1p(−exp(−x)) = log(1 − exp(−x)).
        # -------------------------------------------------------------------

        def neg_loglik(params: np.ndarray) -> float:
            log_k = params[0]
            log_lam0 = params[1]
            coef = params[2:] if p > 0 else np.array([])

            k = np.exp(log_k)
            if X is not None:
                log_lam = log_lam0 + X @ coef
            else:
                log_lam = np.full(n, log_lam0)

            lam = np.exp(log_lam)

            # Log-density of Weibull
            log_f = (
                np.log(k)
                - log_lam
                + (k - 1.0) * (np.log(np.maximum(delay, 1e-15)) - log_lam)
                - (delay / lam) ** k
            )

            # Log-CDF at truncation time: log F(trunc | lam, k)
            z_trunc = (truncation_times / lam) ** k
            # log(1 − exp(−z)) — stable for large and small z
            log_cdf_trunc = np.where(
                z_trunc > 30,
                0.0,  # essentially 1
                np.log(-np.expm1(-z_trunc)),
            )

            ll = np.sum(log_f - log_cdf_trunc)
            return -ll

        # Initial parameters: k=1 (exponential), λ=mean delay
        mean_d = float(np.mean(delay)) if np.mean(delay) > 0 else 1.0
        x0 = np.concatenate([[0.0, np.log(mean_d)], np.zeros(p)])

        result = optimize.minimize(
            neg_loglik,
            x0,
            method="L-BFGS-B",
            options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
        )

        if not result.success:
            warnings.warn(
                f"WeibullInclusionModel optimisation did not converge: "
                f"{result.message}",
                UserWarning,
                stacklevel=2,
            )

        self.shape_ = float(np.exp(result.x[0]))
        self.intercept_ = float(result.x[1])
        self.coef_ = result.x[2:] if p > 0 else None

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_inclusion_prob(
        self,
        tau_minus_t: np.ndarray,
        covariates: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute π̂_i = P(U ≤ τ − T_i | x_i).

        Parameters
        ----------
        tau_minus_t : np.ndarray, shape (n,)
            Time available for reporting: valuation time minus accident time.
        covariates : np.ndarray, shape (n, p), optional
            Covariate matrix. Required if the model was fitted with covariates.

        Returns
        -------
        np.ndarray, shape (n,)
            Inclusion probabilities in (0, 1].
        """
        if self.shape_ is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        tau_minus_t = np.asarray(tau_minus_t, dtype=float)
        n = len(tau_minus_t)

        if self.coef_ is not None and covariates is not None:
            X = np.asarray(covariates, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            log_lam = self.intercept_ + X @ self.coef_
        else:
            log_lam = np.full(n, self.intercept_)

        lam = np.exp(log_lam)
        z = (tau_minus_t / lam) ** self.shape_
        pi = 1.0 - np.exp(-z)

        # Clip to avoid exact 0 (which causes division by zero in IPW)
        return np.clip(pi, 1e-6, 1.0)


# ---------------------------------------------------------------------------
# Helper: chain-ladder development factors → per-claim inclusion probability
# ---------------------------------------------------------------------------


def _cl_inclusion_probs(
    accident_periods: np.ndarray,
    development_factors: dict[Any, float],
) -> np.ndarray:
    """
    Compute π̂_i = 1 / f_k for each claim, where f_k is the development-to-
    ultimate factor for accident period k.

    Parameters
    ----------
    accident_periods : np.ndarray
        Accident period label for each claim.
    development_factors : dict
        Mapping from accident period label to development-to-ultimate factor
        f_k ≥ 1.

    Returns
    -------
    np.ndarray
        Inclusion probabilities, one per claim.
    """
    pi = np.array(
        [1.0 / development_factors[k] for k in accident_periods],
        dtype=float,
    )
    return np.clip(pi, 1e-6, 1.0)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class PopulationSamplingReserve:
    """
    AIPW doubly-robust IBNR reserve estimator.

    Combines an inclusion probability model (reporting delay survival
    function) with a severity model for double robustness. The key identity
    is:

        L̂_IBNR = Σ_{IBNR} Ŷ_i  +  Σ_{reported} [(1−π̂_i)/π̂_i × (Y_i − Ŷ_i)]

    The first term is the micro-level model's prediction for unreported claims
    (estimated count × predicted mean, or claim-level if IBNR claim features
    are available). The second term corrects for bias in the severity model
    caused by the correlation between claim size and reporting speed.

    Parameters
    ----------
    inclusion_model : object or callable, optional
        Fitted model that can estimate π̂_i. Must expose either
        ``predict_inclusion_prob(tau_minus_t, covariates)`` (as
        :class:`WeibullInclusionModel` does) or be a plain callable
        ``(tau_minus_t, covariates) -> ndarray``.
        If None and ``method`` is not ``"chain_ladder"`` or ``"micro"``,
        a :class:`WeibullInclusionModel` is fitted automatically.
    severity_model : object or callable, optional
        Fitted model predicting claim severity. Must expose either
        ``predict(X)`` (sklearn-style) or be a callable ``X -> ndarray``.
        If None and ``method`` is not ``"ipw"`` or ``"chain_ladder"``,
        a lognormal mean (exp(μ̂ + σ̂²/2)) over the reported claims is used.
    method : {"aipw", "ipw", "micro", "chain_ladder"}
        Estimator to use:

        - ``"aipw"``: full doubly-robust estimator (default). Uses both
          inclusion model and severity model.
        - ``"ipw"``: IPW-only estimator. Uses only inclusion model; no
          severity model.
        - ``"micro"``: micro-level model only. Uses only severity model;
          no bias correction.
        - ``"chain_ladder"``: exact chain-ladder. Requires
          ``development_factors`` in ``fit()``.

    Attributes
    ----------
    ibnr_ : float
        IBNR reserve estimate (set after ``fit()``).
    ultimate_ : float
        Ultimate loss estimate (set after ``fit()``).
    diagnostics_ : dict
        Full diagnostics dict (set after ``fit()``).
    """

    def __init__(
        self,
        inclusion_model: Optional[Any] = None,
        severity_model: Optional[Any] = None,
        method: str = "aipw",
    ) -> None:
        valid_methods = {"aipw", "ipw", "micro", "chain_ladder"}
        if method not in valid_methods:
            raise ValueError(
                f"method must be one of {valid_methods}, got {method!r}"
            )
        self.inclusion_model = inclusion_model
        self.severity_model = severity_model
        self.method = method

        self.ibnr_: Optional[float] = None
        self.ultimate_: Optional[float] = None
        self.diagnostics_: Optional[dict] = None

        # Stored after fit for use in other methods
        self._pi_reported: Optional[np.ndarray] = None
        self._y_reported: Optional[np.ndarray] = None
        self._yhat_reported: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        claims_df: pd.DataFrame,
        *,
        accident_time_col: str = "accident_time",
        report_time_col: str = "report_time",
        severity_col: str = "severity",
        valuation_time: Optional[float] = None,
        feature_cols: Optional[list[str]] = None,
        development_factors: Optional[dict] = None,
        n_ibnr: Optional[int] = None,
        ibnr_features: Optional[pd.DataFrame] = None,
    ) -> "PopulationSamplingReserve":
        """
        Fit inclusion probability model (if not provided) and compute reserves.

        The data frame should contain one row per **reported** claim. IBNR
        claims by definition are not in the data; their count is estimated
        from the inclusion probabilities unless ``n_ibnr`` is supplied.

        Parameters
        ----------
        claims_df : pd.DataFrame
            Reported claims. Must contain ``accident_time_col``,
            ``report_time_col``, and ``severity_col``.
        accident_time_col : str
            Column with accident occurrence time (numeric or datetime).
        report_time_col : str
            Column with report time (numeric or datetime).
        severity_col : str
            Column with observed claim amounts (Y_i > 0).
        valuation_time : float, optional
            The triangle cut-off date τ. If None, uses the maximum report
            time observed.
        feature_cols : list of str, optional
            Columns used as covariates in the inclusion and/or severity model.
            For chain-ladder, ignored.
        development_factors : dict, optional
            Required when ``method="chain_ladder"``. Maps accident period
            label to the development-to-ultimate factor f_k ≥ 1. The accident
            period label must match the values in ``accident_time_col``.
        n_ibnr : int, optional
            Known or estimated number of IBNR claims. If None, estimated as
            Σ_{reported} (1 − π̂_i) / π̂_i (Horvitz-Thompson style). Only
            used for the ``"micro"`` method.
        ibnr_features : pd.DataFrame, optional
            Feature matrix for IBNR claim severity predictions (one row per
            IBNR claim). If None, uses the mean of ``feature_cols`` over
            reported claims.

        Returns
        -------
        self
        """
        df = claims_df.copy()

        # ------------------------------------------------------------------
        # Basic validation
        # ------------------------------------------------------------------
        for col in [accident_time_col, report_time_col, severity_col]:
            if col not in df.columns:
                raise ValueError(f"Column {col!r} not found in claims_df.")

        y = df[severity_col].to_numpy(dtype=float)
        if np.any(y <= 0):
            raise ValueError(
                "severity_col must contain strictly positive values."
            )

        # ------------------------------------------------------------------
        # Times
        # ------------------------------------------------------------------
        t_acc = df[accident_time_col].to_numpy(dtype=float)
        t_rep = df[report_time_col].to_numpy(dtype=float)

        if valuation_time is None:
            valuation_time = float(np.max(t_rep))

        tau = float(valuation_time)
        delay = t_rep - t_acc  # U_i for each reported claim
        tau_minus_t = tau - t_acc  # maximum observable delay per claim

        if np.any(delay < 0):
            raise ValueError(
                "report_time must be ≥ accident_time for all claims."
            )
        if np.any(tau_minus_t < delay - 1e-10):
            raise ValueError(
                "valuation_time must be ≥ report_time for all reported claims."
            )

        n_reported = len(df)
        total_reported = float(np.sum(y))

        # ------------------------------------------------------------------
        # Features
        # ------------------------------------------------------------------
        if feature_cols is not None:
            X = df[feature_cols].to_numpy(dtype=float)
        else:
            X = None

        # ------------------------------------------------------------------
        # Compute inclusion probabilities
        # ------------------------------------------------------------------
        if self.method == "chain_ladder":
            if development_factors is None:
                raise ValueError(
                    "development_factors must be provided when "
                    "method='chain_ladder'."
                )
            acc_periods = df[accident_time_col].to_numpy()
            pi_hat = _cl_inclusion_probs(acc_periods, development_factors)

        elif self.method == "micro":
            # No inclusion model needed for micro-only
            pi_hat = None

        else:
            # ipw or aipw: need inclusion model
            if self.inclusion_model is None:
                # Auto-fit a WeibullInclusionModel
                wim = WeibullInclusionModel(fit_covariates=(X is not None))
                wim.fit(delay, covariates=X, truncation_times=tau_minus_t)
                self.inclusion_model = wim

            pi_hat = self._call_inclusion_model(
                self.inclusion_model, tau_minus_t, X
            )

        # ------------------------------------------------------------------
        # Compute severity predictions for reported claims
        # ------------------------------------------------------------------
        if self.method in ("aipw", "micro"):
            if self.severity_model is None:
                # Default: lognormal MLE on log(y), predict mean for everyone
                log_y = np.log(y)
                mu_hat = float(np.mean(log_y))
                sig_hat = float(np.std(log_y, ddof=1))
                lognormal_mean = float(np.exp(mu_hat + 0.5 * sig_hat**2))
                yhat_reported = np.full(n_reported, lognormal_mean)
                self._default_severity_mean = lognormal_mean
            else:
                yhat_reported = self._call_severity_model(
                    self.severity_model, X
                )
        else:
            yhat_reported = None

        # ------------------------------------------------------------------
        # Estimate IBNR component
        # ------------------------------------------------------------------
        if self.method == "chain_ladder":
            # CL: IPW estimator. No severity model.
            # L̂_IBNR = Σ_{reported} [(1−π̂_i)/π̂_i × Y_i]
            ipw_weights = (1.0 - pi_hat) / pi_hat
            ibnr_estimate = float(np.sum(ipw_weights * y))
            augmentation = 0.0
            n_ibnr_est = float(np.sum(ipw_weights))

        elif self.method == "ipw":
            # IPW only
            ipw_weights = (1.0 - pi_hat) / pi_hat
            ibnr_estimate = float(np.sum(ipw_weights * y))
            augmentation = 0.0
            n_ibnr_est = float(np.sum(ipw_weights))

        elif self.method == "micro":
            # Micro model only: Σ_{IBNR} Ŷ_i
            # Estimate n_ibnr if not supplied
            if n_ibnr is not None:
                n_ibnr_est = float(n_ibnr)
            elif pi_hat is not None:
                ipw_weights = (1.0 - pi_hat) / pi_hat
                n_ibnr_est = float(np.sum(ipw_weights))
            else:
                raise ValueError(
                    "For method='micro' without inclusion_model, supply "
                    "n_ibnr explicitly."
                )

            # Predict mean severity for IBNR claims
            if ibnr_features is not None:
                X_ibnr = ibnr_features.to_numpy(dtype=float)
                yhat_ibnr_mean = float(
                    np.mean(self._call_severity_model(self.severity_model, X_ibnr))
                )
            else:
                yhat_ibnr_mean = float(np.mean(yhat_reported))

            ibnr_estimate = n_ibnr_est * yhat_ibnr_mean
            augmentation = 0.0

        else:
            # AIPW: full doubly-robust estimator
            # L̂_IBNR = Σ_{IBNR} Ŷ_i  +  Σ_{reported} [(1−π̂_i)/π̂_i × (Y_i − Ŷ_i)]
            ipw_weights = (1.0 - pi_hat) / pi_hat
            n_ibnr_est = float(np.sum(ipw_weights))

            # IBNR severity component: use mean of predictions from reported
            # claims, scaled by estimated IBNR count
            yhat_ibnr_mean = float(np.mean(yhat_reported))
            micro_ibnr = n_ibnr_est * yhat_ibnr_mean

            # Augmentation (bias correction) term
            augmentation = float(np.sum(ipw_weights * (y - yhat_reported)))

            ibnr_estimate = micro_ibnr + augmentation

        # ------------------------------------------------------------------
        # Store results
        # ------------------------------------------------------------------
        ultimate_estimate = total_reported + ibnr_estimate

        # Weighted balance ratio
        if pi_hat is not None and yhat_reported is not None:
            ipw_w = (1.0 - pi_hat) / pi_hat
            num = float(np.sum(ipw_w * y))
            den = float(np.sum(ipw_w * yhat_reported))
            wb_ratio = num / den if abs(den) > 1e-12 else np.nan
        else:
            wb_ratio = np.nan

        self.ibnr_ = ibnr_estimate
        self.ultimate_ = ultimate_estimate
        self._pi_reported = pi_hat
        self._y_reported = y
        self._yhat_reported = yhat_reported

        self.diagnostics_ = {
            "ibnr_estimate": ibnr_estimate,
            "ultimate_estimate": ultimate_estimate,
            "n_reported": n_reported,
            "n_ibnr_estimated": n_ibnr_est,
            "augmentation_term": augmentation,
            "weighted_balance_ratio": wb_ratio,
            "method": self.method,
            "total_reported_losses": total_reported,
            "valuation_time": tau,
        }

        return self

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def estimate_ibnr(self) -> float:
        """
        Return the IBNR reserve estimate.

        Returns
        -------
        float
            IBNR estimate (losses yet to be reported as of valuation time).

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called.
        """
        if self.ibnr_ is None:
            raise RuntimeError("Call fit() before estimate_ibnr().")
        return self.ibnr_

    def estimate_ultimate(self) -> float:
        """
        Return the ultimate loss estimate.

        Ultimate = reported losses + IBNR estimate.

        Returns
        -------
        float
        """
        if self.ultimate_ is None:
            raise RuntimeError("Call fit() before estimate_ultimate().")
        return self.ultimate_

    def diagnostics(self) -> dict:
        """
        Return the full diagnostics dictionary.

        Keys
        ----
        ibnr_estimate : float
        ultimate_estimate : float
        n_reported : int
        n_ibnr_estimated : float
            Horvitz-Thompson estimate of the IBNR claim count.
        augmentation_term : float
            The IPW-weighted residual correction Σ[(1−π̂_i)/π̂_i × (Y_i − Ŷ_i)].
            Near zero when the severity model satisfies the weighted balance
            property or when the model is well-specified.
        weighted_balance_ratio : float
            b = Σ[(1−π̂_i)/π̂_i × Y_i] / Σ[(1−π̂_i)/π̂_i × Ŷ_i].
            Should be ≈ 1 if the weighted balance property holds.
        method : str
        total_reported_losses : float
        valuation_time : float

        Returns
        -------
        dict
        """
        if self.diagnostics_ is None:
            raise RuntimeError("Call fit() before diagnostics().")
        return dict(self.diagnostics_)

    # ------------------------------------------------------------------
    # Weighted balance adjustment
    # ------------------------------------------------------------------

    def weighted_balance_adjust(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Apply a scalar weighted balance adjustment to severity predictions.

        The weighted balance property (WBP) states that a severity model is
        well-calibrated for IBNR claims when::

            Σ[(1−π̂_i)/π̂_i × Ŷ_i] = Σ[(1−π̂_i)/π̂_i × Y_i]

        When this does not hold, a simple scalar correction brings the model
        into balance::

            b = Σ[(1−π̂_i)/π̂_i × Y_i] / Σ[(1−π̂_i)/π̂_i × Ŷ_i]

        This gives ŷ_WBP = b × ŷ. Calcetero et al. show this scalar
        adjustment achieves near-identical performance to the full AIPW
        estimator in empirical tests on Canadian auto data.

        Parameters
        ----------
        y_pred : np.ndarray
            Severity predictions to adjust.

        Returns
        -------
        np.ndarray
            Adjusted predictions b × y_pred.

        Raises
        ------
        RuntimeError
            If fit() has not been called or the method does not use an
            inclusion model.
        """
        if self._pi_reported is None:
            raise RuntimeError(
                "weighted_balance_adjust requires an inclusion model. "
                "Call fit() with method='aipw', 'ipw', or 'chain_ladder'."
            )
        if self._y_reported is None or self._yhat_reported is None:
            raise RuntimeError("Call fit() before weighted_balance_adjust().")

        pi = self._pi_reported
        y = self._y_reported
        yhat = self._yhat_reported

        ipw_w = (1.0 - pi) / pi
        num = float(np.sum(ipw_w * y))
        den = float(np.sum(ipw_w * yhat))

        if abs(den) < 1e-12:
            raise RuntimeError(
                "Denominator of weighted balance scalar is near zero. "
                "Check that severity predictions are not all zero."
            )

        b = num / den
        return b * np.asarray(y_pred, dtype=float)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _call_inclusion_model(
        model: Any,
        tau_minus_t: np.ndarray,
        X: Optional[np.ndarray],
    ) -> np.ndarray:
        """Dispatch to the inclusion model's prediction interface."""
        if callable(model) and not hasattr(model, "predict_inclusion_prob"):
            # Plain callable: (tau_minus_t, X) -> ndarray
            return np.asarray(model(tau_minus_t, X), dtype=float)
        elif hasattr(model, "predict_inclusion_prob"):
            return np.asarray(
                model.predict_inclusion_prob(tau_minus_t, covariates=X),
                dtype=float,
            )
        elif hasattr(model, "predict_proba"):
            # sklearn-style: create feature matrix [tau_minus_t, X]
            if X is not None:
                feat = np.column_stack([tau_minus_t, X])
            else:
                feat = tau_minus_t.reshape(-1, 1)
            proba = model.predict_proba(feat)
            # Assume binary class: column 1 is P(reported)
            return np.asarray(proba[:, 1], dtype=float)
        else:
            raise TypeError(
                f"inclusion_model must have predict_inclusion_prob(), "
                f"predict_proba(), or be callable. Got {type(model)}."
            )

    @staticmethod
    def _call_severity_model(
        model: Any,
        X: Optional[np.ndarray],
    ) -> np.ndarray:
        """Dispatch to the severity model's prediction interface."""
        if model is None:
            raise RuntimeError("Severity model is None.")
        if callable(model) and not hasattr(model, "predict"):
            # Plain callable: X -> ndarray
            return np.asarray(model(X), dtype=float)
        elif hasattr(model, "predict"):
            if X is not None:
                return np.asarray(model.predict(X), dtype=float)
            else:
                raise ValueError(
                    "severity_model.predict() requires features but "
                    "feature_cols was not supplied."
                )
        else:
            raise TypeError(
                f"severity_model must have predict() or be callable. "
                f"Got {type(model)}."
            )
