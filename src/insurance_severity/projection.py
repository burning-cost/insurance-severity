"""
Projection-to-Ultimate (PtU) factor estimation for individual RBNS claims.

Implements the one-shot individual claims reserving approach of Richman &
Wüthrich (arXiv:2603.11660, March 2026).  The core idea: instead of the
chain-ladder's iterated period-by-period factors, estimate a single PtU
multiplier directly from each claim's current development state — cumulative
paid, claims incurred, development lag, claim status, plus any additional
covariates you pass in.

The empirical finding from the paper: plain OLS outperforms neural networks on
realistic reserving datasets (5x5 triangles, 22k–67k claims).  The balance
property of identity-link regression is partly why — sum of predictions equals
sum of targets in-sample, which removes compounding bias across backward
recursion steps.

Method
------
Two modes are supported:

``method="ols"``
    OLS via numpy least squares.  Fits log(ultimate / paid_to_date) on a
    design matrix of development features and any additional covariates.
    This is the "Listing 2" style of the paper — unweighted regression in
    levels after a log-ratio transform.

``method="ridge"``
    Ridge regression via scipy.  Identical target transformation; useful
    when the design matrix is near-singular (many correlated features) or
    when you have a small learning set at short development lags.

Target variable
---------------
We fit on  y = log(ultimate / paid_to_date)  because:

1.  PtU factors are multiplicative; log scale linearises them.
2.  The ratio is bounded below by 1 for open claims (paid ≤ ultimate).
3.  Residuals on the log-ratio scale are closer to homoscedastic than
    residuals on the raw ultimate.

Predict returns predicted_ultimate = paid_to_date * exp(fitted log-ratio).

References
----------
Richman, R. & Wüthrich, M. V. (2026). "One-Shot Individual Claims Reserving."
arXiv:2603.11660 [stat.AP].

Richman, R. & Wüthrich, M. V. (2026). "Projection-to-Ultimate Factors for
Individual Claims Reserving." arXiv:2602.15385 [stat.AP].
"""

from __future__ import annotations

import warnings
from typing import Literal, Optional

import numpy as np
import polars as pl
from scipy import linalg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_EPSILON = 1e-6  # floor for paid_to_date to avoid log(0)
_DEFAULT_FEATURES = ["dev_month", "log_paid", "claim_age"]


def _add_log_paid(df: pl.DataFrame, paid_col: str) -> pl.DataFrame:
    """Return df with a ``log_paid`` column (log of paid_to_date + epsilon)."""
    return df.with_columns(
        (pl.col(paid_col).clip(lower_bound=_EPSILON).log()).alias("log_paid")
    )


def _build_design_matrix(
    df: pl.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    """
    Extract numeric feature matrix from a Polars DataFrame.

    Categorical columns are label-encoded (integer codes).  Boolean columns
    are cast to float.  All columns are cast to float64.  An intercept column
    of ones is prepended.

    Parameters
    ----------
    df : pl.DataFrame
        Input frame.  Must contain all columns listed in ``feature_cols``.
    feature_cols : list of str
        Feature column names.

    Returns
    -------
    X : np.ndarray, shape (n, 1 + len(feature_cols))
        Design matrix with intercept prepended.
    """
    n = len(df)
    cols: list[np.ndarray] = [np.ones(n, dtype=float)]

    for col in feature_cols:
        series = df[col]
        if series.dtype == pl.Categorical or series.dtype == pl.String:
            # Label-encode: cast to categorical then extract codes
            encoded = series.cast(pl.Categorical).to_physical()
            cols.append(encoded.to_numpy().astype(float))
        elif series.dtype == pl.Boolean:
            cols.append(series.cast(pl.Float64).to_numpy())
        else:
            cols.append(series.cast(pl.Float64).to_numpy())

    return np.column_stack(cols)


def _ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS via numpy least squares.  Returns coefficient vector."""
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return coeffs


def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Ridge regression via normal equations with Tikhonov regularisation.

    Intercept column (first column) is NOT penalised.
    """
    n, p = X.shape
    # Penalty matrix: penalise all but intercept
    penalty = np.eye(p, dtype=float)
    penalty[0, 0] = 0.0
    A = X.T @ X + alpha * penalty
    b = X.T @ y
    coeffs = linalg.solve(A, b, assume_a="pos")
    return coeffs


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class ProjectionToUltimate:
    """
    One-shot Projection-to-Ultimate (PtU) factor estimation for RBNS claims.

    Fits a regression of  log(ultimate / paid_to_date)  on development features
    using the training set, then applies the fitted multiplier to open claims to
    produce predicted ultimate costs.

    Parameters
    ----------
    development_features : list of str
        Column names in the training DataFrame to use as predictors.  These
        should describe how far along the claim is in its development cycle.
        Typical choices: ``["dev_month", "log_paid", "claim_age"]`` (the
        default).  Any extra columns (peril, region, injury_type, ...) are
        appended after computing ``log_paid`` when
        ``auto_add_log_paid=True``.
    method : {"ols", "ridge"}
        Estimation method.  "ols" uses numpy least squares (no new deps).
        "ridge" adds L2 regularisation via scipy (alpha controlled by
        ``ridge_alpha``).
    ridge_alpha : float, default 1.0
        Ridge regularisation strength.  Ignored when ``method="ols"``.
    auto_add_log_paid : bool, default True
        If True and ``"log_paid"`` appears in ``development_features``, the
        class computes ``log(paid_to_date + epsilon)`` automatically.  If
        False, you must supply ``log_paid`` in the DataFrame yourself.
    min_train_rows : int, default 10
        Minimum observations required to fit.  Raises ``ValueError`` if the
        training set is smaller.

    Examples
    --------
    >>> from insurance_severity import ProjectionToUltimate
    >>> ptu = ProjectionToUltimate(
    ...     development_features=["dev_month", "log_paid", "claim_age"],
    ...     method="ols",
    ... )
    >>> ptu.fit(train_df, paid_col="paid_to_date", ultimate_col="ultimate_cost")
    >>> preds = ptu.predict(open_claims_df)
    >>> ptu.summary()

    Notes
    -----
    Zero and negative paid amounts are handled before log-transformation:
    values are clipped to ``1e-6``.  A warning is issued when more than 5% of
    training rows have zero paid (possible subrogation recoveries or data
    errors).

    Predictions for claims where ``paid_to_date = 0`` use a floor so the
    model does not produce undefined multipliers.  The ``ptu_factor`` column
    in the prediction output will be large for these claims — treat them with
    caution.

    References
    ----------
    Richman & Wüthrich (arXiv:2603.11660, March 2026), Algorithm 3.
    """

    def __init__(
        self,
        development_features: Optional[list[str]] = None,
        method: Literal["ols", "ridge"] = "ols",
        ridge_alpha: float = 1.0,
        auto_add_log_paid: bool = True,
        min_train_rows: int = 10,
    ) -> None:
        if method not in ("ols", "ridge"):
            raise ValueError(f"method must be 'ols' or 'ridge', got {method!r}")
        if ridge_alpha <= 0:
            raise ValueError(f"ridge_alpha must be positive, got {ridge_alpha}")
        if min_train_rows < 2:
            raise ValueError(f"min_train_rows must be >= 2, got {min_train_rows}")

        self.development_features: list[str] = (
            list(development_features)
            if development_features is not None
            else list(_DEFAULT_FEATURES)
        )
        self.method = method
        self.ridge_alpha = float(ridge_alpha)
        self.auto_add_log_paid = auto_add_log_paid
        self.min_train_rows = min_train_rows

        # Set after fit()
        self._coeffs: Optional[np.ndarray] = None
        self._feature_names: Optional[list[str]] = None  # incl. intercept
        self._n_train: Optional[int] = None
        self._r2: Optional[float] = None
        self._rmse: Optional[float] = None
        self._residuals: Optional[np.ndarray] = None
        self._paid_col: Optional[str] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pl.DataFrame,
        paid_col: str = "paid_to_date",
        ultimate_col: str = "ultimate_cost",
    ) -> "ProjectionToUltimate":
        """
        Fit the PtU regression on a training set of settled/observed claims.

        The training set should contain claims where the ultimate is known —
        typically fully settled claims from earlier accident cohorts that have
        reached the end of their development in the upper triangle.

        Parameters
        ----------
        df : pl.DataFrame
            Training data.  Must contain ``paid_col``, ``ultimate_col``, and
            all columns in ``development_features``.
        paid_col : str, default "paid_to_date"
            Column name for cumulative paid amount at the evaluation date.
        ultimate_col : str, default "ultimate_cost"
            Column name for the known ultimate claim cost.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If required columns are missing, if training data is too small,
            or if all ultimate values are less than or equal to paid (no open
            development left to estimate).
        """
        if not isinstance(df, pl.DataFrame):
            raise TypeError(f"df must be a polars DataFrame, got {type(df).__name__}")

        required = {paid_col, ultimate_col} | set(self.development_features)
        # Remove "log_paid" from required if it will be auto-added
        if self.auto_add_log_paid and "log_paid" in required:
            required.discard("log_paid")

        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in training DataFrame: {sorted(missing)}")

        # Prepare log_paid if needed
        if self.auto_add_log_paid and "log_paid" in self.development_features:
            df = _add_log_paid(df, paid_col)

        # Check for zero-paid observations
        paid_arr = df[paid_col].to_numpy().astype(float)
        n_zero_paid = int(np.sum(paid_arr <= 0))
        if n_zero_paid > 0.05 * len(paid_arr):
            warnings.warn(
                f"{n_zero_paid} rows ({100 * n_zero_paid / len(paid_arr):.1f}%) "
                f"have zero or negative paid_to_date. "
                "These may be subrogation recoveries or data errors. "
                "Values are clipped to 1e-6 before log-transformation.",
                stacklevel=2,
            )

        paid_clipped = np.maximum(paid_arr, _EPSILON)
        ultimate_arr = df[ultimate_col].to_numpy().astype(float)

        # Validate ultimates
        n_bad_ultimate = int(np.sum(ultimate_arr <= 0))
        if n_bad_ultimate > 0:
            raise ValueError(
                f"{n_bad_ultimate} row(s) have non-positive ultimate_cost. "
                "Ultimate cost must be strictly positive."
            )

        # Target: log(ultimate / paid)
        y = np.log(ultimate_arr / paid_clipped)

        # Drop training rows where paid > ultimate (subrogation / over-payment)
        neg_mask = y < 0
        n_neg = int(np.sum(neg_mask))
        if n_neg > 0:
            warnings.warn(
                f"{n_neg} training row(s) have paid_to_date > ultimate_cost "
                "(subrogation or data issue). These rows are excluded from fitting.",
                stacklevel=2,
            )
            keep = ~neg_mask
            df = df.filter(pl.Series(keep))
            paid_clipped = paid_clipped[keep]
            ultimate_arr = ultimate_arr[keep]
            y = y[keep]

        n = len(y)
        if n < self.min_train_rows:
            raise ValueError(
                f"Training set has {n} rows after filtering, "
                f"but min_train_rows={self.min_train_rows}. "
                "Provide more training data or lower min_train_rows."
            )

        # Build design matrix
        X = _build_design_matrix(df, self.development_features)

        # Fit
        if self.method == "ols":
            coeffs = _ols_fit(X, y)
        else:
            coeffs = _ridge_fit(X, y, self.ridge_alpha)

        # Diagnostics
        y_hat = X @ coeffs
        residuals = y - y_hat
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        rmse = float(np.sqrt(ss_res / n))

        # Store
        self._coeffs = coeffs
        self._feature_names = ["intercept"] + self.development_features
        self._n_train = n
        self._r2 = r2
        self._rmse = rmse
        self._residuals = residuals
        self._paid_col = paid_col

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        df: pl.DataFrame,
        paid_col: Optional[str] = None,
        add_prediction_interval: bool = True,
        pi_coverage: float = 0.90,
    ) -> pl.DataFrame:
        """
        Predict ultimate costs for open (RBNS) claims.

        Parameters
        ----------
        df : pl.DataFrame
            Open claims at the evaluation date.  Must contain all columns in
            ``development_features`` and the paid column (determined from
            ``paid_col`` or the column used during ``fit()``).
        paid_col : str, optional
            Override the paid column name.  Defaults to the column passed at
            fit time.
        add_prediction_interval : bool, default True
            If True, compute a simple prediction interval by adding ±z * RMSE
            on the log-ratio scale, then exponentiating back.  Only meaningful
            when ``method="ridge"`` (where RMSE reflects regularised residuals);
            OLS RMSE on the training set is optimistic.
        pi_coverage : float, default 0.90
            Coverage level for the prediction interval (e.g. 0.90 = 90% PI).

        Returns
        -------
        pl.DataFrame
            Original columns plus:
            - ``predicted_ultimate`` : float — predicted ultimate cost
            - ``ptu_factor`` : float — predicted ultimate / paid_to_date
            - ``log_ptu`` : float — fitted log-ratio (for diagnostics)
            - ``pi_lower`` : float — lower bound of prediction interval
              (only when ``add_prediction_interval=True``)
            - ``pi_upper`` : float — upper bound of prediction interval
              (only when ``add_prediction_interval=True``)

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called.
        """
        self._check_fitted()

        if not isinstance(df, pl.DataFrame):
            raise TypeError(f"df must be a polars DataFrame, got {type(df).__name__}")

        effective_paid_col = paid_col if paid_col is not None else self._paid_col

        # Validate columns
        required_feats = set(self.development_features)
        if self.auto_add_log_paid and "log_paid" in required_feats:
            required_feats.discard("log_paid")
            required_feats.add(effective_paid_col)
        else:
            required_feats.add(effective_paid_col)

        missing = required_feats - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in prediction DataFrame: {sorted(missing)}")

        # Build log_paid if needed
        if self.auto_add_log_paid and "log_paid" in self.development_features:
            df_pred = _add_log_paid(df, effective_paid_col)
        else:
            df_pred = df

        paid_arr = df[effective_paid_col].to_numpy().astype(float)
        paid_clipped = np.maximum(paid_arr, _EPSILON)

        n_zero_pred = int(np.sum(paid_arr <= 0))
        if n_zero_pred > 0:
            warnings.warn(
                f"{n_zero_pred} prediction row(s) have zero or negative paid_to_date. "
                "ptu_factor will be large for these claims; treat with caution.",
                stacklevel=2,
            )

        X = _build_design_matrix(df_pred, self.development_features)
        log_ptu = X @ self._coeffs

        # Clip log_ptu to avoid extreme predictions: PtU factor < 1 means
        # we predict the claim is already past ultimate — set a floor of 0.
        # We warn but do not discard these predictions.
        n_below_zero = int(np.sum(log_ptu < 0))
        if n_below_zero > 0:
            warnings.warn(
                f"{n_below_zero} prediction(s) have negative log-PtU "
                "(predicted ultimate < paid_to_date). "
                "This may indicate subrogation or an out-of-sample claim. "
                "log_ptu is clipped to 0 for these rows.",
                stacklevel=2,
            )
            log_ptu = np.maximum(log_ptu, 0.0)

        ptu_factor = np.exp(log_ptu)
        predicted_ultimate = paid_clipped * ptu_factor

        result = df.with_columns([
            pl.Series("predicted_ultimate", predicted_ultimate),
            pl.Series("ptu_factor", ptu_factor),
            pl.Series("log_ptu", log_ptu),
        ])

        if add_prediction_interval:
            from scipy.stats import norm

            z = float(norm.ppf(0.5 + pi_coverage / 2.0))
            half_width = z * self._rmse  # on log-ratio scale

            pi_lower = paid_clipped * np.exp(log_ptu - half_width)
            pi_upper = paid_clipped * np.exp(log_ptu + half_width)

            result = result.with_columns([
                pl.Series("pi_lower", pi_lower),
                pl.Series("pi_upper", pi_upper),
            ])

        return result

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """
        Return fit diagnostics and coefficient table.

        Returns
        -------
        dict with keys:

        ``n_train`` : int
            Number of training observations used.
        ``r2`` : float
            R² on the log-ratio target (training set).
        ``rmse`` : float
            Root-mean-square error on log-ratio (training set).
        ``method`` : str
            Estimation method ("ols" or "ridge").
        ``ridge_alpha`` : float or None
            Regularisation strength (None when method="ols").
        ``coefficients`` : dict[str, float]
            Feature name -> fitted coefficient.  Includes intercept.
        ``residuals_mean`` : float
            Mean of training residuals.  Should be near zero for OLS.
        ``residuals_std`` : float
            Std of training residuals.
        ``residuals_skewness`` : float
            Skewness of residuals (diagnostic for log-ratio assumption).
        ``development_features`` : list of str
            Feature columns used.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called.
        """
        self._check_fitted()

        from scipy.stats import skew

        residuals = self._residuals
        return {
            "n_train": self._n_train,
            "r2": round(self._r2, 6),
            "rmse": round(self._rmse, 6),
            "method": self.method,
            "ridge_alpha": self.ridge_alpha if self.method == "ridge" else None,
            "coefficients": {
                name: round(float(c), 8)
                for name, c in zip(self._feature_names, self._coeffs)
            },
            "residuals_mean": round(float(np.mean(residuals)), 8),
            "residuals_std": round(float(np.std(residuals)), 8),
            "residuals_skewness": round(float(skew(residuals)), 6),
            "development_features": list(self.development_features),
        }

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self._coeffs is None:
            raise RuntimeError("Call fit() first")

    @property
    def coefficients(self) -> dict[str, float]:
        """Feature name -> fitted coefficient (includes intercept)."""
        self._check_fitted()
        return {
            name: float(c)
            for name, c in zip(self._feature_names, self._coeffs)
        }

    @property
    def r2(self) -> float:
        """R² on log-ratio target (training set)."""
        self._check_fitted()
        return self._r2

    @property
    def rmse(self) -> float:
        """RMSE on log-ratio target (training set)."""
        self._check_fitted()
        return self._rmse

    def __repr__(self) -> str:
        status = "fitted" if self._coeffs is not None else "unfitted"
        return (
            f"ProjectionToUltimate("
            f"method={self.method!r}, "
            f"features={self.development_features}, "
            f"status={status})"
        )
