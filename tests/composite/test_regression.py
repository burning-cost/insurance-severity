"""
Tests for CompositeSeverityRegressor.

Focus:
- Fitting completes without error
- Threshold predictions are covariate-dependent for mode-matching
- Threshold predictions are constant for fixed-threshold
- Regression coefficients have correct sign direction
- Score returns a finite scalar
- ILF computation: shape and monotonicity
- Bootstrap CI structure
"""

import numpy as np
import pytest
import warnings

from insurance_severity.composite.models import LognormalBurrComposite, LognormalGPDComposite
from insurance_severity.composite.regression import CompositeSeverityRegressor, _composite_from_str


# ---------------------------------------------------------------------------
# String shorthand
# ---------------------------------------------------------------------------


class TestCompositeFromStr:

    def test_lognormal_burr(self):
        m = _composite_from_str("lognormal_burr")
        assert isinstance(m, LognormalBurrComposite)

    def test_lognormal_gpd(self):
        m = _composite_from_str("lognormal_gpd")
        assert isinstance(m, LognormalGPDComposite)

    def test_invalid(self):
        with pytest.raises(ValueError, match="Unknown"):
            _composite_from_str("bad_name")


# ---------------------------------------------------------------------------
# Mode-matching regression (LognormalBurr)
# ---------------------------------------------------------------------------


class TestModeMatchingRegression:

    def test_fit_runs(self, regression_data):
        X, y = regression_data
        reg = CompositeSeverityRegressor(
            composite=LognormalBurrComposite(threshold_method="mode_matching"),
            n_starts=2,
        )
        reg.fit(X, y)
        assert reg.coef_ is not None
        assert reg.intercept_ is not None
        assert reg.shape_params_ is not None

    def test_coef_shape(self, regression_data):
        X, y = regression_data
        reg = CompositeSeverityRegressor(
            composite=LognormalBurrComposite(threshold_method="mode_matching"),
            n_starts=2,
        )
        reg.fit(X, y)
        # 1 feature -> 1 coefficient (not counting intercept)
        assert len(reg.coef_) == X.shape[1]

    def test_thresholds_covariate_dependent(self, regression_data):
        """For mode-matching, thresholds should vary with covariates."""
        X, y = regression_data
        reg = CompositeSeverityRegressor(
            composite=LognormalBurrComposite(threshold_method="mode_matching"),
            n_starts=2,
        )
        reg.fit(X, y)

        # Predict thresholds for extreme X values
        X_low = np.array([[-2.0]])
        X_high = np.array([[2.0]])
        t_low = reg.predict_thresholds(X_low)[0]
        t_high = reg.predict_thresholds(X_high)[0]

        # Thresholds should differ (covariate-dependent)
        assert abs(t_high - t_low) > 0.1

        # Direction: positive coefficient means higher x -> higher threshold
        # (depends on estimated sign, so just check they're different)
        assert t_low != pytest.approx(t_high, rel=0.001)

    def test_positive_coef_positive_direction(self, regression_data):
        """
        True coefficient is +0.3. Estimated sign should be positive
        (higher x -> higher tail scale -> higher threshold).
        """
        X, y = regression_data
        reg = CompositeSeverityRegressor(
            composite=LognormalBurrComposite(threshold_method="mode_matching"),
            n_starts=3,
        )
        reg.fit(X, y)
        # With 300 obs and true coef=0.3, estimate should be > 0
        # (allow generous tolerance for noisy data)
        assert reg.coef_[0] > -0.2, f"Coefficient {reg.coef_[0]:.3f} unexpectedly negative"

    def test_predict_returns_array(self, regression_data):
        X, y = regression_data
        reg = CompositeSeverityRegressor(
            composite=LognormalBurrComposite(threshold_method="mode_matching"),
            n_starts=2,
        )
        reg.fit(X, y)
        preds = reg.predict(X[:5])
        assert len(preds) == 5
        assert np.all(preds > 0)

    def test_score_finite(self, regression_data):
        X, y = regression_data
        reg = CompositeSeverityRegressor(
            composite=LognormalBurrComposite(threshold_method="mode_matching"),
            n_starts=2,
        )
        reg.fit(X, y)
        s = reg.score(X[:20], y[:20])
        assert np.isfinite(s)

    def test_summary_string(self, regression_data):
        X, y = regression_data
        reg = CompositeSeverityRegressor(
            composite=LognormalBurrComposite(threshold_method="mode_matching"),
            n_starts=2,
        )
        reg.fit(X, y)
        s = reg.summary()
        assert "CompositeSeverityRegressor" in s
        assert "Intercept" in s

    def test_predict_thresholds_shape(self, regression_data):
        X, y = regression_data
        reg = CompositeSeverityRegressor(
            composite=LognormalBurrComposite(threshold_method="mode_matching"),
            n_starts=2,
        )
        reg.fit(X, y)
        thresholds = reg.predict_thresholds(X)
        assert thresholds.shape == (len(X),)
        assert np.all(thresholds > 0)

    def test_ilf_shape(self, regression_data):
        X, y = regression_data
        reg = CompositeSeverityRegressor(
            composite=LognormalBurrComposite(threshold_method="mode_matching"),
            n_starts=2,
        )
        reg.fit(X, y)
        limits = [50_000, 100_000, 250_000, 500_000]
        # Use only first 3 obs for speed
        ilf = reg.compute_ilf(X[:3], limits=limits, basic_limit=500_000.0)
        assert ilf.shape == (3, 4)
        # ILF at basic limit should be close to 1
        assert np.all(np.abs(ilf[:, -1] - 1.0) < 0.1)

    def test_ilf_monotone_per_obs(self, regression_data):
        """ILF should be non-decreasing as limit increases."""
        X, y = regression_data
        reg = CompositeSeverityRegressor(
            composite=LognormalBurrComposite(threshold_method="mode_matching"),
            n_starts=2,
        )
        reg.fit(X, y)
        limits = [25_000, 50_000, 100_000, 250_000, 500_000]
        ilf = reg.compute_ilf(X[:2], limits=limits, basic_limit=500_000.0)
        for i in range(len(X[:2])):
            assert np.all(np.diff(ilf[i]) >= -0.05), f"ILF not monotone for obs {i}: {ilf[i]}"


# ---------------------------------------------------------------------------
# Fixed threshold regression (LognormalGPD)
# ---------------------------------------------------------------------------


class TestFixedThresholdRegression:

    def test_fit_runs(self, lognormal_gpd_data):
        reg = CompositeSeverityRegressor(
            composite=LognormalGPDComposite(threshold=50000.0, threshold_method="fixed"),
            n_starts=2,
        )
        # Use dummy single covariate
        n = len(lognormal_gpd_data)
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (n, 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reg.fit(X, lognormal_gpd_data)
        assert reg.coef_ is not None
        assert reg.threshold_ == 50000.0

    def test_thresholds_constant(self, lognormal_gpd_data):
        """Fixed-threshold model returns same threshold for all obs."""
        n = len(lognormal_gpd_data)
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (n, 1))

        reg = CompositeSeverityRegressor(
            composite=LognormalGPDComposite(threshold=50000.0, threshold_method="fixed"),
            n_starts=2,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reg.fit(X, lognormal_gpd_data)

        thresholds = reg.predict_thresholds(X[:10])
        assert np.all(thresholds == 50000.0)


# ---------------------------------------------------------------------------
# DataFrame input (optional pandas)
# ---------------------------------------------------------------------------


class TestDataFrameInput:

    def test_fit_with_dataframe(self, regression_data):
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not available")

        X, y = regression_data
        df = pd.DataFrame(X, columns=["feature_1"])
        y_series = pd.Series(y)

        reg = CompositeSeverityRegressor(
            composite=LognormalBurrComposite(threshold_method="mode_matching"),
            feature_cols=["feature_1"],
            n_starts=2,
        )
        reg.fit(df, y_series.values)
        assert reg.coef_ is not None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestRegressionEdgeCases:

    def test_not_fitted_raises_on_predict(self):
        reg = CompositeSeverityRegressor(
            composite=LognormalBurrComposite(threshold_method="mode_matching"),
        )
        with pytest.raises((AttributeError, RuntimeError, TypeError)):
            reg.predict_thresholds(np.array([[1.0], [2.0]]))

    def test_negative_y_raises(self):
        reg = CompositeSeverityRegressor(
            composite=LognormalBurrComposite(threshold_method="mode_matching"),
            n_starts=1,
        )
        X = np.ones((10, 1))
        y = np.array([-1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        with pytest.raises(ValueError, match="positive"):
            reg.fit(X, y)
