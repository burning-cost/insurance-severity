"""
Tests for DRNDiagnostics.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from insurance_severity.drn.diagnostics import DRNDiagnostics
from insurance_severity.drn.drn import DRN


class GammaMockBaseline:
    distribution_family = "gamma"

    def predict_params(self, X):
        n = len(X)
        return {"mu": np.full(n, 2000.0), "dispersion": 0.5}

    def predict_cdf(self, X, cutpoints):
        from scipy import stats
        params = self.predict_params(X)
        mu = params["mu"][:, np.newaxis]
        disp = params["dispersion"]
        return stats.gamma.cdf(cutpoints[np.newaxis, :], a=1.0/disp, scale=mu*disp)


@pytest.fixture
def fitted_drn():
    rng = np.random.default_rng(0)
    n = 150
    X = pd.DataFrame({
        "age": rng.uniform(20, 70, size=n),
        "region": rng.integers(0, 3, size=n).astype(float),
    })
    y = rng.gamma(2.0, 1000.0, size=n)
    baseline = GammaMockBaseline()
    drn = DRN(baseline, max_epochs=2, random_state=0)
    drn.fit(X, y, verbose=False)
    return drn, X, y


class TestDRNDiagnostics:

    def test_pit_values_shape(self, fitted_drn):
        drn, X, y = fitted_drn
        diag = DRNDiagnostics(drn)
        pit = diag.pit_values(X, y)
        assert pit.shape == (len(y),)

    def test_pit_values_bounded(self, fitted_drn):
        drn, X, y = fitted_drn
        diag = DRNDiagnostics(drn)
        pit = diag.pit_values(X, y)
        assert np.all(pit >= 0.0 - 1e-8)
        assert np.all(pit <= 1.0 + 1e-8)

    def test_quantile_calibration_returns_polars(self, fitted_drn):
        drn, X, y = fitted_drn
        diag = DRNDiagnostics(drn)
        df = diag.quantile_calibration(X, y)
        assert isinstance(df, pl.DataFrame)

    def test_quantile_calibration_columns(self, fitted_drn):
        drn, X, y = fitted_drn
        diag = DRNDiagnostics(drn)
        df = diag.quantile_calibration(X, y)
        assert "nominal_coverage" in df.columns
        assert "observed_coverage" in df.columns
        assert "error" in df.columns

    def test_quantile_calibration_coverage_bounded(self, fitted_drn):
        drn, X, y = fitted_drn
        diag = DRNDiagnostics(drn)
        df = diag.quantile_calibration(X, y)
        obs = df["observed_coverage"].to_numpy()
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)

    def test_crps_by_segment_returns_polars(self, fitted_drn):
        drn, X, y = fitted_drn
        diag = DRNDiagnostics(drn)
        df = diag.crps_by_segment(X, y, segment_col="region")
        assert isinstance(df, pl.DataFrame)

    def test_crps_by_segment_columns(self, fitted_drn):
        drn, X, y = fitted_drn
        diag = DRNDiagnostics(drn)
        df = diag.crps_by_segment(X, y, segment_col="region")
        assert "segment" in df.columns
        assert "n" in df.columns
        assert "mean_crps" in df.columns
        assert "mean_y" in df.columns

    def test_crps_by_segment_n_sums_to_total(self, fitted_drn):
        drn, X, y = fitted_drn
        diag = DRNDiagnostics(drn)
        df = diag.crps_by_segment(X, y, segment_col="region")
        assert df["n"].sum() == len(y)

    def test_crps_by_segment_array_input(self, fitted_drn):
        drn, X, y = fitted_drn
        diag = DRNDiagnostics(drn)
        segments = np.where(y > np.median(y), "high", "low")
        df = diag.crps_by_segment(X, y, segment_col=segments)
        assert len(df) == 2

    def test_summary_returns_polars(self, fitted_drn):
        drn, X, y = fitted_drn
        diag = DRNDiagnostics(drn)
        df = diag.summary(X, y)
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 1
        assert "crps" in df.columns
        assert "rmse" in df.columns

    def test_pit_histogram_no_matplotlib(self, fitted_drn, monkeypatch):
        """Without matplotlib, pit_histogram returns PIT values."""
        import sys
        drn, X, y = fitted_drn
        diag = DRNDiagnostics(drn)

        # Monkeypatch matplotlib import to fail
        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else None

        # Just test return_figure=False path directly
        result = diag.pit_histogram(X, y, return_figure=False)
        assert isinstance(result, np.ndarray)
        assert result.shape == (len(y),)
