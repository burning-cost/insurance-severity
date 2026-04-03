"""
Additional tests for insurance_severity.tail_scoring — plot methods and edge cases.

Covers the diagnostic plotting API that is not exercised by test_tail_scoring.py:
- TailCalibration.combined_ratio_plot()
- TailCalibration.pit_histogram()
- BladtTailScore.stability_plot()

Also covers edge cases in TailCalibration.severity_pit() and summary_table()
with scalar cdf_func shapes.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

# Guard: skip if matplotlib not installed
matplotlib = pytest.importorskip("matplotlib")
import matplotlib.pyplot as plt

from insurance_severity.tail_scoring import TailCalibration, BladtTailScore, pareto_qq


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_gpd_tail_calibration(
    n: int = 200,
    xi: float = 0.3,
    sigma: float = 5000.0,
    threshold: float = 10_000.0,
    seed: int = 42,
) -> TailCalibration:
    """
    Build a TailCalibration where the model exactly matches the data DGP.
    cdf_func returns per-observation GPD CDF values (all the same since no covariates).
    """
    rng = np.random.default_rng(seed)
    y = stats.genpareto.rvs(c=xi, scale=sigma, loc=threshold, size=n, random_state=rng)

    def cdf_func(t: float) -> np.ndarray:
        return stats.genpareto.cdf(float(t) - threshold, c=xi, scale=sigma) * np.ones(n)

    tc = TailCalibration(cdf_func=cdf_func, n_obs=n)
    tc.fit(y)
    return tc, y, threshold


# ---------------------------------------------------------------------------
# TailCalibration.combined_ratio_plot()
# ---------------------------------------------------------------------------


class TestCombinedRatioPlot:

    def test_returns_figure(self):
        tc, y, t = _make_gpd_tail_calibration()
        t_levels = np.quantile(y, [0.7, 0.8, 0.9])
        fig = tc.combined_ratio_plot(t_levels)
        assert hasattr(fig, "savefig"), "combined_ratio_plot should return a Figure"
        plt.close("all")

    def test_accepts_existing_axes(self):
        tc, y, t = _make_gpd_tail_calibration()
        t_levels = np.quantile(y, [0.8, 0.9])
        fig_in, ax_in = plt.subplots()
        fig_out = tc.combined_ratio_plot(t_levels, ax=ax_in)
        assert fig_out is fig_in, "should return same figure when ax provided"
        plt.close("all")

    def test_threshold_above_max_warns(self):
        tc, y, t = _make_gpd_tail_calibration()
        t_too_high = np.array([np.max(y) + 1.0])
        with pytest.warns(UserWarning, match="Threshold"):
            tc.combined_ratio_plot(t_too_high)
        plt.close("all")

    def test_raises_before_fit(self):
        def cdf_func(t):
            return np.zeros(10)
        tc = TailCalibration(cdf_func=cdf_func, n_obs=10)
        with pytest.raises(RuntimeError, match="fit"):
            tc.combined_ratio_plot(np.array([1.0, 2.0]))

    def test_calibrated_model_near_diagonal(self):
        """
        For a perfectly calibrated model, the combined ratio should approximate
        the diagonal. Check that the mean absolute deviation from u is small.
        """
        tc, y, t = _make_gpd_tail_calibration(n=400, seed=99)
        t_levels = np.quantile(y, [0.80])

        # We test the underlying computation directly rather than the plot
        # to avoid matplotlib rendering issues on headless systems.
        u_grid = np.linspace(0.0, 1.0, 51)
        t_val = float(t_levels[0])
        Z = tc.severity_pit(t_val)
        n = tc._n_obs
        F_t = tc._get_cdf_at(t_val)
        pred_exc = float(np.mean(1.0 - F_t))
        R_hat = np.array([float(np.sum(Z <= u)) / n / pred_exc for u in u_grid])
        mad = float(np.mean(np.abs(R_hat - u_grid)))
        # For calibrated model with n=400, mean absolute deviation from diagonal
        # should be less than 0.15 (statistical noise)
        assert mad < 0.15, f"Mean deviation from diagonal: {mad:.3f}"


# ---------------------------------------------------------------------------
# TailCalibration.pit_histogram()
# ---------------------------------------------------------------------------


class TestPitHistogram:

    def test_returns_figure(self):
        tc, y, t = _make_gpd_tail_calibration()
        t_val = float(np.quantile(y, 0.8))
        fig = tc.pit_histogram(t_val)
        assert hasattr(fig, "savefig"), "pit_histogram should return a Figure"
        plt.close("all")

    def test_accepts_existing_axes(self):
        tc, y, t = _make_gpd_tail_calibration()
        t_val = float(np.quantile(y, 0.8))
        fig_in, ax_in = plt.subplots()
        fig_out = tc.pit_histogram(t_val, ax=ax_in)
        assert fig_out is fig_in
        plt.close("all")

    def test_warns_small_n_exceedances(self):
        """When fewer than 20 obs exceed threshold, should warn."""
        tc, y, t = _make_gpd_tail_calibration(n=50)
        # High quantile: only a few observations above
        t_val = float(np.quantile(y, 0.98))
        with pytest.warns(UserWarning, match="exceedances"):
            tc.pit_histogram(t_val)
        plt.close("all")

    def test_no_exceedances_returns_figure(self):
        """When no obs exceed threshold, should return figure with empty title."""
        tc, y, t = _make_gpd_tail_calibration(n=100)
        t_val = float(np.max(y) + 1.0)
        fig = tc.pit_histogram(t_val)
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_custom_bins(self):
        tc, y, t = _make_gpd_tail_calibration()
        t_val = float(np.quantile(y, 0.7))
        fig = tc.pit_histogram(t_val, n_bins=20)
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_raises_before_fit(self):
        def cdf_func(t):
            return np.zeros(10)
        tc = TailCalibration(cdf_func=cdf_func, n_obs=10)
        with pytest.raises(RuntimeError, match="fit"):
            tc.pit_histogram(1.0)


# ---------------------------------------------------------------------------
# BladtTailScore.stability_plot()
# ---------------------------------------------------------------------------


class TestStabilityPlot:

    @pytest.fixture(scope="class")
    def pareto_data(self):
        rng = np.random.default_rng(5)
        # Pareto(alpha=2) data: gamma_true=0.5
        u = rng.uniform(0, 1, 300)
        y = 1000.0 * (1.0 - u) ** (-0.5)
        return y

    def test_returns_figure(self, pareto_data):
        bs = BladtTailScore()
        y = pareto_data
        gammas = [0.3, 0.5, 0.7]
        k_grid = np.arange(20, 60, 5)
        fig = bs.stability_plot(y, gammas, k_grid)
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_accepts_existing_axes(self, pareto_data):
        bs = BladtTailScore()
        y = pareto_data
        gammas = [0.4, 0.6]
        k_grid = np.arange(25, 55, 10)
        fig_in, ax_in = plt.subplots()
        fig_out = bs.stability_plot(y, gammas, k_grid, ax=ax_in)
        assert fig_out is fig_in
        plt.close("all")

    def test_single_gamma(self, pareto_data):
        bs = BladtTailScore()
        y = pareto_data
        fig = bs.stability_plot(y, [0.5], np.arange(20, 50, 5))
        assert hasattr(fig, "savefig")
        plt.close("all")


# ---------------------------------------------------------------------------
# TailCalibration edge cases: occurrence_ratio with cdf_func returning scalar
# ---------------------------------------------------------------------------


class TestTailCalibrationScalarCdf:
    """
    Test TailCalibration when cdf_func returns a scalar or 1-element array.
    This can happen when the model has no per-observation variation.
    """

    def test_occurrence_ratio_with_constant_cdf(self):
        """
        If all observations share the same CDF (homogeneous model),
        occurrence_ratio should still compute without error.
        """
        rng = np.random.default_rng(10)
        n = 100
        y = rng.gamma(2.0, 1000.0, n)
        mu = float(np.mean(y))

        def cdf_func(t: float) -> np.ndarray:
            return stats.gamma.cdf(float(t), a=2.0, scale=mu / 2.0) * np.ones(n)

        tc = TailCalibration(cdf_func=cdf_func, n_obs=n)
        tc.fit(y)

        t = float(np.quantile(y, 0.8))
        ratio = tc.occurrence_ratio(t)
        assert np.isfinite(ratio), f"occurrence_ratio should be finite, got {ratio}"
        assert ratio > 0, "occurrence_ratio should be positive"

    def test_severity_pit_returns_bounded_values(self):
        """PIT values from severity_pit should be in [0, 1]."""
        rng = np.random.default_rng(11)
        n = 150
        y = rng.gamma(3.0, 2000.0, n)
        mu = float(np.mean(y))

        def cdf_func(t: float) -> np.ndarray:
            return stats.gamma.cdf(float(t), a=3.0, scale=mu / 3.0) * np.ones(n)

        tc = TailCalibration(cdf_func=cdf_func, n_obs=n)
        tc.fit(y)

        t = float(np.quantile(y, 0.75))
        Z = tc.severity_pit(t)

        assert len(Z) > 0, "Should have some exceedances at 75th percentile"
        assert np.all(Z >= -1e-8), f"PIT values below 0: min={Z.min():.4f}"
        assert np.all(Z <= 1.0 + 1e-8), f"PIT values above 1: max={Z.max():.4f}"

    def test_summary_table_with_provided_thresholds(self):
        """summary_table should work with an explicit array of thresholds."""
        import pandas as pd
        rng = np.random.default_rng(12)
        n = 200
        y = rng.gamma(2.0, 5000.0, n)

        def cdf_func(t: float) -> np.ndarray:
            return stats.gamma.cdf(float(t), a=2.0, scale=2500.0) * np.ones(n)

        tc = TailCalibration(cdf_func=cdf_func, n_obs=n)
        tc.fit(y)

        t_levels = np.quantile(y, [0.70, 0.80, 0.90, 0.95])
        df = tc.summary_table(t_levels)

        assert isinstance(df, pd.DataFrame)
        expected_cols = {"threshold", "n_exceedances", "R_occ", "ks_pvalue"}
        assert expected_cols.issubset(set(df.columns)), (
            f"Missing columns: {expected_cols - set(df.columns)}"
        )
        assert len(df) == 4


# ---------------------------------------------------------------------------
# pareto_qq additional coverage
# ---------------------------------------------------------------------------


class TestParetoQQAdditional:

    def test_custom_k_parameter(self):
        """pareto_qq with explicit k should use that many order statistics."""
        rng = np.random.default_rng(20)
        y = stats.pareto.rvs(b=2.0, scale=10000.0, size=500, random_state=rng)
        # pareto_qq returns a float (R-squared)
        r2 = pareto_qq(y, k=100)
        assert isinstance(r2, float), f"Expected float, got {type(r2)}"
        assert 0.0 <= r2 <= 1.0, f"R2 out of range: {r2}"

    def test_qq_returns_float(self):
        """pareto_qq returns an R-squared float, not a dict."""
        rng = np.random.default_rng(21)
        y = stats.pareto.rvs(b=3.0, scale=5000.0, size=300, random_state=rng)
        r2 = pareto_qq(y)
        assert isinstance(r2, float), f"Expected float, got {type(r2)}"
        assert 0.0 <= r2 <= 1.0

    def test_pareto_qq_r2_not_nan(self):
        """r2 should be finite for valid Pareto data."""
        rng = np.random.default_rng(22)
        y = stats.pareto.rvs(b=1.5, scale=1000.0, size=200, random_state=rng)
        r2 = pareto_qq(y)
        assert np.isfinite(r2), f"r2 is nan/inf: {r2}"
    
    def test_pareto_qq_high_r2_for_pareto_data(self):
        """Pareto data should give high R2 on the QQ plot."""
        rng = np.random.default_rng(23)
        y = stats.pareto.rvs(b=2.0, scale=1000.0, size=500, random_state=rng)
        r2 = pareto_qq(y)
        # Pareto data should give R2 close to 1 on the Pareto QQ plot
        assert r2 > 0.90, f"Expected R2 > 0.90 for Pareto data, got {r2:.4f}"
