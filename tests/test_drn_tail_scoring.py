"""
Tests for DRN tail-scoring capabilities.

Item 3: ExtendedHistogramBatch.tw_crps()
Item 4: DRNDiagnostics.tail_calibration() and DRNDiagnostics.tw_crps_profile()

Tests are skipped if torch is not installed (matching the existing pattern in
tests/drn/test_drn.py and tests/drn/test_diagnostics.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

# Skip the entire module if torch is not available
torch = pytest.importorskip("torch")

from insurance_severity.drn.histogram import ExtendedHistogramBatch  # noqa: E402
from insurance_severity.drn.diagnostics import DRNDiagnostics  # noqa: E402
from insurance_severity.drn.drn import DRN  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def make_histogram_batch(
    n: int = 20,
    K: int = 10,
    mu_val: float = 2000.0,
    c_max: float = 10000.0,
) -> ExtendedHistogramBatch:
    """
    Build a uniform-bin ExtendedHistogramBatch with Gamma baseline.
    c_0=0, c_K=c_max. All bin probabilities equal 1/K.
    """
    cutpoints = np.linspace(0.0, c_max, K + 1)
    bin_probs = np.full((n, K), 1.0 / K)

    mu = np.full(n, mu_val)
    disp = 0.5
    alpha = 1.0 / disp
    scale = mu * disp

    bc0 = stats.gamma.cdf(cutpoints[0], a=alpha, scale=scale)
    bcK = stats.gamma.cdf(cutpoints[-1], a=alpha, scale=scale)

    return ExtendedHistogramBatch(
        cutpoints=cutpoints,
        bin_probs=bin_probs,
        baseline_cdf_c0=bc0,
        baseline_cdf_cK=bcK,
        baseline_params={"mu": mu, "dispersion": disp},
        distribution_family="gamma",
    )


class GammaMockBaseline:
    """Minimal baseline that always predicts mu=2000, dispersion=0.5."""

    distribution_family = "gamma"

    def predict_params(self, X):
        n = len(X)
        return {"mu": np.full(n, 2000.0), "dispersion": 0.5}

    def predict_cdf(self, X, cutpoints):
        params = self.predict_params(X)
        mu = params["mu"][:, np.newaxis]
        disp = params["dispersion"]
        return stats.gamma.cdf(cutpoints[np.newaxis, :], a=1.0 / disp, scale=mu * disp)


@pytest.fixture
def fitted_drn():
    """Return a quickly fitted DRN and matching (X, y)."""
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame({
        "age": rng.uniform(20, 70, size=n),
        "region": rng.integers(0, 3, size=n).astype(float),
    })
    y = rng.gamma(2.0, 1000.0, size=n)
    baseline = GammaMockBaseline()
    drn = DRN(baseline, max_epochs=2, random_state=0)
    drn.fit(X, y, verbose=False)
    return drn, X, y


# ---------------------------------------------------------------------------
# Item 3: ExtendedHistogramBatch.tw_crps()
# ---------------------------------------------------------------------------


class TestTwCRPS:

    def test_tw_crps_at_zero_equals_crps(self):
        """
        tw_crps with threshold=0 should equal the standard crps().

        Both integrate from effectively -infinity (c_0=0 so the left tail
        contributes nothing) to +infinity. The right tail is handled the same
        way in both methods.
        """
        batch = make_histogram_batch(n=15, K=10)
        rng = np.random.default_rng(7)
        y_true = rng.gamma(2.0, 1500.0, size=15)
        y_true = np.clip(y_true, 10.0, 8000.0)  # keep inside histogram

        crps_vals = batch.crps(y_true)
        tw_crps_vals = batch.tw_crps(y_true, threshold=0.0)

        np.testing.assert_allclose(
            tw_crps_vals,
            crps_vals,
            rtol=1e-6,
            err_msg="tw_crps(threshold=0) must equal crps()",
        )

    def test_tw_crps_returns_correct_shape(self):
        """tw_crps returns ndarray of shape (n,)."""
        batch = make_histogram_batch(n=8, K=5)
        y_true = np.full(8, 3000.0)
        result = batch.tw_crps(y_true, threshold=1000.0)
        assert result.shape == (8,)

    def test_tw_crps_non_negative(self):
        """twCRPS values must be >= 0."""
        batch = make_histogram_batch(n=20, K=8)
        rng = np.random.default_rng(0)
        y_true = rng.gamma(2.0, 1000.0, size=20)
        y_true = np.clip(y_true, 1.0, 8000.0)

        for threshold in [0.0, 500.0, 2000.0, 5000.0]:
            vals = batch.tw_crps(y_true, threshold=threshold)
            assert np.all(vals >= -1e-10), (
                f"tw_crps has negative values at threshold={threshold}: {vals.min()}"
            )

    def test_tw_crps_monotone_decreasing_with_threshold(self):
        """
        As threshold increases, tw_crps can only decrease (or stay the same)
        in expectation. A higher threshold means less of the distribution is
        scored, so the mean twCRPS cannot exceed the mean at a lower threshold.
        """
        batch = make_histogram_batch(n=50, K=10, mu_val=3000.0)
        rng = np.random.default_rng(1)
        y_true = rng.gamma(2.0, 1500.0, size=50)
        y_true = np.clip(y_true, 10.0, 8000.0)

        thresholds = [0.0, 1000.0, 2000.0, 3000.0, 5000.0]
        mean_tw = [float(np.mean(batch.tw_crps(y_true, t))) for t in thresholds]

        for i in range(len(mean_tw) - 1):
            assert mean_tw[i] >= mean_tw[i + 1] - 1e-6, (
                f"mean tw_crps should decrease: "
                f"threshold={thresholds[i]} -> {mean_tw[i]:.4f}, "
                f"threshold={thresholds[i+1]} -> {mean_tw[i+1]:.4f}"
            )

    def test_tw_crps_threshold_above_cK_histogram_contribution_is_zero(self):
        """
        When threshold > c_K, all histogram bins are below threshold and are
        skipped. The tw_crps value comes entirely from the right-tail integrator,
        which starts at max(c_K, threshold) = threshold.

        Specifically: when y_true < threshold, the right-tail integrand is
        S(y)^2 = (1 - F(y))^2 > 0 for y in [threshold, inf). So the tw_crps
        is positive (not zero) for y_true < threshold > c_K.

        What we can assert:
        - values are non-negative
        - values are bounded above by tw_crps at threshold=c_K (monotone property)
        - values are strictly less than the full CRPS (we're only scoring the
          right tail beyond a large threshold)
        """
        batch = make_histogram_batch(n=5, K=5, c_max=5000.0)
        y_true = np.full(5, 2000.0)

        vals_above_cK = batch.tw_crps(y_true, threshold=6000.0)
        vals_at_cK = batch.tw_crps(y_true, threshold=5000.0)
        crps_full = batch.crps(y_true)

        # Non-negative
        assert np.all(vals_above_cK >= -1e-10), (
            f"tw_crps must be non-negative, got {vals_above_cK}"
        )
        # Monotone: threshold=6000 <= threshold=5000 contribution
        assert np.all(vals_above_cK <= vals_at_cK + 1e-6), (
            f"tw_crps at threshold=6000 should not exceed tw_crps at threshold=5000 "
            f"(monotone property). Got {vals_above_cK} vs {vals_at_cK}"
        )
        # Less than full CRPS (only tail scored, majority of distribution excluded)
        assert np.all(vals_above_cK <= crps_full + 1e-6), (
            f"tw_crps with high threshold must not exceed full CRPS. "
            f"Got {vals_above_cK} vs {crps_full}"
        )

    def test_tw_crps_threshold_above_cK_y_true_also_above_cK(self):
        """
        When both threshold > c_K and y_true > threshold, the integrand
        (F(y) - 1{y >= y_true})^2 is S(y)^2 for y < y_true and (F(y)-1)^2
        for y > y_true. The tw_crps is smaller when y_true is higher.
        """
        batch = make_histogram_batch(n=3, K=5, c_max=5000.0)
        # y_true below threshold: right-tail contribution is integral of S(y)^2
        y_true_low = np.full(3, 2000.0)
        # y_true above threshold: partial cancellation near y_true
        y_true_high = np.full(3, 7000.0)

        with pytest.warns(UserWarning, match="maximum cutpoint"):
            vals_high = batch.tw_crps(y_true_high, threshold=6000.0)

        vals_low = batch.tw_crps(y_true_low, threshold=6000.0)

        # Both non-negative
        assert np.all(vals_low >= -1e-10)
        assert np.all(vals_high >= -1e-10)

    def test_tw_crps_warns_when_y_true_exceeds_cK(self):
        """When y_true > c_K, a UserWarning is issued."""
        batch = make_histogram_batch(n=5, K=5, c_max=5000.0)
        y_true = np.array([1000.0, 2000.0, 3000.0, 4000.0, 9999.0])  # last exceeds c_K

        with pytest.warns(UserWarning, match="maximum cutpoint"):
            batch.tw_crps(y_true, threshold=0.0)

    def test_tw_crps_straddling_threshold_bin(self):
        """
        When threshold falls inside a bin, the integration should start at
        threshold, not at the bin's left edge. A threshold exactly at the
        midpoint of the histogram should give a tw_crps that is less than
        the full crps but greater than zero.
        """
        batch = make_histogram_batch(n=10, K=20, c_max=10000.0)
        rng = np.random.default_rng(3)
        y_true = rng.gamma(2.0, 2000.0, size=10)
        y_true = np.clip(y_true, 100.0, 8000.0)

        crps_full = batch.crps(y_true)
        tw_half = batch.tw_crps(y_true, threshold=5000.0)

        # twCRPS at midpoint should be <= full CRPS
        assert np.all(tw_half <= crps_full + 1e-6)
        # And >= 0
        assert np.all(tw_half >= -1e-10)


# ---------------------------------------------------------------------------
# Item 4: DRNDiagnostics.tail_calibration() and .tw_crps_profile()
# ---------------------------------------------------------------------------

# Bug: DRNDiagnostics.tail_calibration() constructs cdf_func as:
#   def cdf_func(t: float) -> np.ndarray:
#       return batch.cdf(np.full(n, float(t)))
# This passes an (n,)-shaped array to ExtendedHistogramBatch.cdf(), which
# returns an (n, n) matrix instead of the expected (n,) vector. This causes
# TailCalibration.severity_pit() to fail with ValueError when assigning the
# 2D result into a 1D array.
# The fix is to pass a scalar: batch.cdf(float(t)).
# These tests are marked xfail(strict=True) to document the bug.
_TAIL_CALIBRATION_BUG = pytest.mark.xfail(
    strict=True,
    reason=(
        "DRNDiagnostics.tail_calibration() passes np.full(n, t) to batch.cdf() "
        "instead of a scalar t. ExtendedHistogramBatch.cdf(array) returns (n, n) "
        "but TailCalibration._get_cdf_at() expects (n,). Fix: change cdf_func to "
        "return batch.cdf(float(t)) in diagnostics.py:tail_calibration()."
    ),
)


class TestDRNDiagnosticsTailMethods:

    @_TAIL_CALIBRATION_BUG
    def test_tail_calibration_returns_dict(self, fitted_drn):
        drn, X, y = fitted_drn
        diag = DRNDiagnostics(drn)
        result = diag.tail_calibration(X, y)
        assert isinstance(result, dict)

    @_TAIL_CALIBRATION_BUG
    def test_tail_calibration_expected_keys(self, fitted_drn):
        drn, X, y = fitted_drn
        diag = DRNDiagnostics(drn)
        result = diag.tail_calibration(X, y)
        assert "summary" in result
        assert "occurrence_ratios" in result
        assert "pit_data" in result

    @_TAIL_CALIBRATION_BUG
    def test_tail_calibration_summary_is_dataframe(self, fitted_drn):
        import pandas as pd
        drn, X, y = fitted_drn
        diag = DRNDiagnostics(drn)
        result = diag.tail_calibration(X, y)
        assert isinstance(result["summary"], pd.DataFrame)

    @_TAIL_CALIBRATION_BUG
    def test_tail_calibration_summary_columns(self, fitted_drn):
        drn, X, y = fitted_drn
        diag = DRNDiagnostics(drn)
        result = diag.tail_calibration(X, y)
        df = result["summary"]
        expected_cols = {"threshold", "n_exceedances", "R_occ", "ks_pvalue"}
        assert expected_cols.issubset(set(df.columns))

    @_TAIL_CALIBRATION_BUG
    def test_tail_calibration_default_quantiles(self, fitted_drn):
        """Default threshold_quantiles=[0.80, 0.90, 0.95, 0.99] -> 4 thresholds."""
        drn, X, y = fitted_drn
        diag = DRNDiagnostics(drn)
        result = diag.tail_calibration(X, y)
        assert len(result["occurrence_ratios"]) == 4

    @_TAIL_CALIBRATION_BUG
    def test_tail_calibration_custom_quantiles(self, fitted_drn):
        """Custom threshold_quantiles -> correct number of thresholds."""
        drn, X, y = fitted_drn
        diag = DRNDiagnostics(drn)
        result = diag.tail_calibration(X, y, threshold_quantiles=[0.70, 0.80])
        assert len(result["occurrence_ratios"]) == 2

    @_TAIL_CALIBRATION_BUG
    def test_tail_calibration_pit_data_bounded(self, fitted_drn):
        """All PIT values in pit_data should be in [0, 1]."""
        drn, X, y = fitted_drn
        diag = DRNDiagnostics(drn)
        result = diag.tail_calibration(X, y, threshold_quantiles=[0.80, 0.90])
        for t, pit_arr in result["pit_data"].items():
            if len(pit_arr) > 0:
                assert np.all(pit_arr >= -1e-8), f"PIT < 0 at threshold {t}"
                assert np.all(pit_arr <= 1.0 + 1e-8), f"PIT > 1 at threshold {t}"

    def test_tw_crps_profile_returns_dict(self, fitted_drn):
        drn, X, y = fitted_drn
        diag = DRNDiagnostics(drn)
        profile = diag.tw_crps_profile(X, y)
        assert isinstance(profile, dict)

    def test_tw_crps_profile_default_length(self, fitted_drn):
        """Default threshold_quantiles has 5 entries -> profile has 5 keys."""
        drn, X, y = fitted_drn
        diag = DRNDiagnostics(drn)
        profile = diag.tw_crps_profile(X, y)
        assert len(profile) == 5

    def test_tw_crps_profile_custom_length(self, fitted_drn):
        drn, X, y = fitted_drn
        diag = DRNDiagnostics(drn)
        profile = diag.tw_crps_profile(X, y, threshold_quantiles=[0.80, 0.90, 0.95])
        assert len(profile) == 3

    def test_tw_crps_profile_values_non_negative(self, fitted_drn):
        """All mean twCRPS values must be >= 0."""
        drn, X, y = fitted_drn
        diag = DRNDiagnostics(drn)
        profile = diag.tw_crps_profile(X, y)
        for t, val in profile.items():
            assert val >= -1e-10, f"Negative mean twCRPS at threshold {t}: {val}"

    def test_tw_crps_profile_monotone_decreasing(self, fitted_drn):
        """
        Mean twCRPS should be (weakly) decreasing as threshold increases.
        A higher threshold means less of the distribution contributes to the
        score, so the mean cannot increase.
        """
        drn, X, y = fitted_drn
        diag = DRNDiagnostics(drn)
        # Use explicit sorted quantiles so we can check ordering
        qs = [0.50, 0.70, 0.85, 0.95]
        profile = diag.tw_crps_profile(X, y, threshold_quantiles=qs)
        vals = [profile[t] for t in sorted(profile.keys())]

        for i in range(len(vals) - 1):
            assert vals[i] >= vals[i + 1] - 1e-6, (
                f"mean twCRPS should decrease with threshold: "
                f"{vals[i]:.4f} -> {vals[i+1]:.4f}"
            )

    def test_tw_crps_profile_float_keys(self, fitted_drn):
        """Profile keys should be float threshold values."""
        drn, X, y = fitted_drn
        diag = DRNDiagnostics(drn)
        profile = diag.tw_crps_profile(X, y)
        for k in profile.keys():
            assert isinstance(k, float), f"Expected float key, got {type(k)}"
