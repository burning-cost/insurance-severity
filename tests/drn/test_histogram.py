"""
Tests for ExtendedHistogramBatch — the vectorised predictive distribution.

These tests verify the CDF, quantile, mean, variance, and CRPS calculations
without requiring a fitted DRN. We construct ExtendedHistogramBatch objects
directly with known parameters.
"""

import numpy as np
import pytest
from scipy import stats

from insurance_severity.drn.histogram import ExtendedHistogramBatch


def make_uniform_batch(n: int = 10, K: int = 5, family: str = "gamma") -> ExtendedHistogramBatch:
    """
    Create a batch with uniform bin probabilities (1/K each).
    Cutpoints: 0, 200, 400, 600, 800, 1000 (for K=5).
    Gamma baseline with mu=500, dispersion=0.5.
    """
    cutpoints = np.linspace(0.0, 1000.0, K + 1)
    bin_probs = np.full((n, K), 1.0 / K)

    mu = np.full(n, 500.0)
    alpha = 2.0  # shape = 1/dispersion
    disp = 0.5
    scale = mu * disp

    baseline_cdf_c0 = stats.gamma.cdf(cutpoints[0], a=alpha, scale=scale)
    baseline_cdf_cK = stats.gamma.cdf(cutpoints[-1], a=alpha, scale=scale)

    return ExtendedHistogramBatch(
        cutpoints=cutpoints,
        bin_probs=bin_probs,
        baseline_cdf_c0=baseline_cdf_c0,
        baseline_cdf_cK=baseline_cdf_cK,
        baseline_params={"mu": mu, "dispersion": disp},
        distribution_family=family,
    )


class TestExtendedHistogramBatch:

    def test_repr(self):
        batch = make_uniform_batch(n=5, K=5)
        r = repr(batch)
        assert "ExtendedHistogramBatch" in r
        assert "n=5" in r

    def test_len(self):
        batch = make_uniform_batch(n=7, K=5)
        assert len(batch) == 7

    def test_cdf_at_c0_equals_baseline(self):
        """CDF at c_0 should equal baseline CDF at c_0."""
        batch = make_uniform_batch(n=4, K=5)
        c0 = batch.c_0
        cdf_c0 = batch.cdf(c0)  # shape (n,)
        # At c_0 exactly, we're at the boundary — CDF should be baseline_cdf_c0
        np.testing.assert_allclose(cdf_c0, batch.baseline_cdf_c0, atol=1e-3)

    def test_cdf_at_cK_equals_baseline(self):
        """CDF at c_K should equal baseline CDF at c_K."""
        batch = make_uniform_batch(n=4, K=5)
        cK = batch.c_K
        cdf_cK = batch.cdf(cK)
        np.testing.assert_allclose(cdf_cK, batch.baseline_cdf_cK, atol=1e-3)

    def test_cdf_monotone_within_histogram(self):
        """CDF must be non-decreasing within the histogram region."""
        batch = make_uniform_batch(n=3, K=5)
        y_grid = np.linspace(50.0, 950.0, 50)
        cdf_grid = batch.cdf(y_grid)  # (n, 50)
        diffs = np.diff(cdf_grid, axis=1)
        assert np.all(diffs >= -1e-10), "CDF must be non-decreasing"

    def test_cdf_scalar_input(self):
        """CDF with scalar y returns shape (n,)."""
        batch = make_uniform_batch(n=6, K=5)
        result = batch.cdf(500.0)
        assert result.shape == (6,)

    def test_cdf_array_input(self):
        """CDF with array y returns shape (n, m)."""
        batch = make_uniform_batch(n=6, K=5)
        y = np.array([200.0, 500.0, 800.0])
        result = batch.cdf(y)
        assert result.shape == (6, 3)

    def test_cdf_increasing_with_y(self):
        """For a single observation, CDF(y1) <= CDF(y2) for y1 < y2."""
        batch = make_uniform_batch(n=1, K=5)
        y1, y2 = 300.0, 700.0
        f1 = batch.cdf(y1)[0]
        f2 = batch.cdf(y2)[0]
        assert f1 < f2

    def test_cdf_bounded_zero_one(self):
        """CDF values are in [0, 1]."""
        batch = make_uniform_batch(n=5, K=5)
        y_grid = np.array([0.01, 100.0, 500.0, 999.0, 5000.0])
        cdf = batch.cdf(y_grid)
        assert np.all(cdf >= 0.0 - 1e-10)
        assert np.all(cdf <= 1.0 + 1e-10)

    def test_quantile_median_in_histogram(self):
        """Median should be in the histogram region [c_0, c_K]."""
        batch = make_uniform_batch(n=5, K=5)
        median = batch.quantile(0.50)
        assert median.shape == (5,)
        # With uniform probs and cutpoints 0-1000, median should be near
        # the point where the histogram CDF reaches 0.5
        # Check it's within [0, 1000]
        assert np.all(median >= 0.0)
        assert np.all(median <= 1100.0)  # allow some tail

    def test_quantile_array_input(self):
        """quantile with array alpha returns (n, m)."""
        batch = make_uniform_batch(n=4, K=5)
        alpha = np.array([0.25, 0.50, 0.75])
        result = batch.quantile(alpha)
        assert result.shape == (4, 3)

    def test_quantile_monotone(self):
        """quantile(p1) <= quantile(p2) for p1 < p2."""
        batch = make_uniform_batch(n=3, K=5)
        p_grid = np.linspace(0.05, 0.95, 20)
        q_grid = batch.quantile(p_grid)  # (n, 20)
        diffs = np.diff(q_grid, axis=1)
        assert np.all(diffs >= -1.0)  # allow tiny numerical errors

    def test_quantile_cdf_roundtrip(self):
        """
        For a probability p in the histogram region,
        CDF(quantile(p)) should approximately equal p.
        """
        batch = make_uniform_batch(n=2, K=10)
        p = 0.40
        q = batch.quantile(p)          # (n,)
        cdf_q = np.array([batch.cdf(float(q[i]))[i] for i in range(2)])
        np.testing.assert_allclose(cdf_q, p, atol=0.05)

    def test_mean_positive(self):
        """Mean should be positive for Gamma data."""
        batch = make_uniform_batch(n=5, K=5)
        m = batch.mean()
        assert m.shape == (5,)
        assert np.all(m > 0)

    def test_mean_near_baseline_mean(self):
        """
        With uniform bins over [0, 1000] and Gamma(mu=500),
        the histogram mean should be near 500.
        """
        batch = make_uniform_batch(n=10, K=10)
        # With uniform bins 0..1000 and a lot of mass in histogram,
        # histogram mean ≈ (0+1000)/2 = 500 weighted by histogram mass
        m = batch.mean()
        # Just check it's in a reasonable range
        assert np.all(m > 50)
        assert np.all(m < 3000)

    def test_var_positive(self):
        """Variance should be positive."""
        batch = make_uniform_batch(n=5, K=5)
        v = batch.var()
        assert v.shape == (5,)
        assert np.all(v > 0)

    def test_std_positive(self):
        """Standard deviation is sqrt of variance."""
        batch = make_uniform_batch(n=3, K=5)
        std = batch.std()
        var = batch.var()
        np.testing.assert_allclose(std ** 2, var, rtol=1e-10)

    def test_crps_positive(self):
        """CRPS should be non-negative."""
        batch = make_uniform_batch(n=5, K=5)
        y_true = np.array([100.0, 300.0, 500.0, 700.0, 900.0])
        crps = batch.crps(y_true)
        assert crps.shape == (5,)
        assert np.all(crps >= 0.0)

    def test_crps_histogram_contribution_ordering(self):
        """
        Within the histogram region, the CRPS contribution from a single
        bin is lower when the observation is in the bin vs not in the bin.

        This tests the bin-level CRPS computation directly. For a single
        bin [400, 500] with F_lo=0.4 and F_hi=0.5:
        - Observation y_true=450 (inside the bin) contributes less CRPS
          than observation y_true=100 (which causes the indicator to be 1
          for the entire bin, creating large (F-1)^2 integrand).
        
        We test this property is correctly computed by CRPS.
        """
        n = 1
        from scipy import stats
        # Use a large histogram [0, 10000] so tail contributions are negligible
        K = 20
        cutpoints = np.linspace(0.0, 10000.0, K + 1)  # 500-wide bins
        # Uniform over all bins
        bin_probs = np.full((n, K), 1.0 / K)
        
        alpha, disp = 2.0, 0.5
        mu = np.full(n, 2000.0)  # mean well within histogram
        bc0 = stats.gamma.cdf(0.0, a=alpha, scale=mu * disp)
        bcK = stats.gamma.cdf(10000.0, a=alpha, scale=mu * disp)
        params = {"mu": mu, "dispersion": disp}
        
        batch = ExtendedHistogramBatch(cutpoints, bin_probs, bc0, bcK, params)
        
        # y_true in the middle vs y_true very low: middle should have lower CRPS
        # because F(y_middle) is near 0.5 (less extreme than F(y_low) which causes
        # large (F-1)^2 integral for all bins above y_low)
        crps_mid = batch.crps(np.array([5000.0]))  # middle of histogram
        crps_low = batch.crps(np.array([100.0]))   # very low: indicator=1 everywhere above
        crps_high = batch.crps(np.array([9900.0])) # very high: indicator=0 everywhere below

        # The uniform histogram CRPS should be lower at the median than at extremes
        # At median: sum of min(F^2, (F-1)^2) is minimised
        assert crps_mid[0] < crps_low[0], (
            f"CRPS at middle ({crps_mid[0]:.1f}) should be < CRPS at low extreme ({crps_low[0]:.1f})"
        )
        assert crps_mid[0] < crps_high[0], (
            f"CRPS at middle ({crps_mid[0]:.1f}) should be < CRPS at high extreme ({crps_high[0]:.1f})"
        )

    def test_crps_shape(self):
        batch = make_uniform_batch(n=8, K=5)
        y_true = np.random.default_rng(0).gamma(2, 500, size=8)
        crps = batch.crps(y_true)
        assert crps.shape == (8,)

    def test_expected_shortfall_above_quantile(self):
        """ES should be >= corresponding quantile."""
        batch = make_uniform_batch(n=5, K=10)
        alpha = 0.90
        q90 = batch.quantile(alpha)
        es90 = batch.expected_shortfall(alpha)
        assert np.all(es90 >= q90 - 10.0)  # allow small numerical error

    def test_expected_shortfall_shape(self):
        batch = make_uniform_batch(n=4, K=5)
        es = batch.expected_shortfall(0.995)
        assert es.shape == (4,)

    def test_summary_returns_polars(self):
        import polars as pl
        batch = make_uniform_batch(n=5, K=5)
        df = batch.summary()
        assert isinstance(df, pl.DataFrame)
        assert "mean" in df.columns
        assert "std" in df.columns
        assert len(df) == 5

    def test_bin_probs_approximately_one(self):
        """bin_probs in ExtendedHistogramBatch should sum to 1."""
        batch = make_uniform_batch(n=5, K=5)
        row_sums = batch.bin_probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_gaussian_family(self):
        """ExtendedHistogramBatch works with gaussian family."""
        batch = make_uniform_batch(n=3, K=5, family="gaussian")
        assert batch.distribution_family == "gaussian"
        cdf = batch.cdf(500.0)
        assert cdf.shape == (3,)

    def test_lognormal_family(self):
        """ExtendedHistogramBatch works with lognormal family."""
        n, K = 3, 5
        cutpoints = np.linspace(0.0, 1000.0, K + 1)
        bin_probs = np.full((n, K), 1.0 / K)
        mu = np.full(n, 500.0)
        disp = 0.25
        from scipy import stats
        # For lognormal, mu is raw-scale mean; disp is sigma_log^2
        sigma_log = np.sqrt(disp)
        mu_log = np.log(mu) - 0.5 * sigma_log ** 2
        bc0 = stats.lognorm.cdf(0.0, s=sigma_log, scale=np.exp(mu_log))
        bcK = stats.lognorm.cdf(1000.0, s=sigma_log, scale=np.exp(mu_log))
        params = {"mu": mu, "dispersion": disp}
        batch = ExtendedHistogramBatch(
            cutpoints, bin_probs, bc0, bcK, params, distribution_family="lognormal"
        )
        cdf = batch.cdf(500.0)
        assert cdf.shape == (3,)
        assert np.all(cdf >= 0.0)

    def test_cdf_left_tail_uses_baseline(self):
        """For y < c_0, CDF should be from baseline."""
        n, K = 3, 5
        cutpoints = np.array([100.0, 300.0, 500.0, 700.0, 900.0, 1100.0])
        bin_probs = np.full((n, K), 1.0 / K)
        mu = np.full(n, 500.0)
        disp = 0.5
        from scipy import stats
        alpha = 1.0 / disp
        scale = mu * disp
        bc0 = stats.gamma.cdf(100.0, a=alpha, scale=scale)
        bcK = stats.gamma.cdf(1100.0, a=alpha, scale=scale)
        params = {"mu": mu, "dispersion": disp}
        batch = ExtendedHistogramBatch(cutpoints, bin_probs, bc0, bcK, params)

        # y=50 is below c_0=100 — should use baseline CDF
        cdf_50 = batch.cdf(50.0)
        expected = stats.gamma.cdf(50.0, a=alpha, scale=500.0 * disp)
        np.testing.assert_allclose(cdf_50, expected, rtol=0.05)
