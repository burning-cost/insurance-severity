"""
Tests for distribution building blocks.

Focus:
- Truncated PDFs integrate to 1 over their domain
- Survival functions are consistent with CDFs
- Mode formulas give correct values at known parameters
- MLE recovers approximate parameters on synthetic data
- Edge cases and error conditions
"""

import numpy as np
import pytest
from scipy import stats
from scipy.integrate import quad

from insurance_severity.composite.distributions import (
    LognormalBody,
    GammaBody,
    GPDTail,
    ParetoTail,
    BurrTail,
)


# ---------------------------------------------------------------------------
# LognormalBody
# ---------------------------------------------------------------------------


class TestLognormalBody:

    def test_init(self):
        body = LognormalBody(mu=8.0, sigma=1.5)
        assert body.mu == 8.0
        assert body.sigma == 1.5

    def test_invalid_sigma(self):
        with pytest.raises(ValueError, match="sigma"):
            LognormalBody(mu=0.0, sigma=-1.0)

    def test_params_setter(self):
        body = LognormalBody()
        body.params = np.array([7.0, 0.8])
        assert body.mu == 7.0
        assert body.sigma == 0.8

    def test_truncated_pdf_integrates_to_one(self):
        body = LognormalBody(mu=8.0, sigma=1.2)
        threshold = 5000.0
        result, err = quad(
            lambda x: np.exp(body.logpdf(np.array([x]), threshold)[0]),
            0.01,
            threshold,
        )
        assert abs(result - 1.0) < 1e-4

    def test_logcdf_at_threshold_is_zero(self):
        """log P(X <= t | X <= t) = 0."""
        body = LognormalBody(mu=8.0, sigma=1.2)
        threshold = 5000.0
        lc = body.logcdf(np.array([threshold]), threshold)[0]
        assert abs(lc) < 1e-6

    def test_ppf_inverse_of_cdf(self):
        body = LognormalBody(mu=8.0, sigma=1.2)
        threshold = 5000.0
        q = np.array([0.1, 0.5, 0.9])
        x = body.ppf(q, threshold)
        cdf_x = np.exp(body.logcdf(x, threshold))
        np.testing.assert_allclose(cdf_x, q, atol=1e-4)

    def test_mode(self):
        mu, sigma = 8.0, 1.2
        body = LognormalBody(mu=mu, sigma=sigma)
        expected_mode = np.exp(mu - sigma ** 2)
        assert abs(body.mode(threshold=10000) - expected_mode) < 1.0

    def test_fit_mle_recovers_params(self):
        """MLE on body data should recover sigma within tolerance."""
        rng = np.random.default_rng(99)
        mu_true, sigma_true = 8.5, 1.2
        threshold = np.exp(mu_true)  # roughly at mean of lognormal
        # Generate lognormal data truncated at threshold
        y_body = []
        while len(y_body) < 300:
            batch = stats.lognorm.rvs(s=sigma_true, scale=np.exp(mu_true), size=500, random_state=rng)
            batch = batch[batch <= threshold]
            y_body.extend(batch[:max(0, 300 - len(y_body))])
        y_body = np.array(y_body[:300])

        body = LognormalBody()
        params, info = body.fit_mle(y_body, threshold)

        # sigma should be roughly positive and reasonable
        assert params[1] > 0.3
        assert params[1] < 3.0

    def test_fit_mle_empty_raises(self):
        body = LognormalBody()
        with pytest.raises(ValueError, match="No observations"):
            body.fit_mle(np.array([]), threshold=1000.0)


# ---------------------------------------------------------------------------
# GammaBody
# ---------------------------------------------------------------------------


class TestGammaBody:

    def test_init(self):
        body = GammaBody(shape=3.0, scale=5000.0)
        assert body.params[0] == 3.0

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            GammaBody(shape=-1.0, scale=1.0)

    def test_truncated_pdf_integrates_to_one(self):
        body = GammaBody(shape=3.0, scale=8000.0)
        threshold = 30000.0
        result, err = quad(
            lambda x: np.exp(body.logpdf(np.array([x]), threshold)[0]),
            1.0,
            threshold,
        )
        assert abs(result - 1.0) < 1e-3

    def test_ppf_inverse_of_logcdf(self):
        body = GammaBody(shape=3.0, scale=5000.0)
        threshold = 25000.0
        q = np.array([0.1, 0.3, 0.7, 0.9])
        x = body.ppf(q, threshold)
        cdf_back = np.exp(body.logcdf(x, threshold))
        np.testing.assert_allclose(cdf_back, q, atol=1e-4)

    def test_mode_shape_gt_1(self):
        shape, scale = 4.0, 3000.0
        body = GammaBody(shape=shape, scale=scale)
        expected = (shape - 1.0) * scale
        assert abs(body.mode(threshold=50000) - expected) < 1.0

    def test_mode_shape_lte_1(self):
        body = GammaBody(shape=0.8, scale=1000.0)
        assert body.mode(threshold=5000) == 0.0

    def test_fit_mle(self):
        rng = np.random.default_rng(42)
        shape, scale = 3.0, 8000.0
        threshold = 30000.0
        data = stats.gamma.rvs(a=shape, scale=scale, size=500, random_state=rng)
        data = data[data <= threshold]
        body = GammaBody()
        params, info = body.fit_mle(data, threshold)
        # Shape and scale should be in a reasonable range
        assert params[0] > 1.0
        assert params[1] > 1000.0


# ---------------------------------------------------------------------------
# GPDTail
# ---------------------------------------------------------------------------


class TestGPDTail:

    def test_init(self):
        tail = GPDTail(xi=0.2, sigma=15000.0)
        assert tail._xi == 0.2

    def test_invalid_sigma(self):
        with pytest.raises(ValueError):
            GPDTail(xi=0.2, sigma=-1.0)

    def test_pdf_sums_to_one(self):
        tail = GPDTail(xi=0.25, sigma=20000.0)
        threshold = 50000.0
        result, err = quad(
            lambda x: np.exp(tail.logpdf(np.array([x]), threshold)[0]),
            threshold,
            threshold + 1e7,
        )
        assert abs(result - 1.0) < 1e-3

    def test_logsf_at_threshold_is_zero(self):
        """P(X > t | X > t) = 1, so logsf = 0."""
        tail = GPDTail(xi=0.2, sigma=10000.0)
        logsf = tail.logsf(np.array([50000.0]), threshold=50000.0)
        assert abs(logsf[0]) < 1e-6

    def test_ppf_inverse(self):
        tail = GPDTail(xi=0.2, sigma=15000.0)
        threshold = 50000.0
        q = np.array([0.1, 0.5, 0.9])
        x = tail.ppf(q, threshold)
        assert np.all(x >= threshold)
        cdf_back = tail.cdf(x, threshold)
        np.testing.assert_allclose(cdf_back, q, atol=1e-4)

    def test_mode_value_heavy_tail(self):
        """GPD mode is None for xi >= 0."""
        tail = GPDTail(xi=0.2, sigma=10000.0)
        assert tail.mode_value() is None

    def test_mode_value_light_tail(self):
        """GPD mode is positive for xi < -0.5."""
        tail = GPDTail(xi=-0.6, sigma=10000.0)
        m = tail.mode_value()
        assert m is not None
        assert m > 0

    def test_mode_value_negative_xi_gt_neg_half(self):
        """No positive mode for -0.5 <= xi < 0."""
        tail = GPDTail(xi=-0.3, sigma=10000.0)
        assert tail.mode_value() is None

    def test_mean_finite(self):
        tail = GPDTail(xi=0.5, sigma=10000.0)  # xi < 1
        m = tail.mean(threshold=50000.0)
        assert np.isfinite(m)
        assert m > 50000.0

    def test_mean_infinite_xi_ge_1(self):
        tail = GPDTail(xi=1.0, sigma=10000.0)
        assert tail.mean(threshold=50000.0) == np.inf

    def test_fit_mle(self):
        rng = np.random.default_rng(42)
        threshold = 50000.0
        true_xi, true_sigma = 0.25, 20000.0
        data = stats.genpareto.rvs(c=true_xi, scale=true_sigma, size=300, random_state=rng)
        data += threshold

        tail = GPDTail()
        params, info = tail.fit_mle(data, threshold)

        # Should be in the right ballpark
        assert -0.5 < params[0] < 1.5
        assert params[1] > 1000.0

    def test_fit_mle_warns_small_sample(self):
        rng = np.random.default_rng(42)
        data = np.array([10001, 10500, 12000, 15000, 20000], dtype=float)
        tail = GPDTail()
        with pytest.warns(UserWarning, match="unreliable"):
            tail.fit_mle(data, threshold=10000.0)


# ---------------------------------------------------------------------------
# ParetoTail
# ---------------------------------------------------------------------------


class TestParetoTail:

    def test_pdf_integrates_to_one(self):
        tail = ParetoTail(alpha=2.0, sigma=10000.0)
        threshold = 30000.0
        result, err = quad(
            lambda x: np.exp(tail.logpdf(np.array([x]), threshold)[0]),
            threshold,
            threshold + 1e8,
        )
        assert abs(result - 1.0) < 1e-3

    def test_mode_is_none(self):
        tail = ParetoTail(alpha=2.0, sigma=10000.0)
        assert tail.mode_value() is None

    def test_ppf_roundtrip(self):
        tail = ParetoTail(alpha=2.0, sigma=10000.0)
        threshold = 30000.0
        q = np.array([0.1, 0.5, 0.9])
        x = tail.ppf(q, threshold)
        logsf = tail.logsf(x, threshold)
        sf = np.exp(logsf)
        np.testing.assert_allclose(1.0 - sf, q, atol=1e-6)


# ---------------------------------------------------------------------------
# BurrTail
# ---------------------------------------------------------------------------


class TestBurrTail:

    def test_init(self):
        tail = BurrTail(alpha=2.5, delta=1.2, beta=8000.0)
        assert tail._alpha == 2.5

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            BurrTail(alpha=-1.0, delta=1.0, beta=1000.0)

    def test_pdf_integrates_to_one(self):
        tail = BurrTail(alpha=2.5, delta=1.2, beta=8000.0)
        threshold = 4800.0
        result, err = quad(
            lambda x: np.exp(tail.logpdf(np.array([x]), threshold)[0]),
            threshold,
            threshold + 1e8,
            limit=200,
        )
        assert abs(result - 1.0) < 1e-2

    def test_logsf_at_threshold_is_zero(self):
        tail = BurrTail(alpha=2.5, delta=1.2, beta=8000.0)
        threshold = 4800.0
        logsf = tail.logsf(np.array([threshold]), threshold)
        assert abs(logsf[0]) < 1e-6

    def test_ppf_above_threshold(self):
        tail = BurrTail(alpha=2.5, delta=1.2, beta=8000.0)
        threshold = 4800.0
        q = np.array([0.01, 0.5, 0.99])
        x = tail.ppf(q, threshold)
        assert np.all(x >= threshold)

    def test_ppf_inverse_logsf(self):
        tail = BurrTail(alpha=2.5, delta=1.2, beta=8000.0)
        threshold = 4800.0
        q = np.array([0.1, 0.5, 0.9])
        x = tail.ppf(q, threshold)
        logsf = tail.logsf(x, threshold)
        sf = np.exp(logsf)
        np.testing.assert_allclose(1.0 - sf, q, atol=1e-4)

    def test_mode_exists_delta_gt_1(self):
        """Burr XII mode: beta * [(delta-1)/(alpha*delta+1)]^{1/delta} for delta > 1."""
        alpha, delta, beta = 2.5, 1.5, 8000.0
        tail = BurrTail(alpha=alpha, delta=delta, beta=beta)
        m = tail.mode_value()
        assert m is not None
        assert m > 0
        # Check against correct formula
        ratio = (delta - 1.0) / (alpha * delta + 1.0)
        expected = beta * ratio ** (1.0 / delta)
        assert abs(m - expected) < 1.0

    def test_mode_none_delta_lte_1(self):
        """Mode is None when delta <= 1 (mode existence requires delta > 1)."""
        tail = BurrTail(alpha=2.5, delta=0.8, beta=8000.0)
        assert tail.mode_value() is None

    def test_mode_exists_delta_gt_1(self):
        """Mode exists when delta > 1, regardless of alpha."""
        tail = BurrTail(alpha=0.8, delta=1.5, beta=8000.0)
        assert tail.mode_value() is not None

    def test_mode_is_density_maximum(self):
        """Verify that the analytical mode is near the numerical maximum."""
        alpha, delta, beta = 2.5, 1.2, 8000.0
        tail = BurrTail(alpha=alpha, delta=delta, beta=beta)
        mode = tail.mode_value()
        threshold = 100.0  # low threshold so mode is well above it

        # Find numerical maximum by evaluating logpdf on a fine grid near mode
        x_grid = np.linspace(mode * 0.5, mode * 2.0, 1000)
        logpdf_grid = tail.logpdf(x_grid, threshold)
        numerical_mode = x_grid[np.argmax(logpdf_grid)]

        # Analytical mode should be within 5% of numerical mode
        rel_err = abs(numerical_mode - mode) / mode
        assert rel_err < 0.05, f"Mode error {rel_err:.3f}: analytical={mode:.1f}, numerical={numerical_mode:.1f}"

    def test_fit_mle_without_mode_constraint(self):
        rng = np.random.default_rng(42)
        # Generate from Burr XII
        alpha, delta, beta = 2.5, 1.2, 8000.0
        threshold = 4800.0
        tail_true = BurrTail(alpha=alpha, delta=delta, beta=beta)
        q = rng.uniform(size=300)
        data = tail_true.ppf(q, threshold)

        tail_fit = BurrTail()
        params, info = tail_fit.fit_mle(data, threshold, require_mode=False)
        assert params[0] > 0
        assert params[1] > 0
        assert params[2] > 0

    def test_fit_mle_with_mode_constraint_alpha_gt_1(self):
        """With require_mode=True, fitted delta must be > 1 (mode existence condition)."""
        rng = np.random.default_rng(42)
        # Use delta=1.5 > 1 (required for mode) 
        alpha, delta, beta = 2.5, 1.5, 8000.0
        threshold = np.quantile(
            BurrTail(alpha=alpha, delta=delta, beta=beta).ppf(
                np.linspace(0.01, 0.99, 100), 0.0
            ),
            0.30
        )
        tail_true = BurrTail(alpha=alpha, delta=delta, beta=beta)
        q = rng.uniform(size=200)
        data = tail_true.ppf(q, threshold)

        tail_fit = BurrTail()
        params, info = tail_fit.fit_mle(data, threshold, require_mode=True)
        # delta > 1 must be enforced (the actual mode condition)
        assert params[1] > 1.0, f"delta should be > 1, got {params[1]}"
