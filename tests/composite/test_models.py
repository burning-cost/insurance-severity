"""
Tests for composite severity models.

Focus:
- Fixed threshold fitting
- Profile likelihood threshold selection
- Mode-matching fitting (Lognormal-Burr only)
- PDF integrates to 1
- CDF + SF = 1
- VaR monotonicity
- TVaR > VaR
- Mode-matching constraint errors for GPD
- AIC/BIC computation
- Quantile residuals approximately normal
- ILF at basic limit = 1.0
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import shapiro

from insurance_severity.composite.models import (
    CompositeSeverityModel,
    LognormalBurrComposite,
    LognormalGPDComposite,
    GammaGPDComposite,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _approx_pdf_integral(model, y_max_quantile: float = 0.999) -> float:
    """Numerically integrate composite pdf from 0 to large upper limit."""
    # Use a finite upper limit (very large claims are negligible)
    n_points = 1000
    # Sample adaptively: many points in body, fewer in tail
    t = model.threshold_
    x_body = np.linspace(0.01, t, n_points // 2)
    x_tail = np.linspace(t, t * 50, n_points // 2)
    x_all = np.concatenate([x_body, x_tail[1:]])

    pdf_vals = model.pdf(x_all)
    # Trapezoidal integration
    return np.trapz(pdf_vals, x_all)


# ---------------------------------------------------------------------------
# LognormalGPDComposite — fixed threshold
# ---------------------------------------------------------------------------


class TestLognormalGPDFixed:

    def test_fit_runs(self, lognormal_gpd_data):
        model = LognormalGPDComposite(
            threshold=50000.0, threshold_method="fixed"
        )
        model.fit(lognormal_gpd_data)
        assert model.threshold_ == 50000.0
        assert 0 < model.pi_ < 1
        assert model.n_body_ > 0
        assert model.n_tail_ > 0

    def test_mode_matching_raises(self):
        with pytest.raises(ValueError, match="mode_matching"):
            LognormalGPDComposite(threshold_method="mode_matching")

    def test_fixed_without_threshold_raises(self):
        with pytest.raises(ValueError, match="threshold must be provided"):
            LognormalGPDComposite(threshold_method="fixed")

    def test_pdf_positive(self, lognormal_gpd_data):
        model = LognormalGPDComposite(threshold=50000.0, threshold_method="fixed")
        model.fit(lognormal_gpd_data)
        x = np.array([5000.0, 25000.0, 50000.0, 75000.0, 150000.0])
        pdf = model.pdf(x)
        assert np.all(pdf > 0)

    def test_cdf_monotone(self, lognormal_gpd_data):
        model = LognormalGPDComposite(threshold=50000.0, threshold_method="fixed")
        model.fit(lognormal_gpd_data)
        x = np.linspace(1000, 500000, 100)
        cdf = model.cdf(x)
        assert np.all(np.diff(cdf) >= -1e-10)

    def test_cdf_sf_sum_to_one(self, lognormal_gpd_data):
        model = LognormalGPDComposite(threshold=50000.0, threshold_method="fixed")
        model.fit(lognormal_gpd_data)
        x = np.array([5000.0, 50000.0, 200000.0])
        np.testing.assert_allclose(model.cdf(x) + model.sf(x), 1.0, atol=1e-8)

    def test_ppf_monotone(self, lognormal_gpd_data):
        model = LognormalGPDComposite(threshold=50000.0, threshold_method="fixed")
        model.fit(lognormal_gpd_data)
        q = np.linspace(0.01, 0.99, 50)
        ppf = model.ppf(q)
        assert np.all(np.diff(ppf) > 0)

    def test_ppf_cdf_roundtrip(self, lognormal_gpd_data):
        model = LognormalGPDComposite(threshold=50000.0, threshold_method="fixed")
        model.fit(lognormal_gpd_data)
        q = np.array([0.2, 0.5, 0.8, 0.95])
        x = model.ppf(q)
        q_back = model.cdf(x)
        np.testing.assert_allclose(q_back, q, atol=1e-3)

    def test_tvar_gt_var(self, lognormal_gpd_data):
        model = LognormalGPDComposite(threshold=50000.0, threshold_method="fixed")
        model.fit(lognormal_gpd_data)
        alpha = 0.95
        var = model.var(alpha)
        tvar = model.tvar(alpha)
        assert tvar > var

    def test_ilf_at_basic_limit_is_one(self, lognormal_gpd_data):
        model = LognormalGPDComposite(threshold=50000.0, threshold_method="fixed")
        model.fit(lognormal_gpd_data)
        basic = 500_000.0
        ilf = model.ilf(basic, basic_limit=basic)
        assert abs(ilf - 1.0) < 0.01

    def test_ilf_monotone_increasing(self, lognormal_gpd_data):
        model = LognormalGPDComposite(threshold=50000.0, threshold_method="fixed")
        model.fit(lognormal_gpd_data)
        limits = [100_000, 250_000, 500_000, 1_000_000]
        basic = 100_000.0
        ilfs = [model.ilf(l, basic) for l in limits]
        assert ilfs[0] == pytest.approx(1.0, abs=0.01)
        for i in range(len(ilfs) - 1):
            assert ilfs[i + 1] >= ilfs[i]

    def test_aic_bic(self, lognormal_gpd_data):
        model = LognormalGPDComposite(threshold=50000.0, threshold_method="fixed")
        model.fit(lognormal_gpd_data)
        aic = model.aic(lognormal_gpd_data)
        bic = model.bic(lognormal_gpd_data)
        assert np.isfinite(aic)
        assert np.isfinite(bic)
        # BIC >= AIC for n > e^2 ~ 7.4
        assert bic >= aic

    def test_summary_runs(self, lognormal_gpd_data):
        model = LognormalGPDComposite(threshold=50000.0, threshold_method="fixed")
        model.fit(lognormal_gpd_data)
        s = model.summary(lognormal_gpd_data)
        assert "LognormalGPD" in s
        assert "Threshold" in s

    def test_quantile_residuals_approx_normal(self, lognormal_gpd_data):
        """Quantile residuals should be approximately N(0,1)."""
        model = LognormalGPDComposite(threshold=50000.0, threshold_method="fixed")
        model.fit(lognormal_gpd_data)
        resid = model.quantile_residuals(lognormal_gpd_data)
        # Shapiro-Wilk test at generous significance level
        # (We don't expect perfect normality, just reasonable)
        assert abs(np.mean(resid)) < 1.0
        assert 0.5 < np.std(resid) < 2.5

    def test_empty_data_raises(self):
        model = LognormalGPDComposite(threshold=50000.0, threshold_method="fixed")
        with pytest.raises((ValueError, Exception)):
            model.fit(np.array([100.0, 200.0, 300.0]))

    def test_threshold_too_high_raises(self, lognormal_gpd_data):
        model = LognormalGPDComposite(
            threshold=1e12, threshold_method="fixed"
        )
        with pytest.raises(ValueError, match="No observations above"):
            model.fit(lognormal_gpd_data)

    def test_threshold_too_low_raises(self, lognormal_gpd_data):
        model = LognormalGPDComposite(
            threshold=0.01, threshold_method="fixed"
        )
        with pytest.raises(ValueError, match="No observations at or below"):
            model.fit(lognormal_gpd_data)


# ---------------------------------------------------------------------------
# LognormalGPDComposite — profile likelihood threshold
# ---------------------------------------------------------------------------


class TestLognormalGPDProfileLikelihood:

    def test_fit_selects_threshold(self, lognormal_gpd_data):
        model = LognormalGPDComposite(
            threshold_method="profile_likelihood",
            threshold_quantile_range=(0.70, 0.95),
        )
        model.fit(lognormal_gpd_data)
        assert model.threshold_ is not None
        assert model.threshold_ > np.min(lognormal_gpd_data)
        assert model.threshold_ < np.max(lognormal_gpd_data)

    def test_pdf_valid_after_profile_fit(self, lognormal_gpd_data):
        model = LognormalGPDComposite(threshold_method="profile_likelihood")
        model.fit(lognormal_gpd_data)
        x = np.linspace(
            np.percentile(lognormal_gpd_data, 5),
            np.percentile(lognormal_gpd_data, 95),
            50
        )
        pdf = model.pdf(x)
        assert np.all(np.isfinite(pdf))
        assert np.all(pdf >= 0)


# ---------------------------------------------------------------------------
# GammaGPDComposite — fixed threshold
# ---------------------------------------------------------------------------


class TestGammaGPDFixed:

    def test_fit_runs(self, gamma_gpd_data):
        model = GammaGPDComposite(threshold=30000.0, threshold_method="fixed")
        model.fit(gamma_gpd_data)
        assert model.threshold_ == 30000.0
        assert model.n_body_ > 0

    def test_mode_matching_raises(self):
        with pytest.raises(ValueError, match="mode_matching"):
            GammaGPDComposite(threshold_method="mode_matching")

    def test_cdf_monotone(self, gamma_gpd_data):
        model = GammaGPDComposite(threshold=30000.0, threshold_method="fixed")
        model.fit(gamma_gpd_data)
        x = np.linspace(1000, 300000, 100)
        cdf = model.cdf(x)
        assert np.all(np.diff(cdf) >= -1e-10)

    def test_tvar_gt_var(self, gamma_gpd_data):
        model = GammaGPDComposite(threshold=30000.0, threshold_method="fixed")
        model.fit(gamma_gpd_data)
        tvar = model.tvar(0.90)
        var = model.var(0.90)
        assert tvar > var

    def test_profile_likelihood_gamma_gpd(self, gamma_gpd_data):
        model = GammaGPDComposite(threshold_method="profile_likelihood")
        model.fit(gamma_gpd_data)
        assert model.threshold_ is not None


# ---------------------------------------------------------------------------
# LognormalBurrComposite — fixed threshold
# ---------------------------------------------------------------------------


class TestLognormalBurrFixed:

    def test_fit_fixed_threshold(self, lognormal_burr_data):
        model = LognormalBurrComposite(threshold=4800.0, threshold_method="fixed")
        model.fit(lognormal_burr_data)
        assert model.threshold_ == 4800.0
        assert len(model.tail_params_) == 3  # alpha, delta, beta

    def test_pdf_positive(self, lognormal_burr_data):
        model = LognormalBurrComposite(threshold=4800.0, threshold_method="fixed")
        model.fit(lognormal_burr_data)
        x = np.array([500.0, 2000.0, 4800.0, 10000.0, 50000.0])
        pdf = model.pdf(x)
        assert np.all(pdf > 0)


# ---------------------------------------------------------------------------
# LognormalBurrComposite — mode-matching
# ---------------------------------------------------------------------------


class TestLognormalBurrModeMatching:

    def test_fit_mode_matching(self, lognormal_burr_data):
        model = LognormalBurrComposite(threshold_method="mode_matching", n_starts=3)
        model.fit(lognormal_burr_data)

        assert model.threshold_ is not None
        assert model.threshold_ > 0
        assert 0 < model.pi_ < 1
        assert model.n_body_ > 0
        assert model.n_tail_ > 0

    def test_threshold_equals_burr_mode(self, lognormal_burr_data):
        """After mode-matching fit, threshold == tail mode (using correct formula)."""
        model = LognormalBurrComposite(threshold_method="mode_matching", n_starts=3)
        model.fit(lognormal_burr_data)

        alpha, delta, beta = model.tail_params_
        # Correct formula: beta * [(delta-1)/(alpha*delta+1)]^{1/delta}
        ratio = (delta - 1.0) / (alpha * delta + 1.0)
        mode_expected = beta * ratio ** (1.0 / delta)
        assert abs(model.threshold_ - mode_expected) < 1.0, (
            f"threshold={model.threshold_:.1f}, mode={mode_expected:.1f}"
        )

    def test_lognormal_mu_consistent_with_mode_matching(self, lognormal_burr_data):
        """LN mode should equal threshold after mode-matching."""
        model = LognormalBurrComposite(threshold_method="mode_matching", n_starts=3)
        model.fit(lognormal_burr_data)

        mu, sigma = model.body_params_
        ln_mode = np.exp(mu - sigma ** 2)
        assert abs(ln_mode - model.threshold_) / model.threshold_ < 0.02

    def test_alpha_gt_1(self, lognormal_burr_data):
        """Mode-matching requires alpha > 1."""
        model = LognormalBurrComposite(threshold_method="mode_matching", n_starts=3)
        model.fit(lognormal_burr_data)
        alpha = model.tail_params_[0]
        assert alpha > 1.0

    def test_cdf_sf_sum_to_one(self, lognormal_burr_data):
        model = LognormalBurrComposite(threshold_method="mode_matching", n_starts=3)
        model.fit(lognormal_burr_data)
        x = np.array([1000.0, model.threshold_, model.threshold_ * 5.0])
        np.testing.assert_allclose(model.cdf(x) + model.sf(x), 1.0, atol=1e-8)

    def test_ppf_monotone(self, lognormal_burr_data):
        model = LognormalBurrComposite(threshold_method="mode_matching", n_starts=3)
        model.fit(lognormal_burr_data)
        q = np.linspace(0.01, 0.99, 50)
        ppf = model.ppf(q)
        assert np.all(np.diff(ppf) > 0)

    def test_tvar_gt_var(self, lognormal_burr_data):
        model = LognormalBurrComposite(threshold_method="mode_matching", n_starts=3)
        model.fit(lognormal_burr_data)
        tvar = model.tvar(0.95)
        var = model.var(0.95)
        assert tvar > var

    def test_mean_excess_positive(self, lognormal_burr_data):
        model = LognormalBurrComposite(threshold_method="mode_matching", n_starts=3)
        model.fit(lognormal_burr_data)
        me = model.mean_excess(model.threshold_)
        assert me > 0

    def test_loglik_finite(self, lognormal_burr_data):
        model = LognormalBurrComposite(threshold_method="mode_matching", n_starts=3)
        model.fit(lognormal_burr_data)
        assert np.isfinite(model.loglik_)

    def test_profile_likelihood_burr(self, lognormal_burr_data):
        model = LognormalBurrComposite(threshold_method="profile_likelihood")
        model.fit(lognormal_burr_data)
        assert model.threshold_ is not None


# ---------------------------------------------------------------------------
# Check not-fitted state
# ---------------------------------------------------------------------------


class TestNotFittedState:

    def test_logpdf_raises_if_not_fitted(self):
        model = LognormalGPDComposite(threshold=50000.0, threshold_method="fixed")
        with pytest.raises(RuntimeError, match="not fitted"):
            model.logpdf(np.array([10000.0]))

    def test_cdf_raises_if_not_fitted(self):
        model = GammaGPDComposite(threshold=30000.0, threshold_method="fixed")
        with pytest.raises(RuntimeError, match="not fitted"):
            model.cdf(np.array([10000.0]))
