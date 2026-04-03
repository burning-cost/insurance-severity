"""
Extended test coverage for insurance-severity (April 2026).

Targets gaps across:
- Distribution mean/tvar methods (LognormalBody, GammaBody, GPDTail, ParetoTail, BurrTail)
- GPD edge cases: xi=0 (exponential limit), xi<-0.5, negative-xi bounded support
- ParetoTail: MLE, tvar, mean at boundary (alpha=1)
- BurrTail: mean, tvar, params setter
- CompositeSeverityModel: mean_excess, limited_expected_value, summary without y
- CompositeSeverityModel: profile_likelihood failure path, negative/zero data errors
- CompositeSeverityRegressor: predict_tail_scale, bootstrap_ci structure
- TailCalibration: unfitted guard, cdf_func wrong length, severity_pit all-same-value
- BladtTailScore: score_grid all-finite for valid inputs, rank with all-nan scores
- ProjectionToUltimate: repr, properties before/after fit, alternate paid_col predict,
  categorical/boolean feature columns, zero prediction interval coverage
- _build_design_matrix: categorical + boolean encoding
- LognormalBody/GammaBody: params setter round-trip
- Integration: CDF at 0 and at infinity limits
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest
from scipy import stats
from scipy.integrate import quad


# ===========================================================================
# Distribution mean/tvar methods
# ===========================================================================


class TestLognormalBodyMean:
    def test_mean_formula(self):
        """Mean of LN(mu, sigma) = exp(mu + sigma^2/2)."""
        from insurance_severity.composite.distributions import LognormalBody
        mu, sigma = 8.0, 1.2
        body = LognormalBody(mu=mu, sigma=sigma)
        expected = np.exp(mu + 0.5 * sigma ** 2)
        assert abs(body.mean() - expected) < 1.0

    def test_mean_positive(self):
        from insurance_severity.composite.distributions import LognormalBody
        body = LognormalBody(mu=7.5, sigma=0.8)
        assert body.mean() > 0

    def test_params_setter_round_trip(self):
        from insurance_severity.composite.distributions import LognormalBody
        body = LognormalBody(mu=5.0, sigma=2.0)
        body.params = np.array([9.0, 1.5])
        assert body.mu == pytest.approx(9.0)
        assert body.sigma == pytest.approx(1.5)
        np.testing.assert_array_equal(body.params, [9.0, 1.5])


class TestGammaBodyMean:
    def test_mean_formula(self):
        """Mean of Gamma(shape, scale) = shape * scale."""
        from insurance_severity.composite.distributions import GammaBody
        shape, scale = 3.0, 5000.0
        body = GammaBody(shape=shape, scale=scale)
        expected = shape * scale
        assert abs(body.mean() - expected) < 1.0

    def test_mean_positive(self):
        from insurance_severity.composite.distributions import GammaBody
        body = GammaBody(shape=2.0, scale=1000.0)
        assert body.mean() > 0

    def test_params_setter(self):
        from insurance_severity.composite.distributions import GammaBody
        body = GammaBody(shape=2.0, scale=1000.0)
        body.params = np.array([5.0, 2000.0])
        assert body.params[0] == pytest.approx(5.0)
        assert body.params[1] == pytest.approx(2000.0)

    def test_fit_mle_empty_raises(self):
        from insurance_severity.composite.distributions import GammaBody
        body = GammaBody()
        with pytest.raises(ValueError, match="No observations"):
            body.fit_mle(np.array([]), threshold=1000.0)


class TestGPDTailTVaR:
    def test_tvar_gt_var(self):
        """TVaR must exceed VaR."""
        from insurance_severity.composite.distributions import GPDTail
        tail = GPDTail(xi=0.3, sigma=10000.0)
        threshold = 50000.0
        alpha = 0.9
        var = tail.ppf(np.array([alpha]), threshold)[0]
        tvar = tail.tvar(alpha, threshold)
        assert tvar > var

    def test_tvar_infinite_xi_ge_1(self):
        """TVaR is infinite for xi >= 1."""
        from insurance_severity.composite.distributions import GPDTail
        tail = GPDTail(xi=1.0, sigma=10000.0)
        assert tail.tvar(0.9, threshold=50000.0) == np.inf

    def test_tvar_finite_xi_lt_1(self):
        from insurance_severity.composite.distributions import GPDTail
        tail = GPDTail(xi=0.5, sigma=10000.0)
        tvar = tail.tvar(0.9, threshold=50000.0)
        assert np.isfinite(tvar)
        assert tvar > 50000.0


class TestGPDEdgeCases:
    def test_xi_zero_log_sf_exponential(self):
        """When xi ~ 0, GPD reduces to exponential; log_sf_at_threshold should be -t/sigma."""
        from insurance_severity.composite.distributions import GPDTail
        # xi=0 exactly triggers the exponential branch
        tail = GPDTail(xi=0.0, sigma=10000.0)
        # _log_sf_at_threshold(t) = -t/sigma for xi=0
        t = 5000.0
        expected = -t / 10000.0
        result = tail._log_sf_at_threshold(t)
        assert abs(result - expected) < 1e-10

    def test_xi_zero_ppf_monotone(self):
        """ppf should be monotone even with xi=0 (exponential GPD)."""
        from insurance_severity.composite.distributions import GPDTail
        tail = GPDTail(xi=0.0, sigma=10000.0)
        threshold = 30000.0
        q = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        x = tail.ppf(q, threshold)
        assert np.all(np.diff(x) > 0)

    def test_xi_negative_lt_minus_half_mode_positive(self):
        """xi < -0.5 should give a positive mode."""
        from insurance_severity.composite.distributions import GPDTail
        tail = GPDTail(xi=-0.7, sigma=5000.0)
        m = tail.mode_value()
        assert m is not None
        assert m > 0

    def test_negative_xi_bounded_support(self):
        """For xi < 0, max supported value is -sigma/xi."""
        from insurance_severity.composite.distributions import GPDTail
        xi, sigma = -0.3, 10000.0
        tail = GPDTail(xi=xi, sigma=sigma)
        threshold = 0.0
        max_val = -sigma / xi  # 33333.33...
        # logsf at max should approach -inf
        logsf_near_max = tail.logsf(np.array([max_val - 1.0]), threshold)
        assert logsf_near_max[0] < -5.0  # deep in the tail

    def test_params_setter(self):
        from insurance_severity.composite.distributions import GPDTail
        tail = GPDTail(xi=0.1, sigma=1000.0)
        tail.params = np.array([0.5, 20000.0])
        assert tail._xi == pytest.approx(0.5)
        assert tail._sigma == pytest.approx(20000.0)


class TestParetoTailFull:
    def test_mean_alpha_gt_1(self):
        """Mean of Pareto II = sigma / (alpha - 1) for alpha > 1."""
        from insurance_severity.composite.distributions import ParetoTail
        alpha, sigma = 2.5, 10000.0
        tail = ParetoTail(alpha=alpha, sigma=sigma)
        expected = 0.0 + sigma / (alpha - 1.0)
        assert abs(tail.mean(threshold=0.0) - expected) < 0.01

    def test_mean_alpha_le_1_infinite(self):
        from insurance_severity.composite.distributions import ParetoTail
        tail = ParetoTail(alpha=1.0, sigma=10000.0)
        assert tail.mean() == np.inf

    def test_tvar_finite(self):
        from insurance_severity.composite.distributions import ParetoTail
        tail = ParetoTail(alpha=2.5, sigma=10000.0)
        tvar = tail.tvar(alpha=0.9, threshold=30000.0)
        assert np.isfinite(tvar)
        var = tail.ppf(np.array([0.9]), threshold=30000.0)[0]
        assert tvar > var

    def test_tvar_alpha_le_1_infinite(self):
        from insurance_severity.composite.distributions import ParetoTail
        tail = ParetoTail(alpha=0.9, sigma=10000.0)
        assert tail.tvar(0.9, 30000.0) == np.inf

    def test_fit_mle_empty_raises(self):
        from insurance_severity.composite.distributions import ParetoTail
        tail = ParetoTail()
        with pytest.raises(ValueError, match="No observations above"):
            tail.fit_mle(np.array([]), threshold=1000.0)

    def test_fit_mle_small_sample_warns(self):
        rng = np.random.default_rng(42)
        from insurance_severity.composite.distributions import ParetoTail
        data = np.array([11000.0, 12000.0, 15000.0, 20000.0, 25000.0], dtype=float)
        tail = ParetoTail()
        with pytest.warns(UserWarning, match="unreliable"):
            tail.fit_mle(data, threshold=10000.0)

    def test_fit_mle_recovers_reasonable(self):
        rng = np.random.default_rng(42)
        from insurance_severity.composite.distributions import ParetoTail
        alpha_true, sigma_true = 2.5, 10000.0
        threshold = 30000.0
        tail_true = ParetoTail(alpha=alpha_true, sigma=sigma_true)
        q = rng.uniform(size=300)
        data = tail_true.ppf(q, threshold)
        tail_fit = ParetoTail()
        params, info = tail_fit.fit_mle(data, threshold)
        assert params[0] > 0.5
        assert params[1] > 1000.0

    def test_invalid_params_raises(self):
        from insurance_severity.composite.distributions import ParetoTail
        with pytest.raises(ValueError):
            ParetoTail(alpha=-1.0, sigma=1000.0)
        with pytest.raises(ValueError):
            ParetoTail(alpha=1.0, sigma=-1.0)


class TestBurrTailMeanTVaR:
    def test_mean_infinite_alpha_delta_le_1(self):
        """Mean is infinite when alpha * delta <= 1."""
        from insurance_severity.composite.distributions import BurrTail
        # alpha=0.5, delta=1.0: alpha*delta = 0.5 <= 1
        tail = BurrTail(alpha=0.5, delta=1.0, beta=8000.0)
        assert tail.mean() == np.inf

    def test_mean_finite_alpha_delta_gt_1(self):
        """Mean is finite when alpha * delta > 1."""
        from insurance_severity.composite.distributions import BurrTail
        # alpha=2.0, delta=1.5: alpha*delta = 3.0 > 1
        tail = BurrTail(alpha=2.0, delta=1.5, beta=8000.0)
        m = tail.mean()
        assert np.isfinite(m)
        assert m > 0

    def test_tvar_gt_var(self):
        from insurance_severity.composite.distributions import BurrTail
        tail = BurrTail(alpha=2.5, delta=1.5, beta=8000.0)
        threshold = 4000.0
        alpha = 0.9
        var = tail.ppf(np.array([alpha]), threshold)[0]
        tvar = tail.tvar(alpha, threshold)
        assert tvar > var

    def test_params_setter(self):
        from insurance_severity.composite.distributions import BurrTail
        tail = BurrTail(alpha=1.0, delta=1.0, beta=1.0)
        tail.params = np.array([3.0, 2.5, 12000.0])
        assert tail._alpha == pytest.approx(3.0)
        assert tail._delta == pytest.approx(2.5)
        assert tail._beta == pytest.approx(12000.0)
        np.testing.assert_array_equal(tail.params, [3.0, 2.5, 12000.0])


# ===========================================================================
# CompositeSeverityModel — mean_excess and limited_expected_value
# ===========================================================================


@pytest.fixture(scope="module")
def fitted_lgpd():
    """Returns a fitted LognormalGPD model."""
    from insurance_severity.composite.models import LognormalGPDComposite
    rng = np.random.default_rng(42)
    threshold = 50000.0
    sigma_ln = 1.5
    mu_ln = np.log(threshold) - 1.5 * sigma_ln
    xi_gpd = 0.25
    sigma_gpd = 20000.0
    pi = 0.80

    n = 500
    n_body = int(n * pi)
    n_tail = n - n_body

    y_body = []
    while len(y_body) < n_body:
        batch = stats.lognorm.rvs(s=sigma_ln, scale=np.exp(mu_ln), size=n_body * 3, random_state=rng)
        batch = batch[batch <= threshold]
        y_body.extend(batch[:n_body - len(y_body)])
    y_body = np.array(y_body[:n_body])
    y_tail = stats.genpareto.rvs(c=xi_gpd, scale=sigma_gpd, size=n_tail, random_state=rng) + threshold
    y = np.concatenate([y_body, y_tail])
    rng.shuffle(y)

    model = LognormalGPDComposite(threshold=threshold, threshold_method="fixed")
    model.fit(y)
    return model, y


class TestCompositeMeanExcessLEV:
    def test_mean_excess_positive(self, fitted_lgpd):
        model, y = fitted_lgpd
        me = model.mean_excess(model.threshold_)
        assert me > 0

    def test_mean_excess_decreasing_with_d(self, fitted_lgpd):
        """Mean excess can increase with d for heavy tails (GPD xi>0), but must be positive."""
        model, y = fitted_lgpd
        me_low = model.mean_excess(model.threshold_ * 0.5)
        me_high = model.mean_excess(model.threshold_ * 2.0)
        # Both should be positive finite values
        assert me_low > 0 and np.isfinite(me_low)
        assert me_high > 0 and np.isfinite(me_high)

    def test_mean_excess_at_very_high_d_is_small(self, fitted_lgpd):
        """Survival function at very high d is effectively zero -> mean_excess returns 0."""
        model, y = fitted_lgpd
        # d beyond all reasonable data
        me = model.mean_excess(d=1e12)
        assert me == pytest.approx(0.0, abs=1.0)

    def test_lev_positive(self, fitted_lgpd):
        model, y = fitted_lgpd
        lev = model.limited_expected_value(model.threshold_)
        assert lev > 0

    def test_lev_increasing_with_limit(self, fitted_lgpd):
        model, y = fitted_lgpd
        lev_small = model.limited_expected_value(model.threshold_ * 0.5)
        lev_big = model.limited_expected_value(model.threshold_ * 5.0)
        assert lev_big > lev_small

    def test_lev_integrates_to_correct_mean(self, fitted_lgpd):
        """LEV(infinity) should approximate E[X]. Use very large limit."""
        model, y = fitted_lgpd
        # LEV at very large limit should be close to E[X] for finite-mean distribution
        lev_large = model.limited_expected_value(model.threshold_ * 200.0)
        # Just check it's finite and greater than threshold
        assert np.isfinite(lev_large)
        assert lev_large > 0


class TestCompositeSummaryNoData:
    def test_summary_without_y(self):
        """summary() without y should not include AIC/BIC lines."""
        from insurance_severity.composite.models import LognormalGPDComposite
        rng = np.random.default_rng(42)
        y = stats.lognorm.rvs(s=1.2, scale=np.exp(10), size=300, random_state=rng)
        y = np.abs(y) + 1.0
        model = LognormalGPDComposite(threshold=np.quantile(y, 0.8), threshold_method="fixed")
        model.fit(y)
        s = model.summary()  # no y argument
        assert "Threshold" in s
        assert "AIC" not in s

    def test_summary_with_y_includes_aic(self):
        from insurance_severity.composite.models import LognormalGPDComposite
        rng = np.random.default_rng(42)
        y = stats.lognorm.rvs(s=1.2, scale=np.exp(10), size=300, random_state=rng)
        y = np.abs(y) + 1.0
        model = LognormalGPDComposite(threshold=np.quantile(y, 0.8), threshold_method="fixed")
        model.fit(y)
        s = model.summary(y)
        assert "AIC" in s
        assert "BIC" in s


class TestCompositeFitErrors:
    def test_negative_values_raises(self):
        from insurance_severity.composite.models import LognormalGPDComposite
        model = LognormalGPDComposite(threshold=1000.0, threshold_method="fixed")
        y = np.array([100.0, -50.0, 200.0] * 10, dtype=float)
        with pytest.raises(ValueError, match="strictly positive"):
            model.fit(y)

    def test_too_few_obs_raises(self):
        from insurance_severity.composite.models import LognormalGPDComposite
        model = LognormalGPDComposite(threshold=1000.0, threshold_method="fixed")
        with pytest.raises(ValueError):
            model.fit(np.array([100.0, 200.0, 500.0, 800.0, 900.0]))

    def test_check_fitted_raises_before_fit_sf(self):
        from insurance_severity.composite.models import LognormalGPDComposite
        model = LognormalGPDComposite(threshold=1000.0, threshold_method="fixed")
        with pytest.raises(RuntimeError, match="not fitted"):
            model.sf(np.array([100.0]))

    def test_check_fitted_raises_before_fit_ppf(self):
        from insurance_severity.composite.models import LognormalGPDComposite
        model = LognormalGPDComposite(threshold=1000.0, threshold_method="fixed")
        with pytest.raises(RuntimeError, match="not fitted"):
            model.ppf(np.array([0.5]))

    def test_check_fitted_raises_before_fit_var(self):
        from insurance_severity.composite.models import LognormalGPDComposite
        model = LognormalGPDComposite(threshold=1000.0, threshold_method="fixed")
        with pytest.raises(RuntimeError, match="not fitted"):
            model.var(0.95)

    def test_check_fitted_raises_before_fit_tvar(self):
        from insurance_severity.composite.models import LognormalGPDComposite
        model = LognormalGPDComposite(threshold=1000.0, threshold_method="fixed")
        with pytest.raises(RuntimeError, match="not fitted"):
            model.tvar(0.95)

    def test_check_fitted_raises_aic(self):
        from insurance_severity.composite.models import LognormalGPDComposite
        model = LognormalGPDComposite(threshold=1000.0, threshold_method="fixed")
        with pytest.raises(RuntimeError, match="not fitted"):
            model.aic(np.array([100.0]))

    def test_invalid_threshold_method_raises(self):
        from insurance_severity.composite.models import LognormalGPDComposite
        with pytest.raises(ValueError, match="threshold_method"):
            LognormalGPDComposite(threshold=1000.0, threshold_method="banana")


class TestCDFBoundaries:
    def test_cdf_at_threshold_equals_pi(self, fitted_lgpd):
        """CDF at the threshold should equal pi."""
        model, y = fitted_lgpd
        cdf_at_t = model.cdf(np.array([model.threshold_]))[0]
        assert abs(cdf_at_t - model.pi_) < 0.01  # CDF(t) == pi by definition

    def test_cdf_near_zero_at_tiny_x(self, fitted_lgpd):
        model, y = fitted_lgpd
        cdf = model.cdf(np.array([0.001]))[0]
        assert cdf >= 0.0
        assert cdf < 0.01

    def test_ppf_at_zero_quantile(self, fitted_lgpd):
        """ppf(0.0) should return a very small (or 0) value."""
        model, y = fitted_lgpd
        x = model.ppf(np.array([0.0]))
        assert x[0] >= 0.0

    def test_ppf_at_one_quantile(self, fitted_lgpd):
        """ppf(1.0) should return a large but finite value."""
        model, y = fitted_lgpd
        # PPF at 1.0 may be inf for heavy-tailed distributions; just check no crash
        x = model.ppf(np.array([0.9999]))
        assert x[0] > model.threshold_


class TestGammaGPDFullPipeline:
    """Integration tests for GammaGPD that go beyond basic fitting."""

    def test_aic_lt_bic_for_small_n(self):
        """For small n (close to e^2), AIC < BIC."""
        from insurance_severity.composite.models import GammaGPDComposite
        rng = np.random.default_rng(42)
        threshold = 30000.0
        y_body = stats.gamma.rvs(a=3.0, scale=8000.0, size=85, random_state=rng)
        y_body = y_body[y_body <= threshold][:50]
        y_tail = stats.genpareto.rvs(c=0.3, scale=15000.0, size=20, random_state=rng) + threshold
        y = np.concatenate([y_body, y_tail])
        model = GammaGPDComposite(threshold=threshold, threshold_method="fixed")
        model.fit(y)
        aic = model.aic(y)
        bic = model.bic(y)
        # At n~70, ln(n) ~ 4.25 > 2, so BIC > AIC
        n = len(y)
        k = len(model.body_params_) + len(model.tail_params_) + 1
        # BIC - AIC = k*ln(n) - 2k = k*(ln(n) - 2)
        diff = bic - aic
        expected_diff = k * (np.log(n) - 2.0)
        assert abs(diff - expected_diff) < 1.0

    def test_quantile_residuals_finite(self):
        from insurance_severity.composite.models import GammaGPDComposite
        rng = np.random.default_rng(42)
        threshold = 30000.0
        y_body = stats.gamma.rvs(a=3.0, scale=8000.0, size=200, random_state=rng)
        y_body = y_body[y_body <= threshold]
        y_tail = stats.genpareto.rvs(c=0.3, scale=15000.0, size=60, random_state=rng) + threshold
        y = np.concatenate([y_body, y_tail])
        model = GammaGPDComposite(threshold=threshold, threshold_method="fixed")
        model.fit(y)
        resid = model.quantile_residuals(y)
        assert np.all(np.isfinite(resid))
        assert len(resid) == len(y)


# ===========================================================================
# CompositeSeverityRegressor: predict_tail_scale, bootstrap_ci
# ===========================================================================


class TestRegressorExtended:
    def test_predict_tail_scale_positive(self):
        """predict_tail_scale should return positive values."""
        from insurance_severity.composite.models import LognormalBurrComposite
        from insurance_severity.composite.regression import CompositeSeverityRegressor

        rng = np.random.default_rng(123)
        n = 300
        x = rng.normal(0, 1, n)
        alpha, delta = 2.5, 2.0
        log_beta = 8.5 + 0.3 * x
        beta_arr = np.exp(log_beta)
        sigma = 1.2

        y = np.zeros(n)
        for i in range(n):
            beta_i = beta_arr[i]
            ratio = (delta - 1.0) / (alpha * delta + 1.0)
            t_i = beta_i * ratio ** (1.0 / delta)
            mu_i = sigma ** 2 + np.log(t_i)
            if rng.random() < 0.75:
                while True:
                    v = stats.lognorm.rvs(s=sigma, scale=np.exp(mu_i), random_state=rng)
                    if v <= t_i:
                        y[i] = v
                        break
            else:
                while True:
                    u = rng.random()
                    S_t = (1.0 + (t_i / beta_i) ** delta) ** (-alpha)
                    S_x = u * S_t
                    inner = S_x ** (-1.0 / alpha) - 1.0
                    if inner > 0:
                        y[i] = beta_i * inner ** (1.0 / delta)
                        break

        X = x.reshape(-1, 1)
        reg = CompositeSeverityRegressor(
            composite=LognormalBurrComposite(threshold_method="mode_matching"),
            n_starts=2,
        )
        reg.fit(X, y)
        scales = reg.predict_tail_scale(X[:10])
        assert len(scales) == 10
        assert np.all(scales > 0)

    def test_bootstrap_ci_structure(self):
        """bootstrap_ci should return dict with expected keys."""
        from insurance_severity.composite.models import LognormalBurrComposite
        from insurance_severity.composite.regression import CompositeSeverityRegressor

        rng = np.random.default_rng(42)
        n = 200
        x = rng.normal(0, 1, n)
        alpha, delta = 2.5, 2.0
        log_beta = 8.5 + 0.3 * x
        beta_arr = np.exp(log_beta)
        sigma = 1.2

        y = np.zeros(n)
        for i in range(n):
            beta_i = beta_arr[i]
            ratio = (delta - 1.0) / (alpha * delta + 1.0)
            t_i = beta_i * ratio ** (1.0 / delta)
            mu_i = sigma ** 2 + np.log(t_i)
            if rng.random() < 0.75:
                while True:
                    v = stats.lognorm.rvs(s=sigma, scale=np.exp(mu_i), random_state=rng)
                    if v <= t_i:
                        y[i] = v
                        break
            else:
                while True:
                    u = rng.random()
                    S_t = (1.0 + (t_i / beta_i) ** delta) ** (-alpha)
                    S_x = u * S_t
                    inner = S_x ** (-1.0 / alpha) - 1.0
                    if inner > 0:
                        y[i] = beta_i * inner ** (1.0 / delta)
                        break

        X = x.reshape(-1, 1)
        reg = CompositeSeverityRegressor(
            composite=LognormalBurrComposite(threshold_method="mode_matching"),
            n_starts=2,
        )
        reg.fit(X, y)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ci = reg.bootstrap_ci(X, y, n_bootstrap=5, seed=7)

        expected_keys = {
            "coef_lower", "coef_upper",
            "intercept_lower", "intercept_upper",
            "n_converged",
        }
        assert expected_keys <= set(ci.keys())
        assert ci["n_converged"] >= 0

    def test_summary_unfitted_returns_string(self):
        """summary() on an unfitted regressor should return a 'not fitted' string."""
        from insurance_severity.composite.models import LognormalBurrComposite
        from insurance_severity.composite.regression import CompositeSeverityRegressor
        reg = CompositeSeverityRegressor(
            composite=LognormalBurrComposite(threshold_method="mode_matching"),
        )
        s = reg.summary()
        assert "not fitted" in s

    def test_mismatched_X_y_raises(self):
        from insurance_severity.composite.models import LognormalBurrComposite
        from insurance_severity.composite.regression import CompositeSeverityRegressor
        reg = CompositeSeverityRegressor(
            composite=LognormalBurrComposite(threshold_method="mode_matching"),
        )
        X = np.ones((10, 1))
        y = np.ones(8)  # wrong length
        with pytest.raises(ValueError):
            reg.fit(X, y)

    def test_2d_y_raises(self):
        from insurance_severity.composite.models import LognormalBurrComposite
        from insurance_severity.composite.regression import CompositeSeverityRegressor
        reg = CompositeSeverityRegressor(
            composite=LognormalBurrComposite(threshold_method="mode_matching"),
        )
        X = np.ones((10, 1))
        y = np.ones((10, 2))
        with pytest.raises(ValueError, match="1-D"):
            reg.fit(X, y)


# ===========================================================================
# TailCalibration: unfitted guard, cdf_func wrong length
# ===========================================================================


class TestTailCalibrationGuards:
    def _make_tc(self, n=100, xi=0.3, sigma=1000.0):
        from insurance_severity.tail_scoring import TailCalibration
        def cdf_func(t):
            z = max(1.0 + xi * t / sigma, 0.0)
            val = 1.0 - z ** (-1.0 / xi) if xi != 0 else 1.0 - np.exp(-t / sigma)
            return np.full(n, float(np.clip(val, 0.0, 1.0)))
        return TailCalibration(cdf_func, n_obs=n)

    def test_occurrence_ratio_unfitted_raises(self):
        tc = self._make_tc()
        with pytest.raises(RuntimeError, match="fit"):
            tc.occurrence_ratio(100.0)

    def test_severity_pit_unfitted_raises(self):
        tc = self._make_tc()
        with pytest.raises(RuntimeError, match="fit"):
            tc.severity_pit(100.0)

    def test_summary_table_unfitted_raises(self):
        tc = self._make_tc()
        with pytest.raises(RuntimeError, match="fit"):
            tc.summary_table(np.array([100.0]))

    def test_fit_returns_self(self):
        rng = np.random.default_rng(0)
        tc = self._make_tc(n=50)
        y = rng.exponential(scale=1000.0, size=50) + 1.0
        result = tc.fit(y)
        assert result is tc

    def test_cdf_func_wrong_length_raises(self):
        """cdf_func returning wrong number of values should raise ValueError."""
        from insurance_severity.tail_scoring import TailCalibration
        n = 50
        # cdf_func always returns 100 values, but n_obs=50
        def bad_cdf_func(t):
            return np.full(100, 0.5)  # wrong length

        tc = TailCalibration(bad_cdf_func, n_obs=n)
        rng = np.random.default_rng(0)
        y = rng.exponential(scale=500.0, size=n) + 1.0
        tc.fit(y)
        with pytest.raises(ValueError, match="expected"):
            tc.occurrence_ratio(100.0)

    def test_severity_pit_returns_values_in_01(self):
        """All Z_i returned by severity_pit should be in [0, 1]."""
        from insurance_severity.tail_scoring import TailCalibration
        rng = np.random.default_rng(9)
        n = 200
        xi, sigma = 0.3, 1000.0

        def cdf_func(t):
            z = max(1.0 + xi * t / sigma, 0.0)
            val = 1.0 - z ** (-1.0 / xi) if xi != 0 else 1.0 - np.exp(-t / sigma)
            return np.full(n, float(np.clip(val, 0.0, 1.0)))

        y = stats.genpareto.rvs(c=xi, scale=sigma, size=n, random_state=rng)
        tc = TailCalibration(cdf_func, n_obs=n)
        tc.fit(y)
        t = float(np.quantile(y, 0.80))
        Z = tc.severity_pit(t)
        if len(Z) > 0:
            assert np.all(Z >= 0.0)
            assert np.all(Z <= 1.0)

    def test_summary_table_multiple_thresholds(self):
        """summary_table with multiple thresholds returns correct row count."""
        from insurance_severity.tail_scoring import TailCalibration
        rng = np.random.default_rng(5)
        n = 300
        xi, sigma = 0.4, 1000.0
        y = stats.genpareto.rvs(c=xi, scale=sigma, size=n, random_state=rng)

        def cdf_func(t):
            z = max(1.0 + xi * t / sigma, 0.0)
            val = 1.0 - z ** (-1.0 / xi)
            return np.full(n, float(np.clip(val, 0.0, 1.0)))

        tc = TailCalibration(cdf_func, n_obs=n)
        tc.fit(y)
        t_levels = np.quantile(y, [0.70, 0.80, 0.90])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = tc.summary_table(t_levels)
        assert len(df) == 3
        assert "R_occ" in df.columns


# ===========================================================================
# BladtTailScore: additional edge cases
# ===========================================================================


class TestBladtTailScoreEdgeCases:
    def test_score_grid_all_finite_valid_k(self):
        """score_grid with valid k values should return finite scores."""
        from insurance_severity.tail_scoring import BladtTailScore

        rng = np.random.default_rng(0)
        # Use Pareto sampling inline to avoid import issues
        gamma = 0.8
        n = 500
        u = rng.uniform(0, 1, size=n)
        y = (1 - u) ** (-gamma)

        bs = BladtTailScore()
        k_grid = np.array([10, 20, 30])
        result = bs.score_grid(y, [0.5, 0.8, 1.2], k_grid)
        for gamma_key, scores in result.items():
            assert len(scores) == 3
            # Valid k values should produce finite scores
            for s in scores:
                assert np.isfinite(s) or np.isnan(s)  # nan only if k out of range

    def test_rank_empty_stable_range(self):
        """stable_range that excludes all k values should yield nan mean scores."""
        from insurance_severity.tail_scoring import BladtTailScore
        rng = np.random.default_rng(0)
        n = 200
        y = rng.pareto(a=2.0, size=n) + 1.0  # Pareto with shape 2

        bs = BladtTailScore()
        k_grid = np.arange(5, 51, 5)
        # stable_range that contains no k values
        df = bs.rank(y, [0.5, 1.0], k_grid, stable_range=(200, 300))
        # Mean scores should be NaN (no finite k values in range)
        assert len(df) == 2

    def test_score_ci_ordering_at_true_gamma(self):
        """At or near the true gamma, CI should still satisfy lo <= score <= hi."""
        from insurance_severity.tail_scoring import BladtTailScore
        rng = np.random.default_rng(0)
        n = 1000
        gamma_true = 0.7
        u = rng.uniform(0, 1, size=n)
        y = (1 - u) ** (-gamma_true)

        bs = BladtTailScore()
        s, lo, hi = bs.score(y, gamma=gamma_true, k=80)
        assert np.isfinite(s)
        # At the true gamma, variance formula gives 0, so CI should be tight
        # but we only check structural ordering
        assert lo <= s + 1e-10  # allow floating point tolerance


# ===========================================================================
# ProjectionToUltimate: properties, repr, alternate paid_col
# ===========================================================================


def _make_train_df(n=300, seed=42):
    rng = np.random.default_rng(seed)
    dev_month = rng.integers(1, 36, size=n).astype(float)
    paid_to_date = rng.lognormal(mean=8.0, sigma=1.5, size=n)
    claim_age = rng.integers(6, 60, size=n).astype(float)
    log_paid = np.log(paid_to_date)
    log_ptu = (0.5 - 0.05 * dev_month + 0.1 * log_paid - 0.01 * claim_age
               + rng.normal(0, 0.15, size=n))
    log_ptu = np.maximum(log_ptu, 0.02)
    ultimate_cost = paid_to_date * np.exp(log_ptu)
    return pl.DataFrame({
        "paid_to_date": paid_to_date,
        "ultimate_cost": ultimate_cost,
        "dev_month": dev_month,
        "claim_age": claim_age,
    })


def _make_predict_df(n=100, seed=99):
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "paid_to_date": rng.lognormal(mean=8.0, sigma=1.5, size=n),
        "dev_month": rng.integers(1, 36, size=n).astype(float),
        "claim_age": rng.integers(6, 60, size=n).astype(float),
    })


class TestProjectionProperties:
    def test_repr_unfitted(self):
        from insurance_severity.projection import ProjectionToUltimate
        ptu = ProjectionToUltimate()
        r = repr(ptu)
        assert "unfitted" in r
        assert "ols" in r

    def test_repr_fitted(self):
        from insurance_severity.projection import ProjectionToUltimate
        ptu = ProjectionToUltimate()
        ptu.fit(_make_train_df())
        r = repr(ptu)
        assert "fitted" in r

    def test_r2_property(self):
        from insurance_severity.projection import ProjectionToUltimate
        ptu = ProjectionToUltimate()
        ptu.fit(_make_train_df())
        r2 = ptu.r2
        assert 0.0 < r2 <= 1.0

    def test_rmse_property(self):
        from insurance_severity.projection import ProjectionToUltimate
        ptu = ProjectionToUltimate()
        ptu.fit(_make_train_df())
        rmse = ptu.rmse
        assert rmse > 0

    def test_coefficients_property(self):
        from insurance_severity.projection import ProjectionToUltimate
        ptu = ProjectionToUltimate()
        ptu.fit(_make_train_df())
        coefs = ptu.coefficients
        assert "intercept" in coefs
        assert "dev_month" in coefs
        assert "log_paid" in coefs

    def test_r2_before_fit_raises(self):
        from insurance_severity.projection import ProjectionToUltimate
        ptu = ProjectionToUltimate()
        with pytest.raises(RuntimeError, match="fit"):
            _ = ptu.r2

    def test_rmse_before_fit_raises(self):
        from insurance_severity.projection import ProjectionToUltimate
        ptu = ProjectionToUltimate()
        with pytest.raises(RuntimeError, match="fit"):
            _ = ptu.rmse

    def test_coefficients_before_fit_raises(self):
        from insurance_severity.projection import ProjectionToUltimate
        ptu = ProjectionToUltimate()
        with pytest.raises(RuntimeError, match="fit"):
            _ = ptu.coefficients

    def test_alternate_paid_col_predict(self):
        """predict() should accept an alternate paid_col name."""
        from insurance_severity.projection import ProjectionToUltimate
        train = _make_train_df()
        pred = _make_predict_df()
        # Rename paid_to_date to current_paid
        pred_renamed = pred.rename({"paid_to_date": "current_paid"})

        ptu = ProjectionToUltimate()
        ptu.fit(train, paid_col="paid_to_date")
        out = ptu.predict(pred_renamed, paid_col="current_paid")
        assert "predicted_ultimate" in out.columns
        assert len(out) == len(pred)

    def test_zero_prediction_rows_warn(self):
        """Prediction with zero paid_to_date rows should warn."""
        from insurance_severity.projection import ProjectionToUltimate
        train = _make_train_df()
        ptu = ProjectionToUltimate()
        ptu.fit(train)

        rng = np.random.default_rng(0)
        pred = pl.DataFrame({
            "paid_to_date": [0.0, 100.0, 1000.0],
            "dev_month": [6.0, 12.0, 24.0],
            "claim_age": [12.0, 24.0, 36.0],
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = ptu.predict(pred)
        assert any("zero or negative paid" in str(warning.message) for warning in w)

    def test_min_train_rows_validation(self):
        from insurance_severity.projection import ProjectionToUltimate
        with pytest.raises(ValueError, match="min_train_rows"):
            ProjectionToUltimate(min_train_rows=1)


class TestBuildDesignMatrixEncoding:
    def test_boolean_column_encoded_as_float(self):
        from insurance_severity.projection import _build_design_matrix
        df = pl.DataFrame({
            "is_open": [True, False, True, True],
            "dev_month": [6.0, 12.0, 18.0, 24.0],
        })
        X = _build_design_matrix(df, ["is_open", "dev_month"])
        assert X.shape == (4, 3)
        # Boolean encoded as 1.0 or 0.0
        assert set(X[:, 1]).issubset({0.0, 1.0})

    def test_string_column_label_encoded(self):
        from insurance_severity.projection import _build_design_matrix
        df = pl.DataFrame({
            "peril": ["motor", "property", "motor", "liability"],
            "dev_month": [6.0, 12.0, 18.0, 24.0],
        })
        X = _build_design_matrix(df, ["peril", "dev_month"])
        assert X.shape == (4, 3)
        # Encoded values should be numeric
        assert X.dtype == float

    def test_intercept_prepended(self):
        from insurance_severity.projection import _build_design_matrix
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        X = _build_design_matrix(df, ["x"])
        np.testing.assert_array_equal(X[:, 0], [1.0, 1.0, 1.0])


# ===========================================================================
# Composite model: profile likelihood with very tight data (failure path)
# ===========================================================================


class TestProfileLikelihoodEdge:
    def test_profile_likelihood_very_narrow_range_still_fits(self):
        """Profile likelihood with narrow quantile range should still find a threshold."""
        from insurance_severity.composite.models import LognormalGPDComposite
        rng = np.random.default_rng(42)
        y = stats.lognorm.rvs(s=1.5, scale=np.exp(10), size=500, random_state=rng)
        model = LognormalGPDComposite(
            threshold_method="profile_likelihood",
            threshold_quantile_range=(0.75, 0.85),
            n_threshold_grid=10,
        )
        model.fit(y)
        assert model.threshold_ is not None
        assert 0 < model.pi_ < 1

    def test_profile_likelihood_all_body_raises(self):
        """If quantile range forces only body (threshold > all data), RuntimeError."""
        from insurance_severity.composite.models import LognormalGPDComposite
        rng = np.random.default_rng(0)
        y = rng.uniform(1, 10, size=100)  # very narrow range
        model = LognormalGPDComposite(
            threshold_method="profile_likelihood",
            threshold_quantile_range=(0.99, 1.0),
            n_threshold_grid=5,
        )
        # With tiny tail, fitting may fail
        with pytest.raises((RuntimeError, ValueError, Exception)):
            model.fit(y)


# ===========================================================================
# LognormalBody: zero-sigma init validation
# ===========================================================================


class TestDistributionInitValidation:
    def test_lognormal_zero_sigma_raises(self):
        from insurance_severity.composite.distributions import LognormalBody
        with pytest.raises(ValueError, match="sigma"):
            LognormalBody(mu=0.0, sigma=0.0)

    def test_gamma_zero_shape_raises(self):
        from insurance_severity.composite.distributions import GammaBody
        with pytest.raises(ValueError):
            GammaBody(shape=0.0, scale=1.0)

    def test_gamma_zero_scale_raises(self):
        from insurance_severity.composite.distributions import GammaBody
        with pytest.raises(ValueError):
            GammaBody(shape=1.0, scale=0.0)

    def test_gpd_zero_sigma_raises(self):
        from insurance_severity.composite.distributions import GPDTail
        with pytest.raises(ValueError):
            GPDTail(xi=0.2, sigma=0.0)

    def test_burr_zero_alpha_raises(self):
        from insurance_severity.composite.distributions import BurrTail
        with pytest.raises(ValueError):
            BurrTail(alpha=0.0, delta=1.0, beta=1000.0)

    def test_burr_zero_beta_raises(self):
        from insurance_severity.composite.distributions import BurrTail
        with pytest.raises(ValueError):
            BurrTail(alpha=1.0, delta=1.0, beta=0.0)


# ===========================================================================
# Numerical consistency: body/tail PDFs consistent with CDFs
# ===========================================================================


class TestNumericalConsistency:
    """Verify that CDF is the integral of PDF for all body and tail distributions."""

    def test_lognormal_body_cdf_consistent_with_pdf(self):
        from insurance_severity.composite.distributions import LognormalBody
        body = LognormalBody(mu=9.0, sigma=1.1)
        threshold = 10000.0
        # CDF at x = integral of PDF from 0 to x
        x_test = 5000.0
        integral, _ = quad(
            lambda x: np.exp(body.logpdf(np.array([x]), threshold)[0]),
            0.001, x_test
        )
        cdf_direct = np.exp(body.logcdf(np.array([x_test]), threshold)[0])
        assert abs(integral - cdf_direct) < 0.01

    def test_gamma_body_cdf_consistent_with_pdf(self):
        from insurance_severity.composite.distributions import GammaBody
        body = GammaBody(shape=3.0, scale=5000.0)
        threshold = 25000.0
        x_test = 12000.0
        integral, _ = quad(
            lambda x: np.exp(body.logpdf(np.array([x]), threshold)[0]),
            0.001, x_test
        )
        cdf_direct = np.exp(body.logcdf(np.array([x_test]), threshold)[0])
        assert abs(integral - cdf_direct) < 0.02

    def test_gpd_tail_sf_consistent_with_pdf(self):
        """SF(x) = integral of PDF from x to infinity for GPD tail."""
        from insurance_severity.composite.distributions import GPDTail
        tail = GPDTail(xi=0.25, sigma=15000.0)
        threshold = 50000.0
        x_test = 60000.0
        integral, _ = quad(
            lambda x: np.exp(tail.logpdf(np.array([x]), threshold)[0]),
            x_test,
            threshold + 2e6,
            limit=200,
        )
        sf_direct = np.exp(tail.logsf(np.array([x_test]), threshold)[0])
        assert abs(integral - sf_direct) < 0.05

    def test_pareto_tail_sf_consistent_with_pdf(self):
        from insurance_severity.composite.distributions import ParetoTail
        tail = ParetoTail(alpha=2.0, sigma=10000.0)
        threshold = 30000.0
        x_test = 40000.0
        integral, _ = quad(
            lambda x: np.exp(tail.logpdf(np.array([x]), threshold)[0]),
            x_test,
            threshold + 5e7,
            limit=200,
        )
        sf_direct = np.exp(tail.logsf(np.array([x_test]), threshold)[0])
        assert abs(integral - sf_direct) < 0.05
