"""
Extended test coverage for insurance-severity.

Targets gaps across:
- CMRSAllocator: internal LST functions, single-participant, tilting edge cases,
  lognormal budget balance, validation errors
- TruncatedGPD: properties before fit raise, cdf+sf=1, pdf>=0, sf monotone,
  xi near 0 (exponential limit), isf ordering, threshold stored
- CensoredHillEstimator: _hill_corrected static method edge cases, CI ordering,
  k_opt range, property guards before fit, xi finite
- WeibullTemperedPareto: negative input, sf at large/zero x, cdf monotone,
  parameter signs, isf scalar output
- TailVariableImportance: high alpha -> uniform importances, 1D X raises,
  coefficients keys match, n_obs_tail correctness, threshold = quantile
- ProjectionToUltimate: repr, r2/rmse/coefficients properties, alternate paid_col,
  PI ordering, design matrix boolean/string columns
- PopulationSamplingReserve: report-before-accident raises, callable inclusion,
  callable severity, valuation time default, chain-ladder ultimate identity
- WeibullInclusionModel: no-truncation case, prediction without covariates
- TailCalibration: unfitted raises, fit returns self
- BladtTailScore: score_grid finite, rank correct gamma in top-2
- pareto_qq: without axes, with explicit k
- CMRSAllocator math properties: exponential proportionality, gamma positive,
  larger mean -> larger share, symmetry
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest


# ===========================================================================
# CMRS: internal LST helper correctness
# ===========================================================================


class TestCMRSInternalLSTs:
    """Verify the per-distribution LST and derivative formulae at real arguments."""

    def test_lst_exponential_at_zero(self):
        """L_Exp(rate)(0) = 1."""
        from insurance_severity.cmrs import _lst_exponential
        val = _lst_exponential(complex(0.0, 0.0), rate=2.0)
        assert abs(val - 1.0) < 1e-12

    def test_lst_exponential_known_value(self):
        """L_Exp(rate=2)(t=1) = 2/(2+1) = 2/3."""
        from insurance_severity.cmrs import _lst_exponential
        val = _lst_exponential(complex(1.0, 0.0), rate=2.0)
        assert abs(val - 2.0 / 3.0) < 1e-12

    def test_lst_deriv_exponential_at_zero(self):
        """E[X exp(0)] = E[X] = 1/rate for exponential."""
        from insurance_severity.cmrs import _lst_deriv_exponential
        rate = 3.0
        val = _lst_deriv_exponential(complex(0.0, 0.0), rate=rate)
        assert abs(val - 1.0 / rate) < 1e-12

    def test_lst_gamma_at_zero(self):
        """L_Gamma(0) = 1 for any parameters."""
        from insurance_severity.cmrs import _lst_gamma
        val = _lst_gamma(complex(0.0, 0.0), alpha=3.0, beta=2.0)
        assert abs(val - 1.0) < 1e-12

    def test_lst_deriv_gamma_at_zero(self):
        """E[X] = alpha/beta for Gamma; lst_deriv at t=0 should equal this."""
        from insurance_severity.cmrs import _lst_deriv_gamma
        alpha, beta = 4.0, 2.0
        val = _lst_deriv_gamma(complex(0.0, 0.0), alpha=alpha, beta=beta)
        assert abs(val - alpha / beta) < 1e-12

    def test_euler_inversion_positive_x_required(self):
        """_euler_inversion raises ValueError for x <= 0."""
        from insurance_severity.cmrs import _euler_inversion
        with pytest.raises(ValueError, match="positive"):
            _euler_inversion(lambda s: 1.0 / (1.0 + s), x=0.0)
        with pytest.raises(ValueError, match="positive"):
            _euler_inversion(lambda s: 1.0 / (1.0 + s), x=-1.0)

    def test_euler_inversion_exponential_pdf(self):
        """
        For Exp(rate=1), L(s) = 1/(1+s). The PDF at x is exp(-x).
        Euler inversion should recover this to within 1%.
        """
        from insurance_severity.cmrs import _euler_inversion

        rate = 1.0
        lst_exp = lambda s: rate / (rate + s)

        for x in [0.5, 1.0, 2.0, 3.0]:
            approx = _euler_inversion(lst_exp, x=x, a=18.0, M=15)
            expected = np.exp(-rate * x)
            assert abs(approx - expected) / expected < 0.01, (
                f"Euler inversion failed at x={x}: got {approx:.6f}, expected {expected:.6f}"
            )


# ===========================================================================
# CMRS: additional allocator coverage
# ===========================================================================


class TestCMRSAdditionalCoverage:

    def test_tilt_threshold_invalid_zero(self):
        """tilt_threshold=0.0 should raise."""
        from insurance_severity.cmrs import CMRSAllocator
        with pytest.raises(ValueError, match="tilt_threshold"):
            CMRSAllocator(tilt_threshold=0.0)

    def test_tilt_threshold_invalid_one(self):
        """tilt_threshold=1.0 should raise."""
        from insurance_severity.cmrs import CMRSAllocator
        with pytest.raises(ValueError, match="tilt_threshold"):
            CMRSAllocator(tilt_threshold=1.0)

    def test_fit_gamma_mismatched_shape_raises(self):
        from insurance_severity.cmrs import CMRSAllocator
        alloc = CMRSAllocator(distribution="gamma")
        with pytest.raises(ValueError):
            alloc.fit_gamma(np.array([1.0, 2.0]), np.array([1.0]))

    def test_fit_gamma_negative_alpha_raises(self):
        from insurance_severity.cmrs import CMRSAllocator
        alloc = CMRSAllocator(distribution="gamma")
        with pytest.raises(ValueError, match="positive"):
            alloc.fit_gamma(np.array([-1.0, 2.0]), np.array([1.0, 1.0]))

    def test_fit_gamma_negative_beta_raises(self):
        from insurance_severity.cmrs import CMRSAllocator
        alloc = CMRSAllocator(distribution="gamma")
        with pytest.raises(ValueError, match="positive"):
            alloc.fit_gamma(np.array([1.0, 2.0]), np.array([1.0, -1.0]))

    def test_fit_lognormal_negative_sigma_raises(self):
        from insurance_severity.cmrs import CMRSAllocator
        alloc = CMRSAllocator(distribution="lognormal")
        with pytest.raises(ValueError, match="sigma"):
            alloc.fit_lognormal(np.array([0.0, 1.0]), np.array([-0.5, 0.5]))

    def test_fit_lognormal_mismatched_shape_raises(self):
        from insurance_severity.cmrs import CMRSAllocator
        alloc = CMRSAllocator(distribution="lognormal")
        with pytest.raises(ValueError):
            alloc.fit_lognormal(np.array([0.0, 1.0, 2.0]), np.array([0.5, 0.5]))

    def test_fit_exponential_2d_raises(self):
        from insurance_severity.cmrs import CMRSAllocator
        alloc = CMRSAllocator(distribution="exponential")
        with pytest.raises(ValueError):
            alloc.fit_exponential(np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_allocate_nonpositive_s_raises_array(self):
        from insurance_severity.cmrs import CMRSAllocator
        alloc = CMRSAllocator().fit_gamma(
            np.array([2.0, 3.0]), np.array([1.0, 1.0])
        )
        with pytest.raises(ValueError):
            alloc.allocate(np.array([5.0, -1.0]))

    def test_allocate_zero_s_raises(self):
        from insurance_severity.cmrs import CMRSAllocator
        alloc = CMRSAllocator().fit_gamma(
            np.array([2.0, 3.0]), np.array([1.0, 1.0])
        )
        with pytest.raises(ValueError):
            alloc.allocate(0.0)

    def test_single_participant_gamma(self):
        """Single-participant CMRS: h_1(s) = s (trivially)."""
        from insurance_severity.cmrs import CMRSAllocator
        alloc = CMRSAllocator(distribution="gamma")
        alloc.fit_gamma(np.array([3.0]), np.array([1.0]))
        h = alloc.allocate(5.0)
        assert h.shape == (1,)
        np.testing.assert_allclose(h[0], 5.0, rtol=1e-3)

    def test_single_participant_exponential(self):
        """Single-participant exponential: h_1(s) = s."""
        from insurance_severity.cmrs import CMRSAllocator
        alloc = CMRSAllocator(distribution="exponential")
        alloc.fit_exponential(np.array([2.0]))
        h = alloc.allocate(3.0)
        np.testing.assert_allclose(h[0], 3.0, rtol=1e-3)

    def test_summary_keys_all_present(self):
        """summary() dict has all documented keys."""
        from insurance_severity.cmrs import CMRSAllocator
        alloc = CMRSAllocator(distribution="exponential")
        alloc.fit_exponential(np.array([1.0, 1.0]))
        s = alloc.summary(4.0)
        for key in ("allocations", "aggregate_loss", "total", "tilted", "mean_s"):
            assert key in s

    def test_budget_check_total_near_s(self):
        """budget_check: the allocated total should be near s after normalisation."""
        from insurance_severity.cmrs import CMRSAllocator
        alloc = CMRSAllocator(distribution="gamma")
        alloc.fit_gamma(np.array([2.0, 3.0, 1.5]), np.array([1.0, 1.0, 1.0]))
        result = alloc.budget_check(8.0)
        np.testing.assert_allclose(result["allocations"].sum(), 8.0, rtol=1e-4)
        assert "relative_error" in result

    def test_mean_s_exponential(self):
        """_mean_s for exponential: sum of 1/rates."""
        from insurance_severity.cmrs import CMRSAllocator
        rates = np.array([1.0, 2.0, 4.0])
        alloc = CMRSAllocator(distribution="exponential")
        alloc.fit_exponential(rates)
        expected_mean = np.sum(1.0 / rates)
        assert abs(alloc._mean_s() - expected_mean) < 1e-12

    def test_mean_s_gamma(self):
        """_mean_s for gamma: sum of alpha/beta."""
        from insurance_severity.cmrs import CMRSAllocator
        alphas = np.array([2.0, 3.0])
        betas = np.array([1.0, 2.0])
        alloc = CMRSAllocator(distribution="gamma")
        alloc.fit_gamma(alphas, betas)
        expected = np.sum(alphas / betas)
        assert abs(alloc._mean_s() - expected) < 1e-12

    def test_mean_s_lognormal(self):
        """_mean_s for lognormal: sum of exp(mu + sigma^2/2)."""
        from insurance_severity.cmrs import CMRSAllocator
        mus = np.array([0.0, 0.5])
        sigmas = np.array([0.3, 0.4])
        alloc = CMRSAllocator(distribution="lognormal")
        alloc.fit_lognormal(mus, sigmas)
        expected = np.sum(np.exp(mus + 0.5 * sigmas ** 2))
        assert abs(alloc._mean_s() - expected) < 1e-10

    def test_aggregate_distribution_shape(self):
        """aggregate_distribution returns correct shape."""
        from insurance_severity.cmrs import CMRSAllocator
        alloc = CMRSAllocator(distribution="exponential")
        alloc.fit_exponential(np.array([1.0, 2.0]))
        grid = np.linspace(0.1, 4.0, 20)
        vals = alloc.aggregate_distribution(grid)
        assert vals.shape == (20,)

    def test_aggregate_distribution_zero_for_nonpositive(self):
        """aggregate_distribution returns 0 for non-positive grid points."""
        from insurance_severity.cmrs import CMRSAllocator
        alloc = CMRSAllocator(distribution="exponential")
        alloc.fit_exponential(np.array([1.0]))
        vals = alloc.aggregate_distribution(np.array([-1.0, 0.0, 1.0]))
        assert vals[0] == 0.0
        assert vals[1] == 0.0

    def test_allocate_quantile_shape_exponential(self):
        """allocate_quantile for 1D array of quantile levels returns correct shape."""
        from insurance_severity.cmrs import CMRSAllocator
        alloc = CMRSAllocator(distribution="exponential")
        alloc.fit_exponential(np.array([1.0, 2.0, 3.0]))
        q = np.array([0.25, 0.5, 0.75, 0.95])
        h = alloc.allocate_quantile(q)
        assert h.shape == (4, 3)

    def test_lognormal_budget_balance_array(self):
        """Lognormal allocations budget balance holds for array of s values."""
        from insurance_severity.cmrs import CMRSAllocator
        alloc = CMRSAllocator(distribution="lognormal", n_quad_points=50)
        alloc.fit_lognormal(np.array([0.0, 0.5]), np.array([0.3, 0.5]))
        s_vals = np.array([3.0, 5.0, 8.0])
        h = alloc.allocate(s_vals)
        for i, sv in enumerate(s_vals):
            np.testing.assert_allclose(h[i].sum(), sv, rtol=0.01)

    def test_exponential_equal_rates_symmetric(self):
        """
        For exponential participants with equal rates, h_i(s) = s/n for all i.
        This follows from symmetry: identical participants get identical shares.
        """
        from insurance_severity.cmrs import CMRSAllocator
        rate = 1.0
        n = 3
        rates = np.full(n, rate)

        alloc = CMRSAllocator(distribution="exponential")
        alloc.fit_exponential(rates)
        s = 9.0
        h = alloc.allocate(s)
        np.testing.assert_allclose(h, np.full(n, s / n), rtol=5e-3)

    def test_fit_changes_distribution_attribute(self):
        """fit_gamma() on an exponential-init allocator updates distribution attr."""
        from insurance_severity.cmrs import CMRSAllocator
        alloc = CMRSAllocator(distribution="exponential")
        alloc.fit_gamma(np.array([2.0]), np.array([1.0]))
        assert alloc.distribution == "gamma"

    def test_not_fitted_summary_raises(self):
        from insurance_severity.cmrs import CMRSAllocator
        with pytest.raises(RuntimeError, match="not fitted"):
            CMRSAllocator().summary(5.0)

    def test_not_fitted_budget_check_raises(self):
        from insurance_severity.cmrs import CMRSAllocator
        with pytest.raises(RuntimeError, match="not fitted"):
            CMRSAllocator().budget_check(5.0)

    def test_not_fitted_aggregate_distribution_raises(self):
        from insurance_severity.cmrs import CMRSAllocator
        with pytest.raises(RuntimeError, match="not fitted"):
            CMRSAllocator().aggregate_distribution(np.array([1.0]))

    def test_not_fitted_allocate_quantile_raises(self):
        from insurance_severity.cmrs import CMRSAllocator
        with pytest.raises(RuntimeError, match="not fitted"):
            CMRSAllocator().allocate_quantile(np.array([0.5]))


# ===========================================================================
# TruncatedGPD: additional edge cases
# ===========================================================================


class TestTruncatedGPDExtra:
    """Additional tests for TruncatedGPD edge cases and property behavior."""

    def _sample_gpd(self, xi, sigma, n, rng):
        u = rng.uniform(0, 1, size=n)
        if abs(xi) < 1e-10:
            return -sigma * np.log(1 - u)
        return sigma * ((1 - u) ** (-xi) - 1.0) / xi

    def test_xi_property_raises_before_fit(self):
        from insurance_severity.evt import TruncatedGPD
        model = TruncatedGPD(threshold=0.0)
        with pytest.raises(RuntimeError, match="fit"):
            _ = model.xi

    def test_sigma_property_raises_before_fit(self):
        from insurance_severity.evt import TruncatedGPD
        model = TruncatedGPD(threshold=0.0)
        with pytest.raises(RuntimeError, match="fit"):
            _ = model.sigma

    def test_params_raises_before_fit(self):
        from insurance_severity.evt import TruncatedGPD
        model = TruncatedGPD(threshold=0.0)
        with pytest.raises(RuntimeError):
            _ = model.params

    def test_cdf_plus_sf_equals_one(self):
        """cdf(x) + sf(x) == 1 everywhere."""
        from insurance_severity.evt import TruncatedGPD
        rng = np.random.default_rng(101)
        y = self._sample_gpd(0.4, 2000.0, 400, rng)
        model = TruncatedGPD(threshold=0.0)
        model.fit(y, np.full(len(y), np.inf))
        x_test = np.array([100.0, 500.0, 2000.0, 10000.0])
        np.testing.assert_allclose(
            model.cdf(x_test) + model.sf(x_test), np.ones(4), atol=1e-10
        )

    def test_pdf_nonnegative(self):
        from insurance_severity.evt import TruncatedGPD
        rng = np.random.default_rng(102)
        y = self._sample_gpd(0.5, 5000.0, 300, rng)
        model = TruncatedGPD(threshold=0.0).fit(y, np.full(len(y), np.inf))
        x = np.linspace(1.0, 50000.0, 50)
        assert np.all(model.pdf(x) >= 0)

    def test_sf_monotone_decreasing(self):
        """sf(x) is non-increasing in x."""
        from insurance_severity.evt import TruncatedGPD
        rng = np.random.default_rng(103)
        y = self._sample_gpd(0.3, 3000.0, 400, rng)
        model = TruncatedGPD(threshold=0.0).fit(y, np.full(len(y), np.inf))
        rng2 = np.random.default_rng(1030)
        x = np.sort(np.abs(rng2.normal(0, 10000, 20)))
        x = x[x > 0]
        sf_vals = model.sf(x)
        diffs = np.diff(sf_vals)
        assert np.all(diffs <= 1e-10)

    def test_exponential_limit_xi_near_zero(self):
        """xi near 0 triggers exponential approximation path in _log_lik."""
        from insurance_severity.evt import TruncatedGPD
        model = TruncatedGPD(threshold=0.0)
        rng = np.random.default_rng(104)
        sigma = 5000.0
        y = -sigma * np.log(rng.uniform(0, 1, size=300))
        model.fit(y, np.full(len(y), np.inf))
        assert abs(model.xi) < 0.5
        assert model.sigma > 0

    def test_isf_ordering(self):
        """isf should return larger values for smaller q (more extreme quantiles)."""
        from insurance_severity.evt import TruncatedGPD
        rng = np.random.default_rng(105)
        y = self._sample_gpd(0.5, 2000.0, 500, rng)
        model = TruncatedGPD(threshold=0.0).fit(y, np.full(len(y), np.inf))
        q = np.array([0.5, 0.1, 0.01])
        x = model.isf(q)
        # isf is anti-monotone in q
        assert x[0] < x[1] < x[2]

    def test_threshold_stored(self):
        from insurance_severity.evt import TruncatedGPD
        model = TruncatedGPD(threshold=10_000.0)
        assert model.threshold == 10_000.0

    def test_fit_returns_self(self):
        from insurance_severity.evt import TruncatedGPD
        rng = np.random.default_rng(106)
        y = self._sample_gpd(0.4, 2000.0, 200, rng)
        model = TruncatedGPD(threshold=0.0)
        result = model.fit(y, np.full(len(y), np.inf))
        assert result is model


# ===========================================================================
# CensoredHillEstimator: additional edge cases
# ===========================================================================


class TestCensoredHillExtra:

    def _sample_pareto(self, alpha, xmin, n, rng):
        u = rng.uniform(0, 1, size=n)
        return xmin * (1 - u) ** (-1.0 / alpha)

    def test_hill_corrected_static_nan_at_small_k(self):
        """_hill_corrected should return nan for k < 2."""
        from insurance_severity.evt import CensoredHillEstimator
        x = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        c = np.zeros(5, dtype=bool)
        assert np.isnan(CensoredHillEstimator._hill_corrected(x, c, k=1))

    def test_hill_corrected_static_nan_at_k_equals_n(self):
        """_hill_corrected returns nan when k >= len(x)."""
        from insurance_severity.evt import CensoredHillEstimator
        x = np.array([10.0, 8.0, 6.0])
        c = np.zeros(3, dtype=bool)
        assert np.isnan(CensoredHillEstimator._hill_corrected(x, c, k=3))

    def test_hill_corrected_static_all_censored_top_k(self):
        """_hill_corrected returns nan when all top-k are censored."""
        from insurance_severity.evt import CensoredHillEstimator
        x = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        c = np.array([True, True, False, False, False])
        result = CensoredHillEstimator._hill_corrected(x, c, k=2)
        assert np.isnan(result)

    def test_fit_returns_self(self):
        from insurance_severity.evt import CensoredHillEstimator
        rng = np.random.default_rng(200)
        x = self._sample_pareto(2.0, 1000.0, 200, rng)
        c = np.zeros(200, dtype=bool)
        model = CensoredHillEstimator()
        result = model.fit(x, c, n_bootstrap=20)
        assert result is model

    def test_ci_is_ordered(self):
        """ci lower <= xi <= ci upper."""
        from insurance_severity.evt import CensoredHillEstimator
        rng = np.random.default_rng(201)
        x = self._sample_pareto(2.0, 1000.0, 300, rng)
        c = rng.random(300) < 0.2
        model = CensoredHillEstimator()
        model.fit(x, c, n_bootstrap=50)
        lo, hi = model.ci
        assert lo <= model.xi
        assert model.xi <= hi

    def test_k_opt_in_valid_range(self):
        """k_opt should be in [2, n-1]."""
        from insurance_severity.evt import CensoredHillEstimator
        rng = np.random.default_rng(202)
        n = 200
        x = self._sample_pareto(2.0, 1000.0, n, rng)
        c = np.zeros(n, dtype=bool)
        model = CensoredHillEstimator()
        model.fit(x, c, n_bootstrap=20)
        assert 2 <= model.k_opt <= n - 1

    def test_xi_property_raises_before_fit(self):
        from insurance_severity.evt import CensoredHillEstimator
        model = CensoredHillEstimator()
        with pytest.raises(RuntimeError, match="fit"):
            _ = model.xi

    def test_ci_property_raises_before_fit(self):
        from insurance_severity.evt import CensoredHillEstimator
        model = CensoredHillEstimator()
        with pytest.raises(RuntimeError, match="fit"):
            _ = model.ci

    def test_k_opt_property_raises_before_fit(self):
        from insurance_severity.evt import CensoredHillEstimator
        model = CensoredHillEstimator()
        with pytest.raises(RuntimeError, match="fit"):
            _ = model.k_opt

    def test_xi_finite(self):
        """Fitted xi should always be finite on well-behaved data."""
        from insurance_severity.evt import CensoredHillEstimator
        rng = np.random.default_rng(203)
        x = self._sample_pareto(3.0, 500.0, 400, rng)
        c = np.zeros(400, dtype=bool)
        model = CensoredHillEstimator()
        model.fit(x, c, n_bootstrap=20)
        assert np.isfinite(model.xi)


# ===========================================================================
# WeibullTemperedPareto: additional edge cases
# ===========================================================================


class TestWTPExtra:

    def _sample_pareto(self, alpha, xmin, n, rng):
        u = rng.uniform(0, 1, size=n)
        return xmin * (1 - u) ** (-1.0 / alpha)

    def test_negative_input_raises(self):
        from insurance_severity.evt import WeibullTemperedPareto
        with pytest.raises(ValueError, match="non-positive"):
            WeibullTemperedPareto(threshold=0.0).fit(np.array([1.0, -1.0, 2.0]))

    def test_sf_at_large_x_near_zero(self):
        """SF(x) -> 0 as x -> inf."""
        from insurance_severity.evt import WeibullTemperedPareto
        rng = np.random.default_rng(300)
        x = self._sample_pareto(2.0, 1.0, 300, rng)
        model = WeibullTemperedPareto(threshold=1.0).fit(x[x > 1.0])
        sf_large = model.sf(np.array([1e9]))
        assert sf_large[0] < 1e-3

    def test_sf_at_zero_is_one(self):
        """SF(x) = 1 for x <= 0 by convention."""
        from insurance_severity.evt import WeibullTemperedPareto
        rng = np.random.default_rng(301)
        x = self._sample_pareto(2.0, 1.0, 200, rng)
        model = WeibullTemperedPareto(threshold=1.0).fit(x[x > 1.0])
        assert model.sf(np.array([0.0]))[0] == 1.0
        assert model.sf(np.array([-5.0]))[0] == 1.0

    def test_cdf_monotone_increasing(self):
        """cdf(x) is non-decreasing."""
        from insurance_severity.evt import WeibullTemperedPareto
        rng = np.random.default_rng(302)
        x = self._sample_pareto(2.5, 1.0, 300, rng)
        model = WeibullTemperedPareto(threshold=1.0).fit(x[x > 1.0])
        rng2 = np.random.default_rng(3020)
        x_test = np.sort(rng2.uniform(1.0, 100.0, 20))
        cdf_vals = model.cdf(x_test)
        diffs = np.diff(cdf_vals)
        assert np.all(diffs >= -1e-10)

    def test_alpha_positive(self):
        """Fitted alpha must be positive."""
        from insurance_severity.evt import WeibullTemperedPareto
        rng = np.random.default_rng(303)
        x = self._sample_pareto(2.0, 1.0, 200, rng)
        model = WeibullTemperedPareto(threshold=1.0).fit(x[x > 1.0])
        assert model.alpha > 0

    def test_lam_nonnegative(self):
        """Fitted lambda must be non-negative."""
        from insurance_severity.evt import WeibullTemperedPareto
        rng = np.random.default_rng(304)
        x = self._sample_pareto(2.0, 1.0, 200, rng)
        model = WeibullTemperedPareto(threshold=1.0).fit(x[x > 1.0])
        assert model.lam >= 0

    def test_tau_positive(self):
        """Fitted tau must be positive."""
        from insurance_severity.evt import WeibullTemperedPareto
        rng = np.random.default_rng(305)
        x = self._sample_pareto(2.0, 1.0, 200, rng)
        model = WeibullTemperedPareto(threshold=1.0).fit(x[x > 1.0])
        assert model.tau > 0

    def test_isf_scalar_output(self):
        """isf on scalar probability returns a scalar-compatible value."""
        from insurance_severity.evt import WeibullTemperedPareto
        rng = np.random.default_rng(306)
        x = self._sample_pareto(2.0, 1.0, 200, rng)
        model = WeibullTemperedPareto(threshold=1.0).fit(x[x > 1.0])
        result = model.isf(np.array([0.1]))
        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_cdf_plus_sf_equals_one(self):
        """cdf(x) + sf(x) = 1."""
        from insurance_severity.evt import WeibullTemperedPareto
        rng = np.random.default_rng(307)
        x = self._sample_pareto(2.0, 1.0, 200, rng)
        model = WeibullTemperedPareto(threshold=1.0).fit(x[x > 1.0])
        x_test = np.array([2.0, 5.0, 20.0, 100.0])
        np.testing.assert_allclose(
            model.cdf(x_test) + model.sf(x_test), np.ones(4), atol=1e-10
        )


# ===========================================================================
# TailVariableImportance: additional coverage
# ===========================================================================


class TestTailVIExtra:

    def _make_data(self, n=500, p=5, rng_seed=42):
        rng = np.random.default_rng(rng_seed)
        X = rng.standard_normal((n, p))
        y = np.exp(rng.normal(7.0, 1.0, n))
        return X, y

    def test_high_alpha_uniform_importance(self):
        """With very high alpha, all coefficients zero -> uniform importances."""
        from insurance_severity.evt import TailVariableImportance
        X, y = self._make_data(n=500, p=4)
        tvi = TailVariableImportance(threshold_quantile=0.85, alpha=1000.0)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            tvi.fit(X, y)
        imp = tvi.importances
        expected_each = 1.0 / 4
        for name, val in imp.items():
            assert abs(val - expected_each) < 1e-9, (
                f"Expected uniform importance {expected_each}, got {val} for {name}"
            )

    def test_2d_X_required(self):
        """1D X should raise ValueError."""
        from insurance_severity.evt import TailVariableImportance
        rng = np.random.default_rng(400)
        y = np.exp(rng.normal(7, 1, 100))
        X = np.ones(100)
        with pytest.raises(ValueError, match="2-D"):
            TailVariableImportance().fit(X, y)

    def test_coefficients_keys_match_importances(self):
        """coefficients and importances should have the same feature keys."""
        from insurance_severity.evt import TailVariableImportance
        X, y = self._make_data(n=300, p=3)
        names = ["a", "b", "c"]
        tvi = TailVariableImportance(threshold_quantile=0.85, alpha=0.1)
        tvi.fit(X, y, feature_names=names)
        assert set(tvi.coefficients.keys()) == set(tvi.importances.keys()) == set(names)

    def test_n_obs_tail_correct(self):
        """n_obs_tail should equal number of observations above the threshold."""
        from insurance_severity.evt import TailVariableImportance
        X, y = self._make_data(n=400, p=3)
        q = 0.90
        tvi = TailVariableImportance(threshold_quantile=q, alpha=0.1)
        tvi.fit(X, y)
        threshold = np.quantile(y, q)
        expected_n_tail = int(np.sum(y > threshold))
        assert tvi.summary()["n_obs_tail"] == expected_n_tail

    def test_threshold_equals_quantile(self):
        """The stored threshold should match np.quantile(y, q)."""
        from insurance_severity.evt import TailVariableImportance
        X, y = self._make_data(n=300, p=2)
        q = 0.80
        tvi = TailVariableImportance(threshold_quantile=q, alpha=0.1)
        tvi.fit(X, y)
        expected = float(np.quantile(y, q))
        assert abs(tvi.summary()["threshold"] - expected) < 1e-10


# ===========================================================================
# ProjectionToUltimate: additional coverage
# ===========================================================================


class TestProjectionExtra:
    """Additional tests for ProjectionToUltimate."""

    def _make_train_df(self, n=300, seed=42):
        import polars as pl
        rng = np.random.default_rng(seed)
        paid = rng.lognormal(mean=8.0, sigma=1.5, size=n)
        dev_month = rng.integers(1, 36, size=n).astype(float)
        claim_age = rng.integers(6, 60, size=n).astype(float)
        log_ptu = np.maximum(
            0.5 - 0.05 * dev_month + 0.1 * np.log(paid) - 0.01 * claim_age
            + rng.normal(0, 0.15, n),
            0.02
        )
        return pl.DataFrame({
            "paid_to_date": paid,
            "ultimate_cost": paid * np.exp(log_ptu),
            "dev_month": dev_month,
            "claim_age": claim_age,
        })

    def _make_predict_df(self, n=50, seed=99):
        import polars as pl
        rng = np.random.default_rng(seed)
        return pl.DataFrame({
            "paid_to_date": rng.lognormal(8.0, 1.5, n),
            "dev_month": rng.integers(1, 36, n).astype(float),
            "claim_age": rng.integers(6, 60, n).astype(float),
        })

    def test_repr_unfitted(self):
        from insurance_severity.projection import ProjectionToUltimate
        ptu = ProjectionToUltimate()
        r = repr(ptu)
        assert "unfitted" in r
        assert "ols" in r

    def test_repr_fitted(self):
        from insurance_severity.projection import ProjectionToUltimate
        ptu = ProjectionToUltimate()
        ptu.fit(self._make_train_df())
        r = repr(ptu)
        assert "fitted" in r

    def test_r2_property(self):
        from insurance_severity.projection import ProjectionToUltimate
        ptu = ProjectionToUltimate()
        ptu.fit(self._make_train_df())
        assert 0.0 <= ptu.r2 <= 1.0

    def test_rmse_property(self):
        from insurance_severity.projection import ProjectionToUltimate
        ptu = ProjectionToUltimate()
        ptu.fit(self._make_train_df())
        assert ptu.rmse >= 0.0

    def test_coefficients_property_keys(self):
        from insurance_severity.projection import ProjectionToUltimate
        ptu = ProjectionToUltimate(development_features=["dev_month", "log_paid", "claim_age"])
        ptu.fit(self._make_train_df())
        coefs = ptu.coefficients
        assert "intercept" in coefs
        assert "dev_month" in coefs
        assert "log_paid" in coefs
        assert len(coefs) == 4  # intercept + 3

    def test_coefficients_raises_before_fit(self):
        from insurance_severity.projection import ProjectionToUltimate
        with pytest.raises(RuntimeError):
            _ = ProjectionToUltimate().coefficients

    def test_r2_raises_before_fit(self):
        from insurance_severity.projection import ProjectionToUltimate
        with pytest.raises(RuntimeError):
            _ = ProjectionToUltimate().r2

    def test_rmse_raises_before_fit(self):
        from insurance_severity.projection import ProjectionToUltimate
        with pytest.raises(RuntimeError):
            _ = ProjectionToUltimate().rmse

    def test_predict_alternate_paid_col(self):
        """predict() with explicit paid_col override should work."""
        import polars as pl
        from insurance_severity.projection import ProjectionToUltimate
        train = self._make_train_df(n=200)
        ptu = ProjectionToUltimate(
            development_features=["dev_month", "claim_age"],
            auto_add_log_paid=False,
        )
        ptu.fit(train, paid_col="paid_to_date")
        pred = self._make_predict_df(n=20).rename({"paid_to_date": "my_paid"})
        out = ptu.predict(pred, paid_col="my_paid", add_prediction_interval=False)
        assert "predicted_ultimate" in out.columns
        assert len(out) == 20

    def test_pi_lower_less_than_pi_upper(self):
        """Prediction interval: pi_lower < pi_upper for all rows."""
        from insurance_severity.projection import ProjectionToUltimate
        ptu = ProjectionToUltimate()
        ptu.fit(self._make_train_df())
        out = ptu.predict(self._make_predict_df(), pi_coverage=0.90)
        lowers = out["pi_lower"].to_numpy()
        uppers = out["pi_upper"].to_numpy()
        assert np.all(lowers < uppers)

    def test_min_train_rows_too_small_raises(self):
        from insurance_severity.projection import ProjectionToUltimate
        with pytest.raises(ValueError, match="min_train_rows"):
            ProjectionToUltimate(min_train_rows=1)

    def test_invalid_method_raises(self):
        from insurance_severity.projection import ProjectionToUltimate
        with pytest.raises(ValueError, match="method"):
            ProjectionToUltimate(method="gbm")

    def test_design_matrix_boolean_column(self):
        """_build_design_matrix should handle Boolean columns."""
        import polars as pl
        from insurance_severity.projection import _build_design_matrix
        df = pl.DataFrame({
            "feature_a": [1.0, 2.0, 3.0],
            "is_large": [True, False, True],
        })
        X = _build_design_matrix(df, ["feature_a", "is_large"])
        assert X.shape == (3, 3)
        assert set(X[:, 2].tolist()) <= {0.0, 1.0}

    def test_design_matrix_string_column(self):
        """_build_design_matrix should label-encode String columns."""
        import polars as pl
        from insurance_severity.projection import _build_design_matrix
        df = pl.DataFrame({
            "region": ["north", "south", "north"],
            "age": [1.0, 2.0, 3.0],
        })
        X = _build_design_matrix(df, ["region", "age"])
        assert X.shape == (3, 3)
        assert all(isinstance(v, (float, int)) for v in X[:, 1].tolist())

    def test_summary_ridge_alpha_present_for_ridge(self):
        """summary() ridge_alpha should be set when method='ridge'."""
        from insurance_severity.projection import ProjectionToUltimate
        ptu = ProjectionToUltimate(method="ridge", ridge_alpha=5.0)
        ptu.fit(self._make_train_df())
        s = ptu.summary()
        assert s["ridge_alpha"] == 5.0

    def test_summary_ridge_alpha_none_for_ols(self):
        """summary() ridge_alpha should be None when method='ols'."""
        from insurance_severity.projection import ProjectionToUltimate
        ptu = ProjectionToUltimate(method="ols")
        ptu.fit(self._make_train_df())
        s = ptu.summary()
        assert s["ridge_alpha"] is None


# ===========================================================================
# PopulationSamplingReserve: additional coverage
# ===========================================================================


class TestReservingExtra:

    def _make_df(self, n=100, seed=42):
        import pandas as pd
        rng = np.random.default_rng(seed)
        tau = 24.0
        # accident times well within [0, tau*0.5] so delays fit within tau
        t_acc = rng.uniform(0, tau * 0.5, n)
        delay = rng.exponential(scale=1.5, size=n)
        # Clip so that report_time <= tau - 0.01 (valuation_time check)
        t_rep = np.minimum(t_acc + delay, tau - 0.01)
        y = rng.lognormal(np.log(500), 0.5, n)
        return pd.DataFrame({
            "accident_time": t_acc,
            "report_time": t_rep,
            "severity": y,
        }), tau

    def test_report_before_accident_raises(self):
        """report_time < accident_time should raise."""
        import pandas as pd
        from insurance_severity.reserving import PopulationSamplingReserve
        df = pd.DataFrame({
            "accident_time": [5.0, 3.0],
            "report_time": [2.0, 4.0],  # first claim: report before accident
            "severity": [100.0, 200.0],
        })
        psr = PopulationSamplingReserve(method="ipw")
        pi_model = lambda tau_t, X: np.full(len(tau_t), 0.5)
        psr.inclusion_model = pi_model
        with pytest.raises(ValueError, match="accident_time"):
            psr.fit(df, valuation_time=10.0)

    def test_ipw_with_custom_callable_inclusion(self):
        """IPW with a plain callable inclusion model matches manual formula."""
        from insurance_severity.reserving import PopulationSamplingReserve
        df, tau = self._make_df(n=100)
        pi_val = 0.7
        pi_model = lambda tau_t, X: np.full(len(tau_t), pi_val)
        psr = PopulationSamplingReserve(inclusion_model=pi_model, method="ipw")
        psr.fit(df, valuation_time=tau)
        ibnr = psr.estimate_ibnr()
        expected = float(df["severity"].sum()) * (1.0 - pi_val) / pi_val
        assert abs(ibnr - expected) < 1e-6

    def test_aipw_callable_severity_model(self):
        """AIPW with a callable severity model should work without error."""
        from insurance_severity.reserving import PopulationSamplingReserve
        df, tau = self._make_df(n=80)
        y = df["severity"].to_numpy()
        pi_model = lambda tau_t, X: np.full(len(tau_t), 0.75)
        mean_y = float(np.mean(y))
        sev_model = lambda X: np.full(len(y), mean_y)
        psr = PopulationSamplingReserve(
            inclusion_model=pi_model, severity_model=sev_model, method="aipw"
        )
        psr.fit(df, valuation_time=tau)
        assert np.isfinite(psr.estimate_ibnr())

    def test_valuation_time_defaults_to_max_report_time(self):
        """When valuation_time=None, uses max(report_time)."""
        from insurance_severity.reserving import PopulationSamplingReserve
        df, tau = self._make_df(n=80)
        pi_model = lambda tau_t, X: np.full(len(tau_t), 0.8)
        psr = PopulationSamplingReserve(inclusion_model=pi_model, method="ipw")
        psr.fit(df)  # no valuation_time
        diag = psr.diagnostics()
        expected_tau = float(df["report_time"].max())
        assert abs(diag["valuation_time"] - expected_tau) < 1e-10

    def test_weibull_inclusion_model_no_covariates_prediction(self):
        """WeibullInclusionModel.predict_inclusion_prob without covariates."""
        from insurance_severity.reserving import WeibullInclusionModel
        rng = np.random.default_rng(400)
        # Generate delays that are guaranteed <= truncation_times
        trunc_val = 20.0
        delay_raw = rng.exponential(scale=5.0, size=500)
        delay = delay_raw[delay_raw < trunc_val][:200]  # keep only valid delays
        trunc = np.full(len(delay), trunc_val)
        model = WeibullInclusionModel(fit_covariates=False)
        model.fit(delay, truncation_times=trunc)
        pi = model.predict_inclusion_prob(np.array([5.0, 10.0, 20.0]))
        assert len(pi) == 3
        assert np.all(pi > 0)
        assert np.all(pi <= 1.0)
        assert pi[0] < pi[2]

    def test_weibull_no_truncation_correction(self):
        """Without truncation_times, model falls back to delay as truncation."""
        from insurance_severity.reserving import WeibullInclusionModel
        rng = np.random.default_rng(401)
        delay = rng.exponential(scale=3.0, size=100)
        model = WeibullInclusionModel(fit_covariates=False)
        model.fit(delay)
        assert model.shape_ is not None

    def test_chain_ladder_ultimate_equals_reported_plus_ibnr(self):
        """ultimate = reported + IBNR for chain-ladder method."""
        import pandas as pd
        from insurance_severity.reserving import PopulationSamplingReserve
        rows = []
        for period in [0, 1]:
            for _ in range(10):
                rows.append({
                    "accident_time": float(period),
                    "report_time": float(period) + 1.0,
                    "severity": 100.0,
                })
        df = pd.DataFrame(rows)
        dev_factors = {0: 2.0, 1: 3.0}
        psr = PopulationSamplingReserve(method="chain_ladder")
        psr.fit(df, development_factors=dev_factors, valuation_time=24.0)
        total_reported = float(df["severity"].sum())
        assert abs(psr.estimate_ultimate() - (total_reported + psr.estimate_ibnr())) < 1e-10

    def test_ibnr_positive_for_pi_less_than_one(self):
        """When pi < 1 for all claims, IBNR (IPW) should be positive."""
        from insurance_severity.reserving import PopulationSamplingReserve
        df, tau = self._make_df(n=60)
        pi_model = lambda tau_t, X: np.full(len(tau_t), 0.6)
        psr = PopulationSamplingReserve(inclusion_model=pi_model, method="ipw")
        psr.fit(df, valuation_time=tau)
        assert psr.estimate_ibnr() > 0


# ===========================================================================
# Tail scoring: additional coverage
# ===========================================================================


class TestTailScoringExtra:

    def _sample_pareto(self, gamma, n, rng):
        u = rng.uniform(0, 1, size=n)
        return (1 - u) ** (-gamma)

    def test_pareto_qq_no_axes(self):
        """pareto_qq without passing axes should create its own figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from insurance_severity.tail_scoring import pareto_qq
        rng = np.random.default_rng(500)
        y = self._sample_pareto(0.8, 1000, rng)
        r2 = pareto_qq(y, k=100)
        plt.close("all")
        assert 0.0 <= r2 <= 1.0

    def test_pareto_qq_explicit_k(self):
        """pareto_qq with explicit k should work."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from insurance_severity.tail_scoring import pareto_qq
        rng = np.random.default_rng(501)
        y = self._sample_pareto(1.0, 2000, rng)
        r2 = pareto_qq(y, k=50)
        plt.close("all")
        assert 0.0 <= r2 <= 1.0

    def test_bladt_score_grid_structure(self):
        """score_grid should return dict with correct structure."""
        from insurance_severity.tail_scoring import BladtTailScore
        rng = np.random.default_rng(502)
        y = self._sample_pareto(1.0, 500, rng)
        bs = BladtTailScore()
        k_grid = np.array([10, 20, 30, 40, 50])
        gammas = [0.5, 1.0, 2.0]
        grid = bs.score_grid(y, gammas, k_grid)
        assert set(grid.keys()) == set(gammas)
        for gamma in gammas:
            assert len(grid[gamma]) == len(k_grid)

    def test_bladt_rank_correct_gamma_in_top2(self):
        """With Pareto(gamma=1.0) data, gamma=1.0 should rank in top 2."""
        from insurance_severity.tail_scoring import BladtTailScore
        rng = np.random.default_rng(503)
        gamma_true = 1.0
        y = self._sample_pareto(gamma_true, 3000, rng)
        bs = BladtTailScore()
        gammas = [0.3, 0.7, 1.0, 1.5, 2.0]
        k_grid = np.arange(20, 151, 10)
        df = bs.rank(y, gammas, k_grid, stable_range=(20, 80))
        rank_true = int(df[df["gamma"] == gamma_true]["rank"].values[0])
        assert rank_true <= 2

    def test_tail_calibration_unfitted_occurrence_ratio_raises(self):
        """occurrence_ratio should raise if fit() not called."""
        from insurance_severity.tail_scoring import TailCalibration
        n = 100
        cdf_func = lambda t: np.full(n, 0.5)
        tc = TailCalibration(cdf_func, n_obs=n)
        with pytest.raises(RuntimeError, match="fit"):
            tc.occurrence_ratio(100.0)

    def test_tail_calibration_unfitted_severity_pit_raises(self):
        """severity_pit should raise if fit() not called."""
        from insurance_severity.tail_scoring import TailCalibration
        n = 100
        cdf_func = lambda t: np.full(n, 0.5)
        tc = TailCalibration(cdf_func, n_obs=n)
        with pytest.raises(RuntimeError, match="fit"):
            tc.severity_pit(100.0)

    def test_tail_calibration_fit_returns_self(self):
        """fit() should return self."""
        from insurance_severity.tail_scoring import TailCalibration
        rng = np.random.default_rng(504)
        n = 100
        y = rng.exponential(scale=500.0, size=n)
        cdf_func = lambda t: np.full(n, 0.5)
        tc = TailCalibration(cdf_func, n_obs=n)
        result = tc.fit(y)
        assert result is tc

    def test_bladt_score_returns_three_tuple(self):
        """score() always returns (score, ci_lower, ci_upper)."""
        from insurance_severity.tail_scoring import BladtTailScore
        rng = np.random.default_rng(505)
        y = self._sample_pareto(0.8, 300, rng)
        bs = BladtTailScore()
        result = bs.score(y, gamma=0.8, k=30)
        assert len(result) == 3

    def test_bladt_rank_df_sorted_by_rank(self):
        """rank() should return rows sorted by rank ascending."""
        from insurance_severity.tail_scoring import BladtTailScore
        rng = np.random.default_rng(506)
        y = self._sample_pareto(1.0, 500, rng)
        bs = BladtTailScore()
        df = bs.rank(y, [0.5, 1.0, 1.5], np.arange(10, 61, 10))
        assert list(df["rank"]) == sorted(df["rank"].tolist())


# ===========================================================================
# CMRS: mathematical properties
# ===========================================================================


class TestCMRSMathProperties:
    """Mathematical properties of CMRS allocations."""

    def test_exponential_equal_rates_each_gets_equal_share(self):
        """
        For n exponential participants with equal rates, h_i(s) = s/n for all i.
        Follows from symmetry: identical participants get identical shares.
        Tested at multiple s values to ensure it holds away from E[S].
        """
        from insurance_severity.cmrs import CMRSAllocator
        n = 4
        rate = 2.0
        rates = np.full(n, rate)
        alloc = CMRSAllocator(distribution="exponential")
        alloc.fit_exponential(rates)

        for s in [1.0, 3.0, 8.0]:
            h = alloc.allocate(s)
            np.testing.assert_allclose(h, np.full(n, s / n), rtol=5e-3,
                err_msg=f"Symmetry violated at s={s}")

    def test_gamma_allocation_positive(self):
        """All allocations h_i(s) should be non-negative."""
        from insurance_severity.cmrs import CMRSAllocator
        alloc = CMRSAllocator(distribution="gamma")
        alloc.fit_gamma(
            np.array([1.5, 2.0, 3.0, 1.0]),
            np.array([1.0, 1.0, 1.0, 1.0])
        )
        h = alloc.allocate(12.0)
        assert np.all(h >= 0)

    def test_larger_mean_gets_larger_share(self):
        """
        Participant with larger expected loss gets a larger share near E[S].
        """
        from insurance_severity.cmrs import CMRSAllocator
        # Participant 0 has mean 5, participant 1 has mean 1
        alloc = CMRSAllocator(distribution="exponential")
        alloc.fit_exponential(np.array([0.2, 1.0]))  # means: 5, 1
        s = alloc._mean_s()  # = 6
        h = alloc.allocate(s)
        assert h[0] > h[1], "Participant 0 (mean=5) should get more than participant 1 (mean=1)"

    def test_allocate_symmetry_three_equal_participants(self):
        """Three identical Gamma participants: each gets exactly s/3."""
        from insurance_severity.cmrs import CMRSAllocator
        alloc = CMRSAllocator(distribution="gamma")
        alloc.fit_gamma(
            np.array([2.0, 2.0, 2.0]),
            np.array([1.0, 1.0, 1.0])
        )
        s = 9.0
        h = alloc.allocate(s)
        np.testing.assert_allclose(h[0], h[1], rtol=1e-3)
        np.testing.assert_allclose(h[1], h[2], rtol=1e-3)
        np.testing.assert_allclose(h[0], s / 3, rtol=1e-3)

    def test_budget_balance_holds_across_s_values(self):
        """Budget balance h_1 + h_2 = s for a range of s values."""
        from insurance_severity.cmrs import CMRSAllocator
        alloc = CMRSAllocator(distribution="gamma")
        alloc.fit_gamma(np.array([2.0, 3.0]), np.array([1.0, 2.0]))
        for s in [1.0, 5.0, 10.0, 20.0]:
            h = alloc.allocate(s)
            np.testing.assert_allclose(h.sum(), s, rtol=1e-4)
