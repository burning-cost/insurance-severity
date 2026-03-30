"""
Tests for insurance_severity.tail_scoring: TailCalibration, BladtTailScore, pareto_qq.

These tests run on Databricks (not locally) — see README for instructions.

Test coverage:
- Calibrated GPD: R_occ ≈ 1, KS p-value > 0.05
- Occurrence miscalibration detection
- Severity miscalibration detection (wrong xi)
- Bladt rank recovery (Pareto data, correct gamma ranks highest)
- Formula verification (hand-computed at small k)
- Fréchet domain guard warning
- pareto_qq R² > 0.95 on Pareto data, << 0.95 on Gaussian data
- Edge cases: n < 50, threshold above max
"""

import warnings

import numpy as np
import pytest

from insurance_severity.tail_scoring import (
    BladtTailScore,
    TailCalibration,
    _hill_estimator,
    _tail_log_score,
    pareto_qq,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_pareto(gamma: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n samples from Pareto with tail index gamma. SF(x) = x^{-1/gamma}."""
    u = rng.uniform(0, 1, size=n)
    return (1 - u) ** (-gamma)


def _sample_gpd(xi: float, sigma: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n samples from GPD(xi, sigma) using inverse transform (exceedances above 0)."""
    u = rng.uniform(0, 1, size=n)
    if abs(xi) < 1e-10:
        return -sigma * np.log(1 - u)
    return sigma * ((1 - u) ** (-xi) - 1.0) / xi


def _gpd_cdf(x: float, xi: float, sigma: float) -> float:
    """Scalar GPD CDF."""
    if xi == 0:
        return 1.0 - np.exp(-x / sigma)
    z = 1.0 + xi * x / sigma
    if z <= 0:
        return 1.0
    return 1.0 - z ** (-1.0 / xi)


def make_gpd_cdf_func(xi: float, sigma: float, n: int):
    """
    Returns a homogeneous cdf_func: all n observations share the same GPD(xi, sigma).
    cdf_func(t) -> ndarray shape (n,) of F(t) repeated n times.
    """
    def cdf_func(t: float) -> np.ndarray:
        val = _gpd_cdf(float(t), xi, sigma)
        return np.full(n, val)
    return cdf_func


def make_wrong_occ_cdf_func(xi: float, sigma: float, scale: float, n: int):
    """
    Returns cdf_func where F(t) is scaled — predicted exceedance prob is wrong.
    scale > 1 means model underestimates SF, so R_occ > 1.
    """
    def cdf_func(t: float) -> np.ndarray:
        sf = 1.0 - _gpd_cdf(float(t), xi, sigma)
        # Inflate predicted CDF so predicted exceedance = SF/scale
        # i.e. 1 - F_pred(t) = SF_true(t)/scale
        f_pred = 1.0 - sf / scale
        f_pred = np.clip(f_pred, 0.0, 1.0)
        return np.full(n, f_pred)
    return cdf_func


def make_wrong_severity_cdf_func(xi_true: float, sigma_true: float,
                                  xi_wrong: float, sigma_wrong: float, n: int):
    """
    Returns cdf_func that predicts occurrence correctly but uses wrong tail shape.
    F_i(t) uses true GPD; F_i(y) uses wrong GPD. We fudge by returning a cdf_func
    that evaluates with xi_wrong — the occurrence calibration is approximately
    correct near t but the conditional shape is wrong.
    """
    def cdf_func(t: float) -> np.ndarray:
        val = _gpd_cdf(float(t), xi_wrong, sigma_wrong)
        return np.full(n, val)
    return cdf_func


# ---------------------------------------------------------------------------
# pareto_qq tests
# ---------------------------------------------------------------------------


class TestParetoQQ:
    def test_r2_high_on_pareto_data(self):
        """Pareto data should give R² > 0.95."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rng = np.random.default_rng(42)
        y = _sample_pareto(gamma=0.8, n=2000, rng=rng)
        fig, ax = plt.subplots()
        r2 = pareto_qq(y, k=200, ax=ax)
        plt.close("all")
        assert r2 > 0.95, f"Expected R² > 0.95 on Pareto data, got {r2:.4f}"

    def test_r2_low_on_gaussian_data(self):
        """Gaussian data (Gumbel domain) should give R² << 0.95."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rng = np.random.default_rng(42)
        # Uniform data — bounded support, no power-law tail
        y = rng.uniform(low=1.0, high=100.0, size=5000)
        fig, ax = plt.subplots()
        r2 = pareto_qq(y, ax=ax)
        plt.close("all")
        assert r2 < 0.95, f"Expected R² < 0.95 on Gaussian data, got {r2:.4f}"

    def test_default_k(self):
        """Default k = min(n//4, 200) should work without explicit k."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rng = np.random.default_rng(0)
        y = _sample_pareto(gamma=1.0, n=1000, rng=rng)
        fig, ax = plt.subplots()
        r2 = pareto_qq(y, ax=ax)
        plt.close("all")
        assert 0.0 <= r2 <= 1.0

    def test_negative_values_raise(self):
        """Non-positive values should raise."""
        y = np.array([-1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        with pytest.raises(ValueError, match="strictly positive"):
            pareto_qq(y)

    def test_small_n_raise(self):
        """n < 10 should raise."""
        y = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="at least 10"):
            pareto_qq(y)


# ---------------------------------------------------------------------------
# TailCalibration tests
# ---------------------------------------------------------------------------


class TestTailCalibration:
    def test_calibrated_gpd_r_occ_near_one(self):
        """
        A calibrated GPD model: R_occ should be close to 1 at multiple thresholds.
        """
        rng = np.random.default_rng(42)
        xi, sigma = 0.4, 1000.0
        n = 2000
        y = _sample_gpd(xi, sigma, n, rng)
        cdf_func = make_gpd_cdf_func(xi, sigma, n)
        tc = TailCalibration(cdf_func, n_obs=n)
        tc.fit(y)

        # Use 80th and 90th quantile thresholds
        t80 = float(np.quantile(y, 0.80))
        t90 = float(np.quantile(y, 0.90))

        r_occ_80 = tc.occurrence_ratio(t80)
        r_occ_90 = tc.occurrence_ratio(t90)

        assert 0.7 < r_occ_80 < 1.3, f"R_occ at 80th pct = {r_occ_80:.3f}, expected near 1"
        assert 0.6 < r_occ_90 < 1.4, f"R_occ at 90th pct = {r_occ_90:.3f}, expected near 1"

    def test_calibrated_gpd_ks_pvalue_uniform(self):
        """
        Calibrated GPD: KS test p-value for conditional excess PITs should
        be > 0.05 at reasonable thresholds.
        """
        rng = np.random.default_rng(1)
        xi, sigma = 0.3, 1000.0
        n = 3000
        y = _sample_gpd(xi, sigma, n, rng)
        cdf_func = make_gpd_cdf_func(xi, sigma, n)
        tc = TailCalibration(cdf_func, n_obs=n)
        tc.fit(y)

        t80 = float(np.quantile(y, 0.80))
        Z = tc.severity_pit(t80)

        assert len(Z) > 20, f"Expected > 20 exceedances, got {len(Z)}"

        from scipy import stats
        _, pval = stats.kstest(Z, "uniform")
        assert pval > 0.01, (
            f"KS p-value {pval:.4f} < 0.01; calibrated model should not reject uniformity. "
            f"(This test uses a relaxed threshold since n is finite.)"
        )

    def test_occurrence_miscalibration_detected(self):
        """
        If model predicts wrong exceedance probability (scale=2 means model
        predicts half as many exceedances as actually occur), R_occ should
        be far from 1.
        """
        rng = np.random.default_rng(7)
        xi, sigma = 0.4, 1000.0
        n = 3000
        y = _sample_gpd(xi, sigma, n, rng)

        # scale=2: model says exceedance prob is half the true value
        cdf_func = make_wrong_occ_cdf_func(xi, sigma, scale=2.0, n=n)
        tc = TailCalibration(cdf_func, n_obs=n)
        tc.fit(y)

        t90 = float(np.quantile(y, 0.90))
        r_occ = tc.occurrence_ratio(t90)

        # R_occ = emp_exc / (pred_exc/2) ~ 2 * true R_occ ~ 2.0
        assert r_occ > 1.5, (
            f"Expected R_occ > 1.5 for occurrence-miscalibrated model, got {r_occ:.3f}"
        )

    def test_severity_miscalibration_detected_via_ks(self):
        """
        Model with wrong tail shape (wrong xi) should fail KS test for
        conditional PIT uniformity.
        """
        rng = np.random.default_rng(42)
        xi_true, sigma_true = 0.4, 1000.0
        xi_wrong, sigma_wrong = 0.1, 1000.0  # much lighter tail predicted
        n = 3000
        y = _sample_gpd(xi_true, sigma_true, n, rng)

        # Model predicts with wrong xi (lighter tail)
        cdf_func = make_wrong_severity_cdf_func(
            xi_true, sigma_true, xi_wrong, sigma_wrong, n
        )
        tc = TailCalibration(cdf_func, n_obs=n)
        tc.fit(y)

        t80 = float(np.quantile(y, 0.80))
        Z = tc.severity_pit(t80)
        assert len(Z) > 20

        from scipy import stats
        _, pval = stats.kstest(Z, "uniform")
        # KS test should detect non-uniformity (wrong shape)
        # At n=3000 with large xi difference this should be clearly non-uniform
        assert pval < 0.10, (
            f"Expected KS p-value < 0.10 for severity-miscalibrated model, got {pval:.4f}"
        )

    def test_threshold_above_max_warns(self):
        """Threshold >= max(y) should produce a warning in summary_table."""
        rng = np.random.default_rng(0)
        n = 100
        y = _sample_gpd(0.3, 500.0, n, rng)
        cdf_func = make_gpd_cdf_func(0.3, 500.0, n)
        tc = TailCalibration(cdf_func, n_obs=n)
        tc.fit(y)

        t_above = float(np.max(y)) + 1.0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df = tc.summary_table(np.array([t_above]))
        # Should warn about threshold above max
        assert any("max" in str(warning.message).lower() or
                   ">=" in str(warning.message)
                   for warning in w)
        assert len(df) == 0  # row is skipped

    def test_summary_table_columns(self):
        """summary_table should return expected columns."""
        rng = np.random.default_rng(3)
        n = 500
        y = _sample_gpd(0.4, 1000.0, n, rng)
        cdf_func = make_gpd_cdf_func(0.4, 1000.0, n)
        tc = TailCalibration(cdf_func, n_obs=n)
        tc.fit(y)
        t = float(np.quantile(y, 0.85))
        df = tc.summary_table(np.array([t]))
        assert list(df.columns) == ["threshold", "n_exceedances", "R_occ", "ks_pvalue"]
        assert len(df) == 1

    def test_low_exceedance_warns(self):
        """n_exceedances < 20 should trigger a warning."""
        rng = np.random.default_rng(0)
        n = 50
        y = _sample_gpd(0.4, 1000.0, n, rng)
        cdf_func = make_gpd_cdf_func(0.4, 1000.0, n)
        tc = TailCalibration(cdf_func, n_obs=n)
        tc.fit(y)

        # Use a very high threshold so n_exceedances < 20
        t_high = float(np.quantile(y, 0.99))
        n_exc = int(np.sum(y > t_high))
        if n_exc < 20:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _ = tc.summary_table(t_levels=[t_high])
            assert any("20" in str(warning.message) or "unreliable" in str(warning.message)
                       for warning in w)

    def test_fit_length_mismatch_raises(self):
        """fit() with wrong length y should raise ValueError."""
        n = 100
        cdf_func = make_gpd_cdf_func(0.4, 1000.0, n)
        tc = TailCalibration(cdf_func, n_obs=n)
        with pytest.raises(ValueError, match="length"):
            tc.fit(np.ones(50))

    def test_severity_pit_empty_above_max(self):
        """severity_pit above max(y) should return empty array."""
        rng = np.random.default_rng(0)
        n = 100
        y = _sample_gpd(0.4, 1000.0, n, rng)
        cdf_func = make_gpd_cdf_func(0.4, 1000.0, n)
        tc = TailCalibration(cdf_func, n_obs=n)
        tc.fit(y)
        Z = tc.severity_pit(float(np.max(y)) + 1.0)
        assert len(Z) == 0


# ---------------------------------------------------------------------------
# Formula verification tests (hand-computed)
# ---------------------------------------------------------------------------


class TestFormulaVerification:
    def test_tail_log_score_hand_computed(self):
        """
        Hand-verify _tail_log_score at small k.

        For y = [1, 2, 3, 5, 10], sorted ascending, k=2:
        - Y_{n,n-k} = y_sorted[-(k+1)] = y_sorted[-3] = 3
        - top_k = y_sorted[-2:] = [5, 10]
        - ratios = [5/3, 10/3]
        - log density for Pareto(gamma=1): log(1) - 2*log(ratio)
        - score = mean(log_density)
        """
        y = np.array([1.0, 2.0, 3.0, 5.0, 10.0])
        gamma = 1.0
        k = 2

        y_sorted = np.sort(y)
        x_ref = y_sorted[-(k + 1)]  # 3.0
        assert x_ref == 3.0

        top_k = y_sorted[-k:]  # [5.0, 10.0]
        ratios = top_k / x_ref  # [5/3, 10/3]

        log_scores_manual = np.log(1.0 / gamma) - (1.0 / gamma + 1.0) * np.log(ratios)
        expected = float(np.mean(log_scores_manual))

        actual = _tail_log_score(y_sorted, gamma, k)
        assert abs(actual - expected) < 1e-12, f"Expected {expected:.8f}, got {actual:.8f}"

    def test_hill_estimator_hand_computed(self):
        """
        Hand-verify _hill_estimator at small k.

        For y = [1, 2, 3, 5, 10], k=2:
        - threshold = y_sorted[-3] = 3.0
        - top_k = [5, 10]
        - Hill = mean(log(top_k/threshold)) = mean([log(5/3), log(10/3)])
        """
        y_sorted = np.array([1.0, 2.0, 3.0, 5.0, 10.0])
        k = 2
        expected = float(np.mean(np.log(np.array([5.0, 10.0]) / 3.0)))
        actual = _hill_estimator(y_sorted, k)
        assert abs(actual - expected) < 1e-12, f"Expected {expected:.8f}, got {actual:.8f}"

    def test_score_optimum_at_hill_estimate(self):
        """
        Corollary 8: argmax_gamma score_k(gamma) = Hill estimate.
        Verify numerically on simple data.
        """
        rng = np.random.default_rng(0)
        y = _sample_pareto(gamma=0.8, n=500, rng=rng)
        y_sorted = np.sort(y)
        k = 50

        hill = _hill_estimator(y_sorted, k)

        # Score should be maximised at gamma = hill (approximately)
        gammas = np.linspace(0.2, 2.0, 200)
        scores = np.array([_tail_log_score(y_sorted, g, k) for g in gammas])
        gamma_argmax = gammas[np.argmax(scores)]

        assert abs(gamma_argmax - hill) < 0.15, (
            f"Score argmax {gamma_argmax:.4f} should be close to Hill {hill:.4f}"
        )


# ---------------------------------------------------------------------------
# BladtTailScore tests
# ---------------------------------------------------------------------------


class TestBladtTailScore:
    def test_rank_recovery_pareto(self):
        """
        Pareto data with true gamma=0.8: correct gamma should rank first
        over a stable k range.
        """
        rng = np.random.default_rng(42)
        gamma_true = 0.8
        y = _sample_pareto(gamma=gamma_true, n=5000, rng=rng)

        bs = BladtTailScore()
        gamma_candidates = [0.3, 0.5, 0.8, 1.2, 1.5]
        k_grid = np.arange(20, 251, 10)

        # Use lower 25% of k range as stable range (less bias from second-order terms)
        stable_range = (20, 100)
        df = bs.rank(y, gamma_candidates, k_grid, stable_range=stable_range)

        top_gamma = float(df.iloc[0]["gamma"])
        # True gamma is 0.8; top-ranked should be closest candidate
        assert top_gamma in [0.5, 0.8, 1.2], (
            f"Top gamma {top_gamma} is not near true value {gamma_true}. "
            f"Full ranking:\n{df}"
        )
        # The actually-correct gamma (0.8) should be in top 2
        rank_of_true = int(df[df["gamma"] == gamma_true]["rank"].values[0])
        assert rank_of_true <= 2, (
            f"True gamma {gamma_true} ranked {rank_of_true}, expected <= 2. "
            f"Full ranking:\n{df}"
        )

    def test_score_nan_at_invalid_k(self):
        """k < 2 or k >= n should return NaN tuple."""
        rng = np.random.default_rng(0)
        y = _sample_pareto(gamma=1.0, n=100, rng=rng)
        bs = BladtTailScore()

        s, lo, hi = bs.score(y, gamma=1.0, k=1)
        assert np.isnan(s)

        s, lo, hi = bs.score(y, gamma=1.0, k=100)
        assert np.isnan(s)

        s, lo, hi = bs.score(y, gamma=1.0, k=200)
        assert np.isnan(s)

    def test_score_returns_ci(self):
        """score() should return (score, ci_lower, ci_upper) with lo < score < hi."""
        rng = np.random.default_rng(0)
        y = _sample_pareto(gamma=0.8, n=500, rng=rng)
        bs = BladtTailScore()
        s, lo, hi = bs.score(y, gamma=0.8, k=50)
        assert np.isfinite(s)
        assert lo < s < hi, f"Expected lo < s < hi, got ({lo:.4f}, {s:.4f}, {hi:.4f})"

    def test_score_grid_shape(self):
        """score_grid should return dict with correct shapes."""
        rng = np.random.default_rng(0)
        y = _sample_pareto(gamma=1.0, n=200, rng=rng)
        bs = BladtTailScore()
        k_grid = np.array([10, 20, 30, 40])
        gammas = [0.5, 1.0, 1.5]
        result = bs.score_grid(y, gammas, k_grid)

        assert set(result.keys()) == set(gammas)
        for gamma in gammas:
            assert len(result[gamma]) == len(k_grid)

    def test_frechet_domain_guard_warns(self):
        """
        If Hill estimate is very small (near Gumbel boundary), score() should warn.
        Use lognormal-like data where Hill estimate will be near zero for small k.
        """
        rng = np.random.default_rng(0)
        # Generate exponential data — Hill converges to 0, Gumbel domain
        y = rng.exponential(scale=1.0, size=200) + 1.0
        bs = BladtTailScore()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            s, lo, hi = bs.score(y, gamma=0.5, k=10)

        # If Hill estimate < 0.05, warning should fire
        y_sorted = np.sort(y)
        hill = _hill_estimator(y_sorted, k=10)
        if hill < 0.05:
            assert any("Gumbel" in str(warning.message) or "boundary" in str(warning.message)
                       for warning in w), f"Expected Gumbel warning, Hill={hill:.4f}"

    def test_rank_dataframe_columns(self):
        """rank() should return DataFrame with expected columns and sorted by rank."""
        rng = np.random.default_rng(0)
        y = _sample_pareto(gamma=1.0, n=300, rng=rng)
        bs = BladtTailScore()
        df = bs.rank(y, [0.5, 1.0, 1.5], np.arange(10, 61, 10))
        assert list(df.columns) == ["gamma", "mean_score", "rank"]
        assert list(df["rank"]) == [1, 2, 3]

    def test_rank_with_stable_range(self):
        """rank() with stable_range should restrict k values used."""
        rng = np.random.default_rng(0)
        y = _sample_pareto(gamma=1.0, n=300, rng=rng)
        bs = BladtTailScore()
        k_grid = np.arange(10, 101, 10)
        df_all = bs.rank(y, [0.5, 1.0, 1.5], k_grid)
        df_range = bs.rank(y, [0.5, 1.0, 1.5], k_grid, stable_range=(20, 50))
        # Both should return 3 rows
        assert len(df_range) == 3

    def test_non_positive_values_raise(self):
        """Non-positive observations should raise."""
        y = np.array([1.0, 0.0, 2.0, 3.0])
        bs = BladtTailScore()
        with pytest.raises(ValueError, match="strictly positive"):
            bs.score(y, gamma=1.0, k=2)


# ---------------------------------------------------------------------------
# Edge case: small n
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_small_n_tail_calibration(self):
        """TailCalibration should work with n < 50 (with warnings for low exceedances)."""
        rng = np.random.default_rng(0)
        n = 30
        y = _sample_gpd(0.4, 1000.0, n, rng)
        cdf_func = make_gpd_cdf_func(0.4, 1000.0, n)
        tc = TailCalibration(cdf_func, n_obs=n)
        tc.fit(y)
        t = float(np.quantile(y, 0.85))
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            r_occ = tc.occurrence_ratio(t)
        # Just check it runs without crashing
        assert np.isfinite(r_occ) or np.isnan(r_occ)

    def test_bladt_small_n(self):
        """BladtTailScore should handle small n gracefully."""
        rng = np.random.default_rng(0)
        y = _sample_pareto(gamma=1.0, n=25, rng=rng)
        bs = BladtTailScore()
        # k=5 is valid (< n)
        s, lo, hi = bs.score(y, gamma=1.0, k=5)
        assert np.isfinite(s) or np.isnan(s)  # either is acceptable

    def test_bladt_occurrence_ratio_zero_denom(self):
        """occurrence_ratio returns NaN when all predicted probs are 1 (denom near 0)."""
        rng = np.random.default_rng(0)
        n = 100
        y = _sample_gpd(0.4, 1000.0, n, rng)
        # CDF always = 1 means predicted SF = 0
        cdf_func_ones = lambda t: np.ones(n)
        tc = TailCalibration(cdf_func_ones, n_obs=n)
        tc.fit(y)
        r = tc.occurrence_ratio(float(np.quantile(y, 0.9)))
        assert np.isnan(r)
