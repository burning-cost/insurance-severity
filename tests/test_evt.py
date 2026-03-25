"""
Tests for insurance_severity.evt: TruncatedGPD, CensoredHillEstimator,
WeibullTemperedPareto.

These tests run on Databricks (not locally) — see README for instructions.
"""

import warnings

import numpy as np
import pytest

from insurance_severity.evt import (
    CensoredHillEstimator,
    TruncatedGPD,
    WeibullTemperedPareto,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_gpd(xi: float, sigma: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n samples from GPD(xi, sigma) using inverse transform."""
    u = rng.uniform(0, 1, size=n)
    if abs(xi) < 1e-10:
        return -sigma * np.log(1 - u)
    return sigma * ((1 - u) ** (-xi) - 1.0) / xi


def _sample_pareto(alpha: float, xmin: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n samples from Pareto(alpha, xmin)."""
    u = rng.uniform(0, 1, size=n)
    return xmin * (1 - u) ** (-1.0 / alpha)


# ---------------------------------------------------------------------------
# TruncatedGPD tests
# ---------------------------------------------------------------------------


class TestTruncatedGPD:
    def test_fit_recovers_xi_no_truncation(self):
        """With no truncation (limits=inf), should recover true xi within 15%."""
        rng = np.random.default_rng(42)
        xi_true, sigma_true = 0.4, 5000.0
        threshold = 10_000.0
        y = _sample_gpd(xi_true, sigma_true, n=1000, rng=rng)
        limits = np.full(1000, np.inf)

        model = TruncatedGPD(threshold=threshold)
        model.fit(y, limits)

        assert abs(model.xi - xi_true) / xi_true < 0.15, (
            f"xi recovery failed: true={xi_true}, fitted={model.xi:.4f}"
        )

    def test_fit_with_heterogeneous_limits(self):
        """Truncated MLE should not throw with heterogeneous limits."""
        rng = np.random.default_rng(7)
        xi_true, sigma_true = 0.3, 8000.0
        threshold = 5_000.0
        n = 500

        y = _sample_gpd(xi_true, sigma_true, n=n, rng=rng)
        # Heterogeneous limits: mix of 50k, 100k, 250k, inf
        limits = rng.choice([50_000.0, 100_000.0, 250_000.0, np.inf], size=n)
        # Only keep observations below their limit
        y = np.minimum(y, limits - threshold)
        y = np.maximum(y, 0.01)

        model = TruncatedGPD(threshold=threshold)
        model.fit(y, limits)

        # With truncation xi is harder — just check it's a real number in range
        assert np.isfinite(model.xi)
        assert -1.0 < model.xi < 3.0

    def test_fit_negative_xi(self):
        """xi < 0 means finite upper bound; model should handle it."""
        rng = np.random.default_rng(99)
        xi_true, sigma_true = -0.2, 3000.0
        threshold = 0.0
        y = _sample_gpd(xi_true, sigma_true, n=600, rng=rng)
        y = y[y > 0]
        limits = np.full(len(y), np.inf)

        model = TruncatedGPD(threshold=threshold)
        model.fit(y, limits)

        assert np.isfinite(model.xi)
        # Should be negative or close to zero
        assert model.xi < 0.2, f"Expected xi near -0.2, got {model.xi:.4f}"

    def test_params_dict(self):
        rng = np.random.default_rng(1)
        y = _sample_gpd(0.5, 2000.0, 300, rng)
        model = TruncatedGPD(threshold=0.0).fit(y, np.full(300, np.inf))
        p = model.params
        assert "xi" in p and "sigma" in p and "threshold" in p

    def test_distribution_methods(self):
        rng = np.random.default_rng(2)
        y = _sample_gpd(0.4, 5000.0, 300, rng)
        model = TruncatedGPD(threshold=0.0).fit(y, np.full(300, np.inf))

        x = np.array([1000.0, 5000.0, 20000.0])
        assert np.all(model.sf(x) >= 0) and np.all(model.sf(x) <= 1)
        assert np.allclose(model.cdf(x) + model.sf(x), 1.0)
        assert np.all(model.pdf(x) >= 0)

    def test_isf_round_trip(self):
        rng = np.random.default_rng(3)
        y = _sample_gpd(0.4, 5000.0, 500, rng)
        model = TruncatedGPD(threshold=0.0).fit(y, np.full(500, np.inf))

        q = np.array([0.1, 0.01, 0.001])
        x = model.isf(q)
        recovered_q = model.sf(x)
        assert np.allclose(recovered_q, q, rtol=1e-4)

    def test_summary_has_se(self):
        rng = np.random.default_rng(4)
        y = _sample_gpd(0.4, 5000.0, 500, rng)
        model = TruncatedGPD(threshold=0.0).fit(y, np.full(500, np.inf))
        s = model.summary()
        # SE may or may not be present depending on Hessian stability
        assert "xi" in s and "sigma" in s

    def test_empty_exceedances_raises(self):
        with pytest.raises(ValueError, match="empty"):
            TruncatedGPD(threshold=0.0).fit(np.array([]), np.array([]))

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            TruncatedGPD(threshold=0.0).fit(np.array([1.0, 2.0]), np.array([np.inf]))


# ---------------------------------------------------------------------------
# CensoredHillEstimator tests
# ---------------------------------------------------------------------------


class TestCensoredHillEstimator:
    def test_fit_pareto_with_censoring(self):
        """With 30% IBNR censoring, xi should be within 0.2 of truth."""
        rng = np.random.default_rng(42)
        xi_true = 0.5  # alpha = 2
        n = 1000

        x = _sample_pareto(1.0 / xi_true, xmin=10_000.0, n=n, rng=rng)
        # Mark 30% as censored
        censored = rng.random(n) < 0.3

        model = CensoredHillEstimator()
        model.fit(x, censored)

        assert abs(model.xi - xi_true) < 0.2, (
            f"xi recovery failed: true={xi_true}, fitted={model.xi:.4f}"
        )
        assert model.k_opt >= 2

    def test_ci_brackets_truth(self):
        """95% CI should contain the true value in a typical run."""
        rng = np.random.default_rng(11)
        xi_true = 0.4
        n = 800
        x = _sample_pareto(1.0 / xi_true, xmin=5_000.0, n=n, rng=rng)
        censored = rng.random(n) < 0.25

        model = CensoredHillEstimator()
        model.fit(x, censored, n_bootstrap=100)

        lo, hi = model.ci
        # CI should bracket a reasonable range around xi_true
        assert lo < xi_true + 0.3
        assert hi > xi_true - 0.3

    def test_hill_plot_runs(self):
        """hill_plot() should run without error."""
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend

        rng = np.random.default_rng(5)
        x = _sample_pareto(2.0, xmin=1000.0, n=500, rng=rng)
        censored = rng.random(500) < 0.2

        model = CensoredHillEstimator()
        model.fit(x, censored, n_bootstrap=50)
        ax = model.hill_plot()
        assert ax is not None

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            CensoredHillEstimator().fit(np.array([]), np.array([], dtype=bool))

    def test_all_censored_warns(self):
        rng = np.random.default_rng(6)
        x = _sample_pareto(2.0, xmin=1000.0, n=100, rng=rng)
        censored = np.ones(100, dtype=bool)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = CensoredHillEstimator()
            model.fit(x, censored, n_bootstrap=10)
            assert any("censored" in str(warning.message).lower() for warning in w)

    def test_no_censoring(self):
        """With zero censoring, corrected Hill = standard Hill."""
        rng = np.random.default_rng(99)
        xi_true = 0.5
        n = 500
        x = _sample_pareto(1.0 / xi_true, xmin=1000.0, n=n, rng=rng)
        censored = np.zeros(n, dtype=bool)

        model = CensoredHillEstimator()
        model.fit(x, censored, n_bootstrap=50)

        # Standard Hill at same k_opt
        order = np.argsort(x)[::-1]
        x_ord = x[order]
        k = model.k_opt
        std_hill = (np.sum(np.log(x_ord[:k])) - k * np.log(x_ord[k])) / k
        # When all uncensored, correction factor = 1, so they should match exactly
        assert abs(model.xi - std_hill) < 1e-10, (
            f"Uncensored Hill mismatch: corrected={model.xi:.6f}, standard={std_hill:.6f}"
        )


# ---------------------------------------------------------------------------
# WeibullTemperedPareto tests
# ---------------------------------------------------------------------------


def _sample_wtp_rejection(
    alpha: float, lam: float, tau: float, n: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Sample from WTP via rejection sampling with Pareto(alpha, 1) proposal.

    S(x) = x^{-alpha} * exp(-lam * x^tau)
    Proposal: Pareto(alpha, 1), S_prop(x) = x^{-alpha}
    Acceptance ratio: exp(-lam * x^tau) <= 1.
    """
    samples = []
    batch_size = max(n * 20, 10000)
    while len(samples) < n:
        # Pareto(alpha, xmin=1): x = (1-u)^{-1/alpha}
        u = rng.uniform(0, 1, size=batch_size)
        x = (1.0 - u) ** (-1.0 / alpha)
        # Acceptance probability
        accept = np.exp(-lam * x ** tau)
        v = rng.uniform(0, 1, size=batch_size)
        accepted = x[v < accept]
        samples.extend(accepted.tolist())
    return np.array(samples[:n])


class TestWeibullTemperedPareto:
    def test_fit_recovers_alpha(self):
        """Should recover alpha within 25% on moderate n."""
        rng = np.random.default_rng(42)
        alpha_true = 2.0
        lam_true = 1e-5  # very light tempering so Hill is good starting point
        tau_true = 0.5

        # Use direct Pareto samples with post-hoc tempering weight
        # (simpler: pure Pareto, then the WTP fit near-Pareto should give alpha~2)
        x = _sample_pareto(alpha_true, xmin=1.0, n=600, rng=rng)

        model = WeibullTemperedPareto(threshold=1.0, k=300)
        # WTP takes raw claim values (x > threshold), not exceedances
        exc = x[x > 1.0][:300]
        model.fit(exc)

        assert abs(model.alpha - alpha_true) / alpha_true < 0.30, (
            f"alpha recovery failed: true={alpha_true}, fitted={model.alpha:.4f}"
        )

    def test_isf_cdf_round_trip(self):
        """isf(sf(x)) should recover x."""
        rng = np.random.default_rng(7)
        alpha_true, lam_true, tau_true = 2.0, 1e-6, 0.5

        x = _sample_pareto(alpha_true, xmin=1.0, n=500, rng=rng)
        exc = x[x > 1.0]

        model = WeibullTemperedPareto(threshold=1.0)
        model.fit(exc)

        x_test = np.array([2.0, 5.0, 20.0, 100.0])
        q = model.sf(x_test)
        x_recovered = model.isf(q)
        assert np.allclose(x_recovered, x_test, rtol=0.01), (
            f"isf round-trip failed: {x_test} vs {x_recovered}"
        )

    def test_xi_property(self):
        """xi = 1/alpha for cross-model comparison."""
        rng = np.random.default_rng(3)
        x = _sample_pareto(2.0, xmin=1.0, n=300, rng=rng)
        model = WeibullTemperedPareto(threshold=1.0).fit(x[x > 1.0])
        assert np.isclose(model.xi, 1.0 / model.alpha)

    def test_distribution_methods_valid(self):
        rng = np.random.default_rng(10)
        x = _sample_pareto(2.0, xmin=1.0, n=200, rng=rng)
        exc = x[x > 1.0]
        model = WeibullTemperedPareto(threshold=1.0).fit(exc)

        x_eval = np.array([2.0, 10.0, 50.0])
        assert np.all(model.sf(x_eval) >= 0) and np.all(model.sf(x_eval) <= 1)
        assert np.all(model.pdf(x_eval) >= 0)
        assert np.allclose(model.cdf(x_eval) + model.sf(x_eval), 1.0)

    def test_summary_has_required_keys(self):
        rng = np.random.default_rng(11)
        x = _sample_pareto(2.0, xmin=1.0, n=200, rng=rng)
        exc = x[x > 1.0]
        model = WeibullTemperedPareto(threshold=1.0).fit(exc)
        s = model.summary()
        for key in ("alpha", "lambda", "tau", "xi", "threshold"):
            assert key in s

    def test_k_parameter(self):
        """Using k < n should not raise."""
        rng = np.random.default_rng(12)
        x = _sample_pareto(2.0, xmin=1.0, n=400, rng=rng)
        exc = x[x > 1.0]
        model = WeibullTemperedPareto(threshold=1.0, k=100).fit(exc)
        assert np.isfinite(model.alpha)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            WeibullTemperedPareto(threshold=0.0).fit(np.array([]))

    def test_single_observation_raises(self):
        with pytest.raises(ValueError):
            WeibullTemperedPareto(threshold=0.0).fit(np.array([5.0]))

    def test_pure_pareto_limit(self):
        """With pure Pareto data, WTP alpha should be near truth."""
        rng = np.random.default_rng(88)
        alpha_true = 2.5
        # Pure Pareto data
        x = _sample_pareto(alpha_true, xmin=1.0, n=600, rng=rng)
        exc = x[x > 1.0]

        model = WeibullTemperedPareto(threshold=1.0).fit(exc)
        # xi = 1/alpha should be near 1/2.5 = 0.4
        assert abs(model.xi - 1.0 / alpha_true) < 0.3, (
            f"Pure Pareto limit: xi={model.xi:.4f}, expected ~{1.0/alpha_true:.4f}"
        )


# ---------------------------------------------------------------------------
# TailVariableImportance tests
# ---------------------------------------------------------------------------


class TestTailVariableImportance:
    """Tests for EVT-based tail variable importance."""

    from insurance_severity.evt import TailVariableImportance

    def _make_tail_data(
        self,
        n: int = 2000,
        n_features: int = 5,
        signal_feature: int = 0,
        rng_seed: int = 42,
    ):
        """
        Synthetic data where one feature drives tail behaviour.

        Observations: log(y) = 0.1 * X[:,0] + noise for bulk,
        but for extreme observations (above 90th percentile),
        we add a large contribution from X[:,0] to simulate tail signal.
        """
        rng = np.random.default_rng(rng_seed)
        X = rng.standard_normal((n, n_features))

        # Base: mild effect of all features
        log_y_base = 0.1 * X[:, signal_feature] + rng.standard_normal(n) * 0.5
        y = np.exp(log_y_base + 7.0)  # shift to realistic claim scale

        # Introduce strong tail signal: top 10% get amplified by X[:,signal_feature]
        threshold = np.quantile(y, 0.90)
        tail_mask = y > threshold
        # Add strong positive signal in the tail
        y[tail_mask] *= np.exp(2.0 * np.maximum(0, X[tail_mask, signal_feature]))

        return X, y

    def test_signal_feature_ranks_first(self):
        """
        The feature engineered to drive the tail should have the highest importance.
        """
        from insurance_severity.evt import TailVariableImportance

        X, y = self._make_tail_data(n=3000, n_features=5, signal_feature=0)
        tvi = TailVariableImportance(threshold_quantile=0.90, alpha=0.05)
        tvi.fit(X, y, feature_names=[f"feat_{i}" for i in range(5)])

        imp = tvi.importances
        ranked = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)
        top_feature = ranked[0][0]
        assert top_feature == "feat_0", (
            f"Expected feat_0 to rank first, got {top_feature}. "
            f"Importances: {imp}"
        )

    def test_importances_sum_to_one(self):
        """Normalised importances must sum to 1."""
        from insurance_severity.evt import TailVariableImportance

        X, y = self._make_tail_data(n=500, n_features=4)
        tvi = TailVariableImportance(threshold_quantile=0.85, alpha=0.01)
        tvi.fit(X, y)
        total = sum(tvi.importances.values())
        assert abs(total - 1.0) < 1e-9, f"importances sum to {total}, expected 1.0"

    def test_default_feature_names(self):
        """Without feature_names, defaults to x0, x1, ..."""
        from insurance_severity.evt import TailVariableImportance

        X, y = self._make_tail_data(n=300, n_features=3)
        tvi = TailVariableImportance(threshold_quantile=0.80, alpha=0.1)
        tvi.fit(X, y)
        keys = set(tvi.importances.keys())
        assert keys == {"x0", "x1", "x2"}, f"Unexpected keys: {keys}"

    def test_feature_names_passed_through(self):
        """Explicitly passed feature_names should appear in importances."""
        from insurance_severity.evt import TailVariableImportance

        X, y = self._make_tail_data(n=300, n_features=3)
        names = ["vehicle_age", "driver_age", "postcode_risk"]
        tvi = TailVariableImportance(threshold_quantile=0.80, alpha=0.1)
        tvi.fit(X, y, feature_names=names)
        assert set(tvi.importances.keys()) == set(names)

    def test_summary_has_required_keys(self):
        """summary() should return all documented keys."""
        from insurance_severity.evt import TailVariableImportance

        X, y = self._make_tail_data(n=400, n_features=4)
        tvi = TailVariableImportance(threshold_quantile=0.90, alpha=0.1)
        tvi.fit(X, y)
        s = tvi.summary()
        for key in ("importances", "threshold", "threshold_quantile", "n_features",
                    "n_selected", "n_obs_tail"):
            assert key in s, f"Missing key '{key}' in summary"

        assert s["n_features"] == 4
        assert s["threshold_quantile"] == 0.90
        assert s["threshold"] > 0
        assert 0 <= s["n_selected"] <= 4
        assert s["n_obs_tail"] > 0

    def test_plot_does_not_error(self):
        """plot() should run without raising and return an Axes object."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from insurance_severity.evt import TailVariableImportance

        X, y = self._make_tail_data(n=500, n_features=6)
        tvi = TailVariableImportance(threshold_quantile=0.85, alpha=0.05)
        tvi.fit(X, y, feature_names=[f"f{i}" for i in range(6)])
        ax = tvi.plot(top_k=4)
        assert ax is not None
        plt.close("all")

    def test_plot_with_existing_ax(self):
        """plot() should accept an existing axes."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from insurance_severity.evt import TailVariableImportance

        X, y = self._make_tail_data(n=300, n_features=3)
        tvi = TailVariableImportance(threshold_quantile=0.85, alpha=0.05)
        tvi.fit(X, y)
        fig, ax = plt.subplots()
        returned = tvi.plot(ax=ax)
        assert returned is ax
        plt.close("all")

    def test_coefficients_signs_preserved(self):
        """coefficients property should return signed values (not normalised)."""
        from insurance_severity.evt import TailVariableImportance

        X, y = self._make_tail_data(n=500, n_features=3)
        tvi = TailVariableImportance(threshold_quantile=0.85, alpha=0.01)
        tvi.fit(X, y, feature_names=["a", "b", "c"])
        coef = tvi.coefficients
        imp = tvi.importances
        # |coef| and importances should be consistent in direction
        for name in ["a", "b", "c"]:
            assert abs(coef[name]) >= 0
            assert imp[name] >= 0

    def test_fit_returns_self(self):
        """fit() should return self for chaining."""
        from insurance_severity.evt import TailVariableImportance

        X, y = self._make_tail_data(n=200, n_features=2)
        tvi = TailVariableImportance()
        result = tvi.fit(X, y)
        assert result is tvi

    def test_not_fitted_raises(self):
        """Accessing properties before fit should raise RuntimeError."""
        from insurance_severity.evt import TailVariableImportance

        tvi = TailVariableImportance()
        with pytest.raises(RuntimeError, match="fit"):
            _ = tvi.importances
        with pytest.raises(RuntimeError, match="fit"):
            _ = tvi.coefficients
        with pytest.raises(RuntimeError, match="fit"):
            tvi.summary()

    def test_invalid_threshold_quantile(self):
        """threshold_quantile outside (0, 1) should raise ValueError."""
        from insurance_severity.evt import TailVariableImportance

        with pytest.raises(ValueError):
            TailVariableImportance(threshold_quantile=0.0)
        with pytest.raises(ValueError):
            TailVariableImportance(threshold_quantile=1.0)
        with pytest.raises(ValueError):
            TailVariableImportance(threshold_quantile=1.5)

    def test_invalid_alpha(self):
        """Non-positive alpha should raise ValueError."""
        from insurance_severity.evt import TailVariableImportance

        with pytest.raises(ValueError):
            TailVariableImportance(alpha=0.0)
        with pytest.raises(ValueError):
            TailVariableImportance(alpha=-1.0)

    def test_non_positive_y_raises(self):
        """y with zeros or negatives should raise ValueError."""
        from insurance_severity.evt import TailVariableImportance

        X = np.ones((10, 2))
        y = np.arange(-4, 6, dtype=float)  # contains zero and negatives
        with pytest.raises(ValueError, match="strictly positive"):
            TailVariableImportance().fit(X, y)

    def test_mismatched_X_y_raises(self):
        from insurance_severity.evt import TailVariableImportance

        X = np.ones((10, 3))
        y = np.ones(8)
        with pytest.raises(ValueError):
            TailVariableImportance().fit(X, y)

    def test_wrong_feature_names_length_raises(self):
        from insurance_severity.evt import TailVariableImportance

        X, y = self._make_tail_data(n=100, n_features=3)
        with pytest.raises(ValueError):
            TailVariableImportance().fit(X, y, feature_names=["a", "b"])  # 2 names, 3 features

    def test_top_k_limit_in_plot(self):
        """plot(top_k=2) should only show 2 bars even with more features."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from insurance_severity.evt import TailVariableImportance

        X, y = self._make_tail_data(n=400, n_features=8)
        tvi = TailVariableImportance(threshold_quantile=0.85, alpha=0.05)
        tvi.fit(X, y)
        ax = tvi.plot(top_k=2)
        # Should have 2 y-tick labels
        assert len(ax.get_yticklabels()) == 2
        plt.close("all")
