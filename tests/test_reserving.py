"""
Tests for insurance_severity.reserving — AIPW doubly-robust IBNR estimator.

Test strategy:
- Use fully synthetic data with known data-generating process so we can
  verify the double-robustness property analytically.
- Chain-ladder equivalence verified against manual CL calculation.
- All tests are pure numpy/scipy — no heavy computation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from insurance_severity.reserving import (
    PopulationSamplingReserve,
    WeibullInclusionModel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_claims_df(
    n_reported: int = 200,
    n_ibnr: int = 50,
    mean_severity: float = 1000.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, float]:
    """
    Generate synthetic reported claims with known IBNR true value.

    All claims have the same severity (constant) so the true IBNR is simply
    n_ibnr * mean_severity. We use constant π for simplicity.

    Returns (df_reported, true_ibnr).
    """
    rng = np.random.default_rng(seed)
    tau = 24.0  # valuation at month 24

    # Accident times uniform in [0, 24]
    t_acc_all = rng.uniform(0, tau, n_reported + n_ibnr)
    severity_all = rng.lognormal(mean=np.log(mean_severity), sigma=0.5, size=n_reported + n_ibnr)

    # Reporting delay ~ Exp(1/6) so mean delay = 6 months
    rate = 1.0 / 6.0
    delay_all = rng.exponential(scale=1.0 / rate, size=n_reported + n_ibnr)

    # Reported if delay <= tau - t_acc
    max_delay = tau - t_acc_all
    reported_mask = delay_all <= max_delay

    # We want exactly n_reported; take first n_reported True-flagged
    rep_idx = np.where(reported_mask)[0][:n_reported]

    t_acc_rep = t_acc_all[rep_idx]
    t_rep = t_acc_rep + delay_all[rep_idx]
    y_rep = severity_all[rep_idx]

    df = pd.DataFrame({
        "accident_time": t_acc_rep,
        "report_time": t_rep,
        "severity": y_rep,
    })

    true_ibnr = float(np.sum(severity_all[~reported_mask[:n_reported + n_ibnr]]))
    return df, true_ibnr


def make_simple_triangle_data() -> tuple[pd.DataFrame, dict, float]:
    """
    Two accident periods with known development factors.

    Period 0: 10 claims each worth 100 → reported loss = 1000 → f_k = 2.0
    Period 1: 10 claims each worth 100 → reported loss = 1000 → f_k = 4.0

    CL IBNR = 1000*(2-1) + 1000*(4-1) = 1000 + 3000 = 4000.
    """
    rng = np.random.default_rng(0)
    tau = 12.0

    rows = []
    for period in [0, 1]:
        for i in range(10):
            rows.append({
                "accident_time": float(period),
                "report_time": float(period) + rng.uniform(0.1, 2.0),
                "severity": 100.0,
            })

    df = pd.DataFrame(rows)
    dev_factors = {0: 2.0, 1: 4.0}
    true_ibnr_cl = 1000.0 * (2.0 - 1.0) + 1000.0 * (4.0 - 1.0)
    return df, dev_factors, true_ibnr_cl


# ---------------------------------------------------------------------------
# WeibullInclusionModel tests
# ---------------------------------------------------------------------------


class TestWeibullInclusionModel:

    def test_fit_returns_self(self):
        rng = np.random.default_rng(1)
        delay = rng.exponential(scale=5.0, size=100)
        trunc = delay + rng.uniform(0.5, 2.0, size=100)
        model = WeibullInclusionModel(fit_covariates=False)
        result = model.fit(delay, truncation_times=trunc)
        assert result is model

    def test_shape_and_intercept_fitted(self):
        rng = np.random.default_rng(2)
        delay = rng.weibull(a=1.5, size=300) * 5.0
        trunc = np.full(300, 20.0)  # generous truncation
        model = WeibullInclusionModel(fit_covariates=False)
        model.fit(delay, truncation_times=trunc)
        assert model.shape_ is not None
        assert model.intercept_ is not None
        assert model.shape_ > 0

    def test_predict_inclusion_prob_range(self):
        rng = np.random.default_rng(3)
        delay = rng.exponential(scale=4.0, size=200)
        trunc = np.full(200, 24.0)
        model = WeibullInclusionModel(fit_covariates=False)
        model.fit(delay, truncation_times=trunc)
        tau_minus_t = np.array([1.0, 6.0, 12.0, 24.0])
        pi = model.predict_inclusion_prob(tau_minus_t)
        assert pi.shape == (4,)
        assert np.all(pi > 0)
        assert np.all(pi <= 1.0)
        # Longer horizon → higher inclusion probability
        assert pi[0] < pi[1] < pi[2] < pi[3]

    def test_raises_if_not_fitted(self):
        model = WeibullInclusionModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict_inclusion_prob(np.array([1.0, 2.0]))

    def test_raises_delay_exceeds_truncation(self):
        delay = np.array([5.0, 3.0])
        trunc = np.array([4.0, 5.0])  # first claim violates
        model = WeibullInclusionModel(fit_covariates=False)
        with pytest.raises(ValueError, match="truncation_times"):
            model.fit(delay, truncation_times=trunc)

    def test_with_covariates(self):
        rng = np.random.default_rng(10)
        n = 300
        x = rng.normal(size=(n, 2))
        delay = rng.exponential(scale=np.exp(0.5 + 0.3 * x[:, 0]), size=n)
        trunc = np.full(n, 30.0)
        model = WeibullInclusionModel(fit_covariates=True)
        model.fit(delay, covariates=x, truncation_times=trunc)
        assert model.coef_ is not None
        assert model.coef_.shape == (2,)

        pi = model.predict_inclusion_prob(np.full(n, 15.0), covariates=x)
        assert np.all(pi > 0) and np.all(pi <= 1.0)


# ---------------------------------------------------------------------------
# PopulationSamplingReserve — basic API tests
# ---------------------------------------------------------------------------


class TestPopulationSamplingReserveAPI:

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="method must be one of"):
            PopulationSamplingReserve(method="bayesian")

    def test_fit_returns_self(self):
        df, _ = make_claims_df(n_reported=100, n_ibnr=20, seed=0)
        psr = PopulationSamplingReserve(method="aipw")
        result = psr.fit(df, valuation_time=24.0)
        assert result is psr

    def test_missing_column_raises(self):
        df = pd.DataFrame({"accident_time": [1.0], "severity": [100.0]})
        psr = PopulationSamplingReserve(method="ipw")
        with pytest.raises(ValueError, match="report_time"):
            psr.fit(df, valuation_time=24.0)

    def test_non_positive_severity_raises(self):
        df = pd.DataFrame({
            "accident_time": [1.0, 2.0],
            "report_time": [3.0, 4.0],
            "severity": [100.0, -50.0],
        })
        psr = PopulationSamplingReserve(method="aipw")
        with pytest.raises(ValueError, match="strictly positive"):
            psr.fit(df, valuation_time=10.0)

    def test_estimate_ibnr_raises_before_fit(self):
        psr = PopulationSamplingReserve()
        with pytest.raises(RuntimeError):
            psr.estimate_ibnr()

    def test_estimate_ultimate_raises_before_fit(self):
        psr = PopulationSamplingReserve()
        with pytest.raises(RuntimeError):
            psr.estimate_ultimate()

    def test_diagnostics_raises_before_fit(self):
        psr = PopulationSamplingReserve()
        with pytest.raises(RuntimeError):
            psr.diagnostics()

    def test_diagnostics_keys(self):
        df, _ = make_claims_df(n_reported=100, seed=1)
        psr = PopulationSamplingReserve(method="aipw")
        psr.fit(df, valuation_time=24.0)
        diag = psr.diagnostics()
        required_keys = {
            "ibnr_estimate",
            "ultimate_estimate",
            "n_reported",
            "n_ibnr_estimated",
            "augmentation_term",
            "weighted_balance_ratio",
            "method",
        }
        assert required_keys.issubset(set(diag.keys()))

    def test_ultimate_equals_reported_plus_ibnr(self):
        df, _ = make_claims_df(n_reported=150, seed=2)
        psr = PopulationSamplingReserve(method="aipw")
        psr.fit(df, valuation_time=24.0)
        reported = float(df["severity"].sum())
        assert abs(psr.estimate_ultimate() - (reported + psr.estimate_ibnr())) < 1e-6


# ---------------------------------------------------------------------------
# Chain-ladder equivalence
# ---------------------------------------------------------------------------


class TestChainLadderEquivalence:

    def test_cl_ibnr_exact(self):
        """
        With two accident periods, f_0=2 and f_1=4, each with 10 claims of
        100, CL IBNR = 1000*(2-1) + 1000*(4-1) = 4000.
        The IPW formula gives the same result.
        """
        df, dev_factors, expected_ibnr = make_simple_triangle_data()
        psr = PopulationSamplingReserve(method="chain_ladder")
        psr.fit(df, development_factors=dev_factors, valuation_time=24.0)

        assert abs(psr.estimate_ibnr() - expected_ibnr) < 1e-6

    def test_cl_matches_manual_ipw(self):
        """
        IBNR = Σ_i (1/π_i − 1) × Y_i = Σ_i (f_k − 1) × Y_i.
        Verify this equals the chain-ladder IBNR.
        """
        df, dev_factors, _ = make_simple_triangle_data()
        psr = PopulationSamplingReserve(method="chain_ladder")
        psr.fit(df, development_factors=dev_factors, valuation_time=24.0)

        # Manual IPW
        y = df["severity"].to_numpy()
        acc = df["accident_time"].to_numpy()
        pi = np.array([1.0 / dev_factors[int(a)] for a in acc])
        manual_ibnr = float(np.sum((1.0 - pi) / pi * y))
        assert abs(psr.estimate_ibnr() - manual_ibnr) < 1e-9

    def test_cl_requires_dev_factors(self):
        df, _, _ = make_simple_triangle_data()
        psr = PopulationSamplingReserve(method="chain_ladder")
        with pytest.raises(ValueError, match="development_factors"):
            psr.fit(df, valuation_time=24.0)

    def test_cl_augmentation_zero(self):
        """Chain-ladder has no augmentation term by construction."""
        df, dev_factors, _ = make_simple_triangle_data()
        psr = PopulationSamplingReserve(method="chain_ladder")
        psr.fit(df, development_factors=dev_factors, valuation_time=24.0)
        assert psr.diagnostics()["augmentation_term"] == 0.0


# ---------------------------------------------------------------------------
# Double-robustness property
# ---------------------------------------------------------------------------


class TestDoubleRobustness:

    def test_perfect_severity_model_zero_augmentation(self):
        """
        When the severity model is perfect (Ŷ_i = Y_i for all reported
        claims), the augmentation term should be exactly zero.
        """
        df, _ = make_claims_df(n_reported=200, seed=5)
        y = df["severity"].to_numpy()

        # Perfect model: returns true Y values
        perfect_model = lambda X: y  # noqa: E731

        psr = PopulationSamplingReserve(
            severity_model=perfect_model, method="aipw"
        )
        psr.fit(df, valuation_time=24.0)
        assert abs(psr.diagnostics()["augmentation_term"]) < 1e-6

    def test_correct_inclusion_probs_ipw_unbiased(self):
        """
        With known π̂_i = true π_i, the IPW estimator is unbiased.
        We verify that with constant severity and exact π̂ = 0.8,
        IPW IBNR = Σ (0.2/0.8) * Y_i = (n_reported * mean_Y) / 4.
        """
        rng = np.random.default_rng(7)
        n = 100
        pi_true = 0.8
        mean_y = 500.0
        y = np.full(n, mean_y)
        t_acc = np.zeros(n)
        # For pi = 0.8 and Weibull CDF, τ−T must give exactly 0.8
        # We use a callable inclusion model that returns pi_true for all
        t_rep = rng.uniform(0.1, 5.0, size=n)

        df = pd.DataFrame({
            "accident_time": t_acc,
            "report_time": t_rep,
            "severity": y,
        })

        # Callable that ignores inputs and returns constant pi_true
        pi_model = lambda tau_t, X: np.full(len(tau_t), pi_true)  # noqa: E731

        psr = PopulationSamplingReserve(
            inclusion_model=pi_model, method="ipw"
        )
        psr.fit(df, valuation_time=10.0)

        expected_ibnr = float(n) * mean_y * (1.0 - pi_true) / pi_true
        assert abs(psr.estimate_ibnr() - expected_ibnr) < 1e-6

    def test_aipw_with_biased_severity_but_correct_pi(self):
        """
        AIPW with correct π and biased severity model should still produce
        a correct IBNR estimate. The augmentation term corrects the severity
        bias.

        True IBNR = Σ (1-π)/π × Y_i (same as IPW with correct π).
        AIPW should converge to this value.
        """
        rng = np.random.default_rng(8)
        n = 300
        pi_true = 0.75
        y = rng.lognormal(mean=np.log(800), sigma=0.4, size=n)
        t_rep = rng.uniform(0.1, 3.0, size=n)

        df = pd.DataFrame({
            "accident_time": np.zeros(n),
            "report_time": t_rep,
            "severity": y,
        })

        # Biased severity model (always predicts 100, far from true mean)
        biased_model = lambda X: np.full(n, 100.0)  # noqa: E731

        # Correct π
        pi_model = lambda tau_t, X: np.full(len(tau_t), pi_true)  # noqa: E731

        psr = PopulationSamplingReserve(
            inclusion_model=pi_model,
            severity_model=biased_model,
            method="aipw",
        )
        psr.fit(df, valuation_time=10.0)

        # IPW with correct pi (ground truth)
        ipw_ibnr = float(np.sum((1.0 - pi_true) / pi_true * y))
        assert abs(psr.estimate_ibnr() - ipw_ibnr) < 1e-6


# ---------------------------------------------------------------------------
# Micro-level (severity model only) method
# ---------------------------------------------------------------------------


class TestMicroMethod:

    def test_micro_with_known_n_ibnr(self):
        """
        With constant severity m and known n_ibnr, micro IBNR = n_ibnr * m.
        """
        n = 100
        mean_y = 1000.0
        df = pd.DataFrame({
            "accident_time": np.zeros(n),
            "report_time": np.linspace(0.1, 5.0, n),
            "severity": np.full(n, mean_y),
        })

        # Constant severity model
        const_model = lambda X: np.full(n, mean_y)  # noqa: E731

        psr = PopulationSamplingReserve(severity_model=const_model, method="micro")
        psr.fit(df, valuation_time=10.0, n_ibnr=25)

        assert abs(psr.estimate_ibnr() - 25.0 * mean_y) < 1e-6

    def test_micro_without_n_ibnr_raises(self):
        """
        Micro method with no n_ibnr and no inclusion_model should raise.
        """
        df, _ = make_claims_df(n_reported=50, seed=9)
        psr = PopulationSamplingReserve(method="micro")
        with pytest.raises(ValueError, match="n_ibnr"):
            psr.fit(df, valuation_time=24.0)

    def test_micro_augmentation_zero(self):
        """Micro method has no augmentation by construction."""
        df, _ = make_claims_df(n_reported=80, seed=10)
        psr = PopulationSamplingReserve(method="micro")
        psr.fit(df, valuation_time=24.0, n_ibnr=20)
        assert psr.diagnostics()["augmentation_term"] == 0.0


# ---------------------------------------------------------------------------
# IPW method
# ---------------------------------------------------------------------------


class TestIPWMethod:

    def test_ipw_basic(self):
        """IPW estimate is positive and finite."""
        df, _ = make_claims_df(n_reported=100, seed=11)
        psr = PopulationSamplingReserve(method="ipw")
        psr.fit(df, valuation_time=24.0)
        assert np.isfinite(psr.estimate_ibnr())
        assert psr.estimate_ibnr() >= 0

    def test_ipw_no_severity_needed(self):
        """IPW works without a severity model."""
        df, _ = make_claims_df(n_reported=100, seed=12)
        psr = PopulationSamplingReserve(severity_model=None, method="ipw")
        psr.fit(df, valuation_time=24.0)
        diag = psr.diagnostics()
        assert diag["method"] == "ipw"


# ---------------------------------------------------------------------------
# Weighted balance adjustment
# ---------------------------------------------------------------------------


class TestWeightedBalanceAdjust:

    def test_scalar_b_corrects_mean(self):
        """
        After fitting AIPW, weighted_balance_adjust on the reported
        predictions should produce b * yhat = yhat_wbp. The ratio
        weighted_balance_ratio tells us b. When applied to Ŷ_i, the
        adjusted predictions satisfy the WBP by construction.
        """
        df, _ = make_claims_df(n_reported=200, seed=13)
        y = df["severity"].to_numpy()

        # Biased model
        biased_model = lambda X: np.full(len(y), 200.0)  # noqa: E731
        pi_model = lambda tau_t, X: np.full(len(tau_t), 0.7)  # noqa: E731

        psr = PopulationSamplingReserve(
            inclusion_model=pi_model,
            severity_model=biased_model,
            method="aipw",
        )
        psr.fit(df, valuation_time=24.0)

        b = psr.diagnostics()["weighted_balance_ratio"]
        adjusted = psr.weighted_balance_adjust(np.full(len(y), 200.0))

        # adjusted = b * 200, so mean adjusted = b * 200
        expected = b * 200.0
        assert np.allclose(adjusted, expected)

    def test_wba_raises_if_not_fitted(self):
        psr = PopulationSamplingReserve(method="aipw")
        with pytest.raises(RuntimeError):
            psr.weighted_balance_adjust(np.array([100.0, 200.0]))

    def test_wba_requires_inclusion_model(self):
        """Weighted balance adjust requires an inclusion model (has pi_hat)."""
        df, _ = make_claims_df(n_reported=50, seed=14)
        psr = PopulationSamplingReserve(method="micro")
        psr.fit(df, valuation_time=24.0, n_ibnr=10)
        with pytest.raises(RuntimeError, match="inclusion model"):
            psr.weighted_balance_adjust(np.ones(50))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_all_claims_reported(self):
        """
        When π̂_i ≈ 1 for all claims (all reported), IBNR should be ≈ 0.
        Use a callable that returns π=1 for all claims.
        """
        n = 100
        y = np.full(n, 500.0)
        df = pd.DataFrame({
            "accident_time": np.zeros(n),
            "report_time": np.linspace(0.1, 1.0, n),
            "severity": y,
        })

        # Inclusion probability = 1 for all (IBNR = 0)
        full_inclusion = lambda tau_t, X: np.ones(len(tau_t))  # noqa: E731

        psr = PopulationSamplingReserve(
            inclusion_model=full_inclusion, method="ipw"
        )
        psr.fit(df, valuation_time=5.0)

        # (1-1)/1 * Y = 0 for all claims
        assert abs(psr.estimate_ibnr()) < 1e-9

    def test_single_claim(self):
        """Single reported claim should not raise errors with IPW and known pi."""
        df = pd.DataFrame({
            "accident_time": [1.0],
            "report_time": [3.0],
            "severity": [750.0],
        })
        # Use IPW with a constant known pi to avoid Weibull MLE on 1 claim
        pi_model = lambda tau_t, X: np.full(len(tau_t), 0.8)  # noqa: E731
        psr = PopulationSamplingReserve(method="ipw", inclusion_model=pi_model)
        psr.fit(df, valuation_time=24.0)
        assert np.isfinite(psr.estimate_ibnr())
        assert psr.estimate_ibnr() > 0

    def test_aipw_ibnr_nonnegative_for_reasonable_data(self):
        """
        For well-conditioned data and auto-fitted Weibull model, IBNR should
        be non-negative. (Not guaranteed in theory but expected for normal
        insurance data.)
        """
        rng = np.random.default_rng(99)
        n = 300
        tau = 36.0
        t_acc = rng.uniform(0, tau * 0.8, n)
        delay = rng.exponential(scale=4.0, size=n)
        t_rep = np.minimum(t_acc + delay, tau)
        y = rng.lognormal(mean=np.log(500), sigma=0.6, size=n)

        df = pd.DataFrame({
            "accident_time": t_acc,
            "report_time": t_rep,
            "severity": y,
        })
        psr = PopulationSamplingReserve(method="ipw")
        psr.fit(df, valuation_time=tau)
        assert np.isfinite(psr.estimate_ibnr())
        # IPW with legitimate pi in (0,1) should give positive IBNR
        # (unless every claim has pi=1 which is unlikely here)
        assert psr.estimate_ibnr() >= 0

    def test_feature_cols_passed_through(self):
        """
        With feature_cols specified, fit should succeed and use them for
        the Weibull model.
        """
        rng = np.random.default_rng(20)
        n = 100
        df = pd.DataFrame({
            "accident_time": rng.uniform(0, 12, n),
            "report_time": rng.uniform(1, 14, n),
            "severity": rng.lognormal(np.log(500), 0.5, n),
            "age": rng.integers(20, 70, n).astype(float),
            "region": rng.integers(0, 3, n).astype(float),
        })
        # Ensure report_time >= accident_time
        df["report_time"] = df["accident_time"] + rng.exponential(2.0, n)

        psr = PopulationSamplingReserve(method="aipw")
        psr.fit(df, valuation_time=24.0, feature_cols=["age", "region"])
        assert np.isfinite(psr.estimate_ibnr())


# ---------------------------------------------------------------------------
# Diagnostics completeness
# ---------------------------------------------------------------------------


class TestDiagnostics:

    def test_all_methods_return_complete_diagnostics(self):
        required_keys = {
            "ibnr_estimate",
            "ultimate_estimate",
            "n_reported",
            "n_ibnr_estimated",
            "augmentation_term",
            "weighted_balance_ratio",
            "method",
        }
        df, _ = make_claims_df(n_reported=100, seed=30)
        df2, dev_factors, _ = make_simple_triangle_data()

        configs = [
            (PopulationSamplingReserve(method="aipw"), df, dict(valuation_time=24.0)),
            (PopulationSamplingReserve(method="ipw"), df, dict(valuation_time=24.0)),
            (PopulationSamplingReserve(method="micro"), df, dict(valuation_time=24.0, n_ibnr=20)),
            (
                PopulationSamplingReserve(method="chain_ladder"),
                df2,
                dict(development_factors=dev_factors, valuation_time=24.0),
            ),
        ]

        for psr, data, kwargs in configs:
            psr.fit(data, **kwargs)
            diag = psr.diagnostics()
            assert required_keys.issubset(set(diag.keys())), (
                f"method={psr.method}: missing {required_keys - set(diag.keys())}"
            )

    def test_n_reported_correct(self):
        n = 150
        df, _ = make_claims_df(n_reported=n, seed=31)
        # make_claims_df may return fewer rows than requested if the random sample
        # has fewer reported claims; use len(df) as the expected count
        expected = len(df)
        psr = PopulationSamplingReserve(method="aipw")
        psr.fit(df, valuation_time=24.0)
        assert psr.diagnostics()["n_reported"] == expected

    def test_method_recorded_in_diagnostics(self):
        df, _ = make_claims_df(n_reported=80, seed=32)
        for method in ["aipw", "ipw"]:
            psr = PopulationSamplingReserve(method=method)
            psr.fit(df, valuation_time=24.0)
            assert psr.diagnostics()["method"] == method
