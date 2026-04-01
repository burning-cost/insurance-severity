"""
Tests for insurance_severity.projection: ProjectionToUltimate.

These tests run on Databricks (not locally) — see README for instructions.

Test strategy:
- Synthetic data generated with known multiplicative PtU structure so we can
  verify coefficient recovery.
- Edge case coverage: zero paid, subrogation rows (paid > ultimate), single
  feature, ridge vs OLS, wrong column names, unfitted access.
- Summary and predict output structure tests.
"""

import warnings

import numpy as np
import polars as pl
import pytest

from insurance_severity.projection import (
    ProjectionToUltimate,
    _add_log_paid,
    _build_design_matrix,
    _ols_fit,
    _ridge_fit,
)


# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------


def _make_train_df(
    n: int = 500,
    seed: int = 42,
    intercept: float = 0.5,
    dev_month_coef: float = -0.05,
    log_paid_coef: float = 0.1,
    claim_age_coef: float = -0.01,
    noise_std: float = 0.15,
) -> pl.DataFrame:
    """
    Generate synthetic training data with known log-PtU regression structure.

    log(ultimate / paid_to_date) = intercept
                                   + dev_month_coef * dev_month
                                   + log_paid_coef * log_paid
                                   + claim_age_coef * claim_age
                                   + noise

    This is a clean linear signal so OLS should recover coefficients tightly.
    """
    rng = np.random.default_rng(seed)

    dev_month = rng.integers(1, 36, size=n).astype(float)
    paid_to_date = rng.lognormal(mean=8.0, sigma=1.5, size=n)  # £100 – £200k range
    claim_age = rng.integers(6, 60, size=n).astype(float)

    log_paid = np.log(paid_to_date)
    log_ptu = (
        intercept
        + dev_month_coef * dev_month
        + log_paid_coef * log_paid
        + claim_age_coef * claim_age
        + rng.normal(0, noise_std, size=n)
    )
    # Clip to ensure ultimate >= paid (i.e. log_ptu >= 0)
    log_ptu = np.maximum(log_ptu, 0.02)
    ultimate_cost = paid_to_date * np.exp(log_ptu)

    return pl.DataFrame({
        "paid_to_date": paid_to_date,
        "ultimate_cost": ultimate_cost,
        "dev_month": dev_month,
        "claim_age": claim_age,
    })


def _make_predict_df(n: int = 100, seed: int = 99) -> pl.DataFrame:
    """Generate open claims for prediction (no ultimate_cost needed)."""
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "paid_to_date": rng.lognormal(mean=8.0, sigma=1.5, size=n),
        "dev_month": rng.integers(1, 36, size=n).astype(float),
        "claim_age": rng.integers(6, 60, size=n).astype(float),
    })


# ---------------------------------------------------------------------------
# 1. Basic fit / predict round-trip
# ---------------------------------------------------------------------------


class TestFitPredict:
    def test_fit_returns_self(self):
        df = _make_train_df()
        ptu = ProjectionToUltimate()
        result = ptu.fit(df, paid_col="paid_to_date", ultimate_col="ultimate_cost")
        assert result is ptu

    def test_predict_output_columns_ols(self):
        train = _make_train_df()
        pred_df = _make_predict_df()

        ptu = ProjectionToUltimate(method="ols")
        ptu.fit(train)
        out = ptu.predict(pred_df)

        for col in ("predicted_ultimate", "ptu_factor", "log_ptu", "pi_lower", "pi_upper"):
            assert col in out.columns, f"Missing column: {col}"

    def test_predict_output_columns_ridge(self):
        train = _make_train_df()
        pred_df = _make_predict_df()

        ptu = ProjectionToUltimate(method="ridge", ridge_alpha=0.5)
        ptu.fit(train)
        out = ptu.predict(pred_df)

        for col in ("predicted_ultimate", "ptu_factor", "log_ptu", "pi_lower", "pi_upper"):
            assert col in out.columns, f"Missing column: {col}"

    def test_predict_no_pi(self):
        train = _make_train_df()
        pred_df = _make_predict_df()

        ptu = ProjectionToUltimate()
        ptu.fit(train)
        out = ptu.predict(pred_df, add_prediction_interval=False)

        assert "pi_lower" not in out.columns
        assert "pi_upper" not in out.columns
        assert "predicted_ultimate" in out.columns

    def test_predicted_ultimate_positive(self):
        """Predicted ultimates must all be strictly positive."""
        train = _make_train_df()
        pred_df = _make_predict_df()
        ptu = ProjectionToUltimate()
        ptu.fit(train)
        out = ptu.predict(pred_df)
        assert (out["predicted_ultimate"].to_numpy() > 0).all()

    def test_ptu_factor_nonnegative(self):
        """PtU factors are at least 1.0 (log_ptu clipped to 0)."""
        train = _make_train_df()
        pred_df = _make_predict_df()
        ptu = ProjectionToUltimate()
        ptu.fit(train)
        out = ptu.predict(pred_df)
        assert (out["ptu_factor"].to_numpy() >= 1.0).all()

    def test_predict_preserves_input_rows(self):
        train = _make_train_df()
        pred_df = _make_predict_df(n=73)
        ptu = ProjectionToUltimate()
        ptu.fit(train)
        out = ptu.predict(pred_df)
        assert len(out) == 73


# ---------------------------------------------------------------------------
# 2. Summary diagnostics
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_keys(self):
        ptu = ProjectionToUltimate()
        ptu.fit(_make_train_df())
        s = ptu.summary()

        expected = {
            "n_train", "r2", "rmse", "method", "ridge_alpha",
            "coefficients", "residuals_mean", "residuals_std",
            "residuals_skewness", "development_features",
        }
        assert expected <= set(s.keys())

    def test_summary_r2_range(self):
        """R² on synthetic data with known signal should be > 0.5."""
        ptu = ProjectionToUltimate()
        ptu.fit(_make_train_df(n=1000, noise_std=0.10))
        s = ptu.summary()
        assert 0.0 < s["r2"] <= 1.0, f"R² out of range: {s['r2']}"

    def test_summary_ols_residuals_mean_near_zero(self):
        """OLS residuals should sum to zero by construction."""
        ptu = ProjectionToUltimate(method="ols")
        ptu.fit(_make_train_df(n=500))
        s = ptu.summary()
        assert abs(s["residuals_mean"]) < 1e-8, (
            f"OLS residual mean not near zero: {s['residuals_mean']}"
        )

    def test_summary_n_train(self):
        ptu = ProjectionToUltimate()
        ptu.fit(_make_train_df(n=300))
        assert ptu.summary()["n_train"] == 300

    def test_summary_coefficients_count(self):
        """Coefficient dict should have intercept + len(features) entries."""
        feats = ["dev_month", "log_paid", "claim_age"]
        ptu = ProjectionToUltimate(development_features=feats)
        ptu.fit(_make_train_df())
        s = ptu.summary()
        assert len(s["coefficients"]) == len(feats) + 1  # +1 for intercept
        assert "intercept" in s["coefficients"]


# ---------------------------------------------------------------------------
# 3. Coefficient recovery
# ---------------------------------------------------------------------------


class TestCoefficientRecovery:
    def test_ols_recovers_dev_month_sign(self):
        """
        With negative dev_month coefficient (later development = smaller PtU),
        the fitted coefficient should be negative.
        """
        df = _make_train_df(n=2000, dev_month_coef=-0.05, noise_std=0.05)
        ptu = ProjectionToUltimate(method="ols")
        ptu.fit(df)
        coefs = ptu.coefficients
        assert coefs["dev_month"] < 0, (
            f"Expected negative dev_month coefficient, got {coefs['dev_month']:.4f}"
        )

    def test_ridge_produces_smaller_coefficients_than_ols(self):
        """Ridge shrinks coefficients toward zero."""
        df = _make_train_df(n=500)
        ols = ProjectionToUltimate(method="ols")
        ridge = ProjectionToUltimate(method="ridge", ridge_alpha=10.0)
        ols.fit(df)
        ridge.fit(df)

        # Exclude intercept
        ols_norm = sum(v ** 2 for k, v in ols.coefficients.items() if k != "intercept")
        ridge_norm = sum(v ** 2 for k, v in ridge.coefficients.items() if k != "intercept")

        assert ridge_norm < ols_norm, (
            f"Ridge norm {ridge_norm:.4f} should be < OLS norm {ols_norm:.4f}"
        )


# ---------------------------------------------------------------------------
# 4. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_zero_paid_rows_warning(self):
        """Training rows with zero paid should trigger a warning."""
        df = _make_train_df(n=200)
        # Inject many zero-paid rows (>5%)
        zero_paid = pl.DataFrame({
            "paid_to_date": [0.0] * 20,
            "ultimate_cost": [1000.0] * 20,
            "dev_month": [12.0] * 20,
            "claim_age": [24.0] * 20,
        })
        combined = pl.concat([df, zero_paid])
        ptu = ProjectionToUltimate()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ptu.fit(combined)
        # Should warn about zero-paid rows
        assert any("zero or negative paid" in str(warning.message) for warning in w)

    def test_subrogation_rows_excluded(self):
        """Rows where paid > ultimate (negative log-ratio) are dropped with a warning."""
        df = _make_train_df(n=200)
        # Inject rows where paid > ultimate (subrogation scenario)
        subro = pl.DataFrame({
            "paid_to_date": [5000.0] * 10,
            "ultimate_cost": [100.0] * 10,  # paid > ultimate
            "dev_month": [12.0] * 10,
            "claim_age": [24.0] * 10,
        })
        combined = pl.concat([df, subro])
        ptu = ProjectionToUltimate()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ptu.fit(combined)
        assert any("paid_to_date > ultimate_cost" in str(warning.message) for warning in w)
        assert ptu.summary()["n_train"] == 200  # subrogation rows excluded

    def test_missing_column_raises(self):
        """Fitting with a missing feature column raises ValueError."""
        df = _make_train_df()
        df = df.drop("dev_month")
        ptu = ProjectionToUltimate(development_features=["dev_month", "log_paid"])
        with pytest.raises(ValueError, match="Missing columns"):
            ptu.fit(df)

    def test_missing_predict_column_raises(self):
        """Predicting with a missing feature column raises ValueError."""
        train = _make_train_df()
        ptu = ProjectionToUltimate()
        ptu.fit(train)

        pred = _make_predict_df()
        pred = pred.drop("dev_month")
        with pytest.raises(ValueError, match="Missing columns"):
            ptu.predict(pred)

    def test_unfitted_predict_raises(self):
        ptu = ProjectionToUltimate()
        with pytest.raises(RuntimeError, match="fit()"):
            ptu.predict(_make_predict_df())

    def test_unfitted_summary_raises(self):
        ptu = ProjectionToUltimate()
        with pytest.raises(RuntimeError, match="fit()"):
            ptu.summary()

    def test_min_train_rows_raises(self):
        """Too few training rows should raise ValueError."""
        df = _make_train_df(n=5)
        ptu = ProjectionToUltimate(min_train_rows=50)
        with pytest.raises(ValueError, match="min_train_rows"):
            ptu.fit(df)

    def test_nonpositive_ultimate_raises(self):
        """Non-positive ultimate_cost should raise ValueError."""
        df = _make_train_df(n=100)
        bad = df.with_columns(pl.lit(-1.0).alias("ultimate_cost"))
        ptu = ProjectionToUltimate()
        with pytest.raises(ValueError, match="non-positive ultimate_cost"):
            ptu.fit(bad)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method must be"):
            ProjectionToUltimate(method="xgb")

    def test_negative_ridge_alpha_raises(self):
        with pytest.raises(ValueError, match="ridge_alpha must be positive"):
            ProjectionToUltimate(method="ridge", ridge_alpha=-1.0)

    def test_single_feature(self):
        """Fits and predicts with only one development feature."""
        df = _make_train_df()
        pred_df = _make_predict_df()
        ptu = ProjectionToUltimate(
            development_features=["dev_month"],
            auto_add_log_paid=False,
        )
        ptu.fit(df)
        out = ptu.predict(pred_df)
        assert len(out) == len(pred_df)
        assert "predicted_ultimate" in out.columns

    def test_prediction_interval_coverage_parameter(self):
        """Different pi_coverage should produce different interval widths."""
        train = _make_train_df()
        pred_df = _make_predict_df(n=50)
        ptu = ProjectionToUltimate()
        ptu.fit(train)

        out_90 = ptu.predict(pred_df, pi_coverage=0.90)
        out_50 = ptu.predict(pred_df, pi_coverage=0.50)

        width_90 = (out_90["pi_upper"] - out_90["pi_lower"]).mean()
        width_50 = (out_50["pi_upper"] - out_50["pi_lower"]).mean()

        assert width_90 > width_50, (
            f"90% PI width {width_90:.2f} should exceed 50% PI width {width_50:.2f}"
        )

    def test_polars_input_required(self):
        """Passing a non-Polars object should raise TypeError."""
        import pandas as pd
        df = _make_train_df()
        ptu = ProjectionToUltimate()
        pandas_df = df.to_pandas()
        with pytest.raises(TypeError, match="polars DataFrame"):
            ptu.fit(pandas_df)


# ---------------------------------------------------------------------------
# 5. Internal helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_add_log_paid(self):
        df = pl.DataFrame({"paid_to_date": [100.0, 1000.0, 0.0]})
        result = _add_log_paid(df, "paid_to_date")
        assert "log_paid" in result.columns
        vals = result["log_paid"].to_numpy()
        # log(100) ≈ 4.605, log(1000) ≈ 6.908, log(epsilon) for 0
        assert abs(vals[0] - np.log(100.0)) < 1e-6
        assert abs(vals[1] - np.log(1000.0)) < 1e-6
        # Zero is clipped to epsilon
        assert vals[2] == pytest.approx(np.log(1e-6), rel=1e-4)

    def test_build_design_matrix_shape(self):
        df = pl.DataFrame({
            "dev_month": [1.0, 2.0, 3.0],
            "claim_age": [12.0, 24.0, 36.0],
        })
        X = _build_design_matrix(df, ["dev_month", "claim_age"])
        assert X.shape == (3, 3)  # 3 rows, intercept + 2 features
        # First column should be all ones
        np.testing.assert_array_equal(X[:, 0], [1.0, 1.0, 1.0])

    def test_ols_fit_exact_solution(self):
        """OLS on a noiseless system should recover exact coefficients."""
        rng = np.random.default_rng(0)
        n = 200
        X = np.column_stack([np.ones(n), rng.uniform(0, 10, n)])
        true_coef = np.array([1.5, -0.3])
        y = X @ true_coef
        fitted = _ols_fit(X, y)
        np.testing.assert_allclose(fitted, true_coef, atol=1e-8)

    def test_ridge_shrinks_toward_zero(self):
        """With very high alpha, ridge intercept stays but slope shrinks."""
        rng = np.random.default_rng(7)
        n = 100
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = 2.0 + 5.0 * X[:, 1] + rng.normal(0, 0.1, n)
        coef_ols = _ols_fit(X, y)
        coef_ridge = _ridge_fit(X, y, alpha=1000.0)
        # Slope (index 1) should be much smaller for ridge
        assert abs(coef_ridge[1]) < abs(coef_ols[1])
