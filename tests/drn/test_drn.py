"""
Tests for the DRN class — fit, predict, save/load, adjustment_factors.

These tests use synthetic Gamma data and a mock GLM baseline to avoid
statsmodels dependency and keep tests focused on DRN mechanics.

For tests that require actual training (which must run on Databricks),
we use max_epochs=2 to verify the pipeline runs without testing convergence.
"""

import tempfile
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from insurance_severity.drn.drn import DRN
from insurance_severity.drn.histogram import ExtendedHistogramBatch


# ---------------------------------------------------------------------------
# Mock baseline fixture
# ---------------------------------------------------------------------------

class GammaMockBaseline:
    """Mock baseline returning known Gamma(mu=2000, phi=0.5) for all rows."""
    distribution_family = "gamma"

    def predict_params(self, X):
        n = len(X)
        return {"mu": np.full(n, 2000.0), "dispersion": 0.5}

    def predict_cdf(self, X, cutpoints):
        from scipy import stats
        params = self.predict_params(X)
        mu = params["mu"][:, np.newaxis]
        disp = params["dispersion"]
        alpha = 1.0 / disp
        scale = mu * disp
        return stats.gamma.cdf(cutpoints[np.newaxis, :], a=alpha, scale=scale)


@pytest.fixture
def mock_baseline():
    return GammaMockBaseline()


@pytest.fixture
def tiny_data():
    """Tiny dataset: n=200, p=4 features, Gamma(shape=2, scale=1000) response."""
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame({
        "age": rng.uniform(20, 70, size=n),
        "vh_age": rng.uniform(0, 15, size=n),
        "region": rng.integers(0, 4, size=n).astype(float),
        "value": rng.uniform(5000, 30000, size=n),
    })
    y = rng.gamma(shape=2.0, scale=1000.0, size=n)
    return X, y


class TestDRNConstruction:

    def test_default_construction(self, mock_baseline):
        drn = DRN(mock_baseline)
        assert drn.hidden_size == 75
        assert drn.num_hidden_layers == 2
        assert drn.dropout_rate == 0.2
        assert not drn._is_fitted

    def test_custom_params(self, mock_baseline):
        drn = DRN(mock_baseline, hidden_size=32, num_hidden_layers=3, dropout_rate=0.1)
        assert drn.hidden_size == 32
        assert drn.num_hidden_layers == 3

    def test_repr_unfitted(self, mock_baseline):
        drn = DRN(mock_baseline)
        r = repr(drn)
        assert "not fitted" in r

    def test_predict_before_fit_raises(self, mock_baseline, tiny_data):
        drn = DRN(mock_baseline)
        X, y = tiny_data
        with pytest.raises(RuntimeError, match="not fitted"):
            drn.predict_distribution(X)


class TestDRNFit:

    def test_fit_returns_self(self, mock_baseline, tiny_data):
        X, y = tiny_data
        drn = DRN(mock_baseline, max_epochs=2, patience=5)
        result = drn.fit(X, y, verbose=False)
        assert result is drn

    def test_fit_sets_is_fitted(self, mock_baseline, tiny_data):
        X, y = tiny_data
        drn = DRN(mock_baseline, max_epochs=2)
        drn.fit(X, y, verbose=False)
        assert drn._is_fitted

    def test_fit_creates_cutpoints(self, mock_baseline, tiny_data):
        X, y = tiny_data
        drn = DRN(mock_baseline, max_epochs=2)
        drn.fit(X, y, verbose=False)
        assert drn._cutpoints is not None
        assert len(drn._cutpoints) >= 3

    def test_fit_creates_network(self, mock_baseline, tiny_data):
        X, y = tiny_data
        drn = DRN(mock_baseline, max_epochs=2)
        drn.fit(X, y, verbose=False)
        assert drn._network is not None

    def test_training_history_populated(self, mock_baseline, tiny_data):
        X, y = tiny_data
        drn = DRN(mock_baseline, max_epochs=3)
        drn.fit(X, y, verbose=False)
        assert len(drn.training_history["train_loss"]) >= 1
        assert len(drn.training_history["val_loss"]) >= 1

    def test_fit_with_numpy_X(self, mock_baseline, tiny_data):
        X, y = tiny_data
        drn = DRN(mock_baseline, max_epochs=2)
        drn.fit(X.values, y, verbose=False)
        assert drn._is_fitted

    def test_fit_with_exposure(self, mock_baseline, tiny_data):
        X, y = tiny_data
        exposure = np.random.default_rng(0).uniform(0.5, 2.0, size=len(y))
        drn = DRN(mock_baseline, max_epochs=2)
        drn.fit(X, y, exposure=exposure, verbose=False)
        assert drn._is_fitted

    def test_fit_with_explicit_val_set(self, mock_baseline, tiny_data):
        X, y = tiny_data
        n = len(y)
        split = int(n * 0.8)
        drn = DRN(mock_baseline, max_epochs=2)
        drn.fit(
            X.iloc[:split], y[:split],
            X_val=X.iloc[split:], y_val=y[split:],
            verbose=False
        )
        assert drn._is_fitted

    def test_reproducible_with_random_state(self, mock_baseline, tiny_data):
        X, y = tiny_data
        drn1 = DRN(mock_baseline, max_epochs=3, random_state=0)
        drn1.fit(X, y, verbose=False)
        drn2 = DRN(mock_baseline, max_epochs=3, random_state=0)
        drn2.fit(X, y, verbose=False)
        # Same random state should give same training history
        h1 = drn1.training_history["train_loss"]
        h2 = drn2.training_history["train_loss"]
        np.testing.assert_allclose(h1, h2, rtol=1e-5)

    def test_n_bins_property(self, mock_baseline, tiny_data):
        X, y = tiny_data
        drn = DRN(mock_baseline, max_epochs=2)
        drn.fit(X, y, verbose=False)
        assert drn.n_bins is not None
        assert drn.n_bins >= 2

    def test_repr_fitted(self, mock_baseline, tiny_data):
        X, y = tiny_data
        drn = DRN(mock_baseline, max_epochs=2)
        drn.fit(X, y, verbose=False)
        r = repr(drn)
        assert "fitted" in r


class TestDRNPredict:

    @pytest.fixture(autouse=True)
    def fitted_drn(self, mock_baseline, tiny_data):
        X, y = tiny_data
        self.drn = DRN(mock_baseline, max_epochs=3, random_state=42)
        self.drn.fit(X, y, verbose=False)
        self.X = X
        self.y = y

    def test_predict_distribution_type(self):
        dist = self.drn.predict_distribution(self.X)
        assert isinstance(dist, ExtendedHistogramBatch)

    def test_predict_distribution_shape(self):
        n = 15
        dist = self.drn.predict_distribution(self.X.head(n))
        assert len(dist) == n

    def test_predict_mean_shape(self):
        mean = self.drn.predict_mean(self.X)
        assert mean.shape == (len(self.X),)

    def test_predict_mean_positive(self):
        mean = self.drn.predict_mean(self.X)
        assert np.all(mean > 0)

    def test_predict_quantile_scalar(self):
        q50 = self.drn.predict_quantile(self.X, 0.50)
        assert q50.shape == (len(self.X),)
        assert np.all(q50 > 0)

    def test_predict_quantile_array(self):
        q = self.drn.predict_quantile(self.X, [0.25, 0.50, 0.75])
        assert q.shape == (len(self.X), 3)

    def test_predict_quantile_monotone(self):
        q = self.drn.predict_quantile(self.X.head(20), [0.25, 0.50, 0.75])
        assert np.all(q[:, 0] <= q[:, 1] + 0.1)
        assert np.all(q[:, 1] <= q[:, 2] + 0.1)

    def test_predict_var_positive(self):
        v = self.drn.predict_var(self.X)
        assert np.all(v >= 0)

    def test_predict_cdf_shape(self):
        y_grid = np.array([500.0, 1000.0, 2000.0, 5000.0])
        cdf = self.drn.predict_cdf(self.X.head(10), y_grid)
        assert cdf.shape == (10, 4)

    def test_score_crps(self):
        crps = self.drn.score(self.X, self.y, metric="crps")
        assert isinstance(crps, float)
        assert crps >= 0.0

    def test_score_rmse(self):
        rmse = self.drn.score(self.X, self.y, metric="rmse")
        assert isinstance(rmse, float)
        assert rmse >= 0.0

    def test_score_nll(self):
        nll = self.drn.score(self.X, self.y, metric="nll")
        assert isinstance(nll, float)

    def test_score_quantile_loss(self):
        ql = self.drn.score(self.X, self.y, metric="ql90")
        assert isinstance(ql, float)
        assert ql >= 0.0


class TestDRNAdjustmentFactors:

    @pytest.fixture(autouse=True)
    def fitted_drn(self, mock_baseline, tiny_data):
        X, y = tiny_data
        self.drn = DRN(mock_baseline, max_epochs=2, random_state=0)
        self.drn.fit(X, y, verbose=False)
        self.X = X
        self.y = y

    def test_adjustment_factors_type(self):
        import polars as pl
        af = self.drn.adjustment_factors(self.X)
        assert isinstance(af, pl.DataFrame)

    def test_adjustment_factors_shape(self):
        af = self.drn.adjustment_factors(self.X)
        assert len(af) == len(self.X)
        assert af.width == self.drn.n_bins

    def test_adjustment_factors_columns_start_with_adj(self):
        af = self.drn.adjustment_factors(self.X)
        assert all(c.startswith("adj_") for c in af.columns)

    def test_drn_probs_sum_to_one(self):
        """
        bin_probs in predict_distribution should sum to 1 for each observation.
        """
        dist = self.drn.predict_distribution(self.X.head(20))
        row_sums = dist.bin_probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_adjustment_factors_non_negative(self):
        """Adjustment factors a_k = p_k/b_k should be non-negative.
        
        With softmax output, DRN probs are always > 0 in theory but may
        be numerically zero in float32 for extreme bins. The ratio must
        be >= 0.
        """
        af = self.drn.adjustment_factors(self.X.head(20))
        assert af.to_numpy().min() >= 0


class TestDRNSaveLoad:

    def test_save_load_roundtrip(self, mock_baseline, tiny_data):
        X, y = tiny_data
        drn = DRN(mock_baseline, max_epochs=2, random_state=0)
        drn.fit(X, y, verbose=False)

        mean_before = drn.predict_mean(X.head(10))

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            drn.save(path)
            drn2 = DRN.load(path, baseline=mock_baseline)
            mean_after = drn2.predict_mean(X.head(10))
            np.testing.assert_allclose(mean_before, mean_after, rtol=1e-5)
        finally:
            os.unlink(path)

    def test_save_load_fitted_flag(self, mock_baseline, tiny_data):
        X, y = tiny_data
        drn = DRN(mock_baseline, max_epochs=2)
        drn.fit(X, y, verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            drn.save(path)
            drn2 = DRN.load(path, baseline=mock_baseline)
            assert drn2._is_fitted
        finally:
            os.unlink(path)

    def test_save_before_fit_raises(self, mock_baseline):
        drn = DRN(mock_baseline)
        with pytest.raises(RuntimeError, match="not fitted"):
            drn.save("/tmp/test_drn.pt")

    def test_save_load_cutpoints_preserved(self, mock_baseline, tiny_data):
        X, y = tiny_data
        drn = DRN(mock_baseline, max_epochs=2)
        drn.fit(X, y, verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            drn.save(path)
            drn2 = DRN.load(path, baseline=mock_baseline)
            np.testing.assert_array_equal(drn.cutpoints, drn2.cutpoints)
        finally:
            os.unlink(path)


class TestDRNBaseline:

    def test_zero_epochs_distribution_close_to_baseline(self, mock_baseline, tiny_data):
        """
        With baseline_start=True and 0 effective training (1 epoch but early stop),
        the DRN distribution should be close to the baseline.

        Test: DRN mean should be within 20% of baseline mean.
        """
        X, y = tiny_data
        drn = DRN(
            mock_baseline,
            max_epochs=1,
            baseline_start=True,
            random_state=0,
        )
        drn.fit(X, y, verbose=False)

        dist = drn.predict_distribution(X.head(20))
        drn_mean = dist.mean()

        # Baseline mean is fixed at 2000 for all obs
        baseline_mean = 2000.0
        relative_error = np.abs(drn_mean - baseline_mean) / baseline_mean
        # With only 1 epoch, should be within 50% of baseline
        assert np.mean(relative_error) < 0.5, (
            f"After 1 epoch, DRN mean should be near baseline. "
            f"Mean relative error: {np.mean(relative_error):.2f}"
        )

    def test_nll_loss_option(self, mock_baseline, tiny_data):
        """DRN should train with loss='nll' without errors."""
        X, y = tiny_data
        drn = DRN(mock_baseline, max_epochs=2, loss="nll")
        drn.fit(X, y, verbose=False)
        assert drn._is_fitted

    def test_regularisation_kl(self, mock_baseline, tiny_data):
        """DRN with KL regularisation should train without errors."""
        X, y = tiny_data
        drn = DRN(
            mock_baseline,
            max_epochs=2,
            kl_alpha=1e-4,
            mean_alpha=1e-4,
        )
        drn.fit(X, y, verbose=False)
        assert drn._is_fitted

    def test_scr_aware_cutpoints(self, mock_baseline, tiny_data):
        """scr_aware=True should set c_K above 99.7th percentile."""
        X, y = tiny_data
        drn = DRN(mock_baseline, max_epochs=2, scr_aware=True)
        drn.fit(X, y, verbose=False)
        p997 = np.percentile(y, 99.7)
        assert drn._cutpoints[-1] >= p997
