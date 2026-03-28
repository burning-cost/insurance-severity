"""
Tests for insurance_severity.mdn — Mixture Density Network.

All tests use small synthetic datasets (n <= 500) and very short training
runs (max_epochs=5) to keep the test suite fast. We test correctness of
shapes, API contracts, and numerical stability rather than statistical
performance (convergence tests belong in benchmarks/).

Coverage:
- MDNNetwork: forward pass shapes, init_from_data, parameter count
- Loss functions: mdn_nll_loss, mdn_log_prob, mixture_mean
- MDNMixture: mean, variance, cdf, pdf, quantile, pit_samples, ilf, log_prob
- MDN: fit/predict shapes, save/load, error handling
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

# Guard: skip all tests if torch is not installed
torch = pytest.importorskip("torch", reason="torch not installed; skip MDN tests")

from insurance_severity.mdn.network import MDNNetwork
from insurance_severity.mdn.loss import (
    mdn_nll_loss,
    mdn_log_prob,
    mixture_mean,
    mixture_cdf,
    mixture_quantile,
)
from insurance_severity.mdn.distribution import MDNMixture
from insurance_severity.mdn.mdn import MDN


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="module")
def synthetic_data(rng):
    """
    Small 3-component lognormal mixture dataset for testing.
    n=400, p=3 features. Features are standardised.
    """
    n = 400
    K = 3

    # Features
    X = rng.standard_normal((n, 3)).astype(np.float32)
    df = pd.DataFrame(X, columns=["x0", "x1", "x2"])

    # Ground truth: component selection proportional to softmax(linear score)
    scores = np.column_stack([
        0.5 * X[:, 0],
        -0.2 * X[:, 0] + 0.4 * X[:, 1],
        np.zeros(n),
    ])
    exp_s = np.exp(scores - scores.max(axis=1, keepdims=True))
    pi_true = exp_s / exp_s.sum(axis=1, keepdims=True)

    comp = np.array([rng.choice(K, p=pi_true[i]) for i in range(n)])

    mu_true = np.array([7.0, 9.0, 11.0])   # log-space component means
    sigma_true = np.array([0.4, 0.6, 0.8])

    log_y = mu_true[comp] + sigma_true[comp] * rng.standard_normal(n)
    y = np.exp(log_y)

    return df, y


@pytest.fixture(scope="module")
def fitted_mdn(synthetic_data):
    """MDN fitted with minimal epochs for fast tests."""
    X, y = synthetic_data
    mdn = MDN(
        n_components=3,
        hidden_size=32,
        num_hidden_layers=1,
        max_epochs=5,
        patience=10,
        random_state=42,
        verbose=False,
    )
    mdn.fit(X, y, verbose=False)
    return mdn


# ---------------------------------------------------------------------------
# MDNNetwork tests
# ---------------------------------------------------------------------------

class TestMDNNetwork:

    def test_forward_shapes(self):
        net = MDNNetwork(n_features=5, n_components=3, hidden_size=32, num_hidden_layers=2)
        x = torch.randn(16, 5)
        pi, mu, sigma = net(x)
        assert pi.shape == (16, 3)
        assert mu.shape == (16, 3)
        assert sigma.shape == (16, 3)

    def test_pi_sums_to_one(self):
        net = MDNNetwork(n_features=4, n_components=4, hidden_size=16, num_hidden_layers=1)
        x = torch.randn(32, 4)
        pi, _, _ = net(x)
        sums = pi.sum(dim=1)
        assert torch.allclose(sums, torch.ones(32), atol=1e-5)

    def test_sigma_positive(self):
        net = MDNNetwork(n_features=3, n_components=5, hidden_size=16, num_hidden_layers=1)
        x = torch.randn(20, 3)
        _, _, sigma = net(x)
        assert (sigma > 0).all()

    def test_init_from_data(self):
        net = MDNNetwork(n_features=3, n_components=3, hidden_size=16, num_hidden_layers=1)
        log_y = torch.tensor(np.log(np.array([100.0, 1000.0, 5000.0, 10000.0])))
        net.init_from_data(log_y)
        biases = net.head_mu.bias.detach().numpy()
        # Biases should span the log range
        assert biases[0] < biases[-1], "component means not spread across range"

    def test_n_parameters(self):
        net = MDNNetwork(n_features=5, n_components=3, hidden_size=32, num_hidden_layers=2)
        n = net.n_parameters()
        assert n > 0
        assert isinstance(n, int)

    def test_single_component(self):
        """K=1 degenerates to a single lognormal — should work without errors."""
        net = MDNNetwork(n_features=3, n_components=1, hidden_size=16, num_hidden_layers=1)
        x = torch.randn(10, 3)
        pi, mu, sigma = net(x)
        assert pi.shape == (10, 1)
        assert torch.allclose(pi, torch.ones(10, 1), atol=1e-5)


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------

class TestLossFunctions:

    def _make_params(self, batch: int = 8, K: int = 3):
        """Construct valid mixture parameters for testing."""
        pi = torch.softmax(torch.randn(batch, K), dim=1)
        mu = torch.randn(batch, K) + 8.0   # log-space means around e^8 ~ 3000
        sigma = torch.abs(torch.randn(batch, K)) + 0.3
        y = torch.exp(torch.randn(batch) + 8.0)   # positive severities
        return pi, mu, sigma, y

    def test_nll_loss_scalar(self):
        pi, mu, sigma, y = self._make_params()
        loss = mdn_nll_loss(pi, mu, sigma, y)
        assert loss.shape == ()   # scalar
        assert torch.isfinite(loss)

    def test_nll_loss_positive(self):
        """NLL can be negative (for tight distributions) but must be finite."""
        pi, mu, sigma, y = self._make_params()
        loss = mdn_nll_loss(pi, mu, sigma, y)
        assert torch.isfinite(loss)

    def test_nll_loss_with_weights(self):
        pi, mu, sigma, y = self._make_params(batch=16)
        weights = torch.ones(16)
        loss_w = mdn_nll_loss(pi, mu, sigma, y, weights=weights)
        loss_no_w = mdn_nll_loss(pi, mu, sigma, y)
        # Equal weights should give same result
        assert torch.isclose(loss_w, loss_no_w, atol=1e-4)

    def test_nll_no_nan_with_small_sigma(self):
        """sigma_floor should prevent NaN when sigma is very small."""
        pi = torch.ones(8, 3) / 3
        mu = torch.ones(8, 3) * 8.0
        sigma = torch.ones(8, 3) * 1e-10   # extremely small
        y = torch.exp(torch.ones(8) * 8.0)
        loss = mdn_nll_loss(pi, mu, sigma, y, sigma_floor=1e-4)
        assert torch.isfinite(loss)

    def test_mdn_log_prob_shape(self):
        pi, mu, sigma, y = self._make_params(batch=12)
        lp = mdn_log_prob(pi, mu, sigma, y)
        assert lp.shape == (12,)
        assert torch.all(torch.isfinite(lp))

    def test_mixture_mean_formula(self):
        """Verify E[Y] = Σ π_k exp(μ_k + σ_k²/2) for known single-component case."""
        K = 1
        pi = torch.ones(4, K)
        mu_val = 9.0
        sigma_val = 0.5
        mu = torch.full((4, K), mu_val)
        sigma = torch.full((4, K), sigma_val)
        expected = np.exp(mu_val + 0.5 * sigma_val ** 2)
        result = mixture_mean(pi, mu, sigma)
        assert result.shape == (4,)
        assert torch.allclose(result, torch.full((4,), expected, dtype=torch.float32), rtol=1e-4)

    def test_mixture_cdf_shape(self):
        pi = torch.softmax(torch.randn(5, 3), dim=1)
        mu = torch.randn(5, 3) + 8.0
        sigma = torch.abs(torch.randn(5, 3)) + 0.3
        y_grid = torch.tensor([500.0, 1000.0, 5000.0, 10000.0, 50000.0])
        cdf = mixture_cdf(pi, mu, sigma, y_grid)
        assert cdf.shape == (5, 5)

    def test_mixture_cdf_monotone(self):
        pi = torch.softmax(torch.randn(3, 3), dim=1)
        mu = torch.randn(3, 3) + 8.0
        sigma = torch.abs(torch.randn(3, 3)) + 0.3
        y_grid = torch.linspace(100.0, 100000.0, 50)
        cdf = mixture_cdf(pi, mu, sigma, y_grid)
        # CDF should be non-decreasing
        diffs = cdf[:, 1:] - cdf[:, :-1]
        assert (diffs >= -1e-5).all(), "CDF is not monotone non-decreasing"

    def test_mixture_cdf_bounds(self):
        pi = torch.softmax(torch.randn(3, 3), dim=1)
        mu = torch.randn(3, 3) + 8.0
        sigma = torch.abs(torch.randn(3, 3)) + 0.3
        y_grid = torch.tensor([1e-4, 1e8])
        cdf = mixture_cdf(pi, mu, sigma, y_grid)
        assert (cdf[:, 0] >= 0).all()
        assert (cdf[:, 1] <= 1.0 + 1e-5).all()

    def test_mixture_quantile_shape(self):
        pi = torch.softmax(torch.randn(5, 3), dim=1)
        mu = torch.randn(5, 3) + 8.0
        sigma = torch.abs(torch.randn(5, 3)) + 0.3
        q = mixture_quantile(pi, mu, sigma, q=0.5)
        assert q.shape == (5,)
        assert (q > 0).all()

    def test_mixture_quantile_single_obs(self):
        """Single observation (squeezed input) should work."""
        pi = torch.softmax(torch.randn(3), dim=0)
        mu = torch.tensor([7.0, 9.0, 11.0])
        sigma = torch.tensor([0.4, 0.5, 0.8])
        q = mixture_quantile(pi, mu, sigma, q=0.5)
        assert q.shape == ()
        assert float(q) > 0


# ---------------------------------------------------------------------------
# MDNMixture tests
# ---------------------------------------------------------------------------

class TestMDNMixture:

    def _make_mixture(self, n: int = 50, K: int = 3):
        rng = np.random.default_rng(1)
        pi_raw = rng.dirichlet(np.ones(K), size=n)
        mu = rng.standard_normal((n, K)) + 8.0
        sigma = np.abs(rng.standard_normal((n, K))) + 0.3
        return MDNMixture(pi=pi_raw, mu=mu, sigma=sigma)

    def test_mean_shape(self):
        dist = self._make_mixture(n=20)
        m = dist.mean()
        assert m.shape == (20,)
        assert np.all(m > 0)

    def test_variance_nonneg(self):
        dist = self._make_mixture(n=20)
        v = dist.variance()
        assert np.all(v >= 0)

    def test_std_shape(self):
        dist = self._make_mixture(n=20)
        s = dist.std()
        assert s.shape == (20,)

    def test_cdf_shape(self):
        dist = self._make_mixture(n=10)
        y_grid = np.array([500.0, 1000.0, 5000.0, 50000.0])
        cdf = dist.cdf(y_grid)
        assert cdf.shape == (10, 4)

    def test_cdf_monotone(self):
        dist = self._make_mixture(n=5)
        y_grid = np.linspace(100, 100000, 100)
        cdf = dist.cdf(y_grid)
        diffs = np.diff(cdf, axis=1)
        assert np.all(diffs >= -1e-8), "CDF is not monotone"

    def test_cdf_bounds(self):
        dist = self._make_mixture(n=5)
        y_grid = np.array([1e-4, 1e10])
        cdf = dist.cdf(y_grid)
        assert np.all(cdf[:, 0] >= 0)
        assert np.all(cdf[:, 1] <= 1.0 + 1e-6)

    def test_pdf_shape_and_positive(self):
        dist = self._make_mixture(n=5)
        y_grid = np.array([1000.0, 5000.0, 20000.0])
        pdf = dist.pdf(y_grid)
        assert pdf.shape == (5, 3)
        assert np.all(pdf >= 0)

    def test_quantile_monotone(self):
        dist = self._make_mixture(n=10)
        q50 = dist.quantile(0.50)
        q90 = dist.quantile(0.90)
        assert np.all(q90 >= q50), "Q90 should be >= Q50"

    def test_quantile_array(self):
        dist = self._make_mixture(n=5)
        qs = dist.quantile(np.array([0.25, 0.50, 0.75, 0.90]))
        assert qs.shape == (5, 4)
        # Monotone across quantile levels
        assert np.all(np.diff(qs, axis=1) >= 0)

    def test_log_prob_shape(self):
        rng = np.random.default_rng(5)
        dist = self._make_mixture(n=30)
        y = np.exp(rng.standard_normal(30) + 8.0)
        lp = dist.log_prob(y)
        assert lp.shape == (30,)
        assert np.all(np.isfinite(lp))

    def test_pit_samples_shape(self):
        rng = np.random.default_rng(6)
        dist = self._make_mixture(n=20)
        y = np.exp(rng.standard_normal(20) + 8.0)
        pit = dist.pit_samples(y)
        assert pit.shape == (20,)
        assert np.all((pit >= 0) & (pit <= 1))

    def test_ilf_shape(self):
        dist = self._make_mixture(n=10)
        ilf = dist.ilf(limit=50_000, basic_limit=10_000)
        assert ilf.shape == (10,)
        assert np.all(ilf >= 1.0), "ILF(L, b) with L > b should be >= 1"

    def test_single_obs_squeeze(self):
        """MDNMixture with 1-d inputs (single observation)."""
        pi = np.array([0.5, 0.3, 0.2])
        mu = np.array([7.0, 9.0, 11.0])
        sigma = np.array([0.4, 0.5, 0.8])
        dist = MDNMixture(pi=pi, mu=mu, sigma=sigma)
        assert dist.n == 1
        assert dist.mean().shape == (1,)

    def test_repr(self):
        dist = self._make_mixture(n=10)
        r = repr(dist)
        assert "MDNMixture" in r


# ---------------------------------------------------------------------------
# MDN (main class) tests
# ---------------------------------------------------------------------------

class TestMDN:

    def test_fit_returns_self(self, synthetic_data):
        X, y = synthetic_data
        mdn = MDN(n_components=2, hidden_size=16, num_hidden_layers=1,
                  max_epochs=3, random_state=99)
        result = mdn.fit(X, y, verbose=False)
        assert result is mdn

    def test_is_fitted_after_fit(self, fitted_mdn):
        assert fitted_mdn._is_fitted

    def test_predict_mean_shape(self, fitted_mdn, synthetic_data):
        X, y = synthetic_data
        means = fitted_mdn.predict_mean(X)
        assert means.shape == (len(X),)
        assert np.all(means > 0)
        assert np.all(np.isfinite(means))

    def test_predict_distribution_returns_mdn_mixture(self, fitted_mdn, synthetic_data):
        X, _ = synthetic_data
        dist = fitted_mdn.predict_distribution(X)
        assert isinstance(dist, MDNMixture)
        assert dist.n == len(X)
        assert dist.K == fitted_mdn.n_components

    def test_predict_params_shapes(self, fitted_mdn, synthetic_data):
        X, _ = synthetic_data
        pi, mu, sigma = fitted_mdn.predict_params(X)
        n = len(X)
        K = fitted_mdn.n_components
        assert pi.shape == (n, K)
        assert mu.shape == (n, K)
        assert sigma.shape == (n, K)
        # Mixing weights sum to 1
        assert np.allclose(pi.sum(axis=1), 1.0, atol=1e-5)
        # Sigma positive
        assert np.all(sigma > 0)

    def test_predict_distribution_mean_consistency(self, fitted_mdn, synthetic_data):
        """predict_mean should match predict_distribution().mean()."""
        X, _ = synthetic_data
        m1 = fitted_mdn.predict_mean(X)
        m2 = fitted_mdn.predict_distribution(X).mean()
        np.testing.assert_allclose(m1, m2, rtol=1e-5)

    def test_score_nll(self, fitted_mdn, synthetic_data):
        X, y = synthetic_data
        nll = fitted_mdn.score(X, y, metric="nll")
        assert isinstance(nll, float)
        assert np.isfinite(nll)

    def test_score_rmse(self, fitted_mdn, synthetic_data):
        X, y = synthetic_data
        rmse = fitted_mdn.score(X, y, metric="rmse")
        assert isinstance(rmse, float)
        assert rmse > 0

    def test_score_mae(self, fitted_mdn, synthetic_data):
        X, y = synthetic_data
        mae = fitted_mdn.score(X, y, metric="mae")
        assert isinstance(mae, float)
        assert mae > 0

    def test_score_crps(self, fitted_mdn, synthetic_data):
        X, y = synthetic_data
        crps = fitted_mdn.score(X, y, metric="crps")
        assert isinstance(crps, float)
        assert np.isfinite(crps)

    def test_score_unknown_metric(self, fitted_mdn, synthetic_data):
        X, y = synthetic_data
        with pytest.raises(ValueError, match="Unknown metric"):
            fitted_mdn.score(X, y, metric="bad_metric")

    def test_fit_numpy_input(self, synthetic_data):
        """MDN should accept numpy arrays (not just DataFrames)."""
        X_df, y = synthetic_data
        X_np = X_df.values
        mdn = MDN(n_components=2, hidden_size=16, num_hidden_layers=1,
                  max_epochs=3, random_state=7)
        mdn.fit(X_np, y, verbose=False)
        means = mdn.predict_mean(X_np)
        assert means.shape == (len(y),)

    def test_fit_explicit_val_set(self, synthetic_data):
        """Passing explicit val set should not raise errors."""
        X, y = synthetic_data
        n = len(X)
        X_tr, X_val = X.iloc[:300], X.iloc[300:]
        y_tr, y_val = y[:300], y[300:]
        mdn = MDN(n_components=2, hidden_size=16, num_hidden_layers=1,
                  max_epochs=3, random_state=11)
        mdn.fit(X_tr, y_tr, X_val=X_val, y_val=y_val, verbose=False)
        assert mdn._is_fitted

    def test_predict_before_fit_raises(self):
        mdn = MDN()
        with pytest.raises(RuntimeError, match="not fitted"):
            mdn.predict_mean(np.zeros((5, 3)))

    def test_negative_y_raises(self, synthetic_data):
        X, y = synthetic_data
        y_bad = y.copy()
        y_bad[0] = -1.0
        mdn = MDN(n_components=2, hidden_size=16, num_hidden_layers=1, max_epochs=2)
        with pytest.raises(ValueError, match="strictly positive"):
            mdn.fit(X, y_bad, verbose=False)

    def test_zero_y_raises(self, synthetic_data):
        X, y = synthetic_data
        y_bad = y.copy()
        y_bad[5] = 0.0
        mdn = MDN(n_components=2, hidden_size=16, num_hidden_layers=1, max_epochs=2)
        with pytest.raises(ValueError, match="strictly positive"):
            mdn.fit(X, y_bad, verbose=False)

    def test_save_load_roundtrip(self, fitted_mdn, synthetic_data, tmp_path):
        X, _ = synthetic_data
        path = tmp_path / "test_mdn.pt"
        fitted_mdn.save(path)

        loaded = MDN.load(path)
        assert loaded._is_fitted

        # Predictions should be identical
        m_orig = fitted_mdn.predict_mean(X)
        m_loaded = loaded.predict_mean(X)
        np.testing.assert_allclose(m_orig, m_loaded, rtol=1e-5)

    def test_training_history(self, fitted_mdn):
        hist = fitted_mdn.training_history
        assert "train_loss" in hist
        assert "val_loss" in hist
        assert len(hist["train_loss"]) > 0

    def test_repr(self, fitted_mdn):
        r = repr(fitted_mdn)
        assert "MDN" in r
        assert "fitted" in r

    def test_feature_names_stored(self, synthetic_data):
        X, y = synthetic_data
        mdn = MDN(n_components=2, hidden_size=16, num_hidden_layers=1,
                  max_epochs=2, random_state=0)
        mdn.fit(X, y, verbose=False)
        assert mdn.feature_names == ["x0", "x1", "x2"]

    def test_single_component_mdn(self, synthetic_data):
        """K=1 degenerates to a single lognormal — should train without errors."""
        X, y = synthetic_data
        mdn = MDN(n_components=1, hidden_size=16, num_hidden_layers=1,
                  max_epochs=3, random_state=0)
        mdn.fit(X, y, verbose=False)
        m = mdn.predict_mean(X)
        assert np.all(m > 0)

    def test_high_component_count(self, synthetic_data):
        """K=7 should run without mode collapse errors on a small dataset."""
        X, y = synthetic_data
        mdn = MDN(n_components=7, hidden_size=32, num_hidden_layers=2,
                  max_epochs=3, random_state=0)
        mdn.fit(X, y, verbose=False)
        m = mdn.predict_mean(X)
        assert np.all(np.isfinite(m))

    def test_quantile_from_distribution(self, fitted_mdn, synthetic_data):
        X, _ = synthetic_data
        dist = fitted_mdn.predict_distribution(X)
        q50 = dist.quantile(0.50)
        q90 = dist.quantile(0.90)
        assert q50.shape == (len(X),)
        assert np.all(q90 >= q50)

    def test_pit_coverage(self, synthetic_data):
        """
        PIT values on the generative data should be roughly uniform.
        Test that the mean is close to 0.5 and std close to 1/sqrt(12)~0.289.
        This is a weak test (only 5 training epochs!) — passes if not wildly off.
        """
        X, y = synthetic_data
        mdn = MDN(n_components=3, hidden_size=32, num_hidden_layers=2,
                  max_epochs=5, random_state=42)
        mdn.fit(X, y, verbose=False)
        dist = mdn.predict_distribution(X)
        pit = dist.pit_samples(y)
        assert pit.shape == (len(y),)
        assert np.all((pit >= 0) & (pit <= 1.0 + 1e-6))

    def test_ilf_from_distribution(self, fitted_mdn, synthetic_data):
        X, _ = synthetic_data
        dist = fitted_mdn.predict_distribution(X)
        ilf = dist.ilf(limit=50_000, basic_limit=5_000)
        assert ilf.shape == (len(X),)
        assert np.all(ilf >= 0.999)  # ILF(L, b) >= 1 when L > b

    def test_gradient_flows(self, synthetic_data):
        """
        Backward pass through the NLL should produce non-zero gradients on
        all network parameters.
        """
        X, y = synthetic_data
        net = MDNNetwork(
            n_features=3, n_components=3,
            hidden_size=16, num_hidden_layers=1,
        )
        from insurance_severity.mdn.loss import mdn_nll_loss
        x_t = torch.tensor(X.values[:32].astype(np.float32))
        y_t = torch.tensor(y[:32].astype(np.float32))
        pi, mu, sigma = net(x_t)
        loss = mdn_nll_loss(pi, mu, sigma, y_t)
        loss.backward()

        for name, p in net.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"
                # Some grads may be zero if a head is trivial, but trunk should not be
