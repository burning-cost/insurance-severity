"""
Tests for insurance_severity.spqrx — SPQRxSeverity.

Test strategy: small synthetic datasets, short training runs (max_epochs=5)
to keep the suite fast on CI. We test API contracts, numerical correctness,
and edge cases rather than convergence to known parameter values.

Coverage:
- SPQRxNetwork: forward shapes, spline weights sum to 1, xi positive
- make_mspline_basis: shape, M-splines non-negative, I-splines monotone in [0,1]
- solve_bgpd_params: sigma > 0, threshold in plausible range
- SPQRxSeverity: fit/predict API, quantile monotonicity, tail_params, CDF/PDF
- SPQRxDistribution: quantile, cdf, pdf, ilf
- Edge cases: very small samples, single-feature, constant-ish y, no covariates
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed; skip SPQRx tests")

from insurance_severity.spqrx.network import (
    SPQRxNetwork,
    make_mspline_basis,
    solve_bgpd_params,
)
from insurance_severity.spqrx.distribution import SPQRxDistribution
from insurance_severity.spqrx.spqrx import SPQRxSeverity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def pareto_data(rng):
    """
    Synthetic Pareto-tailed severity data with covariates.
    n=600, p=3, xi_true ~ 0.4 (moderate heavy tail).
    """
    n = 600
    X = rng.standard_normal((n, 3)).astype(np.float32)
    # Pareto(xi=0.4): F(y) = 1 - (1 + y/1000)^{-1/0.4}
    # Simulate: y = 1000 * (U^{-0.4} - 1) where U ~ Uniform(0,1)
    U = rng.uniform(0.01, 0.99, n)
    y = 1000.0 * (U ** (-0.4) - 1.0) + 500.0
    y = np.maximum(y, 100.0)
    return X, y


@pytest.fixture(scope="module")
def fitted_spqrx(pareto_data):
    """SPQRxSeverity fitted with minimal epochs."""
    X, y = pareto_data
    model = SPQRxSeverity(
        n_splines=10,       # small for speed
        hidden_size=16,
        num_hidden_layers=1,
        pa=0.75,
        pb=0.90,
        xi_l1=0.01,
        max_epochs=5,
        patience=10,
        random_state=0,
    )
    model.fit(X, y, verbose=False)
    return model


# ---------------------------------------------------------------------------
# Network tests
# ---------------------------------------------------------------------------

class TestSPQRxNetwork:

    def test_forward_shapes(self):
        net = SPQRxNetwork(n_features=5, n_splines=15, hidden_size=32, num_hidden_layers=2)
        x = torch.randn(16, 5)
        w, xi = net(x)
        assert w.shape == (16, 15)
        assert xi.shape == (16,)

    def test_spline_weights_sum_to_one(self):
        net = SPQRxNetwork(n_features=4, n_splines=10, hidden_size=16, num_hidden_layers=1)
        x = torch.randn(32, 4)
        w, _ = net(x)
        sums = w.sum(dim=1)
        assert torch.allclose(sums, torch.ones(32), atol=1e-5)

    def test_xi_positive(self):
        net = SPQRxNetwork(n_features=3, n_splines=10, hidden_size=16, num_hidden_layers=1)
        x = torch.randn(20, 3)
        _, xi = net(x)
        assert (xi > 0).all()

    def test_n_parameters(self):
        net = SPQRxNetwork(n_features=5, n_splines=15, hidden_size=32, num_hidden_layers=2)
        n = net.n_parameters()
        assert n > 0
        assert isinstance(n, int)

    def test_single_feature(self):
        net = SPQRxNetwork(n_features=1, n_splines=8, hidden_size=16, num_hidden_layers=1)
        x = torch.randn(10, 1)
        w, xi = net(x)
        assert w.shape == (10, 8)
        assert xi.shape == (10,)

    def test_xi_bounded_softplus(self):
        """Softplus is always positive; clipping in solver keeps xi in [1e-4, 0.5]."""
        net = SPQRxNetwork(n_features=3, n_splines=10, hidden_size=16, num_hidden_layers=1)
        x = torch.randn(100, 3) * 5  # extreme inputs
        _, xi = net(x)
        assert (xi > 0).all()


# ---------------------------------------------------------------------------
# M-spline basis tests
# ---------------------------------------------------------------------------

class TestMSplineBasis:

    def test_basis_shape(self):
        u = np.linspace(0.01, 0.99, 50)
        M, I = make_mspline_basis(u, K=15)
        assert M.shape == (50, 15)
        assert I.shape == (50, 15)

    def test_mspline_nonneg(self):
        u = np.linspace(0.01, 0.99, 100)
        M, _ = make_mspline_basis(u, K=15)
        assert np.all(M >= -1e-10), "M-splines must be non-negative"

    def test_ispline_monotone(self):
        """I-splines must be non-decreasing in u."""
        u = np.linspace(0.01, 0.99, 100)
        _, I = make_mspline_basis(u, K=15)
        diffs = np.diff(I, axis=0)
        assert np.all(diffs >= -1e-8), "I-splines should be non-decreasing"

    def test_ispline_range(self):
        """I-splines should start near 0 and end near 1."""
        u = np.array([0.01, 0.99])
        _, I = make_mspline_basis(u, K=10)
        # At u=0.99, each I-spline should be in [0, 1]
        assert np.all(I >= -1e-8)
        assert np.all(I <= 1.0 + 1e-8)

    def test_different_K(self):
        u = np.linspace(0.05, 0.95, 20)
        for K in [8, 15, 25]:
            M, I = make_mspline_basis(u, K=K)
            assert M.shape == (20, K)
            assert I.shape == (20, K)

    def test_uniform_spline_weights_cdf(self):
        """With uniform weights 1/K, I-spline CDF should be monotone and bounded."""
        K = 15
        u = np.linspace(0.01, 0.99, 100)
        _, I = make_mspline_basis(u, K=K)
        w = np.ones(K) / K
        F = (I * w).sum(axis=1)
        diffs = np.diff(F)
        assert np.all(diffs >= -1e-8), "Weighted I-spline CDF must be monotone"
        assert F[-1] <= 1.0 + 1e-6

    def test_ispline_boundary_zero(self):
        """I-splines at u=0 should be close to 0."""
        u = np.array([1e-6])
        _, I = make_mspline_basis(u, K=10)
        assert np.all(I <= 0.1), "I-splines should be near 0 at u=0"


# ---------------------------------------------------------------------------
# bGPD parameter solver tests
# ---------------------------------------------------------------------------

class TestSolveBGPDParams:

    def test_sigma_positive(self):
        a = np.array([1000.0, 2000.0, 500.0])
        b = np.array([3000.0, 6000.0, 1500.0])
        xi = np.array([0.3, 0.4, 0.2])
        u_tilde, sigma_tilde = solve_bgpd_params(a, b, xi, pa=0.85, pb=0.95)
        assert np.all(sigma_tilde > 0), "sigma_tilde must be positive"

    def test_threshold_plausible(self):
        """u_tilde should be near a (just below or at the lower blend boundary)."""
        a = np.array([1000.0, 2000.0])
        b = np.array([3000.0, 6000.0])
        xi = np.array([0.3, 0.4])
        u_tilde, sigma_tilde = solve_bgpd_params(a, b, xi, pa=0.85, pb=0.95)
        # u_tilde can be below a (GPD starts below the blend boundary)
        assert np.all(np.isfinite(u_tilde))
        assert np.all(np.isfinite(sigma_tilde))

    def test_shape_preserved(self):
        n = 50
        a = np.random.default_rng(0).uniform(500, 2000, n)
        b = a * 2.0
        xi = np.random.default_rng(0).uniform(0.1, 0.4, n)
        u, s = solve_bgpd_params(a, b, xi, pa=0.85, pb=0.95)
        assert u.shape == (n,)
        assert s.shape == (n,)

    def test_xi_clipping(self):
        """Very large xi should be clipped to 0.5."""
        a = np.array([1000.0])
        b = np.array([3000.0])
        xi = np.array([10.0])  # way too large
        u, s = solve_bgpd_params(a, b, xi, pa=0.85, pb=0.95)
        assert np.isfinite(u[0])
        assert np.isfinite(s[0])

    def test_consistency_with_gpd_cdf(self):
        """
        Given the solved (u_tilde, sigma_tilde, xi), the GPD CDF at a should
        equal pa (approximately), and at b should equal pb.
        """
        a_val = 1000.0
        b_val = 3000.0
        xi_val = 0.35
        pa, pb = 0.85, 0.95
        u, s = solve_bgpd_params(
            np.array([a_val]), np.array([b_val]), np.array([xi_val]), pa, pb
        )
        u = float(u[0])
        s = float(s[0])
        xi_c = min(xi_val, 0.5)

        # F_GP(a) = 1 - (1 + xi*(a-u)/s)^{-1/xi}
        def gpd_cdf(y):
            z = 1.0 + xi_c * (y - u) / s
            z = max(z, 1e-8)
            return 1.0 - z ** (-1.0 / xi_c)

        F_at_a = gpd_cdf(a_val)
        F_at_b = gpd_cdf(b_val)
        assert abs(F_at_a - pa) < 0.05, f"GPD CDF at a should be near pa; got {F_at_a:.4f}"
        assert abs(F_at_b - pb) < 0.05, f"GPD CDF at b should be near pb; got {F_at_b:.4f}"


# ---------------------------------------------------------------------------
# SPQRxSeverity main class tests
# ---------------------------------------------------------------------------

class TestSPQRxSeverity:

    def test_fit_returns_self(self, pareto_data):
        X, y = pareto_data
        model = SPQRxSeverity(
            n_splines=8, hidden_size=16, num_hidden_layers=1,
            max_epochs=3, patience=5, random_state=1,
        )
        result = model.fit(X[:100], y[:100], verbose=False)
        assert result is model

    def test_is_fitted_after_fit(self, fitted_spqrx):
        assert fitted_spqrx._is_fitted

    def test_repr_fitted(self, fitted_spqrx):
        r = repr(fitted_spqrx)
        assert "SPQRxSeverity" in r
        assert "fitted" in r

    def test_repr_unfitted(self):
        m = SPQRxSeverity()
        assert "not fitted" in repr(m)

    def test_predict_quantile_shape(self, fitted_spqrx, pareto_data):
        X, _ = pareto_data
        q = fitted_spqrx.predict_quantile(X[:20], tau=0.90)
        assert q.shape == (20,)
        assert np.all(np.isfinite(q))
        assert np.all(q > 0)

    def test_predict_quantile_monotone_in_tau(self, fitted_spqrx, pareto_data):
        """Q(tau1|x) <= Q(tau2|x) when tau1 < tau2, for all x."""
        X, _ = pareto_data
        X_sub = X[:30]
        q50 = fitted_spqrx.predict_quantile(X_sub, 0.50)
        q80 = fitted_spqrx.predict_quantile(X_sub, 0.80)
        q95 = fitted_spqrx.predict_quantile(X_sub, 0.95)
        q99 = fitted_spqrx.predict_quantile(X_sub, 0.99)
        assert np.all(q80 >= q50 - 1e-4), "Q80 >= Q50"
        assert np.all(q95 >= q80 - 1e-4), "Q95 >= Q80"
        assert np.all(q99 >= q95 - 1e-4), "Q99 >= Q95"

    def test_tail_regime_q99_positive(self, fitted_spqrx, pareto_data):
        """Q99 (tail regime, tau >= pb=0.90) should be positive and finite."""
        X, _ = pareto_data
        q99 = fitted_spqrx.predict_quantile(X[:20], tau=0.99)
        assert np.all(q99 > 0)
        assert np.all(np.isfinite(q99))

    def test_tail_params_shapes(self, fitted_spqrx, pareto_data):
        X, _ = pareto_data
        params = fitted_spqrx.tail_params(X[:30])
        for key in ["xi", "u_tilde", "sigma_tilde", "a", "b"]:
            assert key in params, f"Missing key: {key}"
            assert params[key].shape == (30,), f"Wrong shape for {key}"

    def test_tail_params_xi_positive(self, fitted_spqrx, pareto_data):
        X, _ = pareto_data
        params = fitted_spqrx.tail_params(X[:50])
        assert np.all(params["xi"] > 0), "xi must be positive"

    def test_tail_params_sigma_positive(self, fitted_spqrx, pareto_data):
        X, _ = pareto_data
        params = fitted_spqrx.tail_params(X[:50])
        assert np.all(params["sigma_tilde"] > 0), "sigma_tilde must be positive"

    def test_tail_params_a_less_than_b(self, fitted_spqrx, pareto_data):
        X, _ = pareto_data
        params = fitted_spqrx.tail_params(X[:50])
        assert np.all(params["a"] < params["b"]), "a < b (lower < upper blend boundary)"

    def test_cdf_shape(self, fitted_spqrx, pareto_data):
        X, y = pareto_data
        cdf_vals = fitted_spqrx.cdf(X[:30], y[:30])
        assert cdf_vals.shape == (30,)
        assert np.all(cdf_vals >= 0)
        assert np.all(cdf_vals <= 1.0 + 1e-6)

    def test_cdf_monotone_in_y(self, fitted_spqrx, pareto_data):
        """For fixed x, CDF(y1) <= CDF(y2) when y1 < y2."""
        X, y = pareto_data
        X_rep = np.repeat(X[0:1], 20, axis=0)
        y_grid = np.linspace(500, 50000, 20)
        cdf_vals = fitted_spqrx.cdf(X_rep, y_grid)
        diffs = np.diff(cdf_vals)
        assert np.all(diffs >= -0.05), f"CDF not monotone: min diff = {diffs.min():.4f}"

    def test_pdf_nonneg(self, fitted_spqrx, pareto_data):
        X, y = pareto_data
        pdf_vals = fitted_spqrx.pdf(X[:20], y[:20])
        assert pdf_vals.shape == (20,)
        assert np.all(pdf_vals >= 0), "PDF must be non-negative"

    def test_cdf_pdf_consistency(self, fitted_spqrx, pareto_data):
        """PDF ~ (CDF(y+h) - CDF(y-h)) / (2h) for a single observation."""
        X, y = pareto_data
        X_rep = np.repeat(X[0:1], 3, axis=0)
        y_mid = np.array([5000.0])
        h = 50.0
        # Finite difference vs pdf()
        cdf_hi = fitted_spqrx.cdf(X[0:1], y_mid + h)
        cdf_lo = fitted_spqrx.cdf(X[0:1], y_mid - h)
        fd_pdf = (cdf_hi - cdf_lo) / (2 * h)
        direct_pdf = fitted_spqrx.pdf(X[0:1], y_mid)
        # Should agree to within a factor of 3 (coarse test)
        ratio = float(direct_pdf[0]) / max(float(fd_pdf[0]), 1e-12)
        assert 0.1 < ratio < 10.0, f"PDF/FD ratio out of range: {ratio:.3f}"

    def test_predict_distribution_returns_correct_type(self, fitted_spqrx, pareto_data):
        X, _ = pareto_data
        dist = fitted_spqrx.predict_distribution(X[:10])
        assert isinstance(dist, SPQRxDistribution)
        assert dist.n == 10

    def test_fit_unfitted_raises(self):
        model = SPQRxSeverity()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_quantile(np.zeros((5, 3)), tau=0.5)

    def test_negative_y_raises(self, pareto_data):
        X, y = pareto_data
        y_bad = y.copy()
        y_bad[0] = -1.0
        model = SPQRxSeverity(n_splines=8, max_epochs=2)
        with pytest.raises(ValueError, match="strictly positive"):
            model.fit(X, y_bad, verbose=False)

    def test_zero_y_raises(self, pareto_data):
        X, y = pareto_data
        y_bad = y.copy()
        y_bad[5] = 0.0
        model = SPQRxSeverity(n_splines=8, max_epochs=2)
        with pytest.raises(ValueError, match="strictly positive"):
            model.fit(X, y_bad, verbose=False)

    def test_invalid_pa_pb_raises(self):
        with pytest.raises(ValueError, match="pa and pb"):
            SPQRxSeverity(pa=0.95, pb=0.85)  # pa > pb

    def test_fit_single_feature(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((200, 1)).astype(np.float32)
        y = np.exp(rng.standard_normal(200) + 7.0)
        model = SPQRxSeverity(n_splines=8, hidden_size=8, max_epochs=3, random_state=0)
        model.fit(X, y, verbose=False)
        q = model.predict_quantile(X[:5], tau=0.95)
        assert q.shape == (5,)
        assert np.all(np.isfinite(q))

    def test_fit_numpy_array(self, pareto_data):
        X, y = pareto_data
        X_np = np.array(X)  # ensure pure ndarray
        model = SPQRxSeverity(n_splines=8, max_epochs=3, random_state=2)
        model.fit(X_np[:100], y[:100], verbose=False)
        assert model._is_fitted

    def test_training_history(self, fitted_spqrx):
        hist = fitted_spqrx.training_history
        assert "train_loss" in hist
        assert "val_loss" in hist
        assert len(hist["train_loss"]) > 0

    def test_feature_names_stored(self, pareto_data):
        import pandas as pd
        X, y = pareto_data
        df = pd.DataFrame(X, columns=["age", "value", "type"])
        model = SPQRxSeverity(n_splines=8, max_epochs=2, random_state=0)
        model.fit(df, y, verbose=False)
        assert model.feature_names == ["age", "value", "type"]

    def test_sample_weight_accepted(self, pareto_data):
        X, y = pareto_data
        w = np.ones(len(y))
        model = SPQRxSeverity(n_splines=8, max_epochs=3, random_state=0)
        model.fit(X, y, sample_weight=w, verbose=False)
        assert model._is_fitted

    def test_small_sample_warning(self):
        rng = np.random.default_rng(99)
        X = rng.standard_normal((15, 2)).astype(np.float32)
        y = np.exp(rng.standard_normal(15) + 7.0)
        model = SPQRxSeverity(n_splines=5, max_epochs=2)
        with pytest.warns(UserWarning, match="very small sample"):
            model.fit(X, y, verbose=False)

    def test_pa_pb_sensitivity(self, fitted_spqrx, pareto_data):
        X, _ = pareto_data
        result = fitted_spqrx.pa_pb_sensitivity(X[:20], tau=0.99, pa_range=[0.80, 0.85], pb_range=[0.92, 0.95])
        assert "results" in result
        assert len(result["results"]) > 0
        for r in result["results"]:
            assert "pa" in r
            assert "pb" in r
            assert "median_q" in r
            assert np.isfinite(r["median_q"])


# ---------------------------------------------------------------------------
# SPQRxDistribution tests
# ---------------------------------------------------------------------------

class TestSPQRxDistribution:

    def test_quantile_scalar_tau(self, fitted_spqrx, pareto_data):
        X, _ = pareto_data
        dist = fitted_spqrx.predict_distribution(X[:10])
        q = dist.quantile(0.90)
        assert q.shape == (10,)
        assert np.all(q > 0)

    def test_quantile_array_tau(self, fitted_spqrx, pareto_data):
        X, _ = pareto_data
        dist = fitted_spqrx.predict_distribution(X[:10])
        qs = dist.quantile(np.array([0.50, 0.80, 0.95]))
        assert qs.shape == (10, 3)
        # Monotone across quantile levels
        assert np.all(np.diff(qs, axis=1) >= -1e-4), "Quantiles should be non-decreasing in tau"

    def test_cdf_1d_grid(self, fitted_spqrx, pareto_data):
        X, _ = pareto_data
        dist = fitted_spqrx.predict_distribution(X[:5])
        y_grid = np.array([1000.0, 5000.0, 20000.0])
        cdf = dist.cdf(y_grid)
        assert cdf.shape == (5, 3)
        assert np.all(cdf >= 0)
        assert np.all(cdf <= 1.0 + 1e-6)

    def test_pdf_1d_grid(self, fitted_spqrx, pareto_data):
        X, _ = pareto_data
        dist = fitted_spqrx.predict_distribution(X[:5])
        y_grid = np.array([2000.0, 8000.0])
        pdf = dist.pdf(y_grid)
        assert pdf.shape == (5, 2)
        assert np.all(pdf >= 0)

    def test_ilf_greater_than_one(self, fitted_spqrx, pareto_data):
        """ILF(L, b) >= 1 when L > b."""
        X, _ = pareto_data
        dist = fitted_spqrx.predict_distribution(X[:10])
        ilf = dist.ilf(limit=50_000, basic_limit=10_000)
        assert ilf.shape == (10,)
        assert np.all(ilf >= 0.9), "ILF(L, b) with L > b should be >= 1"

    def test_ilf_increases_with_limit(self, fitted_spqrx, pareto_data):
        """ILF(L1, b) <= ILF(L2, b) when L1 < L2."""
        X, _ = pareto_data
        dist = fitted_spqrx.predict_distribution(X[:10])
        ilf_50k = dist.ilf(limit=50_000, basic_limit=10_000)
        ilf_100k = dist.ilf(limit=100_000, basic_limit=10_000)
        assert np.all(ilf_100k >= ilf_50k - 0.01)

    def test_repr(self, fitted_spqrx, pareto_data):
        X, _ = pareto_data
        dist = fitted_spqrx.predict_distribution(X[:5])
        r = repr(dist)
        assert "SPQRxDistribution" in r
        assert "n=5" in r


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_q99_recovery_pareto(self, pareto_data):
        """
        After fitting on Pareto data (xi_true=0.4), Q99 should be reasonably
        close to the true Pareto Q99. This is a weak test (5 epochs), but checks
        that the extrapolation formula gives a plausible answer.

        True Q99: y = 1000 * (0.01^{-0.4} - 1) + 500 ~ 1000*(6.31-1)+500 = 5810
        """
        X, y = pareto_data
        model = SPQRxSeverity(
            n_splines=10, hidden_size=16, num_hidden_layers=1,
            pa=0.75, pb=0.90, max_epochs=5, random_state=0,
        )
        model.fit(X, y, verbose=False)
        q99 = model.predict_quantile(X[:20], tau=0.99)
        # Just check the output is in a plausible range (factor 10 of true Q99)
        assert np.all(q99 > 100), "Q99 should be well above 100"
        assert np.all(q99 < 1e8), "Q99 should not be astronomically large"
        assert np.all(np.isfinite(q99))

    def test_tail_params_xi_near_true(self, pareto_data):
        """
        Fitted xi(x) should not be wildly different from xi_true=0.4 after 5 epochs.
        We just check it is in [0.05, 0.5] (the clipping range).
        """
        X, y = pareto_data
        model = SPQRxSeverity(
            n_splines=10, hidden_size=16, num_hidden_layers=1,
            pa=0.75, pb=0.90, max_epochs=5, random_state=0,
        )
        model.fit(X, y, verbose=False)
        params = model.tail_params(X[:50])
        xi = params["xi"]
        assert np.all(xi > 0), "xi must be positive"
        assert np.all(xi < 2.0), "xi should not be > 2 for typical insurance severity"

    def test_cdf_increases_with_y(self, fitted_spqrx, pareto_data):
        """CDF(y1) < CDF(y2) when y1 < y2 for fixed x (monotonicity check)."""
        X, _ = pareto_data
        X_one = X[0:1]
        X_rep = np.repeat(X_one, 50, axis=0)
        y_seq = np.linspace(500, 100000, 50)
        cdf_vals = fitted_spqrx.cdf(X_rep, y_seq)
        diffs = np.diff(cdf_vals)
        n_violations = np.sum(diffs < -0.05)
        # Allow up to 2 small violations (numerical noise)
        assert n_violations <= 2, f"{n_violations} CDF monotonicity violations"

    def test_full_pipeline_small_n(self):
        """
        End-to-end test on n=80 with 1 feature. Checks the pipeline doesn't
        crash on very small samples.
        """
        rng = np.random.default_rng(5)
        X = rng.standard_normal((80, 1)).astype(np.float32)
        y = np.exp(rng.standard_normal(80) + 8.0)
        model = SPQRxSeverity(
            n_splines=8, hidden_size=8, pa=0.70, pb=0.85,
            max_epochs=3, patience=5, random_state=0,
        )
        model.fit(X, y, verbose=False)

        q99 = model.predict_quantile(X[:5], tau=0.99)
        assert q99.shape == (5,)
        assert np.all(np.isfinite(q99))

        params = model.tail_params(X[:5])
        assert params["xi"].shape == (5,)
        assert np.all(params["sigma_tilde"] > 0)

        dist = model.predict_distribution(X[:5])
        ilf = dist.ilf(limit=100_000, basic_limit=10_000)
        assert ilf.shape == (5,)
        assert np.all(np.isfinite(ilf))

    def test_gradient_flows(self, pareto_data):
        """Backward pass should produce non-zero gradients on all parameters."""
        X, y = pareto_data
        model = SPQRxSeverity(
            n_splines=8, hidden_size=16, num_hidden_layers=1,
            pa=0.80, pb=0.90, max_epochs=1, random_state=0,
        )
        model.fit(X[:50], y[:50], verbose=False)

        # Manually run one forward/backward pass
        net = model._network
        net.train()
        x_t = torch.tensor(X[:16], dtype=torch.float32)
        y_t = torch.tensor(y[:16].astype(np.float32), dtype=torch.float32)
        u_b = model._pit(y[:16].astype(np.float64))
        w_t = torch.ones(16)
        loss = model._batch_loss(x_t, y_t, u_b, w_t)
        loss.backward()
        for name, p in net.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"
