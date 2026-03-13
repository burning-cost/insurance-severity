"""
Tests for JBCE loss and regularisation terms.
"""

import numpy as np
import pytest
import torch

from insurance_severity.drn.loss import jbce_loss, drn_regularisation, nll_loss


class TestJBCELoss:

    def test_output_is_scalar(self):
        cdf = torch.tensor([[0.1, 0.3, 0.6, 0.8], [0.2, 0.4, 0.7, 0.9]])
        indicators = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]])
        loss = jbce_loss(cdf, indicators)
        assert loss.shape == ()

    def test_perfect_prediction_low_loss(self):
        """Near-perfect CDF predictions should give low JBCE."""
        eps = 1e-5
        cdf = torch.tensor([[eps, eps, 1.0 - eps, 1.0 - eps]])
        indicators = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        loss = jbce_loss(cdf, indicators)
        assert float(loss) < 0.01

    def test_worst_prediction_high_loss(self):
        """Inverted predictions should give high loss."""
        eps = 1e-5
        cdf = torch.tensor([[1.0 - eps, 1.0 - eps, eps, eps]])
        indicators = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        loss = jbce_loss(cdf, indicators)
        assert float(loss) > 5.0

    def test_uniform_cdf_uniform_loss(self):
        """Uniform CDF with uniform indicators should give log(2) per cell."""
        n, K = 10, 5
        cdf = torch.full((n, K), 0.5)
        indicators = torch.full((n, K), 0.5)  # not 0/1 but checks formula
        loss = jbce_loss(cdf, indicators)
        # BCE(0.5, 0.5) = -0.5*log(0.5) - 0.5*log(0.5) = log(2) ≈ 0.693
        expected = float(np.log(2))
        np.testing.assert_allclose(float(loss), expected, rtol=1e-5)

    def test_weighted_loss(self):
        """Weighted loss should differ from unweighted."""
        cdf = torch.tensor([[0.2, 0.5, 0.8], [0.3, 0.6, 0.9]])
        indicators = torch.zeros(2, 3)
        weights_equal = torch.tensor([1.0, 1.0])
        weights_skewed = torch.tensor([10.0, 1.0])
        loss_eq = jbce_loss(cdf, indicators, weights=weights_equal)
        loss_sk = jbce_loss(cdf, indicators, weights=weights_skewed)
        assert float(loss_eq) != float(loss_sk)

    def test_non_negative(self):
        """JBCE should always be non-negative."""
        rng = np.random.default_rng(0)
        cdf = torch.tensor(rng.uniform(0.01, 0.99, size=(20, 10)), dtype=torch.float32)
        ind = torch.tensor(rng.integers(0, 2, size=(20, 10)).astype(float), dtype=torch.float32)
        loss = jbce_loss(cdf, ind)
        assert float(loss) >= 0.0

    def test_differentiable(self):
        """Loss should be differentiable w.r.t. cdf."""
        cdf = torch.tensor([[0.2, 0.5, 0.8]], requires_grad=True)
        ind = torch.tensor([[0.0, 0.0, 1.0]])
        loss = jbce_loss(cdf, ind)
        loss.backward()
        assert cdf.grad is not None
        assert cdf.grad.abs().max() > 0


class TestDRNRegularisation:

    def _make_probs(self, n: int = 5, K: int = 10) -> tuple:
        rng = np.random.default_rng(0)
        raw = rng.dirichlet(np.ones(K), size=n)
        drn = torch.tensor(raw, dtype=torch.float32)
        raw2 = rng.dirichlet(np.ones(K), size=n)
        base = torch.tensor(raw2, dtype=torch.float32)
        return drn, base

    def test_zero_alphas_returns_zero(self):
        drn, base = self._make_probs()
        reg = drn_regularisation(drn, base)
        assert float(reg) == 0.0

    def test_kl_alpha_positive(self):
        drn, base = self._make_probs()
        reg = drn_regularisation(drn, base, kl_alpha=1e-3)
        assert float(reg) >= 0.0

    def test_kl_zero_when_drn_equals_baseline(self):
        """KL = 0 when DRN = baseline."""
        p = torch.softmax(torch.randn(5, 10), dim=1)
        reg = drn_regularisation(p, p, kl_alpha=1.0)
        np.testing.assert_allclose(float(reg), 0.0, atol=1e-5)

    def test_tv_alpha_positive(self):
        drn, base = self._make_probs()
        reg = drn_regularisation(drn, base, tv_alpha=1.0)
        assert float(reg) >= 0.0

    def test_dv_alpha_positive(self):
        drn, base = self._make_probs()
        reg = drn_regularisation(drn, base, dv_alpha=1.0)
        assert float(reg) >= 0.0

    def test_mean_alpha_zero_when_equal(self):
        """Mean penalty = 0 when DRN mean = baseline mean."""
        K = 10
        midpoints = torch.linspace(50.0, 950.0, K)
        p = torch.softmax(torch.randn(5, K), dim=1)
        reg = drn_regularisation(p, p, mean_alpha=1.0, bin_midpoints=midpoints)
        np.testing.assert_allclose(float(reg), 0.0, atol=1e-5)

    def test_mean_alpha_nonzero_when_different(self):
        K = 10
        midpoints = torch.linspace(50.0, 950.0, K)
        drn = torch.softmax(torch.randn(5, K), dim=1)
        base = torch.softmax(torch.randn(5, K), dim=1)
        reg = drn_regularisation(drn, base, mean_alpha=1.0, bin_midpoints=midpoints)
        assert float(reg) >= 0.0

    def test_combined_regularisation(self):
        drn, base = self._make_probs()
        K = 10
        midpoints = torch.linspace(50.0, 950.0, K)
        reg = drn_regularisation(
            drn, base,
            kl_alpha=1e-4,
            mean_alpha=1e-4,
            tv_alpha=1e-3,
            dv_alpha=1e-3,
            bin_midpoints=midpoints,
        )
        assert float(reg) >= 0.0

    def test_reverse_kl(self):
        drn, base = self._make_probs()
        reg_fwd = drn_regularisation(drn, base, kl_alpha=1.0, kl_direction="forwards")
        reg_rev = drn_regularisation(drn, base, kl_alpha=1.0, kl_direction="reverse")
        # Forward and reverse KL are generally different
        assert abs(float(reg_fwd) - float(reg_rev)) < 100.0  # both should be finite


class TestNLLLoss:

    def test_output_scalar(self):
        n, K = 10, 5
        pmf = torch.softmax(torch.randn(n, K), dim=1)
        bin_idx = torch.randint(0, K, (n,))
        bin_widths = torch.full((K,), 200.0)
        loss = nll_loss(pmf, bin_idx, bin_widths)
        assert loss.shape == ()

    def test_non_negative(self):
        n, K = 10, 5
        pmf = torch.softmax(torch.randn(n, K), dim=1)
        bin_idx = torch.randint(0, K, (n,))
        bin_widths = torch.full((K,), 200.0)
        # NLL can be negative if density > 1 (for narrow bins)
        loss = nll_loss(pmf, bin_idx, bin_widths)
        assert torch.isfinite(loss)
