"""
Tests for DRNNetwork — the PyTorch feedforward neural network.
"""

import numpy as np
import pytest
import torch

from insurance_severity.drn.network import DRNNetwork


class TestDRNNetwork:

    def test_forward_shape(self):
        net = DRNNetwork(n_features=5, n_bins=20)
        x = torch.randn(10, 5)
        out = net(x)
        assert out.shape == (10, 20)

    def test_output_is_log_adjustments_not_probs(self):
        """Output should not be probabilities — it's unbounded log-adjustments."""
        net = DRNNetwork(n_features=3, n_bins=10)
        x = torch.randn(5, 3)
        out = net(x)
        # Not constrained to [0, 1] or sum-to-one
        assert not torch.all((out >= 0) & (out <= 1)), (
            "Output should be unbounded log-adjustments, not probabilities"
        )

    def test_baseline_start_zeros_output(self):
        """After reset_to_baseline(), output should be all zeros."""
        net = DRNNetwork(n_features=4, n_bins=15)
        net.reset_to_baseline()
        x = torch.randn(8, 4)
        out = net(x)
        # With zero output layer, output is all zeros regardless of input
        np.testing.assert_allclose(out.detach().numpy(), 0.0, atol=1e-6)

    def test_reset_to_baseline_recovers_baseline(self):
        """
        After reset_to_baseline(), softmax(log(b_k) + 0) = b_k.
        DRN reduces to baseline exactly.
        """
        net = DRNNetwork(n_features=3, n_bins=8)
        net.reset_to_baseline()
        baseline_probs = torch.softmax(torch.randn(5, 8), dim=1)  # random valid probs
        log_adj = net(torch.randn(5, 3))
        drn_pmf = torch.softmax(torch.log(baseline_probs) + log_adj, dim=1)
        np.testing.assert_allclose(
            drn_pmf.detach().numpy(),
            baseline_probs.detach().numpy(),
            atol=1e-5,
        )

    def test_n_parameters_positive(self):
        net = DRNNetwork(n_features=10, n_bins=50)
        assert net.n_parameters() > 0

    def test_n_parameters_scales_with_hidden(self):
        net_small = DRNNetwork(n_features=5, n_bins=10, hidden_size=16)
        net_large = DRNNetwork(n_features=5, n_bins=10, hidden_size=128)
        assert net_large.n_parameters() > net_small.n_parameters()

    def test_multiple_hidden_layers(self):
        net = DRNNetwork(n_features=5, n_bins=10, num_hidden_layers=4)
        x = torch.randn(3, 5)
        out = net(x)
        assert out.shape == (3, 10)

    def test_no_dropout(self):
        """dropout_rate=0 should work without errors."""
        net = DRNNetwork(n_features=5, n_bins=10, dropout_rate=0.0)
        x = torch.randn(5, 5)
        out = net(x)
        assert out.shape == (5, 10)

    def test_single_hidden_layer(self):
        net = DRNNetwork(n_features=3, n_bins=5, num_hidden_layers=1)
        x = torch.randn(4, 3)
        out = net(x)
        assert out.shape == (4, 5)

    def test_eval_mode_no_dropout(self):
        """In eval mode, output should be deterministic (no dropout stochasticity)."""
        net = DRNNetwork(n_features=5, n_bins=10, dropout_rate=0.5)
        net.eval()
        x = torch.randn(10, 5)
        with torch.no_grad():
            out1 = net(x)
            out2 = net(x)
        np.testing.assert_allclose(out1.numpy(), out2.numpy())

    def test_gradient_flows(self):
        """Gradients should flow through the network."""
        net = DRNNetwork(n_features=5, n_bins=10)
        net.train()
        x = torch.randn(8, 5, requires_grad=True)
        out = net(x)
        loss = out.mean()
        loss.backward()
        # Check that at least some parameters have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().max() > 0
            for p in net.parameters()
        )
        assert has_grad
