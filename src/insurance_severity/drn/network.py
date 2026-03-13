"""
DRNNetwork: the PyTorch feedforward neural network component.

This module contains only the neural network — no baseline coupling,
no training logic, no data handling. The DRN class in drn.py
orchestrates everything else.

Architecture:
    Input(p) -> [Linear(hidden_size) -> LeakyReLU -> Dropout] * n_layers -> Linear(K)

Output is K log-adjustment values (one per histogram bin). The DRN class
combines these with log(baseline_probs) and applies softmax to get the
refined bin probabilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DRNNetwork(nn.Module):
    """
    Feedforward neural network producing K log-adjustment values.

    Parameters
    ----------
    n_features : int
        Number of input features.
    n_bins : int
        Number of histogram bins K. Output dimension.
    hidden_size : int
        Width of each hidden layer.
    num_hidden_layers : int
        Number of hidden layers (excluding input and output).
    dropout_rate : float
        Dropout probability applied after each hidden layer activation.
        Set to 0.0 to disable dropout.

    Notes
    -----
    The output layer has no activation — it outputs raw log-adjustment values.
    These are added to log(baseline_probs) before softmax, following
    the DRN formulation: p_k = softmax(log(b_k) + delta_k(x)).

    When all weights are zero (at initialisation with baseline_start=True),
    the output is all zeros, and softmax(log(b_k) + 0) = softmax(log(b_k)) = b_k.
    This means the DRN reduces to the GLM baseline at initialisation.
    """

    def __init__(
        self,
        n_features: int,
        n_bins: int,
        hidden_size: int = 75,
        num_hidden_layers: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_bins = n_bins
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate

        layers: list[nn.Module] = []

        # Input -> first hidden layer
        layers.append(nn.Linear(n_features, hidden_size))
        layers.append(nn.LeakyReLU(negative_slope=0.01))
        if dropout_rate > 0:
            layers.append(nn.Dropout(p=dropout_rate))

        # Additional hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))

        # Output layer
        layers.append(nn.Linear(hidden_size, n_bins))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_features)

        Returns
        -------
        torch.Tensor, shape (batch, n_bins)
            Raw log-adjustment values (no activation applied).
        """
        return self.net(x)

    def reset_to_baseline(self) -> None:
        """
        Zero-initialise the output layer so the network starts at the baseline.

        When the output layer weights and biases are zero, forward() returns
        zeros for all inputs, and the DRN recovers the baseline distribution
        exactly (before any training). This is the recommended initialisation
        for training stability.
        """
        # Zero-initialise only the last Linear layer (output layer)
        last_linear = None
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                last_linear = module
        if last_linear is not None:
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)

    def n_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
