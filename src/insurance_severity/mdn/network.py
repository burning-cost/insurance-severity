"""
MDNNetwork: PyTorch neural network producing mixture density parameters.

Architecture follows Bishop (1994): a feedforward network whose output layer
is split into three heads — mixing logits, component means, and component
log-scales — corresponding to the K Gaussian mixture components.

The network operates on log-transformed targets. If the raw severity is y,
the network models log(y) with Gaussian mixture components, which is
equivalent to modelling y with a lognormal mixture. This is the recommended
approach for insurance severity: positive support, right-skewed, log-space
Gaussian components are well-behaved.

References
----------
Bishop, C.M. (1994). 'Mixture Density Networks.'
    Technical Report NCRG/94/004, Aston University.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MDNNetwork(nn.Module):
    """
    Feedforward network producing K-component Gaussian mixture parameters.

    The output is split into three heads:

    - **Mixing logits** (K values) — passed through softmax to give mixing
      weights π_k(x), summing to 1.
    - **Component means** (K values) — raw linear outputs μ_k(x). When the
      target is log(y), these are means of the log-scale components.
    - **Log-scales** (K values) — log σ_k(x). Exponentiated inside the MDN
      to enforce positivity. Initialised with a positive bias so initial
      variances are non-zero.

    Parameters
    ----------
    n_features : int
        Number of input features.
    n_components : int
        Number of mixture components K.
    hidden_size : int
        Width of each hidden layer.
    num_hidden_layers : int
        Number of hidden layers (excluding input projection and output heads).
    dropout_rate : float
        Dropout probability applied after each hidden activation. 0 disables.

    Notes
    -----
    Output layer weight initialisations:

    - Mixing logits: zeros (uniform mixture weights at start).
    - Component means: small random weights; biases spread across the data
      range — but the MDN class re-initialises biases via
      ``init_from_data()`` to span the observed range.
    - Log-scales: zeros with positive bias (0.5) so exp(0.5) ≈ 1.65 is
      the initial scale. This avoids the NaN-from-zero-variance failure mode.
    """

    def __init__(
        self,
        n_features: int,
        n_components: int,
        hidden_size: int = 64,
        num_hidden_layers: int = 2,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_components = n_components
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate

        # Shared trunk
        trunk_layers: list[nn.Module] = []
        trunk_layers.append(nn.Linear(n_features, hidden_size))
        trunk_layers.append(nn.Tanh())
        if dropout_rate > 0:
            trunk_layers.append(nn.Dropout(p=dropout_rate))

        for _ in range(num_hidden_layers - 1):
            trunk_layers.append(nn.Linear(hidden_size, hidden_size))
            trunk_layers.append(nn.Tanh())
            if dropout_rate > 0:
                trunk_layers.append(nn.Dropout(p=dropout_rate))

        self.trunk = nn.Sequential(*trunk_layers)

        # Three output heads — each is a single linear layer
        self.head_pi = nn.Linear(hidden_size, n_components)      # mixing logits
        self.head_mu = nn.Linear(hidden_size, n_components)      # component means
        self.head_log_sigma = nn.Linear(hidden_size, n_components)  # log-scales

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise output heads for training stability."""
        # Mixing logits: zeros → uniform weights at epoch 0
        nn.init.zeros_(self.head_pi.weight)
        nn.init.zeros_(self.head_pi.bias)

        # Means: small random weights; biases will be overridden by init_from_data
        nn.init.xavier_normal_(self.head_mu.weight, gain=0.1)
        nn.init.zeros_(self.head_mu.bias)

        # Log-scales: zeros with positive bias → exp(0.5) ≈ 1.65 initial scale
        nn.init.zeros_(self.head_log_sigma.weight)
        nn.init.constant_(self.head_log_sigma.bias, 0.5)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_features)
            Input features.

        Returns
        -------
        pi : torch.Tensor, shape (batch, n_components)
            Mixing weights — sum to 1 along component dimension.
        mu : torch.Tensor, shape (batch, n_components)
            Component means (in log-space when target is log-transformed).
        sigma : torch.Tensor, shape (batch, n_components)
            Component standard deviations (positive, with variance floor
            applied in the loss function).
        """
        h = self.trunk(x)
        pi = torch.softmax(self.head_pi(h), dim=1)
        mu = self.head_mu(h)
        sigma = torch.exp(self.head_log_sigma(h))
        return pi, mu, sigma

    def init_from_data(self, log_y: torch.Tensor) -> None:
        """
        Initialise component mean biases to span the data range.

        Spreads the K component mean biases evenly between the 10th and 90th
        percentile of ``log_y``. This reduces early-training mode collapse by
        ensuring components start at different locations.

        Parameters
        ----------
        log_y : torch.Tensor, shape (n,)
            Log-transformed observed severities.
        """
        lo = float(torch.quantile(log_y, 0.10).item())
        hi = float(torch.quantile(log_y, 0.90).item())
        if lo >= hi:
            hi = lo + 1.0
        spread = torch.linspace(lo, hi, self.n_components)
        with torch.no_grad():
            self.head_mu.bias.copy_(spread)

    def n_parameters(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
