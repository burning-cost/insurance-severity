"""
SPQRxNetwork: PyTorch neural network for SPQRx severity modelling.

Architecture:
- Shared MLP backbone with tanh activations
- Spline head: K softmax outputs (mixing weights for M-splines)
- Xi head: single softplus output (GPD tail shape parameter)

Also contains M-spline / I-spline basis construction (de Boor recursion)
and the analytic bGPD parameter solver (Eq 5 of Majumder & Richards 2025).

References
----------
Majumder, S. & Richards, J. (2025). 'Semi-parametric bulk and tail regression
    using spline-based neural networks.' arXiv:2504.19994.
Xu, G. & Reich, B.J. (2021). 'Bayesian nonparametric quantile process
    regression.' JASA.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# M-spline basis (pure numpy, computed once and reused)
# ---------------------------------------------------------------------------


def make_mspline_basis(u: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute M-spline and I-spline basis matrices for probability-integral-
    transformed values u in [0, 1].

    M-splines are third-order (cubic) B-splines normalised to integrate to 1
    over [0, 1].  I-splines are their integrals, giving a monotone non-
    decreasing basis for modelling CDFs.

    Parameters
    ----------
    u : np.ndarray, shape (n,)
        PIT-transformed values in [0, 1].
    K : int
        Number of spline basis functions.

    Returns
    -------
    M : np.ndarray, shape (n, K)  — M-spline values.
    I : np.ndarray, shape (n, K)  — I-spline values (antiderivatives).
    """
    order = 3  # cubic
    n_knots_inner = K - order + 1  # interior knots including boundaries
    # Uniform knot sequence with clamped end knots
    inner = np.linspace(0.0, 1.0, n_knots_inner)
    t = np.concatenate([
        np.zeros(order - 1),
        inner,
        np.ones(order - 1),
    ])  # total length = K + order - 1 + 2*(order-1) -- wait, let me recalculate
    # Actually: len(t) = K + order (for clamped B-splines)
    # K basis functions of order m need K + m knots
    n_knots_inner_strict = K - order  # interior knots (excluding boundaries)
    inner_strict = np.linspace(0.0, 1.0, n_knots_inner_strict + 2)  # includes 0 and 1
    t = np.concatenate([
        np.zeros(order),
        inner_strict[1:-1],
        np.ones(order),
    ])
    # t has length: order + (K - order) + order = K + order ... that's K+3 for order=3

    u_clipped = np.clip(u, 0.0, 1.0 - 1e-9)  # avoid boundary issues
    M = _bspline_basis(u_clipped, t, K, order)    # shape (n, K)
    I = _ispline_basis(u_clipped, t, K, order)    # shape (n, K)
    return M, I


def _bspline_basis(x: np.ndarray, t: np.ndarray, K: int, order: int) -> np.ndarray:
    """
    B-spline basis of given order via de Boor recursion (Cox-de Boor).

    Returns normalised M-spline values M_k(x) = B_k(x) * order / (t_{k+order} - t_k).
    M-splines integrate to 1 over their support.

    Parameters
    ----------
    x : np.ndarray, shape (n,)
    t : np.ndarray — knot sequence.
    K : int — number of basis functions.
    order : int — spline order (3 = cubic).

    Returns
    -------
    np.ndarray, shape (n, K)
    """
    n = len(x)
    # Start: order-1 = 0 (indicator functions)
    # B_{k,1}(x) = 1 if t_k <= x < t_{k+1} else 0
    B = np.zeros((n, len(t) - 1), dtype=np.float64)
    for k in range(len(t) - 1):
        B[:, k] = ((x >= t[k]) & (x < t[k + 1])).astype(np.float64)

    # Recursion for orders 2 up to `order`
    for d in range(2, order + 1):
        B_new = np.zeros((n, len(t) - d), dtype=np.float64)
        for k in range(len(t) - d):
            denom1 = t[k + d - 1] - t[k]
            denom2 = t[k + d] - t[k + 1]
            term1 = 0.0
            term2 = 0.0
            if denom1 > 1e-12:
                term1 = (x - t[k]) / denom1 * B[:, k]
            if denom2 > 1e-12:
                term2 = (t[k + d] - x) / denom2 * B[:, k + 1]
            B_new[:, k] = term1 + term2
        B = B_new  # shape (n, len(t) - d)

    B_final = B[:, :K]  # shape (n, K) — take first K columns

    # Convert to M-splines: M_k = B_k * order / (t_{k+order} - t_k)
    M = np.zeros_like(B_final)
    for k in range(K):
        denom = t[k + order] - t[k]
        if denom > 1e-12:
            M[:, k] = B_final[:, k] * order / denom
        # else M[:, k] = 0 (zero-support interval)

    return M


def _ispline_basis(x: np.ndarray, t: np.ndarray, K: int, order: int) -> np.ndarray:
    """
    I-spline basis: antiderivatives of M-splines.

    I_k(x) = integral_0^x M_k(u) du  in [0, 1].

    Computed via the recurrence: for each evaluation point x, sum all M-spline
    contributions from the left using the numerical antiderivative property of
    B-splines.

    Approximation: integrate each M_k numerically using a 200-point quadrature.
    This is computed once at training time and cached as a precomputed basis.

    Parameters
    ----------
    x : np.ndarray, shape (n,)
    t : np.ndarray
    K : int
    order : int

    Returns
    -------
    np.ndarray, shape (n, K)
    """
    # Use 200-point Gauss-Legendre quadrature nodes on [0, 1] for each x
    # I_k(x) = integral_0^x M_k(u) du ~ sum_j w_j * M_k(x_j) for x_j <= x
    n_quad = 200
    quad_nodes = np.linspace(0.0, 1.0, n_quad + 1)[:-1] + 0.5 / n_quad  # midpoints
    quad_w = np.ones(n_quad) / n_quad  # uniform weights

    M_quad = _bspline_basis(quad_nodes, t, K, order)  # (n_quad, K)

    # For each x[i], I_k(x[i]) = sum_{j: quad_nodes[j] <= x[i]} quad_w[j] * M_quad[j, k]
    # Vectorised: cumulative sum over quad axis
    cum_I = np.cumsum(M_quad * quad_w[:, np.newaxis], axis=0)  # (n_quad, K)

    n = len(x)
    I = np.zeros((n, K), dtype=np.float64)
    for i in range(n):
        # Find largest j where quad_nodes[j] <= x[i]
        j = np.searchsorted(quad_nodes, x[i], side="right") - 1
        if j >= 0:
            I[i] = cum_I[min(j, n_quad - 1)]
    return I


def make_mspline_basis_torch(
    u_np: np.ndarray, K: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert numpy M/I-spline basis to torch tensors.

    Parameters
    ----------
    u_np : np.ndarray, shape (n,)
    K : int
    device : torch.device

    Returns
    -------
    M_t : torch.Tensor, shape (n, K)
    I_t : torch.Tensor, shape (n, K)
    """
    M, I = make_mspline_basis(u_np, K)
    M_t = torch.tensor(M, dtype=torch.float32, device=device)
    I_t = torch.tensor(I, dtype=torch.float32, device=device)
    return M_t, I_t


# ---------------------------------------------------------------------------
# bGPD parameter solver (Equation 5 of Majumder & Richards 2025)
# ---------------------------------------------------------------------------


def solve_bgpd_params(
    a: np.ndarray,
    b: np.ndarray,
    xi: np.ndarray,
    pa: float,
    pb: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve analytically for GPD threshold ũ(x) and scale sigma_tilde(x).

    The GPD CDF must match the bulk CDF at both blending boundaries:
        F_GP(a | ũ, sigma, xi) = pa
        F_GP(b | ũ, sigma, xi) = pb

    Derivation (two-point GPD constraint):
        F_GP(y) = 1 - (1 + xi*(y - ũ)/sigma)^{-1/xi}
        So 1 - pa = (1 + xi*(a - ũ)/sigma)^{-1/xi}
           1 - pb = (1 + xi*(b - ũ)/sigma)^{-1/xi}

    Let r_a = (1 - pa)^{-xi} - 1 and r_b = (1 - pb)^{-xi} - 1. Then:
        sigma = xi * (b - a) / (r_b - r_a)
        ũ = a - (sigma / xi) * r_a

    Parameters
    ----------
    a : np.ndarray, shape (n,)  lower blend boundary Q(pa|x)
    b : np.ndarray, shape (n,)  upper blend boundary Q(pb|x)
    xi : np.ndarray, shape (n,)  tail shape (positive)
    pa : float  lower quantile level
    pb : float  upper quantile level

    Returns
    -------
    u_tilde : np.ndarray, shape (n,)
    sigma_tilde : np.ndarray, shape (n,)
    """
    xi = np.clip(xi, 1e-4, 0.5)  # constrain xi to valid range
    r_a = np.power(1.0 - pa, -xi) - 1.0
    r_b = np.power(1.0 - pb, -xi) - 1.0
    denom = r_b - r_a
    # Clamp denom away from zero
    denom = np.where(np.abs(denom) < 1e-8, 1e-8, denom)
    sigma_tilde = xi * (b - a) / denom
    sigma_tilde = np.maximum(sigma_tilde, 1e-4)  # must be positive
    u_tilde = a - (sigma_tilde / xi) * r_a
    return u_tilde, sigma_tilde


def solve_bgpd_params_torch(
    a: torch.Tensor,
    b: torch.Tensor,
    xi: torch.Tensor,
    pa: float,
    pb: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Torch version of solve_bgpd_params for use in the training loop.
    """
    xi = torch.clamp(xi, min=1e-4, max=0.5)
    r_a = (1.0 - pa) ** (-xi) - 1.0
    r_b = (1.0 - pb) ** (-xi) - 1.0
    denom = r_b - r_a
    denom = torch.where(denom.abs() < 1e-8, torch.full_like(denom, 1e-8), denom)
    sigma_tilde = xi * (b - a) / denom
    sigma_tilde = torch.clamp(sigma_tilde, min=1e-4)
    u_tilde = a - (sigma_tilde / xi) * r_a
    return u_tilde, sigma_tilde


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


class SPQRxNetwork(nn.Module):
    """
    Neural network for SPQRxSeverity.

    Shared MLP backbone with two output heads:

    - **Spline head**: K logits -> softmax -> M-spline mixing weights w(x).
      The bulk CDF is F_bulk(u|x) = sum_k w_k(x) * I_k(u).
    - **Xi head**: 1 logit -> softplus -> xi(x) > 0 (GPD tail shape).
      Softplus naturally constrains xi > 0; the bGPD solver further clips to [1e-4, 0.5].

    Parameters
    ----------
    n_features : int
    n_splines : int  — K M-spline basis functions.
    hidden_size : int
    num_hidden_layers : int
    dropout_rate : float
    """

    def __init__(
        self,
        n_features: int,
        n_splines: int,
        hidden_size: int = 32,
        num_hidden_layers: int = 2,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_splines = n_splines
        self.hidden_size = hidden_size

        # Shared backbone
        trunk: list[nn.Module] = []
        trunk.append(nn.Linear(n_features, hidden_size))
        trunk.append(nn.Tanh())
        if dropout_rate > 0:
            trunk.append(nn.Dropout(p=dropout_rate))
        for _ in range(num_hidden_layers - 1):
            trunk.append(nn.Linear(hidden_size, hidden_size))
            trunk.append(nn.Tanh())
            if dropout_rate > 0:
                trunk.append(nn.Dropout(p=dropout_rate))
        self.trunk = nn.Sequential(*trunk)

        # Output heads
        self.head_spline = nn.Linear(hidden_size, n_splines)
        self.head_xi = nn.Linear(hidden_size, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        # Uniform spline weights at initialisation
        nn.init.zeros_(self.head_spline.weight)
        nn.init.zeros_(self.head_spline.bias)
        # xi: small positive initial value (softplus(0.5) ~ 0.97, clip -> 0.5)
        nn.init.zeros_(self.head_xi.weight)
        nn.init.constant_(self.head_xi.bias, -1.0)  # softplus(-1) ~ 0.31

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_features)

        Returns
        -------
        w : torch.Tensor, shape (batch, n_splines) — softmax mixing weights
        xi : torch.Tensor, shape (batch,) — tail shape parameter
        """
        h = self.trunk(x)
        w = torch.softmax(self.head_spline(h), dim=1)
        xi = torch.nn.functional.softplus(self.head_xi(h)).squeeze(-1)
        return w, xi

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
