"""
Tests for CMRSAllocator (insurance_severity.cmrs).

Pure scipy/numpy implementation — all tests run without torch.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def exp_2():
    """Two exponential participants: rate1=1, rate2=2. Means 1 and 0.5."""
    from insurance_severity.cmrs import CMRSAllocator
    alloc = CMRSAllocator(distribution="exponential", n_euler_terms=15, euler_a=18.0)
    alloc.fit_exponential(rates=np.array([1.0, 2.0]))
    return alloc


@pytest.fixture
def gamma_5():
    """Five gamma participants with distinct parameters (Lloyd's syndicate scenario)."""
    from insurance_severity.cmrs import CMRSAllocator
    alloc = CMRSAllocator(distribution="gamma", n_euler_terms=15, euler_a=18.0)
    alloc.fit_gamma(
        alphas=np.array([2.0, 3.0, 1.5, 4.0, 2.5]),
        betas=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
    )
    return alloc


@pytest.fixture
def gamma_equal():
    """Two identical gamma participants: h_1 = h_2 = s/2 by symmetry."""
    from insurance_severity.cmrs import CMRSAllocator
    alloc = CMRSAllocator(distribution="gamma", n_euler_terms=15, euler_a=18.0)
    alloc.fit_gamma(
        alphas=np.array([2.0, 2.0]),
        betas=np.array([1.0, 1.0]),
    )
    return alloc


@pytest.fixture
def lognormal_2():
    """Two lognormal participants."""
    from insurance_severity.cmrs import CMRSAllocator
    alloc = CMRSAllocator(distribution="lognormal", n_euler_terms=15, euler_a=18.0, n_quad_points=50)
    alloc.fit_lognormal(
        mus=np.array([0.5, 1.0]),
        sigmas=np.array([0.3, 0.5]),
    )
    return alloc


# ---------------------------------------------------------------------------
# Helpers: closed-form CMRS for two exponentials with rates [1, 2]
# ---------------------------------------------------------------------------

def _exp_cmrs_h1(s: float) -> float:
    """True E[X_1 | S=s] for X_1 ~ Exp(1), X_2 ~ Exp(2), S = X_1 + X_2.

    Derived analytically from the conditional density:
        f_{X_1|S}(x|s) ∝ exp(-x) * exp(-2*(s-x))  for 0 < x < s
    integrating x * f_{X_1|S}(x|s) gives:
        E[X_1|S=s] = [(s - 1) + exp(-s)] / (1 - exp(-s))

    This is NOT mean-proportional: the allocation fraction changes with s,
    approaching 1/1 (participant 1 gets nearly all) as s → ∞ and approaching
    2/3 (mean-proportional) as s → 0.
    """
    return ((s - 1.0) + np.exp(-s)) / (1.0 - np.exp(-s))


# ---------------------------------------------------------------------------
# 1. Import and construction
# ---------------------------------------------------------------------------

def test_import():
    from insurance_severity.cmrs import CMRSAllocator
    assert CMRSAllocator is not None


def test_top_level_import():
    """CMRSAllocator should be importable from the package top level."""
    import insurance_severity
    assert hasattr(insurance_severity, "CMRSAllocator")


def test_init_invalid_distribution():
    from insurance_severity.cmrs import CMRSAllocator
    with pytest.raises(ValueError, match="distribution"):
        CMRSAllocator(distribution="weibull_custom")


def test_not_fitted_raises():
    from insurance_severity.cmrs import CMRSAllocator
    alloc = CMRSAllocator(distribution="gamma")
    with pytest.raises(RuntimeError, match="not fitted"):
        alloc.allocate(10.0)


# ---------------------------------------------------------------------------
# 2. Fitting methods
# ---------------------------------------------------------------------------

def test_fit_exponential():
    from insurance_severity.cmrs import CMRSAllocator
    alloc = CMRSAllocator(distribution="exponential")
    alloc.fit_exponential(np.array([1.0, 2.0, 3.0]))
    assert alloc._is_fitted
    assert alloc._n == 3


def test_fit_gamma():
    from insurance_severity.cmrs import CMRSAllocator
    alloc = CMRSAllocator(distribution="gamma")
    alloc.fit_gamma(np.array([2.0, 3.0]), np.array([1.0, 2.0]))
    assert alloc._is_fitted
    assert alloc._n == 2


def test_fit_lognormal():
    from insurance_severity.cmrs import CMRSAllocator
    alloc = CMRSAllocator(distribution="lognormal")
    alloc.fit_lognormal(np.array([0.0, 1.0]), np.array([0.5, 0.5]))
    assert alloc._is_fitted
    assert alloc._n == 2


def test_fit_exponential_negative_rate_raises():
    from insurance_severity.cmrs import CMRSAllocator
    alloc = CMRSAllocator(distribution="exponential")
    with pytest.raises(ValueError):
        alloc.fit_exponential(np.array([1.0, -0.5]))


# ---------------------------------------------------------------------------
# 3. Exponential closed-form check
# ---------------------------------------------------------------------------

def test_exponential_single_s(exp_2):
    """CMRS allocations for two exponentials match the analytical conditional expectation.

    For X_1 ~ Exp(1), X_2 ~ Exp(2), S = X_1 + X_2:
        E[X_1 | S=s] = [(s-1) + exp(-s)] / (1 - exp(-s))
        E[X_2 | S=s] = s - E[X_1 | S=s]

    This is NOT mean-proportional: CMRS gives the richer participant (smaller rate =
    larger mean, here X_1 with rate 1) a larger share that increases with s.
    """
    s = 3.0
    h = exp_2.allocate(s)
    h1_true = _exp_cmrs_h1(s)   # ≈ 2.157
    h2_true = s - h1_true        # ≈ 0.843
    expected = np.array([h1_true, h2_true])
    np.testing.assert_allclose(h, expected, rtol=1e-3)


def test_exponential_closed_form_multiple_s(exp_2):
    """CMRS allocation fractions for exponentials change with s (not constant).

    The CMRS fraction h_1/s is NOT constant across s values — it grows from
    ~2/3 (mean-proportional, the limit as s→0) toward 1 as s→∞.
    We verify against the exact analytical formula at several s values.
    """
    s_vals = np.array([1.0, 5.0, 10.0])
    h = exp_2.allocate(s_vals)  # (3, 2)
    for i, sv in enumerate(s_vals):
        h1_true = _exp_cmrs_h1(sv)
        h2_true = sv - h1_true
        expected = np.array([h1_true, h2_true])
        np.testing.assert_allclose(h[i], expected, rtol=1e-3)


# ---------------------------------------------------------------------------
# 4. Budget balance
# ---------------------------------------------------------------------------

def test_budget_balance_exponential(exp_2):
    """sum_i h_i(s) = s for exponential participants."""
    for s in [1.0, 5.0, 20.0]:
        h = exp_2.allocate(s)
        np.testing.assert_allclose(h.sum(), s, rtol=1e-4)


def test_budget_balance_gamma(gamma_5):
    """sum_i h_i(s) = s for gamma participants."""
    for s in [2.0, 10.0, 50.0]:
        h = gamma_5.allocate(s)
        np.testing.assert_allclose(h.sum(), s, rtol=1e-3)


def test_budget_balance_lognormal(lognormal_2):
    """sum_i h_i(s) = s for lognormal participants."""
    h = lognormal_2.allocate(5.0)
    np.testing.assert_allclose(h.sum(), 5.0, rtol=5e-3)


def test_budget_check_structure(exp_2):
    result = exp_2.budget_check(4.0)
    assert "s" in result
    assert "allocations" in result
    assert "absolute_error" in result
    assert "relative_error" in result
    assert "tilted" in result
    np.testing.assert_allclose(result["s"], 4.0)


# ---------------------------------------------------------------------------
# 5. Symmetry: equal participants
# ---------------------------------------------------------------------------

def test_gamma_symmetry(gamma_equal):
    """Two identical gamma participants: each should get exactly s/2."""
    s = 8.0
    h = gamma_equal.allocate(s)
    np.testing.assert_allclose(h[0], h[1], rtol=1e-3)
    np.testing.assert_allclose(h[0], s / 2, rtol=1e-3)


# ---------------------------------------------------------------------------
# 6. Shape tests
# ---------------------------------------------------------------------------

def test_allocate_scalar_returns_1d(exp_2):
    h = exp_2.allocate(3.0)
    assert h.shape == (2,)


def test_allocate_array_returns_2d(exp_2):
    s_arr = np.linspace(1.0, 10.0, 20)
    h = exp_2.allocate(s_arr)
    assert h.shape == (20, 2)


def test_allocate_quantile_shape(gamma_5):
    h = gamma_5.allocate_quantile(np.array([0.5, 0.9, 0.95]))
    assert h.shape == (3, 5)


def test_aggregate_distribution_shape(exp_2):
    s_grid = np.linspace(0.1, 5.0, 30)
    f_vals = exp_2.aggregate_distribution(s_grid)
    assert f_vals.shape == (30,)
    assert np.all(f_vals >= -1e-6)  # densities should be non-negative


# ---------------------------------------------------------------------------
# 7. Monotonicity: h_i(s) non-decreasing in s
# ---------------------------------------------------------------------------

def test_monotonicity_exponential(exp_2):
    """h_i(s) should be non-decreasing in s for exponential participants."""
    s_vals = np.linspace(0.5, 10.0, 30)
    h = exp_2.allocate(s_vals)  # (30, 2)
    for col in range(2):
        diffs = np.diff(h[:, col])
        assert np.all(diffs >= -1e-4), f"h_{col+1} not monotone"


def test_monotonicity_gamma(gamma_5):
    """h_i(s) non-decreasing for gamma participants."""
    s_vals = np.linspace(1.0, 30.0, 20)
    h = gamma_5.allocate(s_vals)
    for col in range(5):
        diffs = np.diff(h[:, col])
        assert np.all(diffs >= -1e-3), f"h_{col+1} not monotone"


# ---------------------------------------------------------------------------
# 8. Summary output
# ---------------------------------------------------------------------------

def test_summary_structure(gamma_5):
    result = gamma_5.summary(10.0)
    assert "allocations" in result
    assert "aggregate_loss" in result
    assert "total" in result
    assert "tilted" in result
    assert "mean_s" in result
    np.testing.assert_allclose(result["total"], 10.0, rtol=1e-3)


# ---------------------------------------------------------------------------
# 9. Lognormal vs Monte Carlo
# ---------------------------------------------------------------------------

def test_lognormal_vs_mc(lognormal_2):
    """Lognormal allocations should match Monte Carlo E[X_i | S=s] within 5%.

    Uses a large MC sample with kernel density estimation to approximate E[X_i | S near s].
    """
    rng = np.random.default_rng(0)
    n_sim = 200_000
    mu1, sigma1 = 0.5, 0.3
    mu2, sigma2 = 1.0, 0.5
    X1 = rng.lognormal(mean=mu1, sigma=sigma1, size=n_sim)
    X2 = rng.lognormal(mean=mu2, sigma=sigma2, size=n_sim)
    S = X1 + X2

    s_target = np.exp(mu1 + mu2)  # near the product of medians
    bandwidth = 0.1 * np.std(S)
    weights = np.exp(-0.5 * ((S - s_target) / bandwidth) ** 2)
    weights /= weights.sum()

    mc_h1 = np.dot(weights, X1)
    mc_h2 = np.dot(weights, X2)

    h = lognormal_2.allocate(s_target)

    np.testing.assert_allclose(h[0], mc_h1, rtol=0.05)
    np.testing.assert_allclose(h[1], mc_h2, rtol=0.05)


# ---------------------------------------------------------------------------
# 10. Repr
# ---------------------------------------------------------------------------

def test_repr_fitted(exp_2):
    r = repr(exp_2)
    assert "CMRSAllocator" in r
    assert "exponential" in r
    assert "n=2" in r


def test_repr_unfitted():
    from insurance_severity.cmrs import CMRSAllocator
    alloc = CMRSAllocator(distribution="gamma")
    r = repr(alloc)
    assert "not fitted" in r
