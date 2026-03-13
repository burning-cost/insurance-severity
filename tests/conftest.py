"""
Shared fixtures for insurance-severity tests.

Merged from insurance-composite and insurance-drn conftest files.
No fixture name collisions — composite fixtures use severity data,
DRN fixtures use gamma/Pandas data.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats


# ---------------------------------------------------------------------------
# Composite fixtures (from insurance-composite)
# ---------------------------------------------------------------------------


def _lognormal_burr_composite(n: int, seed: int = 42) -> np.ndarray:
    """
    Generate from LognormalBurr composite with known parameters.

    Parameters: sigma=1.2 (body), alpha=2.5, delta=2.0, beta=10000 (tail)
    threshold = beta * [(delta-1)/(alpha*delta+1)]^{1/delta} ~ 4082

    Note: delta > 1 is required for the Burr XII mode to exist.
    The correct mode formula is beta * [(delta-1)/(alpha*delta+1)]^{1/delta}.
    """
    rng = np.random.default_rng(seed)
    # Tail Burr XII params (delta=2.0 > 1 required for mode)
    alpha, delta, beta = 2.5, 2.0, 10000.0
    ratio = (delta - 1.0) / (alpha * delta + 1.0)
    threshold = beta * ratio ** (1.0 / delta)
    sigma = 1.2
    mu = sigma ** 2 + np.log(threshold)
    pi = 0.75  # 75% in body

    n_body = int(n * pi)
    n_tail = n - n_body

    # Body: lognormal truncated at threshold
    y_body = []
    while len(y_body) < n_body:
        batch = stats.lognorm.rvs(s=sigma, scale=np.exp(mu), size=n_body * 3, random_state=rng)
        batch = batch[batch <= threshold]
        y_body.extend(batch[:n_body - len(y_body)])
    y_body = np.array(y_body[:n_body])

    # Tail: Burr XII, shifted above threshold
    y_tail = []
    while len(y_tail) < n_tail:
        # Sample from Burr XII using ppf
        u = rng.uniform(size=n_tail * 3)
        # Burr XII ppf: x = beta * (u^{-1/alpha} - 1)^{1/delta}
        x = beta * (u ** (-1.0 / alpha) - 1.0) ** (1.0 / delta)
        x = x[x > threshold]
        y_tail.extend(x[:n_tail - len(y_tail)])
    y_tail = np.array(y_tail[:n_tail])

    y = np.concatenate([y_body, y_tail])
    rng.shuffle(y)
    return y


def _lognormal_gpd_composite(n: int, seed: int = 42) -> np.ndarray:
    """Generate from LognormalGPD composite with known threshold."""
    rng = np.random.default_rng(seed)
    threshold = 50_000.0
    sigma_ln = 1.5
    mu_ln = np.log(threshold) - 1.5 * sigma_ln
    xi_gpd = 0.25
    sigma_gpd = 20_000.0
    pi = 0.80

    n_body = int(n * pi)
    n_tail = n - n_body

    y_body = []
    while len(y_body) < n_body:
        batch = stats.lognorm.rvs(s=sigma_ln, scale=np.exp(mu_ln), size=n_body * 3, random_state=rng)
        batch = batch[batch <= threshold]
        y_body.extend(batch[:n_body - len(y_body)])
    y_body = np.array(y_body[:n_body])

    y_tail = stats.genpareto.rvs(c=xi_gpd, scale=sigma_gpd, size=n_tail, random_state=rng) + threshold

    y = np.concatenate([y_body, y_tail])
    rng.shuffle(y)
    return y


def _gamma_gpd_composite(n: int, seed: int = 42) -> np.ndarray:
    """Generate from GammaGPD composite."""
    rng = np.random.default_rng(seed)
    threshold = 30_000.0
    shape_gamma = 3.0
    scale_gamma = 8_000.0
    xi_gpd = 0.3
    sigma_gpd = 15_000.0
    pi = 0.85

    n_body = int(n * pi)
    n_tail = n - n_body

    y_body = []
    while len(y_body) < n_body:
        batch = stats.gamma.rvs(a=shape_gamma, scale=scale_gamma, size=n_body * 3, random_state=rng)
        batch = batch[batch <= threshold]
        y_body.extend(batch[:n_body - len(y_body)])
    y_body = np.array(y_body[:n_body])

    y_tail = stats.genpareto.rvs(c=xi_gpd, scale=sigma_gpd, size=n_tail, random_state=rng) + threshold

    y = np.concatenate([y_body, y_tail])
    rng.shuffle(y)
    return y


@pytest.fixture(scope="session")
def lognormal_burr_data():
    """Session-scoped fixture: 500 obs from LognormalBurr composite."""
    return _lognormal_burr_composite(n=500, seed=42)


@pytest.fixture(scope="session")
def lognormal_gpd_data():
    """Session-scoped fixture: 500 obs from LognormalGPD composite."""
    return _lognormal_gpd_composite(n=500, seed=42)


@pytest.fixture(scope="session")
def gamma_gpd_data():
    """Session-scoped fixture: 500 obs from GammaGPD composite."""
    return _gamma_gpd_composite(n=500, seed=42)


@pytest.fixture(scope="session")
def regression_data():
    """
    Session-scoped: 300 obs with a single covariate.
    Covariate enters tail scale: log(beta_i) = 8.5 + 0.3 * x_i.
    """
    rng = np.random.default_rng(123)
    n = 300
    x = rng.normal(0, 1, n)

    # delta=2.0 > 1 required for Burr XII mode existence
    alpha, delta = 2.5, 2.0
    log_beta = 8.5 + 0.3 * x
    beta_arr = np.exp(log_beta)
    sigma = 1.2

    y = np.zeros(n)
    for i in range(n):
        beta_i = beta_arr[i]
        # Correct mode formula: (delta-1)/(alpha*delta+1)
        ratio = (delta - 1.0) / (alpha * delta + 1.0)
        t_i = beta_i * ratio ** (1.0 / delta)
        mu_i = sigma ** 2 + np.log(t_i)

        # Draw body or tail
        if rng.random() < 0.75:
            # Body
            while True:
                v = stats.lognorm.rvs(s=sigma, scale=np.exp(mu_i), random_state=rng)
                if v <= t_i:
                    y[i] = v
                    break
        else:
            # Tail
            while True:
                u = rng.random()
                S_t = (1.0 + (t_i / beta_i) ** delta) ** (-alpha)
                S_x = u * S_t  # S(x) = u * S(t) => x > t
                inner = S_x ** (-1.0 / alpha) - 1.0
                if inner > 0:
                    y[i] = beta_i * inner ** (1.0 / delta)
                    break

    return x.reshape(-1, 1), y


# ---------------------------------------------------------------------------
# DRN fixtures (from insurance-drn)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def gamma_data(rng):
    """
    Synthetic Gamma severity dataset.
    shape=2.0, scale=1000.0 => mean=2000, CV=0.707
    n=2000 observations, 3 features.
    """
    n = 2000
    X = pd.DataFrame({
        "age": rng.integers(18, 75, size=n).astype(float),
        "vehicle_age": rng.integers(0, 15, size=n).astype(float),
        "region": rng.integers(0, 5, size=n).astype(float),
    })
    # True log-mu depends on features
    log_mu = (
        7.5
        + 0.005 * (X["age"] - 40)
        - 0.02 * X["vehicle_age"]
        + 0.1 * X["region"]
    )
    mu = np.exp(log_mu)
    shape = 2.0
    scale = mu / shape
    y = rng.gamma(shape=shape, scale=scale)
    return X, y, mu


@pytest.fixture(scope="session")
def small_gamma_data(rng):
    """Smaller dataset for faster tests — n=300."""
    n = 300
    X = pd.DataFrame({
        "age": rng.integers(18, 75, size=n).astype(float),
        "region": rng.integers(0, 3, size=n).astype(float),
    })
    mu = np.exp(7.0 + 0.003 * (X["age"] - 40))
    shape = 2.0
    scale = mu / shape
    y = rng.gamma(shape=shape, scale=scale)
    return X, y, mu


@pytest.fixture(scope="session")
def mock_glm_baseline(gamma_data):
    """
    A mock GLM baseline that returns known Gamma parameters.
    Avoids statsmodels dependency for core unit tests.
    """
    X, y, mu = gamma_data

    class MockGLMBaseline:
        distribution_family = "gamma"

        def predict_params(self, X_new):
            n = len(X_new)
            return {"mu": np.full(n, float(np.mean(mu))), "dispersion": 0.5}

        def predict_cdf(self, X_new, cutpoints):
            from scipy import stats
            params = self.predict_params(X_new)
            mu_val = params["mu"][:, np.newaxis]
            disp = params["dispersion"]
            alpha = 1.0 / disp
            scale = mu_val * disp
            return stats.gamma.cdf(cutpoints[np.newaxis, :], a=alpha, scale=scale)

    return MockGLMBaseline()


@pytest.fixture(scope="session")
def mock_glm_baseline_small(small_gamma_data):
    """Mock GLM baseline for small dataset."""
    X, y, mu = small_gamma_data

    class MockGLMBaseline:
        distribution_family = "gamma"

        def predict_params(self, X_new):
            n = len(X_new)
            return {"mu": np.full(n, float(np.mean(mu))), "dispersion": 0.5}

        def predict_cdf(self, X_new, cutpoints):
            from scipy import stats
            params = self.predict_params(X_new)
            mu_val = params["mu"][:, np.newaxis]
            disp = params["dispersion"]
            alpha = 1.0 / disp
            scale = mu_val * disp
            return stats.gamma.cdf(cutpoints[np.newaxis, :], a=alpha, scale=scale)

    return MockGLMBaseline()
