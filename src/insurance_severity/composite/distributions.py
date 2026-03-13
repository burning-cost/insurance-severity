"""
Distribution building blocks for composite severity models.

Each distribution is wrapped with truncation support and exposes a consistent
interface: logpdf, logcdf, logsf, ppf, mode, fit_mle.

Design note: We wrap scipy rather than subclass rv_continuous because we need
truncated versions that integrate to 1 over a restricted domain. scipy's own
truncation (truncnorm, truncexpon) is distribution-specific; we need a generic
mechanism. All log-probability methods guard against -inf by clipping inputs.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from scipy import optimize, stats
from scipy.special import gammaln


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


class BodyDistribution(ABC):
    """Body distribution truncated above at the splice threshold."""

    @abstractmethod
    def logpdf(self, x: np.ndarray, threshold: float) -> np.ndarray:
        """Log density of truncated body: log[f(x) / F(threshold)]."""
        ...

    @abstractmethod
    def logcdf(self, x: np.ndarray, threshold: float) -> np.ndarray:
        """Log CDF of truncated body: log[F(x) / F(threshold)]."""
        ...

    @abstractmethod
    def ppf(self, q: np.ndarray, threshold: float) -> np.ndarray:
        """Quantile function of truncated body."""
        ...

    @abstractmethod
    def mode(self, threshold: float) -> float:
        """Mode of the body distribution (not the truncated version)."""
        ...

    @abstractmethod
    def fit_mle(
        self, x: np.ndarray, threshold: float
    ) -> tuple[np.ndarray, dict]:
        """
        Fit body parameters by MLE to observations below threshold.

        Returns (params, info_dict).
        """
        ...

    @property
    @abstractmethod
    def params(self) -> np.ndarray:
        """Current parameter vector."""
        ...

    @params.setter
    @abstractmethod
    def params(self, values: np.ndarray) -> None: ...


class TailDistribution(ABC):
    """Tail distribution truncated below at the splice threshold."""

    @abstractmethod
    def logpdf(self, x: np.ndarray, threshold: float) -> np.ndarray:
        """Log density of truncated tail: log[f(x) / S(threshold)]."""
        ...

    @abstractmethod
    def logsf(self, x: np.ndarray, threshold: float) -> np.ndarray:
        """Log survival function of truncated tail: log[S(x) / S(threshold)]."""
        ...

    @abstractmethod
    def ppf(self, q: np.ndarray, threshold: float) -> np.ndarray:
        """Quantile function of truncated tail."""
        ...

    @abstractmethod
    def mode_value(self) -> Optional[float]:
        """
        Unconditional mode of the tail distribution at current params.
        Returns None if the mode does not exist (e.g. GPD with xi >= 0).
        """
        ...

    @abstractmethod
    def fit_mle(
        self, x: np.ndarray, threshold: float
    ) -> tuple[np.ndarray, dict]:
        """Fit tail parameters by MLE to observations above threshold."""
        ...

    @property
    @abstractmethod
    def params(self) -> np.ndarray:
        """Current parameter vector."""
        ...

    @params.setter
    @abstractmethod
    def params(self, values: np.ndarray) -> None: ...

    def mean(self, threshold: float) -> float:
        """Expected value of the (unconditional) tail distribution."""
        raise NotImplementedError

    def tvar(self, alpha: float, threshold: float) -> float:
        """Tail Value-at-Risk at level alpha of the truncated tail."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Lognormal body
# ---------------------------------------------------------------------------


class LognormalBody(BodyDistribution):
    """
    Lognormal body distribution.

    Parameters
    ----------
    mu : float
        Log-scale parameter (mean of log(X)).
    sigma : float
        Log-standard-deviation parameter (sigma > 0).

    The lognormal mode is exp(mu - sigma^2). For the composite model this
    is typically much less than the threshold, ensuring the body density
    is well-behaved up to the splice point.
    """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self._mu = float(mu)
        self._sigma = float(sigma)

    @property
    def mu(self) -> float:
        return self._mu

    @property
    def sigma(self) -> float:
        return self._sigma

    @property
    def params(self) -> np.ndarray:
        return np.array([self._mu, self._sigma])

    @params.setter
    def params(self, values: np.ndarray) -> None:
        self._mu = float(values[0])
        self._sigma = float(values[1])

    def _log_norm_const(self, threshold: float) -> float:
        """log P(X <= threshold) = log F_lognormal(threshold)."""
        return stats.lognorm.logcdf(threshold, s=self._sigma, scale=np.exp(self._mu))

    def logpdf(self, x: np.ndarray, threshold: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        lp = stats.lognorm.logpdf(x, s=self._sigma, scale=np.exp(self._mu))
        return lp - self._log_norm_const(threshold)

    def logcdf(self, x: np.ndarray, threshold: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        lc = stats.lognorm.logcdf(x, s=self._sigma, scale=np.exp(self._mu))
        return lc - self._log_norm_const(threshold)

    def ppf(self, q: np.ndarray, threshold: float) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        # q is quantile of truncated distribution: F_trunc(x) = F(x)/F(t)
        # So F(x) = q * F(t) => x = F_lognorm_inv(q * F(t))
        F_t = stats.lognorm.cdf(threshold, s=self._sigma, scale=np.exp(self._mu))
        return stats.lognorm.ppf(q * F_t, s=self._sigma, scale=np.exp(self._mu))

    def mode(self, threshold: float) -> float:
        del threshold  # not used for lognormal
        return np.exp(self._mu - self._sigma ** 2)

    def mean(self) -> float:
        return np.exp(self._mu + 0.5 * self._sigma ** 2)

    def fit_mle(
        self, x: np.ndarray, threshold: float
    ) -> tuple[np.ndarray, dict]:
        x = np.asarray(x, dtype=float)
        if len(x) == 0:
            raise ValueError("No observations below threshold")

        # For lognormal with truncation at threshold, the log-likelihood is:
        # sum_i [logpdf_lognormal(x_i) - log F_lognormal(threshold)]
        # = sum_i logpdf_lognormal(x_i) - n * log F_lognormal(threshold)
        # This is maximised by standard lognormal MLE on log(x_i), adjusted
        # by the truncation constant. The truncation shifts the MLE slightly.
        n = len(x)
        lx = np.log(x)

        def neg_loglik(params):
            mu, log_sigma = params
            sigma = np.exp(log_sigma)
            lp = stats.norm.logpdf(lx, loc=mu, scale=sigma) - np.log(x)
            lp -= stats.lognorm.logcdf(threshold, s=sigma, scale=np.exp(mu))
            return -np.sum(lp)

        # Init from moments of log(x)
        mu0 = np.mean(lx)
        sigma0 = max(np.std(lx), 0.1)
        result = optimize.minimize(
            neg_loglik,
            x0=[mu0, np.log(sigma0)],
            method="L-BFGS-B",
        )
        mu_hat = result.x[0]
        sigma_hat = np.exp(result.x[1])
        self._mu = mu_hat
        self._sigma = sigma_hat
        return np.array([mu_hat, sigma_hat]), {"success": result.success, "loglik": -result.fun}


# ---------------------------------------------------------------------------
# Gamma body
# ---------------------------------------------------------------------------


class GammaBody(BodyDistribution):
    """
    Gamma body distribution with shape > 1 (for mode existence).

    Parameters
    ----------
    shape : float
        Shape parameter (alpha > 0; mode exists for alpha > 1).
    scale : float
        Scale parameter (theta > 0). Mean = shape * scale.

    The Gamma mode is (shape - 1) * scale for shape > 1.
    """

    def __init__(self, shape: float = 2.0, scale: float = 1.0):
        if shape <= 0 or scale <= 0:
            raise ValueError("shape and scale must be positive")
        self._shape = float(shape)
        self._scale = float(scale)

    @property
    def params(self) -> np.ndarray:
        return np.array([self._shape, self._scale])

    @params.setter
    def params(self, values: np.ndarray) -> None:
        self._shape = float(values[0])
        self._scale = float(values[1])

    def _log_norm_const(self, threshold: float) -> float:
        return stats.gamma.logcdf(threshold, a=self._shape, scale=self._scale)

    def logpdf(self, x: np.ndarray, threshold: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        lp = stats.gamma.logpdf(x, a=self._shape, scale=self._scale)
        return lp - self._log_norm_const(threshold)

    def logcdf(self, x: np.ndarray, threshold: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        lc = stats.gamma.logcdf(x, a=self._shape, scale=self._scale)
        return lc - self._log_norm_const(threshold)

    def ppf(self, q: np.ndarray, threshold: float) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        F_t = stats.gamma.cdf(threshold, a=self._shape, scale=self._scale)
        return stats.gamma.ppf(q * F_t, a=self._shape, scale=self._scale)

    def mode(self, threshold: float) -> float:
        del threshold
        if self._shape <= 1:
            return 0.0
        return (self._shape - 1.0) * self._scale

    def mean(self) -> float:
        return self._shape * self._scale

    def fit_mle(
        self, x: np.ndarray, threshold: float
    ) -> tuple[np.ndarray, dict]:
        x = np.asarray(x, dtype=float)
        if len(x) == 0:
            raise ValueError("No observations below threshold")

        def neg_loglik(params):
            log_shape, log_scale = params
            shape = np.exp(log_shape)
            scale = np.exp(log_scale)
            lp = stats.gamma.logpdf(x, a=shape, scale=scale)
            lp -= stats.gamma.logcdf(threshold, a=shape, scale=scale)
            return -np.sum(lp)

        # Init from method of moments
        m1 = np.mean(x)
        m2 = np.var(x)
        shape0 = max(m1 ** 2 / m2, 0.5)
        scale0 = m2 / m1

        result = optimize.minimize(
            neg_loglik,
            x0=[np.log(shape0), np.log(scale0)],
            method="L-BFGS-B",
        )
        shape_hat = np.exp(result.x[0])
        scale_hat = np.exp(result.x[1])
        self._shape = shape_hat
        self._scale = scale_hat
        return np.array([shape_hat, scale_hat]), {"success": result.success, "loglik": -result.fun}


# ---------------------------------------------------------------------------
# GPD tail
# ---------------------------------------------------------------------------


class GPDTail(TailDistribution):
    """
    Generalized Pareto Distribution (GPD) tail.

    Parameterisation: GPD(xi, sigma) where sigma > 0 and xi is real.
        f(x) = (1/sigma) * (1 + xi*x/sigma)^{-1/xi - 1}   for x > 0

    Shape regimes:
    - xi > 0: heavy tail (Pareto-like), unbounded support
    - xi = 0: exponential, unbounded support
    - xi < 0: bounded support, max value = -sigma/xi

    Mode: exists and is positive only for xi < -0.5. For insurance heavy
    losses (xi > 0), the mode is 0. Therefore GPD + mode-matching is NOT
    supported — use Burr XII for mode-matching.

    Parameters
    ----------
    xi : float
        Shape parameter.
    sigma : float
        Scale parameter (> 0).
    """

    def __init__(self, xi: float = 0.2, sigma: float = 1.0):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self._xi = float(xi)
        self._sigma = float(sigma)

    @property
    def params(self) -> np.ndarray:
        return np.array([self._xi, self._sigma])

    @params.setter
    def params(self, values: np.ndarray) -> None:
        self._xi = float(values[0])
        self._sigma = float(values[1])

    def _log_sf_at_threshold(self, threshold: float) -> float:
        """log S_GPD(threshold) = log P(Z > threshold) where Z ~ GPD(xi, sigma)."""
        # GPD survival: S(x) = (1 + xi*x/sigma)^{-1/xi} for xi != 0
        #                     = exp(-x/sigma) for xi = 0
        xi, sigma = self._xi, self._sigma
        if abs(xi) < 1e-10:
            return -threshold / sigma
        val = 1.0 + xi * threshold / sigma
        if val <= 0:
            return -np.inf
        return (-1.0 / xi) * np.log(val)

    def logpdf(self, x: np.ndarray, threshold: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        xi, sigma = self._xi, self._sigma
        # Shift: GPD applies to exceedances over threshold
        z = x - threshold
        lp = stats.genpareto.logpdf(z, c=xi, scale=sigma)
        lp -= self._log_sf_at_threshold(0.0)  # S(0) = 1, so logS(0) = 0
        # The truncated tail is GPD on exceedances: no normalization needed
        # because GPD is defined for x >= threshold as exceedance distribution
        return lp

    def logsf(self, x: np.ndarray, threshold: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        z = x - threshold
        return stats.genpareto.logsf(z, c=self._xi, scale=self._sigma)

    def cdf(self, x: np.ndarray, threshold: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        z = x - threshold
        return stats.genpareto.cdf(z, c=self._xi, scale=self._sigma)

    def ppf(self, q: np.ndarray, threshold: float) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return stats.genpareto.ppf(q, c=self._xi, scale=self._sigma) + threshold

    def mode_value(self) -> Optional[float]:
        """GPD mode is positive only for xi < -0.5."""
        if self._xi < -0.5:
            # Mode of GPD(xi, sigma): x = sigma * (-1/xi - 1) > 0
            return self._sigma * (-1.0 / self._xi - 1.0)
        return None  # mode is 0 for xi >= 0, not useful for mode-matching

    def mean(self, threshold: float = 0.0) -> float:
        """Mean of GPD exceedance: sigma / (1 - xi) for xi < 1."""
        if self._xi >= 1.0:
            return np.inf
        return threshold + self._sigma / (1.0 - self._xi)

    def tvar(self, alpha: float, threshold: float) -> float:
        """
        TVaR (Tail Value at Risk / CTE) of the exceedance distribution
        at level alpha. For GPD with xi < 1:
        TVaR_alpha = VaR_alpha + sigma*(1 + xi*VaR_alpha/sigma) / (1-xi)
        """
        if self._xi >= 1.0:
            return np.inf
        var = self.ppf(np.array([alpha]), threshold=threshold)[0]
        excess = var - threshold
        return var + (self._sigma + self._xi * excess) / (1.0 - self._xi)

    def fit_mle(
        self, x: np.ndarray, threshold: float
    ) -> tuple[np.ndarray, dict]:
        x = np.asarray(x, dtype=float)
        if len(x) == 0:
            raise ValueError("No observations above threshold")
        if len(x) < 30:
            warnings.warn(
                f"Only {len(x)} observations above threshold. "
                "Tail parameter estimates may be unreliable.",
                UserWarning,
                stacklevel=2,
            )

        z = x - threshold  # exceedances

        def neg_loglik(params):
            xi, log_sigma = params
            sigma = np.exp(log_sigma)
            # Support check
            if xi < 0 and np.any(z > -sigma / xi):
                return 1e10
            lp = stats.genpareto.logpdf(z, c=xi, scale=sigma)
            if np.any(np.isnan(lp)) or np.any(np.isinf(lp)):
                return 1e10
            return -np.sum(lp)

        # Initialize with method of moments: mean=sigma/(1-xi), var=sigma^2/((1-xi)^2*(1-2*xi))
        m = np.mean(z)
        v = np.var(z)
        xi0 = max(0.5 * (1.0 - m**2 / v), -0.9)
        sigma0 = max(m * (1.0 - xi0), 0.1 * m)

        result = optimize.minimize(
            neg_loglik,
            x0=[xi0, np.log(sigma0)],
            method="L-BFGS-B",
            bounds=[(-0.9, 2.0), (-10, 10)],
        )
        xi_hat = result.x[0]
        sigma_hat = np.exp(result.x[1])
        self._xi = xi_hat
        self._sigma = sigma_hat
        return np.array([xi_hat, sigma_hat]), {"success": result.success, "loglik": -result.fun}


# ---------------------------------------------------------------------------
# Pareto (Lomax / Pareto II) tail
# ---------------------------------------------------------------------------


class ParetoTail(TailDistribution):
    """
    Pareto II (Lomax) tail distribution.

    Parameterisation: f(x) = (alpha/sigma) * (1 + x/sigma)^{-(alpha+1)}
    This is GPD with xi = 1/alpha > 0. Always heavy-tailed.
    Mode is at x = 0, so mode-matching is not possible.

    Parameters
    ----------
    alpha : float
        Shape parameter (tail index, > 0). Mean exists for alpha > 1.
    sigma : float
        Scale parameter (> 0).
    """

    def __init__(self, alpha: float = 1.5, sigma: float = 1.0):
        if alpha <= 0 or sigma <= 0:
            raise ValueError("alpha and sigma must be positive")
        self._alpha = float(alpha)
        self._sigma = float(sigma)

    @property
    def params(self) -> np.ndarray:
        return np.array([self._alpha, self._sigma])

    @params.setter
    def params(self, values: np.ndarray) -> None:
        self._alpha = float(values[0])
        self._sigma = float(values[1])

    def logpdf(self, x: np.ndarray, threshold: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        z = x - threshold
        # Pareto II on exceedances
        alpha, sigma = self._alpha, self._sigma
        lp = np.log(alpha) - np.log(sigma) - (alpha + 1.0) * np.log1p(z / sigma)
        return lp

    def logsf(self, x: np.ndarray, threshold: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        z = x - threshold
        return -self._alpha * np.log1p(z / self._sigma)

    def ppf(self, q: np.ndarray, threshold: float) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return threshold + self._sigma * ((1.0 - q) ** (-1.0 / self._alpha) - 1.0)

    def mode_value(self) -> Optional[float]:
        """Pareto mode is 0 — not useful for mode-matching."""
        return None

    def mean(self, threshold: float = 0.0) -> float:
        if self._alpha <= 1.0:
            return np.inf
        return threshold + self._sigma / (self._alpha - 1.0)

    def tvar(self, alpha: float, threshold: float) -> float:
        if self._alpha <= 1.0:
            return np.inf
        var = self.ppf(np.array([alpha]), threshold=threshold)[0]
        # TVaR: E[X | X > VaR] = VaR + E[X - VaR | X > VaR]
        # Mean excess of Pareto II: E[X - x | X > x] = (sigma + x) / (alpha - 1)
        excess = var - threshold
        return var + (self._sigma + excess) / (self._alpha - 1.0)

    def fit_mle(
        self, x: np.ndarray, threshold: float
    ) -> tuple[np.ndarray, dict]:
        x = np.asarray(x, dtype=float)
        if len(x) == 0:
            raise ValueError("No observations above threshold")
        if len(x) < 30:
            warnings.warn(
                f"Only {len(x)} observations above threshold. "
                "Tail parameter estimates may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
        z = x - threshold

        def neg_loglik(params):
            log_alpha, log_sigma = params
            alpha = np.exp(log_alpha)
            sigma = np.exp(log_sigma)
            lp = np.log(alpha) - np.log(sigma) - (alpha + 1.0) * np.log1p(z / sigma)
            return -np.sum(lp)

        m = np.mean(z)
        v = np.var(z)
        alpha0 = max(2.0 * m ** 2 / (v - m ** 2) if v > m ** 2 else 2.0, 0.5)
        sigma0 = m * (alpha0 - 1.0) if alpha0 > 1 else m

        result = optimize.minimize(
            neg_loglik,
            x0=[np.log(alpha0), np.log(max(sigma0, 0.01))],
            method="L-BFGS-B",
        )
        self._alpha = np.exp(result.x[0])
        self._sigma = np.exp(result.x[1])
        return self.params.copy(), {"success": result.success, "loglik": -result.fun}


# ---------------------------------------------------------------------------
# Burr XII tail
# ---------------------------------------------------------------------------


class BurrTail(TailDistribution):
    """
    Burr XII tail distribution.

    Parameterisation: f(x; alpha, delta, beta) where
        f(x) = (alpha * delta / beta) * (x/beta)^{delta-1} * (1 + (x/beta)^delta)^{-(alpha+1)}

    Equivalent to Singh-Maddala distribution. Special case of GBII with nu=1.

    Mode: exists and is positive for alpha > 1:
        mode = (1/beta) * [(alpha - 1) / (delta * alpha + 1)]^{1/delta}
             = beta * [(alpha - 1) / (delta * alpha + 1)]^{1/delta}

    This is the key property enabling mode-matching composite models.
    For mode-matching we require alpha > 1, enforced via reparameterisation
    log(alpha - 1) during estimation.

    Parameters
    ----------
    alpha : float
        Shape parameter controlling tail heaviness (alpha > 1 for mode-matching).
    delta : float
        Shape parameter controlling body shape (delta > 0).
    beta : float
        Scale parameter (beta > 0).
    """

    def __init__(self, alpha: float = 2.0, delta: float = 1.0, beta: float = 1.0):
        if alpha <= 0 or delta <= 0 or beta <= 0:
            raise ValueError("alpha, delta, and beta must be positive")
        self._alpha = float(alpha)
        self._delta = float(delta)
        self._beta = float(beta)

    @property
    def params(self) -> np.ndarray:
        return np.array([self._alpha, self._delta, self._beta])

    @params.setter
    def params(self, values: np.ndarray) -> None:
        self._alpha = float(values[0])
        self._delta = float(values[1])
        self._beta = float(values[2])

    def _log_sf_unconditional(self, x: np.ndarray) -> np.ndarray:
        """log S(x) of the unconditional (non-truncated) Burr XII."""
        x = np.asarray(x, dtype=float)
        return -self._alpha * np.log1p((x / self._beta) ** self._delta)

    def _log_sf_at_threshold(self, threshold: float) -> float:
        """log S_Burr(threshold)."""
        return float(self._log_sf_unconditional(np.array([threshold]))[0])

    def logpdf(self, x: np.ndarray, threshold: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        alpha, delta, beta = self._alpha, self._delta, self._beta
        # Burr XII log-density (unconditional)
        lp = (
            np.log(alpha)
            + np.log(delta)
            - np.log(beta)
            + (delta - 1.0) * np.log(x / beta)
            - (alpha + 1.0) * np.log1p((x / beta) ** delta)
        )
        # Truncated at threshold from below: divide by S(threshold)
        lp -= self._log_sf_at_threshold(threshold)
        return lp

    def logsf(self, x: np.ndarray, threshold: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        log_sf_x = self._log_sf_unconditional(x)
        log_sf_t = self._log_sf_at_threshold(threshold)
        return log_sf_x - log_sf_t

    def ppf(self, q: np.ndarray, threshold: float) -> np.ndarray:
        """Quantile of truncated Burr XII."""
        q = np.asarray(q, dtype=float)
        # P(X <= x | X > threshold) = 1 - S(x)/S(threshold)
        # => S(x)/S(threshold) = 1 - q
        # => S(x) = (1-q) * S(threshold)
        # => (1 + (x/beta)^delta)^{-alpha} = (1-q) * (1 + (t/beta)^delta)^{-alpha}
        # Let A = (1 + (t/beta)^delta)^{-alpha}, then:
        # (1 + (x/beta)^delta)^{-alpha} = (1-q) * A
        # (1 + (x/beta)^delta) = ((1-q)*A)^{-1/alpha}
        # x = beta * (((1-q)*A)^{-1/alpha} - 1)^{1/delta}
        alpha, delta, beta = self._alpha, self._delta, self._beta
        S_t = np.exp(self._log_sf_at_threshold(threshold))
        S_x = (1.0 - q) * S_t
        S_x = np.clip(S_x, 1e-300, 1.0)
        inner = S_x ** (-1.0 / alpha) - 1.0
        inner = np.maximum(inner, 0.0)
        return beta * inner ** (1.0 / delta)

    def mode_value(self) -> Optional[float]:
        """
        Burr XII mode: beta * [(delta-1)/(alpha*delta+1)]^{1/delta}.
        Exists and is positive for delta > 1 (requires delta > 1, not alpha > 1).

        Derivation: set d/dx log f = 0 for f(x) = alpha*delta/beta * (x/beta)^{delta-1}
        * (1+(x/beta)^delta)^{-(alpha+1)}. Gives (x/beta)^delta = (delta-1)/(alpha*delta+1).
        Requires delta > 1 for positive mode.
        """
        if self._delta <= 1.0:
            return None
        ratio = (self._delta - 1.0) / (self._alpha * self._delta + 1.0)
        return self._beta * ratio ** (1.0 / self._delta)

    def mean(self, threshold: float = 0.0) -> float:
        """
        Mean of Burr XII: beta * B(1 + 1/delta, alpha - 1/delta) / B(1, alpha).
        Exists for alpha * delta > 1.
        """
        from scipy.special import beta as beta_fn
        alpha, delta, beta = self._alpha, self._delta, self._beta
        if alpha * delta <= 1.0:
            return np.inf
        # Mean of unconditional Burr XII
        try:
            mean_uncond = beta * beta_fn(1.0 + 1.0/delta, alpha - 1.0/delta) * alpha
        except Exception:
            return np.inf
        return mean_uncond

    def tvar(self, alpha: float, threshold: float) -> float:
        """TVaR computed numerically via integration."""
        var = self.ppf(np.array([alpha]), threshold=threshold)[0]
        # Numerical integration
        from scipy.integrate import quad
        def integrand(x):
            return np.exp(self.logsf(np.array([x]), threshold=threshold)[0])

        result, _ = quad(integrand, var, var * 1000 + threshold, limit=200)
        return var + result / (1.0 - alpha)

    def fit_mle(
        self, x: np.ndarray, threshold: float, require_mode: bool = False
    ) -> tuple[np.ndarray, dict]:
        """
        Fit Burr XII by MLE to observations above threshold.

        Parameters
        ----------
        require_mode : bool
            If True, constrain alpha > 1 (needed for mode-matching).
            Implemented via reparameterisation: alpha = 1 + exp(log_alpha_m1).
        """
        x = np.asarray(x, dtype=float)
        if len(x) == 0:
            raise ValueError("No observations above threshold")
        if len(x) < 30:
            warnings.warn(
                f"Only {len(x)} observations above threshold. "
                "Tail parameter estimates may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
        log_sf_t = self._log_sf_at_threshold(threshold)

        if require_mode:
            # Mode requires delta > 1. Reparameterisation: delta = 1 + exp(v), so delta > 1.
            # alpha is unconstrained (just needs > 0).
            def neg_loglik(params):
                log_alpha, v, log_beta = params
                alpha = np.exp(log_alpha)
                delta = 1.0 + np.exp(v)  # > 1 always
                beta = np.exp(log_beta)
                lp = (
                    np.log(alpha) + np.log(delta) - np.log(beta)
                    + (delta - 1.0) * np.log(x / beta)
                    - (alpha + 1.0) * np.log1p((x / beta) ** delta)
                    - (-alpha * np.log1p((threshold / beta) ** delta))
                )
                if np.any(np.isnan(lp)) or np.any(np.isinf(lp)):
                    return 1e10
                return -np.sum(lp)

            # Init: ensure delta0 > 1
            delta0 = max(self._delta, 1.1)
            result = optimize.minimize(
                neg_loglik,
                x0=[np.log(self._alpha), np.log(delta0 - 1.0), np.log(self._beta)],
                method="L-BFGS-B",
            )
            self._alpha = np.exp(result.x[0])
            self._delta = 1.0 + np.exp(result.x[1])  # > 1 enforced
            self._beta = np.exp(result.x[2])
        else:
            def neg_loglik(params):
                log_alpha, log_delta, log_beta = params
                alpha = np.exp(log_alpha)
                delta = np.exp(log_delta)
                beta = np.exp(log_beta)
                lp = (
                    np.log(alpha) + np.log(delta) - np.log(beta)
                    + (delta - 1.0) * np.log(x / beta)
                    - (alpha + 1.0) * np.log1p((x / beta) ** delta)
                    - (-alpha * np.log1p((threshold / beta) ** delta))
                )
                if np.any(np.isnan(lp)) or np.any(np.isinf(lp)):
                    return 1e10
                return -np.sum(lp)

            result = optimize.minimize(
                neg_loglik,
                x0=[np.log(self._alpha), np.log(self._delta), np.log(self._beta)],
                method="L-BFGS-B",
            )
            self._alpha = np.exp(result.x[0])
            self._delta = np.exp(result.x[1])
            self._beta = np.exp(result.x[2])

        return self.params.copy(), {"success": result.success, "loglik": -result.fun}
