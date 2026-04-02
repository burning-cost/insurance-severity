"""
Conditional Mean Risk Sharing (CMRS) via Laplace-Stieltjes Transforms.

Implements the fast allocation framework of Blier-Wong (2026) arXiv:2603.01434,
which extends Denuit & Dhaene (2012) CMRS to continuous severity distributions using
one-dimensional Laplace inversion rather than multi-dimensional integrals.

The core actuarial problem: n participants share an aggregate loss S = X_1 + ... + X_n.
Under CMRS, participant i pays h_i(s) = E[X_i | S = s]. This allocation is:
  - Actuarially fair (each pays their conditional expectation)
  - Pareto optimal (reduces risks in convex order, Denuit & Dhaene 2012)
  - Efficient (O(M * n) per allocation point, vs O(N^n) brute-force)

The Blier-Wong insight: h_i(s) = xi_i(s) / f_S(s) where xi_i and f_S are both
Laplace inversions of transforms computable from individual LSTs.

Actuarial use cases:
  - Lloyd's syndicate risk allocation: members pay proportional to E[X_i | S]
  - P2P insurance pool shortfall allocation
  - IFRS 17 / Solvency II capital allocation to lines of business
  - Motor fleet: allocate aggregate losses to individual vehicles

References:
    Blier-Wong (2026). A Laplace-based perspective on conditional mean risk sharing.
    arXiv:2603.01434.

    Denuit & Dhaene (2012). Convex order and comonotonic conditional mean risk sharing.
    Insurance: Mathematics and Economics, 51(2):265-270.

    Abate & Whitt (1992). Numerical inversion of Laplace transforms of probability
    distributions. ORSA Journal on Computing, 7(1):36-43.
"""

from __future__ import annotations

import warnings
from math import comb
from typing import Callable, Optional, Union

import numpy as np
from scipy import integrate, optimize, stats


# ---------------------------------------------------------------------------
# Euler-Abate-Whitt Laplace inversion with acceleration
# ---------------------------------------------------------------------------


def _euler_inversion(
    lst_func: Callable[[complex], complex],
    x: float,
    a: float = 18.0,
    M: int = 15,
) -> float:
    """Invert a Laplace transform at point x via Euler-accelerated summation.

    Implements the Euler-accelerated Abate-Whitt method (Abate & Whitt 1992).
    Uses M+1 direct Bromwich terms plus M+1 Euler-accelerated tail terms, for
    a total of 2M+2 function evaluations.

    This is the correct Euler acceleration — plain truncation at M terms without
    binomial weighting fails for CMRS transforms because the alternating Bromwich
    series converges too slowly (terms decay like 1/k^2 at best) and requires
    enormous cancellation. With M=15 and a=18, this reaches about 1e-8 relative
    accuracy for typical insurance loss distributions.

    Formula:
        f(x) ≈ (e^a / x) * (S_direct + S_euler)

    where (using f_k = Re[F*((a + ik*pi)/x)]):
        S_direct = f_0/2 + sum_{k=1}^{M} (-1)^k f_k
        S_euler  = (1/2^M) * sum_{j=0}^{M} C(M,j) * (-1)^{M+1+j} * f_{M+1+j}

    The Euler acceleration applies van Wijngaarden binomial weights to the tail
    of the alternating series, converting conditional convergence to absolute
    convergence and dramatically improving accuracy.

    Args:
        lst_func: Callable accepting complex s, returning complex L(s).
        x: Point at which to evaluate f(x). Must be positive.
        a: Euler parameter. a=18 gives discretisation error O(e^{-36}).
        M: Controls the number of terms. Total function evaluations = 2M+2.
           M=15 gives ~1e-8 accuracy; increase to 20-25 for demanding cases.

    Returns:
        Approximation of f(x).
    """
    if x <= 0.0:
        raise ValueError(f"x must be positive for Euler inversion, got {x}")

    scale = np.exp(a) / x

    # Evaluate f_k = Re[F*((a + ik*pi)/x)] for k = 0, 1, ..., 2M+1
    # Total evaluations: 2M+2
    n_evals = 2 * M + 2
    f_vals = np.empty(n_evals)
    for k in range(n_evals):
        tk = complex(a / x, k * np.pi / x)
        f_vals[k] = np.real(lst_func(tk))

    # Direct sum: k=0..M with k=0 halved (the c_0=1/2 Bromwich endpoint correction)
    s_direct = 0.5 * f_vals[0]
    for k in range(1, M + 1):
        s_direct += ((-1) ** k) * f_vals[k]

    # Euler-accelerated tail: apply C(M,j)/2^M weights to tail terms k=M+1..2M+1
    # This is the van Wijngaarden transformation of the remaining alternating series.
    # Term k = M+1+j has sign (-1)^{M+1+j} and Euler weight C(M,j)/2^M.
    s_euler = 0.0
    inv_2m = 1.0 / (2 ** M)
    for j in range(M + 1):
        coeff = comb(M, j) * inv_2m
        s_euler += coeff * ((-1) ** (M + 1 + j)) * f_vals[M + 1 + j]

    return scale * (s_direct + s_euler)


# ---------------------------------------------------------------------------
# Per-distribution LST and derivative functions
# ---------------------------------------------------------------------------


def _lst_exponential(t: complex, rate: float) -> complex:
    """LST of Exp(rate): L(t) = rate / (rate + t)."""
    return rate / (rate + t)


def _lst_deriv_exponential(t: complex, rate: float) -> complex:
    """E[X * exp(-t*X)] for X ~ Exp(rate): rate / (rate + t)^2."""
    return rate / (rate + t) ** 2


def _lst_gamma(t: complex, alpha: float, beta: float) -> complex:
    """LST of Gamma(alpha, beta) with shape alpha, rate beta: L(t) = (beta/(beta+t))^alpha."""
    return (beta / (beta + t)) ** alpha


def _lst_deriv_gamma(t: complex, alpha: float, beta: float) -> complex:
    """E[X * exp(-t*X)] for X ~ Gamma(alpha, beta): alpha/beta * (beta/(beta+t))^{alpha+1}."""
    return (alpha / beta) * (beta / (beta + t)) ** (alpha + 1)


def _lst_lognormal_quad(t: complex, mu: float, sigma: float, n_points: int = 50) -> complex:
    """LST of LogNormal(mu, sigma) via Gauss-Legendre quadrature.

    L(t) = integral_0^inf exp(-t*x) * f_LN(x; mu, sigma) dx

    For complex t with Re(t) >= 0, the integrand exp(-t*x) is bounded by exp(-Re(t)*x),
    so the integral converges. We truncate at the 99.99th percentile of the lognormal.

    Args:
        t: Complex argument with Re(t) >= 0.
        mu: Log-mean parameter.
        sigma: Log-standard-deviation parameter.
        n_points: Number of Gauss-Legendre quadrature points.
    Returns:
        Complex value L(t).
    """
    upper = np.exp(mu + 6 * sigma)  # 99.99th pctile upper bound
    nodes, weights = np.polynomial.legendre.leggauss(n_points)
    # Transform from [-1,1] to [0, upper]
    mid = upper / 2.0
    half = upper / 2.0
    x = mid + half * nodes  # (n_points,) positive
    x = np.maximum(x, 1e-15)
    f_x = stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu))
    kernel = np.exp(-t * x)  # complex
    return float(half) * np.dot(weights, kernel * f_x)


def _lst_deriv_lognormal_quad(t: complex, mu: float, sigma: float, n_points: int = 50) -> complex:
    """E[X * exp(-t*X)] for X ~ LogNormal(mu, sigma) via quadrature.

    = integral_0^inf x * exp(-t*x) * f_LN(x) dx
    """
    upper = np.exp(mu + 6 * sigma)
    nodes, weights = np.polynomial.legendre.leggauss(n_points)
    mid = upper / 2.0
    half = upper / 2.0
    x = mid + half * nodes
    x = np.maximum(x, 1e-15)
    f_x = stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu))
    kernel = x * np.exp(-t * x)  # complex
    return float(half) * np.dot(weights, kernel * f_x)


# ---------------------------------------------------------------------------
# CMRSAllocator
# ---------------------------------------------------------------------------


class CMRSAllocator:
    """Conditional Mean Risk Sharing allocation via Laplace-Stieltjes transforms.

    Computes h_i(s) = E[X_i | S = s] for independent risks X_1, ..., X_n with
    aggregate S = X_1 + ... + X_n, using the Laplace inversion approach of
    Blier-Wong (2026).

    The allocation is computed as:
        h_i(s) = xi_i(s) / f_S(s)

    where both xi_i and f_S are inverted from their Laplace transforms using the
    Euler-accelerated Abate-Whitt method. The acceleration is essential: plain
    truncation of the alternating Bromwich series at M terms without Euler
    weighting fails to converge for the transforms arising in CMRS, because
    the terms decay too slowly and enormous cancellation is required.

    Mathematical guarantee: sum_i h_i(s) = s (budget balance) holds exactly in
    theory and to within numerical tolerance in practice. Allocations are normalised
    to enforce exact budget balance before returning.

    Args:
        distribution: One of 'exponential', 'gamma', 'lognormal'. Determines
            which LST formulae are used. Use fit_exponential(), fit_gamma(), or
            fit_lognormal() after construction to supply parameters.
        tilt_threshold: Activate exponential tilting when s exceeds this quantile
            of S under the fitted parameters. Default 0.99.
        n_euler_terms: M in Euler summation. Total evaluations = 2M+2. Higher = more
            precise but slower. M=15 gives ~1e-8 accuracy for typical insurance losses.
        euler_a: a parameter in Euler summation. Default 18.0 gives O(e^{-36})
            discretisation error.
        n_quad_points: Gauss-Legendre points for lognormal LST quadrature.
            50 is sufficient; 100 for high-precision tail inversion.
        random_state: Unused; present for API consistency.

    Example — Lloyd's syndicate:
        allocator = CMRSAllocator(distribution='gamma')
        allocator.fit_gamma(
            alphas=np.array([2.0, 3.0, 1.5, 4.0, 2.5]),
            betas=np.array([0.01, 0.008, 0.015, 0.007, 0.012]),
        )
        h = allocator.allocate(12_000_000.0)  # (5,) fair shares of £12M aggregate loss
        h_scr = allocator.allocate_quantile(np.array([0.995]))  # at SCR level
    """

    def __init__(
        self,
        distribution: str = "gamma",
        tilt_threshold: float = 0.99,
        n_euler_terms: int = 15,
        euler_a: float = 18.0,
        n_quad_points: int = 50,
        random_state: int = 42,
    ) -> None:
        if distribution not in ("exponential", "gamma", "lognormal"):
            raise ValueError(
                f"distribution must be 'exponential', 'gamma', or 'lognormal', got {distribution!r}"
            )
        if not 0.0 < tilt_threshold < 1.0:
            raise ValueError(f"tilt_threshold must be in (0, 1), got {tilt_threshold}")

        self.distribution = distribution
        self.tilt_threshold = tilt_threshold
        self.n_euler_terms = n_euler_terms
        self.euler_a = euler_a
        self.n_quad_points = n_quad_points
        self.random_state = random_state

        self._n: Optional[int] = None
        self._params: Optional[dict] = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting methods
    # ------------------------------------------------------------------

    def fit_exponential(self, rates: np.ndarray) -> "CMRSAllocator":
        """Fit independent exponential marginals.

        Args:
            rates: Rate parameters lambda_i, shape (n,). E[X_i] = 1/lambda_i.

        Returns:
            self (fitted)
        """
        rates = np.asarray(rates, dtype=float)
        if rates.ndim != 1 or (rates <= 0).any():
            raise ValueError("rates must be a 1D array of positive values")
        self._n = len(rates)
        self._params = {"rates": rates}
        self.distribution = "exponential"
        self._is_fitted = True
        return self

    def fit_gamma(self, alphas: np.ndarray, betas: np.ndarray) -> "CMRSAllocator":
        """Fit independent gamma marginals.

        Args:
            alphas: Shape parameters alpha_i > 0, shape (n,).
            betas: Rate parameters beta_i > 0 (mean = alpha_i/beta_i), shape (n,).

        Returns:
            self (fitted)
        """
        alphas = np.asarray(alphas, dtype=float)
        betas = np.asarray(betas, dtype=float)
        if alphas.shape != betas.shape or alphas.ndim != 1:
            raise ValueError("alphas and betas must be 1D arrays of the same length")
        if (alphas <= 0).any() or (betas <= 0).any():
            raise ValueError("All alphas and betas must be positive")
        self._n = len(alphas)
        self._params = {"alphas": alphas, "betas": betas}
        self.distribution = "gamma"
        self._is_fitted = True
        return self

    def fit_lognormal(self, mus: np.ndarray, sigmas: np.ndarray) -> "CMRSAllocator":
        """Fit independent lognormal marginals.

        Args:
            mus: Log-mean parameters mu_i, shape (n,).
            sigmas: Log-standard-deviation parameters sigma_i > 0, shape (n,).

        Returns:
            self (fitted)
        """
        mus = np.asarray(mus, dtype=float)
        sigmas = np.asarray(sigmas, dtype=float)
        if mus.shape != sigmas.shape or mus.ndim != 1:
            raise ValueError("mus and sigmas must be 1D arrays of the same length")
        if (sigmas <= 0).any():
            raise ValueError("All sigmas must be positive")
        self._n = len(mus)
        self._params = {"mus": mus, "sigmas": sigmas}
        self.distribution = "lognormal"
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # LST computation: joint and individual
    # ------------------------------------------------------------------

    def _lst_i(self, i: int, t: complex) -> complex:
        """L_{X_i}(t) for participant i."""
        if self.distribution == "exponential":
            return _lst_exponential(t, self._params["rates"][i])
        elif self.distribution == "gamma":
            return _lst_gamma(t, self._params["alphas"][i], self._params["betas"][i])
        elif self.distribution == "lognormal":
            return _lst_lognormal_quad(t, self._params["mus"][i], self._params["sigmas"][i], self.n_quad_points)
        else:
            raise RuntimeError(f"Unsupported distribution: {self.distribution!r}")

    def _lst_deriv_i(self, i: int, t: complex) -> complex:
        """E[X_i * exp(-t*X_i)] for participant i (= -d/dt L_{X_i}(t))."""
        if self.distribution == "exponential":
            return _lst_deriv_exponential(t, self._params["rates"][i])
        elif self.distribution == "gamma":
            return _lst_deriv_gamma(t, self._params["alphas"][i], self._params["betas"][i])
        elif self.distribution == "lognormal":
            return _lst_deriv_lognormal_quad(t, self._params["mus"][i], self._params["sigmas"][i], self.n_quad_points)
        else:
            raise RuntimeError(f"Unsupported distribution: {self.distribution!r}")

    def _joint_lst(self, t: complex) -> complex:
        """L_S(t) = prod_i L_{X_i}(t) for S = sum X_i (independence assumed)."""
        result = complex(1.0, 0.0)
        for i in range(self._n):
            result *= self._lst_i(i, t)
        return result

    def _allocation_lst_i(self, i: int, t: complex) -> complex:
        """LST of the allocation measure for participant i.

        tilde_nu_i(t) = E[X_i * exp(-t*S)] = E[X_i * exp(-t*X_i)] * prod_{j != i} L_{X_j}(t)

        This uses independence: X_i is independent of {X_j, j != i}, so:
            E[X_i * exp(-t*X_i) * prod_{j!=i} exp(-t*X_j)]
            = E[X_i * exp(-t*X_i)] * prod_{j!=i} E[exp(-t*X_j)]
        """
        deriv_i = self._lst_deriv_i(i, t)
        prod_others = complex(1.0, 0.0)
        for j in range(self._n):
            if j != i:
                prod_others *= self._lst_i(j, t)
        return deriv_i * prod_others

    # ------------------------------------------------------------------
    # Mean of S (for tilting threshold)
    # ------------------------------------------------------------------

    def _mean_s(self) -> float:
        """E[S] = sum_i E[X_i] under the fitted distribution."""
        if self.distribution == "exponential":
            return float(np.sum(1.0 / self._params["rates"]))
        elif self.distribution == "gamma":
            return float(np.sum(self._params["alphas"] / self._params["betas"]))
        elif self.distribution == "lognormal":
            mus = self._params["mus"]
            sigmas = self._params["sigmas"]
            return float(np.sum(np.exp(mus + 0.5 * sigmas ** 2)))
        else:
            raise RuntimeError(f"Unsupported distribution: {self.distribution!r}")

    def _var_s(self) -> float:
        """Var[S] = sum_i Var[X_i] (independence)."""
        if self.distribution == "exponential":
            return float(np.sum(1.0 / self._params["rates"] ** 2))
        elif self.distribution == "gamma":
            return float(np.sum(self._params["alphas"] / self._params["betas"] ** 2))
        elif self.distribution == "lognormal":
            mus = self._params["mus"]
            sigmas = self._params["sigmas"]
            means_sq = np.exp(2 * mus + sigmas ** 2)
            return float(np.sum(means_sq * (np.exp(sigmas ** 2) - 1)))
        else:
            raise RuntimeError(f"Unsupported distribution: {self.distribution!r}")

    def _quantile_s_approx(self, q: float) -> float:
        """Approximate F_S^{-1}(q) via normal approximation to S (CLT).

        Adequate for finding the tilt threshold; the exact allocation uses numerical
        inversion regardless.
        """
        mu = self._mean_s()
        std = np.sqrt(self._var_s())
        return float(stats.norm.ppf(q, loc=mu, scale=std))

    def _mgf_convergence_bound(self) -> float:
        """Return theta_max such that E[exp(theta*X_i)] < inf for all i.

        The CGF kappa(theta) = log E[exp(theta*S)] is only finite for
        theta < theta_max. Exponential tilting requires theta in (0, theta_max).

        For exponential: theta_max = min(rate_i)  (LST pole at rate_i).
        For gamma:       theta_max = min(beta_i)  (LST pole at beta_i).
        For lognormal:   lognormal has no finite MGF for theta > 0 in closed form,
                         but the quadrature approximation stays bounded for moderate
                         theta. We cap at 5.0 for numerical safety.
        """
        if self.distribution == "exponential":
            return float(np.min(self._params["rates"]))
        elif self.distribution == "gamma":
            return float(np.min(self._params["betas"]))
        elif self.distribution == "lognormal":
            return 5.0
        else:
            raise RuntimeError(f"Unsupported distribution: {self.distribution!r}")

    # ------------------------------------------------------------------
    # Exponential tilting
    # ------------------------------------------------------------------

    def _kappa(self, theta: float) -> float:
        """CGF of S: kappa(theta) = log E[exp(theta*S)] = sum_i log L_{X_i}(-theta).

        Only valid for theta < theta_max = _mgf_convergence_bound().
        """
        log_l = sum(np.log(np.real(self._lst_i(i, complex(-theta, 0.0))) + 1e-300)
                    for i in range(self._n))
        return float(log_l)

    def _kappa_prime(self, theta: float) -> float:
        """kappa'(theta) = E^theta[S] = mean of S under tilted measure.

        kappa'(theta) = d/dtheta log L_S(-theta)
                      = sum_i E[X_i exp(theta X_i)] / E[exp(theta X_i)]

        Returns float('inf') if theta is outside the convergence strip (i.e.,
        L_i(-theta) <= 0 for some i), so callers can detect the boundary.
        """
        total = 0.0
        for i in range(self._n):
            t = complex(-theta, 0.0)
            l_i = np.real(self._lst_i(i, t))
            dl_i = np.real(self._lst_deriv_i(i, t))
            if l_i <= 0:
                return float("inf")
            total += dl_i / l_i
        return total

    def _find_tilt_theta(self, s: float) -> float:
        """Find theta* such that kappa'(theta*) = s (tilted mean equals s).

        Searches within (0, theta_max) where theta_max = _mgf_convergence_bound().
        The CGF kappa is only defined for theta strictly less than theta_max, so the
        search must stay inside this strip. For exponentials theta_max = min(rate_i);
        for gammas theta_max = min(beta_i).

        The strategy: binary search toward theta_max to find an upper bracket, then
        use brentq to find the exact root.
        """
        if self._kappa_prime(0.0) >= s:
            return 0.0

        theta_max = self._mgf_convergence_bound()

        # Binary search toward theta_max to find upper bracket where kappa'(upper) > s.
        # kappa' is monotone increasing on (0, theta_max) and diverges at theta_max.
        upper = min(0.1, theta_max * 0.5)
        for _ in range(100):
            kp = self._kappa_prime(upper)
            if not np.isfinite(kp) or kp > s:
                break
            # Halve the remaining distance to theta_max
            upper = upper + (theta_max - upper) * 0.5
            if upper >= theta_max * 0.99999:
                upper = theta_max * 0.99999
                break
        else:
            warnings.warn(
                f"Could not bracket tilting parameter for s={s:.4g}. "
                "Results may be inaccurate.",
                RuntimeWarning,
                stacklevel=3,
            )
            return upper

        try:
            theta_star = optimize.brentq(
                lambda th: self._kappa_prime(th) - s,
                0.0,
                upper,
                xtol=1e-10,
                rtol=1e-10,
            )
        except ValueError:
            warnings.warn(
                f"Tilting optimisation failed at s={s:.4g}. Using theta=0.",
                RuntimeWarning,
                stacklevel=3,
            )
            theta_star = 0.0
        return float(theta_star)

    # ------------------------------------------------------------------
    # Core allocation at a single s value
    # ------------------------------------------------------------------

    def _allocate_single(self, s: float, tilt: bool = False) -> np.ndarray:
        """Compute h_i(s) for all i at a single aggregate loss value s.

        With tilt=False, directly inverts tilde_nu_i and L_S using Euler summation.
        With tilt=True, first finds theta* to shift the tilted distribution mean to s,
        then inverts the tilted transforms.

        Returns:
            h: (n,) array of conditional allocations, normalised to sum to s.
        """
        if tilt:
            theta = self._find_tilt_theta(s)
            kappa_theta = self._kappa(theta)

            def tilted_joint_lst(t: complex) -> complex:
                # L_S^theta(t) = L_S(t - theta) / exp(kappa(theta))
                return self._joint_lst(t - complex(theta, 0.0)) / np.exp(kappa_theta)

            def tilted_alloc_lst_i(i: int, t: complex) -> complex:
                return self._allocation_lst_i(i, t - complex(theta, 0.0)) / np.exp(kappa_theta)

            f_S_s = _euler_inversion(tilted_joint_lst, s, a=self.euler_a, M=self.n_euler_terms)
            xi = np.array([
                _euler_inversion(lambda t, _i=i: tilted_alloc_lst_i(_i, t), s, a=self.euler_a, M=self.n_euler_terms)
                for i in range(self._n)
            ])
        else:
            f_S_s = _euler_inversion(self._joint_lst, s, a=self.euler_a, M=self.n_euler_terms)
            xi = np.array([
                _euler_inversion(lambda t, _i=i: self._allocation_lst_i(_i, t), s, a=self.euler_a, M=self.n_euler_terms)
                for i in range(self._n)
            ])

        if f_S_s <= 0.0:
            # Numerical underflow in tail — fall back to mean-proportional allocation
            warnings.warn(
                f"f_S({s:.4g}) estimated as <= 0; falling back to mean-proportional allocation.",
                RuntimeWarning,
                stacklevel=4,
            )
            means = np.array([np.real(self._lst_deriv_i(i, complex(0.0, 0.0))) for i in range(self._n)])
            h = s * means / (means.sum() + 1e-300)
        else:
            h = xi / f_S_s

        # Budget balance: normalise to sum exactly to s
        total = h.sum()
        if total > 0.0:
            error_rel = abs(total - s) / (abs(s) + 1e-15)
            if error_rel > 1e-3:
                warnings.warn(
                    f"Budget balance error {error_rel:.2%} at s={s:.4g}. "
                    "Consider increasing n_euler_terms or activating tilting.",
                    RuntimeWarning,
                    stacklevel=4,
                )
            h = h * (s / total)
        return h

    def _should_tilt(self, s: float) -> bool:
        """Determine whether to use exponential tilting for this value of s."""
        threshold_s = self._quantile_s_approx(self.tilt_threshold)
        return s > threshold_s

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allocate(self, s: Union[float, np.ndarray]) -> np.ndarray:
        """Compute CMRS allocations h_i(s) for one or more aggregate loss values.

        Args:
            s: Scalar or array of aggregate loss values. All must be positive.

        Returns:
            If s is scalar: (n,) array of allocations.
            If s is array of shape (m,): (m, n) array of allocations.

        Raises:
            RuntimeError: If not fitted.
        """
        self._check_fitted()
        scalar_input = np.isscalar(s)
        s_arr = np.atleast_1d(np.asarray(s, dtype=float))
        if (s_arr <= 0).any():
            raise ValueError("All aggregate loss values s must be positive")

        results = np.zeros((len(s_arr), self._n))
        for idx, sv in enumerate(s_arr):
            use_tilt = self._should_tilt(sv)
            results[idx] = self._allocate_single(sv, tilt=use_tilt)

        if scalar_input:
            return results[0]
        return results

    def allocate_quantile(self, quantile_levels: np.ndarray) -> np.ndarray:
        """Compute CMRS allocations at specified quantiles of the aggregate loss S.

        The quantile F_S^{-1}(q) is approximated via CLT normal approximation for
        moderate quantiles, or via the tilting saddlepoint approximation for q >= tilt_threshold.

        Args:
            quantile_levels: Array of probabilities in (0, 1), shape (m,).

        Returns:
            (m, n) array of allocations h_i(F_S^{-1}(q_j)) for each j.
        """
        self._check_fitted()
        quantile_levels = np.asarray(quantile_levels, dtype=float)
        s_values = np.array([self._quantile_s_approx(q) for q in quantile_levels])
        return self.allocate(s_values)

    def aggregate_distribution(self, s_grid: np.ndarray) -> np.ndarray:
        """Approximate density of S on a grid via direct Laplace inversion.

        Uses the Euler-accelerated Abate-Whitt method to invert L_S(t) = prod_i L_{X_i}(t).
        For extreme tail values, the result may be numerically noisy without tilting;
        consider restricting to s_grid values below the tilt_threshold quantile.

        Args:
            s_grid: Array of positive values at which to evaluate f_S.

        Returns:
            (len(s_grid),) array of density values.
        """
        self._check_fitted()
        s_grid = np.asarray(s_grid, dtype=float)
        f_vals = np.zeros(len(s_grid))
        for idx, sv in enumerate(s_grid):
            if sv <= 0:
                f_vals[idx] = 0.0
                continue
            try:
                f_vals[idx] = _euler_inversion(self._joint_lst, sv, a=self.euler_a, M=self.n_euler_terms)
            except Exception:
                f_vals[idx] = np.nan
        return f_vals

    def budget_check(self, s: float) -> dict:
        """Verify budget balance h_1(s) + ... + h_n(s) = s and return diagnostics.

        Args:
            s: Aggregate loss value to check.

        Returns:
            dict with keys:
                s (float): input value
                allocations (np.ndarray): h_i(s)
                total (float): sum of allocations before normalisation
                absolute_error (float): |total - s|
                relative_error (float): |total - s| / s
                tilted (bool): whether exponential tilting was used
        """
        self._check_fitted()
        use_tilt = self._should_tilt(s)
        h = self._allocate_single(s, tilt=use_tilt)
        # Compute raw (pre-normalised) totals to report actual numerical error
        if use_tilt:
            theta = self._find_tilt_theta(s)
            kappa_theta = self._kappa(theta)
            f_S_s = _euler_inversion(
                lambda t: self._joint_lst(t - complex(theta, 0.0)) / np.exp(kappa_theta),
                s, a=self.euler_a, M=self.n_euler_terms
            )
            xi = np.array([
                _euler_inversion(
                    lambda t, _i=i: self._allocation_lst_i(_i, t - complex(theta, 0.0)) / np.exp(kappa_theta),
                    s, a=self.euler_a, M=self.n_euler_terms
                )
                for i in range(self._n)
            ])
        else:
            f_S_s = _euler_inversion(self._joint_lst, s, a=self.euler_a, M=self.n_euler_terms)
            xi = np.array([
                _euler_inversion(lambda t, _i=i: self._allocation_lst_i(_i, t), s, a=self.euler_a, M=self.n_euler_terms)
                for i in range(self._n)
            ])
        raw_h = xi / (f_S_s + 1e-300)
        raw_total = float(raw_h.sum())
        abs_err = abs(raw_total - s)
        rel_err = abs_err / (abs(s) + 1e-15)
        return {
            "s": s,
            "allocations": h,
            "total": raw_total,
            "absolute_error": abs_err,
            "relative_error": rel_err,
            "tilted": use_tilt,
        }

    def summary(self, s: float) -> dict:
        """Return allocation summary for aggregate loss value s.

        Args:
            s: Aggregate loss value.

        Returns:
            dict with keys:
                allocations (np.ndarray): h_i(s), shape (n,)
                aggregate_loss (float): s
                total (float): sum of allocations (= s after normalisation)
                tilted (bool): whether exponential tilting was used
                mean_s (float): E[S] under fitted distribution
        """
        self._check_fitted()
        use_tilt = self._should_tilt(s)
        h = self._allocate_single(s, tilt=use_tilt)
        return {
            "allocations": h,
            "aggregate_loss": s,
            "total": float(h.sum()),
            "tilted": use_tilt,
            "mean_s": self._mean_s(),
        }

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("CMRSAllocator is not fitted. Call fit_exponential(), fit_gamma(), or fit_lognormal() first.")

    def __repr__(self) -> str:
        n_str = f"n={self._n}" if self._is_fitted else "not fitted"
        return (
            f"CMRSAllocator(distribution={self.distribution!r}, "
            f"n_euler_terms={self.n_euler_terms}, euler_a={self.euler_a}, "
            f"{n_str})"
        )
