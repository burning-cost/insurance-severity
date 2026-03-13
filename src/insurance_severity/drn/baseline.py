"""
Baseline distribution wrappers.

BaselineDistribution is a Protocol — any object satisfying the interface works.
GLMBaseline wraps a fitted statsmodels GLM result. Supports Gamma, Gaussian,
LogNormal (via statsmodels Gaussian on log-y), and InverseGaussian families.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from scipy import stats


@runtime_checkable
class BaselineDistribution(Protocol):
    """
    Protocol for frozen baseline distributions consumed by DRN.

    Every concrete baseline must be fitted before being passed to DRN.
    The DRN never modifies the baseline — it only reads CDF values.
    """

    distribution_family: str  # 'gamma' | 'gaussian' | 'lognormal' | 'inversegaussian'

    def predict_params(self, X: pd.DataFrame) -> dict:
        """
        Return per-observation distributional parameters.

        Must include at minimum 'mu' (the mean). Family-specific extras:
        - Gamma: 'dispersion' (1/shape = phi, so shape=1/phi)
        - Gaussian: 'sigma'
        - LogNormal: 'sigma' (sigma of the log-scale)
        - InverseGaussian: 'dispersion'

        Parameters
        ----------
        X : pd.DataFrame, shape (n, p)

        Returns
        -------
        dict of str -> np.ndarray | float
        """
        ...

    def predict_cdf(self, X: pd.DataFrame, cutpoints: np.ndarray) -> np.ndarray:
        """
        CDF at each cutpoint for each observation.

        Parameters
        ----------
        X : pd.DataFrame, shape (n, p)
        cutpoints : np.ndarray, shape (K+1,)  — sorted ascending

        Returns
        -------
        np.ndarray, shape (n, K+1)
            F_baseline(c_k | x_i) for each i, k.
        """
        ...


class GLMBaseline:
    """
    Wraps a fitted statsmodels GLM result as a DRN baseline.

    Supports Gamma, Gaussian, LogNormal, and InverseGaussian. The family is
    inferred from the statsmodels result by default, or can be overridden.

    Parameters
    ----------
    smf_result : statsmodels GLMResultsWrapper
        A fitted GLM — the output of ``glm.fit()``.
    family : str, optional
        One of 'gamma', 'gaussian', 'lognormal', 'inversegaussian'.
        If None, inferred from the statsmodels family class name.
    dispersion : float, optional
        Override the dispersion estimate. By default, uses the Pearson
        chi-squared dispersion from the GLM result. For Gamma this is phi
        (the inverse shape parameter). For Gaussian this is sigma^2.

    Notes
    -----
    For LogNormal: the convention here is that the GLM is fitted on log(y)
    with a Gaussian family (identity link), giving mu=E[log y], sigma=sqrt(dispersion).
    predict_cdf then uses the LogNormal CDF correctly.

    Alternatively, pass family='lognormal' with a Gamma-linked GLM fitted on
    the raw scale — the library will use the GLM mu as the lognormal mean and
    derive sigma from dispersion via the variance relationship.
    """

    def __init__(
        self,
        smf_result,
        family: str | None = None,
        dispersion: float | None = None,
    ):
        self._result = smf_result
        self.distribution_family = family or self._infer_family(smf_result)
        self._dispersion_override = dispersion

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict_params(self, X: pd.DataFrame) -> dict:
        """Return per-row distribution parameters given features X."""
        mu = self._predict_mu(X)
        disp = self._get_dispersion()
        return {"mu": mu, "dispersion": disp}

    def predict_cdf(self, X: pd.DataFrame, cutpoints: np.ndarray) -> np.ndarray:
        """
        Vectorised CDF for all (observation, cutpoint) pairs.

        Returns
        -------
        np.ndarray, shape (n, len(cutpoints))
        """
        params = self.predict_params(X)
        mu = params["mu"]          # (n,)
        disp = params["dispersion"]  # scalar or (n,)
        n = len(mu)
        K = len(cutpoints)

        # Broadcast: mu -> (n, 1), cutpoints -> (1, K)
        mu_2d = mu[:, np.newaxis]          # (n, 1)
        c_2d = cutpoints[np.newaxis, :]    # (1, K)

        return self._cdf_vectorised(mu_2d, disp, c_2d, n, K)

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        family: str = "gamma",
        **fit_kwargs,
    ) -> "GLMBaseline":
        """
        Fit a GLM and return a GLMBaseline in one call.

        Parameters
        ----------
        formula : str
            Patsy formula, e.g. ``'claims ~ age + C(region)'``
        data : pd.DataFrame
        family : str
            'gamma', 'gaussian', 'lognormal', 'inversegaussian'
        **fit_kwargs
            Passed to ``glm.fit()``.

        Returns
        -------
        GLMBaseline
        """
        try:
            import statsmodels.api as sm
            import statsmodels.formula.api as smf
        except ImportError as e:
            raise ImportError(
                "statsmodels is required for GLMBaseline.from_formula(). "
                "Install it with: pip install insurance-drn[glm]"
            ) from e

        sm_family = _sm_family(family)
        result = smf.glm(formula, data=data, family=sm_family).fit(**fit_kwargs)
        return cls(result, family=family)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _predict_mu(self, X: pd.DataFrame) -> np.ndarray:
        """Get fitted means from GLM for new data X."""
        try:
            mu = self._result.predict(X)
        except Exception:
            # Some statsmodels results need exog directly
            mu = self._result.predict(X.values)
        return np.asarray(mu, dtype=np.float64)

    def _get_dispersion(self) -> float:
        """Return dispersion estimate (phi for Gamma, sigma^2 for Gaussian)."""
        if self._dispersion_override is not None:
            return float(self._dispersion_override)
        # Pearson chi-squared dispersion from fitted result
        try:
            disp = self._result.scale
            return float(disp)
        except AttributeError:
            return 1.0

    def _cdf_vectorised(
        self,
        mu_2d: np.ndarray,
        disp: float,
        c_2d: np.ndarray,
        n: int,
        K: int,
    ) -> np.ndarray:
        """Dispatch to family-specific CDF."""
        fam = self.distribution_family

        if fam == "gamma":
            # Gamma(alpha, beta) where alpha = 1/phi, mean = alpha/beta = mu
            # So alpha = 1/disp, beta = 1/(disp * mu)
            alpha = 1.0 / disp  # shape — scalar
            # scale = mu * disp  (n, 1) * scalar
            scale = mu_2d * disp
            # CDF: scipy gamma uses shape + scale parameterisation
            return stats.gamma.cdf(c_2d, a=alpha, scale=scale)

        elif fam == "gaussian":
            sigma = np.sqrt(disp)
            return stats.norm.cdf(c_2d, loc=mu_2d, scale=sigma)

        elif fam == "lognormal":
            # mu here is E[Y] on raw scale (from GLM with log link)
            # sigma^2 = log(1 + disp / mu^2) but we treat disp as sigma on log scale
            # Convention: disp IS sigma (std of log Y), mu is the GLM prediction
            # E[Y] = exp(mu_log + 0.5*sigma^2) for lognormal
            # Here we accept mu as raw-scale mean and convert:
            sigma_log = float(np.sqrt(disp)) if disp < 10 else float(disp)
            # mu_log = log(mu) - 0.5*sigma_log^2
            mu_log = np.log(mu_2d) - 0.5 * sigma_log ** 2
            return stats.lognorm.cdf(c_2d, s=sigma_log, scale=np.exp(mu_log))

        elif fam == "inversegaussian":
            # InverseGaussian(mu, lambda) where lambda = mu^2 / (disp * mu) = mu/disp
            lam = mu_2d / disp
            return stats.invgauss.cdf(c_2d, mu=mu_2d / lam, scale=lam)

        else:
            raise ValueError(
                f"Unknown distribution family: {fam!r}. "
                "Supported: 'gamma', 'gaussian', 'lognormal', 'inversegaussian'."
            )

    @staticmethod
    def _infer_family(result) -> str:
        """Infer family string from statsmodels GLM result family class."""
        try:
            family_class = type(result.family).__name__.lower()
        except AttributeError:
            return "gamma"

        if "gamma" in family_class:
            return "gamma"
        elif "gaussian" in family_class:
            return "gaussian"
        elif "inversegaussian" in family_class or "inverse_gaussian" in family_class:
            return "inversegaussian"
        else:
            return "gamma"  # safe default for insurance severity


def _sm_family(family: str):
    """Return statsmodels family object for a family name string."""
    try:
        import statsmodels.api as sm
    except ImportError as e:
        raise ImportError("statsmodels required") from e

    families = {
        "gamma": sm.families.Gamma(sm.families.links.Log()),
        "gaussian": sm.families.Gaussian(sm.families.links.Log()),
        "lognormal": sm.families.Gaussian(sm.families.links.Identity()),
        "inversegaussian": sm.families.InverseGaussian(sm.families.links.Log()),
    }
    if family not in families:
        raise ValueError(f"Unknown family {family!r}. Choose from: {list(families)}")
    return families[family]
