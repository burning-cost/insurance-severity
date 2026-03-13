"""
CatBoost baseline wrapper.

CatBoostBaseline wraps either:
  (a) an insurance-distributional model that exposes predict_cdf(), or
  (b) a plain CatBoost regressor that predicts the mean, combined with a
      dispersion estimate to form a parametric distribution.

catboost is an optional dependency — this module is lazy-imported so the
rest of the library works without it installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


class CatBoostBaseline:
    """
    Wraps a CatBoost model as a DRN baseline distribution.

    Two modes:

    1. **insurance-distributional mode**: if the wrapped model has a
       ``predict_cdf(X, cutpoints)`` method, that is called directly.
       This is the preferred mode — CatBoost handles all feature interactions,
       and the cdf is already parametric-family-aware.

    2. **plain-mean mode**: if the model only predicts a scalar mean (standard
       CatBoost regressor), combine with ``family`` and ``dispersion`` to
       form a full parametric CDF. Dispersion can be estimated from training
       residuals via ``fit_dispersion(y_train, X_train)``.

    Parameters
    ----------
    model : CatBoost model or insurance-distributional model
        Must implement ``.predict(X)`` returning (n,) means. Optionally also
        ``predict_cdf(X, cutpoints)`` returning (n, K+1).
    family : str
        'gamma' | 'gaussian' | 'lognormal' | 'inversegaussian'.
        Used only in plain-mean mode.
    dispersion : float, optional
        Dispersion parameter (phi for Gamma, sigma^2 for Gaussian).
        If None, defaults to 1.0 in plain-mean mode.
    """

    def __init__(
        self,
        model: Any,
        family: str = "gamma",
        dispersion: float | None = None,
    ):
        self.model = model
        self.distribution_family = family
        self._dispersion = dispersion if dispersion is not None else 1.0
        self._has_predict_cdf = hasattr(model, "predict_cdf")

    def predict_params(self, X: pd.DataFrame) -> dict:
        """Return {'mu': array, 'dispersion': float}."""
        mu = np.asarray(self.model.predict(X), dtype=np.float64)
        return {"mu": mu, "dispersion": self._dispersion}

    def predict_cdf(self, X: pd.DataFrame, cutpoints: np.ndarray) -> np.ndarray:
        """
        CDF at each cutpoint for each observation.

        Delegates to model.predict_cdf if available; otherwise uses a
        parametric distribution with the predicted mean.

        Returns
        -------
        np.ndarray, shape (n, len(cutpoints))
        """
        if self._has_predict_cdf:
            result = self.model.predict_cdf(X, cutpoints)
            return np.asarray(result, dtype=np.float64)

        # Plain-mean mode: build parametric CDF
        params = self.predict_params(X)
        mu = params["mu"][:, np.newaxis]   # (n, 1)
        disp = self._dispersion
        c = cutpoints[np.newaxis, :]       # (1, K)

        return self._parametric_cdf(mu, disp, c)

    def fit_dispersion(self, y_train: np.ndarray, X_train: pd.DataFrame) -> "CatBoostBaseline":
        """
        Estimate dispersion from training residuals (Pearson method).

        For Gamma: phi = mean((y - mu)^2 / mu^2) (Pearson / n).
        For Gaussian: sigma^2 = mean((y - mu)^2).

        Modifies self._dispersion in place and returns self for chaining.
        """
        mu = np.asarray(self.model.predict(X_train), dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64)

        if self.distribution_family == "gamma":
            self._dispersion = float(np.mean(((y - mu) / mu) ** 2))
        elif self.distribution_family in ("gaussian", "lognormal"):
            self._dispersion = float(np.var(y - mu))
        elif self.distribution_family == "inversegaussian":
            self._dispersion = float(np.mean(((y - mu) ** 2) / (mu ** 3)))
        else:
            self._dispersion = float(np.var(y - mu))
        return self

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _parametric_cdf(self, mu_2d: np.ndarray, disp: float, c_2d: np.ndarray) -> np.ndarray:
        fam = self.distribution_family
        if fam == "gamma":
            alpha = 1.0 / disp
            scale = mu_2d * disp
            return stats.gamma.cdf(c_2d, a=alpha, scale=scale)
        elif fam == "gaussian":
            sigma = np.sqrt(disp)
            return stats.norm.cdf(c_2d, loc=mu_2d, scale=sigma)
        elif fam == "lognormal":
            sigma_log = float(np.sqrt(disp))
            mu_log = np.log(mu_2d) - 0.5 * sigma_log ** 2
            return stats.lognorm.cdf(c_2d, s=sigma_log, scale=np.exp(mu_log))
        elif fam == "inversegaussian":
            lam = mu_2d / disp
            return stats.invgauss.cdf(c_2d, mu=mu_2d / lam, scale=lam)
        else:
            raise ValueError(f"Unknown family: {fam!r}")
