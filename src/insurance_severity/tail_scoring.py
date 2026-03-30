"""
Tail calibration diagnostics and tail scoring rules for severity models.

Two complementary approaches to evaluating how well your model captures
extreme claims:

1. **TailCalibration** — Allen et al. (2025, JASA) tail calibration diagnostics.
   Works across all EVT domains (Fréchet, Gumbel, Weibull). Checks whether your
   model's predicted exceedance probabilities and conditional tail shapes are
   correct. This is an absolute reliability check — it tells you whether your
   model is calibrated in the tail, not just which of several candidates is best.

2. **BladtTailScore** — Bladt & Øhlenschlæger (arXiv:2603.24122) tail log-score.
   Fréchet domain only (regular variation, xi > 0). Provides a strictly proper
   scoring rule for ranking competing Pareto-family models on tail fit. The key
   insight: evaluate on normalised upper order statistics rather than the full
   sample. Score optimisation recovers the Hill estimator as a special case.

3. **pareto_qq** — Pareto QQ plot with R² statistic. Use this before applying
   BladtTailScore to confirm you are in the Fréchet domain.

References
----------
Allen, Koh, Segers, Ziegel (2025). Tail calibration of probabilistic forecasts.
JASA. arXiv:2407.03167.

Bladt & Øhlenschlæger (2026). Tail scoring rules for heavy-tailed distributions.
arXiv:2603.24122.

Segers (2001). Residual estimators. J. Stat. Plan. Inf. 98: 15–27.
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Pareto QQ plot
# ---------------------------------------------------------------------------


def pareto_qq(
    y: np.ndarray,
    k: Optional[int] = None,
    ax=None,
) -> float:
    """
    Pareto QQ plot for Fréchet domain verification.

    Plots log(Y_{n,n-i+1}) against log((n+1)/i) for i = 1, ..., k.
    In the Fréchet domain (regularly varying tail), the upper order statistics
    follow Pareto, so this plot should be linear. Non-linearity indicates
    Gumbel or Weibull domain, or that Bladt tail scoring should not be used.

    Parameters
    ----------
    y : array-like, shape (n,)
        Observations (strictly positive).
    k : int, optional
        Number of upper order statistics to include. Defaults to
        min(n // 4, 200) — show the top quarter of the data.
    ax : matplotlib.axes.Axes, optional
        If None, creates a new figure.

    Returns
    -------
    r_squared : float
        R² of linear fit to the plotted points. R² > 0.95 suggests
        Fréchet domain is plausible. R² << 0.95 (e.g. < 0.85) strongly
        suggests Gumbel domain — do not apply BladtTailScore.
    """
    import matplotlib.pyplot as plt

    y = np.asarray(y, dtype=float)
    if np.any(y <= 0):
        raise ValueError("pareto_qq requires strictly positive observations")

    n = len(y)
    if n < 10:
        raise ValueError(f"Need at least 10 observations, got {n}")

    if k is None:
        k = min(n // 4, 200)
    k = min(k, n - 1)
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")

    y_sorted = np.sort(y)  # ascending
    # Top-k order statistics: Y_{n,n-1}, ..., Y_{n,n-k}
    top_k = y_sorted[-(k):][::-1]  # descending: largest first

    # Theoretical Pareto quantiles: log((n+1)/i) for i = 1,...,k
    i_vals = np.arange(1, k + 1)
    x_theoretical = np.log((n + 1) / i_vals)
    y_observed = np.log(top_k)

    # Linear fit
    slope, intercept, r_value, p_value, _ = stats.linregress(x_theoretical, y_observed)
    r_squared = float(r_value ** 2)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(x_theoretical, y_observed, s=12, color="steelblue", alpha=0.7, label="Data")
    x_line = np.array([x_theoretical.min(), x_theoretical.max()])
    ax.plot(x_line, intercept + slope * x_line, color="tomato", lw=1.5, label=f"Fit (R²={r_squared:.3f})")
    ax.set_xlabel("Theoretical Pareto quantile [log((n+1)/i)]")
    ax.set_ylabel("log(Y_{(n-i+1)})")
    ax.set_title(f"Pareto QQ plot (k={k}, n={n})")
    ax.legend(fontsize=9)

    return r_squared


# ---------------------------------------------------------------------------
# TailCalibration
# ---------------------------------------------------------------------------


class TailCalibration:
    """
    Allen et al. (2025, JASA) tail calibration diagnostics.

    Checks whether a probabilistic forecast is calibrated in the tail,
    using two diagnostic conditions as threshold t increases:

    1. **Occurrence calibration:** The model-predicted probability of
       exceeding threshold t should match the empirical exceedance rate.
       Occurrence ratio R_occ(t) = mean(y > t) / mean(1 - F_i(t)) → 1.

    2. **Severity calibration:** Given that Y > t, the model's conditional
       excess distribution should be correctly calibrated. Test via PIT
       of the conditional excess: Z_i = (F_i(y_i) - F_i(t)) / (1 - F_i(t))
       for obs with y_i > t. These should be Uniform(0,1).

    The combined ratio R_hat_t(u) combines both: it equals u for a calibrated
    model, and should track the diagonal in a combined ratio plot.

    Works across all EVT domains (Fréchet, Gumbel, Weibull) — unlike
    BladtTailScore which requires Fréchet domain.

    Parameters
    ----------
    cdf_func : callable
        Function cdf_func(t) -> ndarray shape (n,). Returns the model CDF
        evaluated at scalar threshold t for all n observations. F_i(t) is the
        probability that observation i is <= t under its individual predictive
        distribution.
    n_obs : int
        Number of observations (must match cdf_func output length).

    Notes
    -----
    Composite or non-covariate models use the same CDF for every observation,
    so in-sample exceedance rate will trivially equal mean(1 - F(t)). Warn
    about this. Occurrence calibration is only non-trivial when F_i varies
    across observations (covariate-indexed models).

    References
    ----------
    Allen, S., Koh, J., Segers, J., Ziegel, J. (2025). Tail calibration of
    probabilistic forecasts. JASA. arXiv:2407.03167.
    """

    def __init__(
        self,
        cdf_func: Callable[[float], np.ndarray],
        n_obs: int,
    ) -> None:
        self._cdf_func = cdf_func
        self._n_obs = int(n_obs)
        self._y: Optional[np.ndarray] = None

    def fit(self, y: np.ndarray) -> "TailCalibration":
        """
        Store observed actuals.

        Parameters
        ----------
        y : array-like, shape (n,)
            Observed claim amounts. Must have length equal to n_obs.

        Returns
        -------
        self
        """
        y = np.asarray(y, dtype=float)
        if len(y) != self._n_obs:
            raise ValueError(
                f"y length {len(y)} does not match n_obs={self._n_obs}"
            )
        self._y = y
        return self

    def _check_fitted(self) -> None:
        if self._y is None:
            raise RuntimeError("Call fit(y) first")

    def _get_cdf_at(self, t: float) -> np.ndarray:
        """Return F_i(t) for all i, shape (n,)."""
        f = np.asarray(self._cdf_func(t), dtype=float)
        if len(f) != self._n_obs:
            raise ValueError(
                f"cdf_func returned {len(f)} values, expected {self._n_obs}"
            )
        return f

    def occurrence_ratio(self, t: float) -> float:
        """
        Occurrence ratio R_occ(t) = mean(y > t) / mean(1 - F_i(t)).

        Should be close to 1.0 for a calibrated model. Values > 1 indicate
        the model underestimates tail exceedance probabilities; < 1 indicates
        overestimation.

        Parameters
        ----------
        t : float
            Threshold value.

        Returns
        -------
        float
            R_occ(t). Returns NaN if mean(1 - F_i(t)) is near zero.
        """
        self._check_fitted()
        y = self._y
        F_t = self._get_cdf_at(t)
        emp_exc = float(np.mean(y > t))
        pred_exc = float(np.mean(1.0 - F_t))
        if pred_exc < 1e-10:
            return float("nan")
        return emp_exc / pred_exc

    def severity_pit(self, t: float) -> np.ndarray:
        """
        Conditional excess PITs for observations with y_i > t.

        Z_i = (F_i(y_i) - F_i(t)) / (1 - F_i(t))

        These should be distributed Uniform(0,1) if the model's conditional
        tail shape is correct. Departures indicate tail shape miscalibration
        (wrong tail index, wrong body shape carried too far into tail, etc.).

        Parameters
        ----------
        t : float
            Threshold value.

        Returns
        -------
        np.ndarray
            Array of PIT values Z_i for all obs with y_i > t. May be empty
            if no observations exceed t.
        """
        self._check_fitted()
        y = self._y
        mask = y > t
        if not np.any(mask):
            return np.array([])

        y_exc = y[mask]
        F_t = self._get_cdf_at(t)[mask]

        # Evaluate CDF at each exceedance value y_i
        # We need F_i(y_i) for each exceedance — we evaluate cdf_func at
        # each observed y_i value, returning n values, then select masked obs.
        # For efficiency, compute column-by-column for unique y values.
        F_y = np.empty(len(y_exc))
        # We need F_i(y_i) only for i in mask. Since cdf_func takes a scalar t
        # and returns all n values, we need one call per unique y_exc value.
        # For small exceedance sets this is fine. For large sets, batch.
        unique_y, inv_idx = np.unique(y_exc, return_inverse=True)
        masked_indices = np.where(mask)[0]
        for j, yval in enumerate(unique_y):
            # cdf_func(yval) returns shape (n,); select only masked observations
            F_yval_all = self._get_cdf_at(float(yval))
            where_this_y = np.where(inv_idx == j)[0]
            for idx_in_exc in where_this_y:
                orig_i = masked_indices[idx_in_exc]
                F_y[idx_in_exc] = F_yval_all[orig_i]

        # PIT: Z_i = (F_i(y_i) - F_i(t)) / (1 - F_i(t))
        denom = 1.0 - F_t
        # Clip to avoid division by zero for near-certain exceedances
        denom = np.where(denom < 1e-10, 1e-10, denom)
        Z = (F_y - F_t) / denom
        # Clip to [0,1] due to numerical issues
        Z = np.clip(Z, 0.0, 1.0)
        return Z

    def combined_ratio_plot(
        self,
        t_levels: np.ndarray,
        ax=None,
    ):
        """
        Combined ratio plot R_hat_t(u) for a grid of thresholds.

        For each threshold t, plots R_hat_t(u) vs u where:
            R_hat_t(u) = count(Z_i <= u) / n / mean(1 - F_i(t))

        A calibrated model produces R_hat_t(u) ≈ u (the diagonal).
        Deviations indicate occurrence or severity miscalibration.

        Parameters
        ----------
        t_levels : array-like
            Threshold values to assess.
        ax : matplotlib.axes.Axes, optional
            If None, creates a new figure.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        self._check_fitted()
        t_levels = np.asarray(t_levels, dtype=float)

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 6))
        else:
            fig = ax.get_figure()

        u_grid = np.linspace(0.0, 1.0, 101)
        n = self._n_obs
        y = self._y

        cmap = plt.get_cmap("viridis")
        n_levels = len(t_levels)

        for i, t in enumerate(t_levels):
            if t >= np.max(y):
                warnings.warn(
                    f"Threshold t={t:.3g} >= max(y); skipping.",
                    stacklevel=2,
                )
                continue
            F_t = self._get_cdf_at(t)
            pred_exc = float(np.mean(1.0 - F_t))
            if pred_exc < 1e-10:
                continue

            Z = self.severity_pit(t)
            if len(Z) < 5:
                continue

            R_hat = np.array([
                float(np.sum(Z <= u)) / n / pred_exc
                for u in u_grid
            ])

            color = cmap(i / max(n_levels - 1, 1))
            ax.plot(u_grid, R_hat, lw=1.2, color=color, alpha=0.8,
                    label=f"t={t:.3g}")

        ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Ideal (calibrated)")
        ax.set_xlabel("u")
        ax.set_ylabel(r"$\hat{R}_t(u)$")
        ax.set_title("Combined ratio plot (Allen et al. 2025)")
        ax.legend(fontsize=8, loc="upper left")
        return fig

    def pit_histogram(
        self,
        t: float,
        ax=None,
        n_bins: int = 10,
    ):
        """
        PIT histogram for conditional excesses above threshold t.

        Plots the distribution of Z_i = (F_i(y_i) - F_i(t)) / (1 - F_i(t))
        for obs with y_i > t. Should be approximately uniform (flat) if the
        model's conditional tail shape is correct.

        Parameters
        ----------
        t : float
            Threshold value.
        ax : matplotlib.axes.Axes, optional
            If None, creates a new figure.
        n_bins : int
            Number of histogram bins. Default 10.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        self._check_fitted()

        Z = self.severity_pit(t)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        else:
            fig = ax.get_figure()

        n_exc = len(Z)

        if n_exc == 0:
            ax.set_title(f"PIT histogram (t={t:.3g}): no exceedances")
            return fig

        if n_exc < 20:
            warnings.warn(
                f"Only {n_exc} exceedances above t={t:.3g}; "
                "PIT histogram unreliable (n < 20).",
                stacklevel=2,
            )

        ax.hist(Z, bins=n_bins, range=(0, 1), density=True,
                color="steelblue", edgecolor="white", alpha=0.75)
        ax.axhline(1.0, color="tomato", ls="--", lw=1.5, label="Uniform")
        ax.set_xlabel("PIT value Z")
        ax.set_ylabel("Density")
        ax.set_title(f"Conditional excess PIT (t={t:.3g}, n={n_exc})")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=9)
        return fig

    def summary_table(
        self,
        t_levels: np.ndarray,
    ) -> pd.DataFrame:
        """
        Summary table of tail calibration diagnostics across thresholds.

        For each threshold t computes:
        - n_exceedances: number of obs with y > t
        - R_occ: occurrence ratio (should be 1.0)
        - ks_pvalue: KS test p-value for uniformity of conditional excess PITs
          (should be > 0.05 for calibrated model)

        Parameters
        ----------
        t_levels : array-like
            Threshold values to assess.

        Returns
        -------
        pd.DataFrame
            Columns: threshold, n_exceedances, R_occ, ks_pvalue
        """
        self._check_fitted()
        t_levels = np.asarray(t_levels, dtype=float)

        rows = []
        y = self._y

        for t in t_levels:
            if t >= float(np.max(y)):
                warnings.warn(
                    f"Threshold t={t:.3g} >= max(y); skipping.",
                    stacklevel=2,
                )
                continue

            n_exc = int(np.sum(y > t))
            r_occ = self.occurrence_ratio(t)
            Z = self.severity_pit(t)

            if n_exc < 20:
                warnings.warn(
                    f"Only {n_exc} exceedances above t={t:.3g}; "
                    "KS test unreliable (n < 20).",
                    stacklevel=2,
                )

            if len(Z) >= 2:
                ks_stat, ks_pvalue = stats.kstest(Z, "uniform")
            else:
                ks_pvalue = float("nan")

            rows.append({
                "threshold": float(t),
                "n_exceedances": n_exc,
                "R_occ": r_occ,
                "ks_pvalue": ks_pvalue,
            })

        return pd.DataFrame(rows, columns=["threshold", "n_exceedances", "R_occ", "ks_pvalue"])


# ---------------------------------------------------------------------------
# BladtTailScore
# ---------------------------------------------------------------------------


def _hill_estimator(y_sorted_asc: np.ndarray, k: int) -> float:
    """
    Hill (1975) tail index estimator.

    y_sorted_asc : sorted ascending
    k            : number of upper order statistics (k >= 2, k < n)

    Returns gamma_hat = (1/k) * sum log(Y_{n,n-i+1} / Y_{n,n-k})
    """
    n = len(y_sorted_asc)
    if k < 2 or k >= n:
        return float("nan")
    x_ref = y_sorted_asc[-(k + 1)]
    top_k = y_sorted_asc[-k:]
    return float(np.mean(np.log(top_k / x_ref)))


def _tail_log_score(y_sorted_asc: np.ndarray, gamma: float, k: int) -> float:
    """
    Empirical tail log-score S_k(Pareto(gamma)), eq. (4.1) of arXiv:2603.24122.

    S_k(gamma) = (1/k) * sum_{i=1}^{k} log f_{Pareto(gamma)}(Y_{n,n-i+1} / Y_{n,n-k})

    where f_{Pareto(gamma)}(z) = (1/gamma) * z^{-(1/gamma + 1)} for z >= 1.

    Higher score = better tail fit.
    """
    n = len(y_sorted_asc)
    if k < 2 or k >= n:
        return float("nan")
    if gamma <= 0:
        return float("nan")
    x_ref = y_sorted_asc[-(k + 1)]
    top_k = y_sorted_asc[-k:]
    ratios = top_k / x_ref  # in [1, inf)
    # log density: log(1/gamma) - (1/gamma + 1) * log(z)
    log_scores = np.log(1.0 / gamma) - (1.0 / gamma + 1.0) * np.log(ratios)
    return float(np.mean(log_scores))


class BladtTailScore:
    """
    Bladt & Øhlenschlæger (arXiv:2603.24122) tail log-score for model ranking.

    Provides a strictly proper scoring rule for comparing Pareto-family severity
    models in the Fréchet domain (xi > 0, regular variation). The approach
    evaluates candidates on normalised upper order statistics rather than the
    full sample — this is what gives it discriminating power in the tail.

    Mathematical basis:
    - The top-k normalised ratios Y_{n,n-i+1}/Y_{n,n-k} converge to i.i.d.
      Pareto(1/gamma_G) as k,n/k -> inf (Pickands-Balkema-de Haan).
    - The tail log-score is the average log-density of these ratios under a
      candidate Pareto(1/gamma), higher = better fit.
    - Score optimisation over gamma recovers the Hill estimator (Theorem 6,
      Corollary 8 of the paper) — Hill is score-optimal for log-scoring.
    - Asymptotic normality (Theorem 3) gives CIs: Var = (1/gamma - 1)^2 * gamma_G^2.

    Limitations:
    - Fréchet domain only (power-law tails, xi > 0). NOT valid for lognormal
      (Gumbel domain). Use pareto_qq() R² > 0.95 to confirm domain first.
    - Sample requirements: n >= 10,000 for reliable discrimination. At n ~ 1,000,
      ranking is suggestive not decisive.
    - Does not handle policy-limit truncation or IBNR censoring.

    References
    ----------
    Bladt & Øhlenschlæger (2026). arXiv:2603.24122.
    Segers (2001). Residual estimators. J. Stat. Plan. Inf. 98: 15–27.
    """

    def score(
        self,
        y: np.ndarray,
        gamma: float,
        k: int,
    ) -> tuple[float, float, float]:
        """
        Compute tail log-score for a single (gamma, k) pair, with CI.

        Parameters
        ----------
        y : array-like
            Observations (strictly positive, unsorted).
        gamma : float
            Candidate tail index (> 0). gamma = xi in GPD notation.
        k : int
            Number of upper order statistics. Should satisfy k < len(y)
            and k/n → 0 (intermediate sequence). Typical range: n*0.05 to n*0.25.

        Returns
        -------
        (score, ci_lower, ci_upper) : tuple of float
            score     : mean tail log-score (higher = better)
            ci_lower  : lower bound of 95% asymptotic CI
            ci_upper  : upper bound of 95% asymptotic CI

        Notes
        -----
        CI uses Theorem 3 / Corollary 4 of the paper:
            Var(LogS(Pareto(gamma), Y^circ)) = (1/gamma - 1)^2 * gamma_G^2
        where gamma_G is estimated by Hill. At gamma = gamma_G (true model),
        variance = 0 — the CI degenerates. This is correct: the score is
        asymptotically constant under the true model.
        """
        y = np.asarray(y, dtype=float)
        if np.any(y <= 0):
            raise ValueError("y must be strictly positive")
        n = len(y)
        if k < 2 or k >= n:
            return (float("nan"), float("nan"), float("nan"))

        y_sorted = np.sort(y)
        s = _tail_log_score(y_sorted, gamma, k)
        if not np.isfinite(s):
            return (s, float("nan"), float("nan"))

        gamma_G_hat = _hill_estimator(y_sorted, k)
        if not np.isfinite(gamma_G_hat) or gamma_G_hat < 1e-10:
            return (s, float("nan"), float("nan"))

        if gamma_G_hat < 0.05:
            warnings.warn(
                f"Hill estimate gamma_G_hat={gamma_G_hat:.4f} is near the "
                "Gumbel boundary (xi ≈ 0). Fréchet domain may not hold; "
                "BladtTailScore results are unreliable.",
                stacklevel=2,
            )

        # Asymptotic variance: Var = (1/gamma - 1)^2 * gamma_G^2
        var = (1.0 / gamma - 1.0) ** 2 * gamma_G_hat ** 2
        se = np.sqrt(var / k)
        z95 = stats.norm.ppf(0.975)
        return (s, s - z95 * se, s + z95 * se)

    def score_grid(
        self,
        y: np.ndarray,
        gamma_candidates: list[float],
        k_grid: np.ndarray,
    ) -> dict[float, np.ndarray]:
        """
        Compute tail log-scores over a grid of k for each candidate gamma.

        Parameters
        ----------
        y : array-like
            Observations.
        gamma_candidates : list of float
            Candidate tail index values to evaluate.
        k_grid : array-like of int
            Values of k to evaluate at.

        Returns
        -------
        dict mapping gamma -> np.ndarray of scores (shape len(k_grid))
            NaN where k is out of valid range.
        """
        y = np.asarray(y, dtype=float)
        k_grid = np.asarray(k_grid, dtype=int)
        y_sorted = np.sort(y)

        result = {}
        for gamma in gamma_candidates:
            scores = np.array([
                _tail_log_score(y_sorted, gamma, int(k))
                for k in k_grid
            ])
            result[gamma] = scores
        return result

    def stability_plot(
        self,
        y: np.ndarray,
        gamma_candidates: list[float],
        k_grid: np.ndarray,
        ax=None,
    ):
        """
        Plot tail log-scores vs k for each candidate gamma.

        The stability diagnostic: look for a range of k where the relative
        ranking of candidates is consistent. In that range, the normalised
        order statistics have converged to their Pareto limit and the scores
        are reliable. Use that range to make model selection decisions.

        Parameters
        ----------
        y : array-like
            Observations.
        gamma_candidates : list of float
            Candidate tail indices.
        k_grid : array-like of int
            k values to evaluate at.
        ax : matplotlib.axes.Axes, optional
            If None, creates a new figure.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        scores = self.score_grid(y, gamma_candidates, k_grid)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.get_figure()

        cmap = plt.get_cmap("tab10")
        for i, gamma in enumerate(gamma_candidates):
            ax.plot(k_grid, scores[gamma], lw=1.5, color=cmap(i % 10),
                    label=f"γ={gamma}")

        ax.set_xlabel("k (number of upper order statistics)")
        ax.set_ylabel("Tail log-score (higher = better)")
        ax.set_title("Bladt tail score stability plot")
        ax.legend(fontsize=9)
        return fig

    def rank(
        self,
        y: np.ndarray,
        gamma_candidates: list[float],
        k_grid: np.ndarray,
        stable_range: Optional[tuple[int, int]] = None,
    ) -> pd.DataFrame:
        """
        Rank candidate tail indices by average tail log-score over stable k range.

        Parameters
        ----------
        y : array-like
            Observations.
        gamma_candidates : list of float
            Candidate tail indices.
        k_grid : array-like of int
            k values to evaluate at.
        stable_range : (k_min, k_max), optional
            Restrict averaging to k values in this range. If None, use all
            valid (finite) scores. Recommendation: choose this visually from
            the stability_plot, or use a heuristic like the lower 25% of k_grid.

        Returns
        -------
        pd.DataFrame
            Columns: gamma, mean_score, rank (1 = best)
            Sorted by mean_score descending (best first).
        """
        y = np.asarray(y, dtype=float)
        k_grid = np.asarray(k_grid, dtype=int)
        scores = self.score_grid(y, gamma_candidates, k_grid)

        rows = []
        for gamma in gamma_candidates:
            s = scores[gamma]
            if stable_range is not None:
                k_lo, k_hi = stable_range
                mask = (k_grid >= k_lo) & (k_grid <= k_hi)
                s_range = s[mask]
            else:
                s_range = s

            finite = s_range[np.isfinite(s_range)]
            mean_score = float(np.mean(finite)) if len(finite) > 0 else float("nan")
            rows.append({"gamma": gamma, "mean_score": mean_score})

        df = pd.DataFrame(rows).sort_values("mean_score", ascending=False)
        df = df.reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
        return df
