"""
DRNDiagnostics: calibration diagnostics for fitted DRN models.

Three standard actuarial validation checks:
1. PIT histogram — tests distributional calibration (should be Uniform(0,1))
2. Quantile calibration — observed vs nominal coverage at multiple alpha levels
3. CRPS by segment — identifies where the DRN performs better/worse than baseline

These diagnostics are standard in the actuarial literature on distributional
forecasting (Gneiting & Raftery 2007, Wüthrich 2023). All return Polars
DataFrames for easy downstream use; plot methods return matplotlib Figures.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl

if TYPE_CHECKING:
    from insurance_severity.drn.drn import DRN


class DRNDiagnostics:
    """
    Distributional calibration diagnostics for a fitted DRN.

    All methods accept a fitted DRN, feature matrix X, and actuals y.
    The DRN does not need to be refitted — diagnostics use predict_distribution.

    Parameters
    ----------
    drn : DRN
        A fitted DRN model.
    """

    def __init__(self, drn: "DRN"):
        self.drn = drn

    # ------------------------------------------------------------------
    # PIT histogram
    # ------------------------------------------------------------------

    def pit_values(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Probability Integral Transform (PIT) values.

        PIT_i = F_DRN(y_i | x_i) — the model CDF evaluated at the actual.
        For a well-calibrated model, PIT values should be Uniform(0, 1).

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
        y : np.ndarray, shape (n,)

        Returns
        -------
        np.ndarray, shape (n,)
            PIT values in [0, 1].
        """
        dist = self.drn.predict_distribution(X)
        y_arr = np.asarray(y, dtype=np.float64)
        # Evaluate CDF at each y_i
        pit = np.array([dist.cdf(float(y_arr[i]))[i] for i in range(len(y_arr))])
        return pit

    def pit_histogram(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        n_bins: int = 10,
        return_figure: bool = True,
    ):
        """
        PIT uniformity plot.

        A well-calibrated model produces a flat histogram (all bars near 1.0
        on the relative frequency axis). Systematic deviations indicate:
        - U-shaped: overdispersion (DRN too wide)
        - Hump-shaped: underdispersion (DRN too narrow)
        - Left-heavy: DRN overpredicts (actuals tend to be lower)
        - Right-heavy: DRN underpredicts

        Parameters
        ----------
        X, y : features and actuals
        n_bins : int
            Number of PIT histogram bins. Default: 10.
        return_figure : bool
            If True, return matplotlib Figure. If False, return PIT values.

        Returns
        -------
        matplotlib.Figure or np.ndarray
        """
        pit = self.pit_values(X, y)

        if not return_figure:
            return pit

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not installed. Returning PIT values instead.")
            return pit

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(pit, bins=n_bins, range=(0, 1), density=True, color="steelblue", alpha=0.8,
                edgecolor="white", linewidth=0.5)
        ax.axhline(1.0, color="crimson", linestyle="--", linewidth=1.5, label="Uniform(0,1)")
        ax.set_xlabel("PIT value")
        ax.set_ylabel("Density")
        ax.set_title("PIT Histogram (distributional calibration)")
        ax.legend()
        ax.set_xlim(0, 1)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Quantile calibration
    # ------------------------------------------------------------------

    def quantile_calibration(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        alpha_levels: list[float] | None = None,
    ) -> pl.DataFrame:
        """
        Observed vs nominal coverage at multiple quantile levels.

        For a well-calibrated model, the fraction of actuals below the alpha-th
        predicted quantile should be approximately alpha.

        Parameters
        ----------
        X, y : features and actuals
        alpha_levels : list of float, optional
            Quantile levels to evaluate. Defaults to a standard grid from
            0.01 to 0.99.

        Returns
        -------
        pl.DataFrame with columns:
            - nominal_coverage : the requested alpha level
            - observed_coverage : fraction of actuals below predicted quantile
            - error : observed - nominal (positive = DRN underpredicts)

        Notes
        -----
        Plot with::

            import matplotlib.pyplot as plt
            df = diag.quantile_calibration(X, y)
            plt.plot(df['nominal_coverage'], df['observed_coverage'])
            plt.plot([0,1],[0,1], 'k--')
        """
        if alpha_levels is None:
            alpha_levels = [
                0.01, 0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50,
                0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99,
            ]

        dist = self.drn.predict_distribution(X)
        y_arr = np.asarray(y, dtype=np.float64)
        n = len(y_arr)

        nominals, observed, errors = [], [], []
        for alpha in alpha_levels:
            q = dist.quantile(alpha)  # (n,)
            cov = float(np.mean(y_arr <= q))
            nominals.append(alpha)
            observed.append(cov)
            errors.append(cov - alpha)

        return pl.DataFrame({
            "nominal_coverage": nominals,
            "observed_coverage": observed,
            "error": errors,
        })

    def quantile_calibration_plot(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        alpha_levels: list[float] | None = None,
    ):
        """
        Plot observed vs nominal coverage (reliability diagram).

        Returns matplotlib Figure.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plots. pip install insurance-drn[plots]")

        df = self.quantile_calibration(X, y, alpha_levels)
        nom = df["nominal_coverage"].to_numpy()
        obs = df["observed_coverage"].to_numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Reliability diagram
        ax = axes[0]
        ax.plot(nom, obs, "o-", color="steelblue", label="DRN")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
        ax.set_xlabel("Nominal coverage")
        ax.set_ylabel("Observed coverage")
        ax.set_title("Quantile Reliability Diagram")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Error plot
        ax2 = axes[1]
        ax2.bar(nom, df["error"].to_numpy(), width=0.03, color="steelblue", alpha=0.8)
        ax2.axhline(0, color="k", linewidth=1)
        ax2.set_xlabel("Nominal coverage")
        ax2.set_ylabel("Observed - Nominal")
        ax2.set_title("Coverage Error")

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # CRPS by segment
    # ------------------------------------------------------------------

    def crps_by_segment(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        segment_col: str | np.ndarray,
    ) -> pl.DataFrame:
        """
        CRPS broken down by segment (e.g. vehicle age band, region).

        Useful for identifying where the DRN adds the most distributional
        value compared to the baseline GLM.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
        y : np.ndarray
        segment_col : str or np.ndarray
            If str: column name in X (X must be DataFrame).
            If np.ndarray: segment labels of shape (n,).

        Returns
        -------
        pl.DataFrame with columns:
            - segment : segment label
            - n : number of observations
            - mean_crps : mean CRPS for this segment
            - mean_y : average actual for reference
        """
        dist = self.drn.predict_distribution(X)
        y_arr = np.asarray(y, dtype=np.float64)
        crps_vals = dist.crps(y_arr)

        if isinstance(segment_col, str):
            if not isinstance(X, pd.DataFrame):
                raise ValueError(
                    "segment_col as string requires X to be a pd.DataFrame"
                )
            segments = np.asarray(X[segment_col])
        else:
            segments = np.asarray(segment_col)

        unique_segs = np.unique(segments)
        rows = []
        for seg in unique_segs:
            mask = segments == seg
            rows.append({
                "segment": str(seg),
                "n": int(np.sum(mask)),
                "mean_crps": float(np.mean(crps_vals[mask])),
                "mean_y": float(np.mean(y_arr[mask])),
            })

        return pl.DataFrame(rows).sort("mean_crps", descending=True)

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------

    def summary(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
    ) -> pl.DataFrame:
        """
        One-row summary of model performance metrics.

        Returns
        -------
        pl.DataFrame
            Columns: crps, rmse, nll, coverage_90, coverage_99
        """
        dist = self.drn.predict_distribution(X)
        y_arr = np.asarray(y, dtype=np.float64)

        crps_mean = float(np.mean(dist.crps(y_arr)))
        rmse = float(np.sqrt(np.mean((dist.mean() - y_arr) ** 2)))
        nll = self.drn._score_nll(dist, y_arr)

        q90 = dist.quantile(0.90)
        q99 = dist.quantile(0.99)
        cov90 = float(np.mean(y_arr <= q90))
        cov99 = float(np.mean(y_arr <= q99))

        return pl.DataFrame({
            "crps": [crps_mean],
            "rmse": [rmse],
            "nll": [nll],
            "coverage_90": [cov90],
            "coverage_99": [cov99],
        })
