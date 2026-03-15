"""
DRN: Distributional Refinement Network (main class).

Orchestrates the full pipeline:
  1. Compute cutpoints from training data
  2. Compute baseline bin probabilities (frozen GLM/GBM)
  3. Train DRNNetwork to refine bin probabilities via JBCE loss
  4. Wrap predictions in ExtendedHistogramBatch for downstream use

References
----------
Avanzi, Dong, Laub, Wong (2024). "Distributional Refinement Network:
Distributional Forecasting via Deep Learning." arXiv:2406.00998.
"""

from __future__ import annotations

import copy
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.optim as optim
# DataLoader/TensorDataset removed — mini-batching done with torch.randperm

from insurance_severity.drn.baseline import BaselineDistribution
from insurance_severity.drn.cutpoints import drn_cutpoints
from insurance_severity.drn.histogram import ExtendedHistogramBatch
from insurance_severity.drn.loss import jbce_loss, drn_regularisation, nll_loss
from insurance_severity.drn.network import DRNNetwork


class DRN:
    """
    Distributional Refinement Network.

    Takes a frozen baseline distribution (GLM or GBM) and refines it into a
    full predictive distribution using a neural network. The key insight:
    the neural network outputs *adjustments* to the baseline, not the full
    distribution from scratch. This means the DRN inherits the GLM's
    actuarial calibration and only refines where the data supports it.

    Parameters
    ----------
    baseline : BaselineDistribution
        A fitted baseline model. Use ``GLMBaseline`` or ``CatBoostBaseline``.
    hidden_size : int
        Width of each hidden layer in the neural network. Default: 75.
    num_hidden_layers : int
        Number of hidden layers. Default: 2.
    dropout_rate : float
        Dropout probability during training. Default: 0.2.
    proportion : float
        Target fraction of observations per histogram bin (for cutpoint
        selection). Default: 0.1 (approximately 10 bins per dataset).
    min_obs : int
        Minimum observations per bin before merging. Default: 1.
    loss : str
        'jbce' (recommended) or 'nll'. JBCE is more stable for histograms.
    kl_alpha : float
        KL regularisation strength. 0 disables. Typical: 1e-4.
    mean_alpha : float
        Mean consistency regularisation. 0 disables. Typical: 1e-4.
    tv_alpha : float
        Total variation regularisation. 0 disables.
    dv_alpha : float
        Discrete variation (roughness) regularisation. Typical: 1e-3.
    kl_direction : str
        'forwards' = KL(baseline || DRN). 'reverse' = KL(DRN || baseline).
    lr : float
        Adam learning rate. Default: 1e-3.
    batch_size : int
        Training batch size. Default: 256.
    max_epochs : int
        Maximum training epochs. Default: 500.
    patience : int
        Early stopping patience (epochs without val improvement). Default: 30.
    baseline_start : bool
        If True, zero-initialise the output layer so DRN starts at baseline.
        Strongly recommended for training stability. Default: True.
    scr_aware : bool
        If True, set c_K above the 99.7th percentile so the Solvency II SCR
        (99.5th VaR) falls within the histogram region. Default: False.
    device : str
        'cpu', 'cuda', or 'mps'. Default: 'cpu'.
    random_state : int | None
        Seed for reproducibility. Default: None.

    Examples
    --------
    >>> from insurance_drn import GLMBaseline, DRN
    >>> import statsmodels.formula.api as smf
    >>> import statsmodels.api as sm
    >>> glm = smf.glm(
    ...     "claims ~ age + vehicle_age",
    ...     data=df_train,
    ...     family=sm.families.Gamma(sm.families.links.Log())
    ... ).fit()
    >>> baseline = GLMBaseline(glm)
    >>> drn = DRN(baseline, hidden_size=64, max_epochs=200, patience=20)
    >>> drn.fit(X_train, y_train)
    >>> dist = drn.predict_distribution(X_test)
    >>> print(dist.mean())           # (n,) array of expected claims
    >>> print(dist.quantile(0.995))  # 99.5th percentile for SCR
    """

    def __init__(
        self,
        baseline: BaselineDistribution,
        hidden_size: int = 75,
        num_hidden_layers: int = 2,
        dropout_rate: float = 0.2,
        proportion: float = 0.1,
        min_obs: int = 1,
        loss: str = "jbce",
        kl_alpha: float = 0.0,
        mean_alpha: float = 0.0,
        tv_alpha: float = 0.0,
        dv_alpha: float = 0.0,
        kl_direction: str = "forwards",
        lr: float = 1e-3,
        batch_size: int = 256,
        max_epochs: int = 500,
        patience: int = 30,
        baseline_start: bool = True,
        scr_aware: bool = False,
        device: str = "cpu",
        random_state: int | None = None,
    ):
        self.baseline = baseline
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        self.proportion = proportion
        self.min_obs = min_obs
        self.loss = loss
        self.kl_alpha = kl_alpha
        self.mean_alpha = mean_alpha
        self.tv_alpha = tv_alpha
        self.dv_alpha = dv_alpha
        self.kl_direction = kl_direction
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.baseline_start = baseline_start
        self.scr_aware = scr_aware
        self.device = torch.device(device)
        self.random_state = random_state

        # Set after fit()
        self._cutpoints: np.ndarray | None = None
        self._network: DRNNetwork | None = None
        self._feature_names: list[str] | None = None
        self._n_features: int | None = None
        self._train_history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        exposure: np.ndarray | None = None,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        exposure_val: np.ndarray | None = None,
        val_fraction: float = 0.2,
        verbose: bool = True,
    ) -> "DRN":
        """
        Fit the DRN to training data.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray, shape (n, p)
            Feature matrix. If DataFrame, column names are stored.
        y : np.ndarray, shape (n,)
            Observed response (must be positive — use positive claims only).
        exposure : np.ndarray, shape (n,), optional
            Observation-level weights (e.g. earned premium, policy years).
            Used as sample weights in JBCE loss.
        X_val : pd.DataFrame or np.ndarray, optional
            Validation features. If None, split from training data.
        y_val : np.ndarray, optional
            Validation response.
        exposure_val : np.ndarray, optional
            Validation exposure.
        val_fraction : float
            Fraction of training data to hold out for validation (used only
            if X_val is not provided). Default: 0.2.
        verbose : bool
            Print training progress. Default: True.

        Returns
        -------
        self
        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        X_df, y_arr = self._validate_inputs(X, y)
        y_arr = np.asarray(y_arr, dtype=np.float64)

        # Train/val split if no explicit val set
        if X_val is None:
            X_df, X_val_df, y_arr, y_val_arr, exp_arr, exp_val_arr = self._train_val_split(
                X_df, y_arr, exposure, val_fraction
            )
        else:
            X_val_df, y_val_arr = self._validate_inputs(X_val, y_val)
            exp_arr = exposure
            exp_val_arr = exposure_val

        # Step 1: compute cutpoints from training y
        self._cutpoints = drn_cutpoints(
            y_arr,
            proportion=self.proportion,
            min_obs=self.min_obs,
            scr_aware=self.scr_aware,
        )
        K = len(self._cutpoints) - 1  # number of bins

        # Step 2: get baseline bin probabilities
        # predict_cdf returns (n, K+1) — CDF at each cutpoint
        baseline_cdf_train = self.baseline.predict_cdf(X_df, self._cutpoints)  # (n, K+1)
        baseline_cdf_val = self.baseline.predict_cdf(X_val_df, self._cutpoints)

        # Bin probs = diff of CDF at adjacent cutpoints
        baseline_probs_train = np.diff(baseline_cdf_train, axis=1)  # (n, K)
        baseline_probs_val = np.diff(baseline_cdf_val, axis=1)

        # Clip for numerical stability
        baseline_probs_train = np.clip(baseline_probs_train, 1e-10, 1.0)
        baseline_probs_val = np.clip(baseline_probs_val, 1e-10, 1.0)

        # Step 3: prepare features
        X_np = X_df.values if isinstance(X_df, pd.DataFrame) else X_df
        X_val_np = X_val_df.values if isinstance(X_val_df, pd.DataFrame) else X_val_df
        self._n_features = X_np.shape[1]

        if isinstance(X_df, pd.DataFrame):
            self._feature_names = list(X_df.columns)

        # Step 4: build and initialise network
        self._network = DRNNetwork(
            n_features=self._n_features,
            n_bins=K,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            dropout_rate=self.dropout_rate,
        ).to(self.device)

        if self.baseline_start:
            self._network.reset_to_baseline()

        # Step 5: prepare training tensors
        X_t = torch.tensor(X_np, dtype=torch.float32, device=self.device)
        bp_t = torch.tensor(baseline_probs_train, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y_arr, dtype=torch.float32, device=self.device)
        exp_t = (
            torch.tensor(exp_arr, dtype=torch.float32, device=self.device)
            if exp_arr is not None
            else None
        )

        X_val_t = torch.tensor(X_val_np, dtype=torch.float32, device=self.device)
        bp_val_t = torch.tensor(baseline_probs_val, dtype=torch.float32, device=self.device)
        y_val_t = torch.tensor(y_val_arr, dtype=torch.float32, device=self.device)

        # Pre-compute y_indicators (binary: is y_i <= c_k for each cutpoint k)
        # Interior cutpoints only (exclude c_0 which all y exceed, and c_K)
        interior_cuts = self._cutpoints[1:-1]  # shape (K-1,) — for JBCE we use K interior bins
        # Actually JBCE uses all K+1 cutpoints but we evaluate CDF *between* them
        # Following the paper: for K bins, we evaluate CDF at K-1 interior cutpoints
        # Let's use all interior cutpoints c_1, ..., c_{K-1}
        # y_indicators[i, k] = 1 if y_i <= c_{k+1} (the right edge of bin k)
        # Using cumulative bin probs = CDF at right edge of each bin

        # Precompute indicators at interior cutpoints (c_1 to c_{K-1})
        y_ind_train = self._compute_y_indicators(y_arr, self._cutpoints)  # (n, K-1)
        y_ind_val = self._compute_y_indicators(y_val_arr, self._cutpoints)

        y_ind_t = torch.tensor(y_ind_train, dtype=torch.float32, device=self.device)
        y_ind_val_t = torch.tensor(y_ind_val, dtype=torch.float32, device=self.device)

        # Bin midpoints for mean regularisation
        bin_midpoints = torch.tensor(
            0.5 * (self._cutpoints[:-1] + self._cutpoints[1:]),
            dtype=torch.float32,
            device=self.device,
        )

        # Bin widths for NLL loss
        bin_widths_t = torch.tensor(
            np.diff(self._cutpoints),
            dtype=torch.float32,
            device=self.device,
        )

        # Step 6: train
        optimizer = optim.Adam(self._network.parameters(), lr=self.lr)
        best_val_loss = float("inf")
        best_weights = copy.deepcopy(self._network.state_dict())
        epochs_no_improve = 0

        for epoch in range(self.max_epochs):
            self._network.train()

            # Mini-batch training
            indices = torch.randperm(len(X_t), device=self.device)
            train_loss_epoch = 0.0
            n_batches = 0

            for start in range(0, len(X_t), self.batch_size):
                idx = indices[start: start + self.batch_size]
                x_b = X_t[idx]
                bp_b = bp_t[idx]
                yi_b = y_ind_t[idx]
                w_b = exp_t[idx] if exp_t is not None else None

                # Forward pass
                log_adj = self._network(x_b)          # (batch, K)
                drn_logits = torch.log(bp_b) + log_adj
                drn_pmf = torch.softmax(drn_logits, dim=1)

                # CDF at interior cutpoints: cumsum of pmf, excluding the last bin
                # CDF at c_k = sum_{j=0}^{k} p_j, for k = 0..K-2 (interior cuts c_1..c_{K-1})
                cdf_pred = torch.cumsum(drn_pmf, dim=1)[:, :-1]  # (batch, K-1)

                # Primary loss
                if self.loss == "jbce":
                    loss_val = jbce_loss(cdf_pred, yi_b, weights=w_b)
                else:
                    # NLL: need bin indices
                    y_b = y_t[idx]
                    bin_idx_b = self._get_bin_indices(y_b, bin_widths_t)
                    loss_val = nll_loss(drn_pmf, bin_idx_b, bin_widths_t, weights=w_b)

                # Regularisation
                if any([self.kl_alpha, self.mean_alpha, self.tv_alpha, self.dv_alpha]):
                    reg = drn_regularisation(
                        drn_pmf, bp_b,
                        kl_alpha=self.kl_alpha,
                        mean_alpha=self.mean_alpha,
                        tv_alpha=self.tv_alpha,
                        dv_alpha=self.dv_alpha,
                        kl_direction=self.kl_direction,
                        bin_midpoints=bin_midpoints,
                    )
                    loss_val = loss_val + reg

                optimizer.zero_grad()
                loss_val.backward()
                torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss_epoch += float(loss_val.item())
                n_batches += 1

            train_loss_epoch /= max(n_batches, 1)

            # Validation loss
            self._network.eval()
            with torch.no_grad():
                log_adj_val = self._network(X_val_t)
                drn_logits_val = torch.log(bp_val_t) + log_adj_val
                drn_pmf_val = torch.softmax(drn_logits_val, dim=1)
                cdf_pred_val = torch.cumsum(drn_pmf_val, dim=1)[:, :-1]
                if self.loss == "nll":
                    bin_idx_val = self._get_bin_indices_from_cutpoints(
                        y_val_t, self._cutpoints
                    )
                    val_loss = float(nll_loss(drn_pmf_val, bin_idx_val, bin_widths_t).item())
                else:
                    val_loss = float(jbce_loss(cdf_pred_val, y_ind_val_t).item())

            self._train_history["train_loss"].append(train_loss_epoch)
            self._train_history["val_loss"].append(val_loss)

            if val_loss < best_val_loss - 1e-8:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(self._network.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if verbose and (epoch % 50 == 0 or epoch < 10):
                print(
                    f"Epoch {epoch:4d} | train_loss={train_loss_epoch:.6f} "
                    f"| val_loss={val_loss:.6f}"
                    + (" *" if epochs_no_improve == 0 else "")
                )

            if epochs_no_improve >= self.patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (patience={self.patience})")
                break

        # Restore best weights
        self._network.load_state_dict(best_weights)
        self._network.eval()
        self._is_fitted = True

        if verbose:
            print(f"Training complete. Best val_loss={best_val_loss:.6f}")

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_distribution(
        self,
        X: pd.DataFrame | np.ndarray,
        exposure: np.ndarray | None = None,
    ) -> ExtendedHistogramBatch:
        """
        Full predictive distribution for each observation.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray, shape (n, p)
        exposure : np.ndarray, shape (n,), optional
            Not used in distribution prediction (exposure is a training
            concept). Included for API consistency.

        Returns
        -------
        ExtendedHistogramBatch
            Vectorised batch of extended histogram distributions.
        """
        self._check_fitted()
        X_df, _ = self._validate_inputs(X, None)
        X_np = X_df.values if isinstance(X_df, pd.DataFrame) else X_df

        n = X_np.shape[0]

        # Baseline bin probabilities
        baseline_cdf = self.baseline.predict_cdf(X_df, self._cutpoints)  # (n, K+1)
        baseline_probs = np.diff(baseline_cdf, axis=1)                    # (n, K)
        baseline_probs = np.clip(baseline_probs, 1e-10, 1.0)

        # DRN refinement
        X_t = torch.tensor(X_np, dtype=torch.float32, device=self.device)
        self._network.eval()
        with torch.no_grad():
            log_adj = self._network(X_t)                            # (n, K)
            drn_logits = torch.log(torch.tensor(baseline_probs, dtype=torch.float32, device=self.device)) + log_adj
            drn_pmf = torch.softmax(drn_logits, dim=1)

        drn_pmf_np = drn_pmf.cpu().numpy()

        # Baseline params for tail computation
        baseline_params = self.baseline.predict_params(X_df)

        return ExtendedHistogramBatch(
            cutpoints=self._cutpoints,
            bin_probs=drn_pmf_np,
            baseline_cdf_c0=baseline_cdf[:, 0],
            baseline_cdf_cK=baseline_cdf[:, -1],
            baseline_params=baseline_params,
            distribution_family=self.baseline.distribution_family,
        )

    def predict_mean(
        self,
        X: pd.DataFrame | np.ndarray,
        exposure: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Expected value for each observation. Shape (n,).

        Uses the histogram midpoint approximation for the interior region
        plus analytical formulas for the tails.
        """
        return self.predict_distribution(X, exposure).mean()

    def predict_quantile(
        self,
        X: pd.DataFrame | np.ndarray,
        quantiles: float | list[float] | np.ndarray,
        exposure: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Quantile predictions.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray, shape (n, p)
        quantiles : float or list of floats
            Quantile level(s) in (0, 1).
        exposure : np.ndarray, optional

        Returns
        -------
        np.ndarray
            Shape (n,) for scalar quantiles, (n, len(quantiles)) for array.
        """
        scalar = np.isscalar(quantiles)
        q_arr = np.atleast_1d(np.asarray(quantiles, dtype=np.float64))
        dist = self.predict_distribution(X, exposure)
        result = dist.quantile(q_arr)
        if scalar:
            return result[:, 0]
        return result

    def predict_var(
        self,
        X: pd.DataFrame | np.ndarray,
        exposure: np.ndarray | None = None,
    ) -> np.ndarray:
        """Variance for each observation. Shape (n,)."""
        return self.predict_distribution(X, exposure).var()

    def predict_cdf(
        self,
        X: pd.DataFrame | np.ndarray,
        y_grid: np.ndarray,
        exposure: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        CDF evaluated at each point in y_grid for all observations.

        Returns
        -------
        np.ndarray, shape (n, len(y_grid))
        """
        dist = self.predict_distribution(X, exposure)
        return dist.cdf(y_grid)

    def score(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        metric: str = "crps",
    ) -> float:
        """
        Evaluate the DRN on test data.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
        y : np.ndarray
        metric : str
            'crps' | 'nll' | 'rmse' | 'ql095' | 'ql099'

        Returns
        -------
        float
        """
        dist = self.predict_distribution(X)
        y_arr = np.asarray(y, dtype=np.float64)

        if metric == "crps":
            return float(np.mean(dist.crps(y_arr)))
        elif metric == "rmse":
            return float(np.sqrt(np.mean((dist.mean() - y_arr) ** 2)))
        elif metric == "nll":
            # Evaluate NLL: -log(density at y)
            # Use histogram PDF: p_k / bin_width for the bin containing y
            return self._score_nll(dist, y_arr)
        elif metric.startswith("ql"):
            alpha = float(metric[2:]) / 100.0
            q = dist.quantile(alpha)
            return float(np.mean(np.where(y_arr >= q, alpha * (y_arr - q), (1 - alpha) * (q - y_arr))))
        else:
            raise ValueError(f"Unknown metric {metric!r}. Choose: 'crps', 'nll', 'rmse', 'ql<xx>'.")

    # ------------------------------------------------------------------
    # Interpretability
    # ------------------------------------------------------------------

    def adjustment_factors(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> pl.DataFrame:
        """
        Per-bin adjustment factors a_k = p_k^DRN / b_k for each observation.

        Values > 1 mean the DRN assigns more probability to bin k than the
        GLM baseline. This is the primary interpretability output of the DRN.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray, shape (n, p)

        Returns
        -------
        pl.DataFrame, shape (n, K)
            Columns named by bin midpoint (e.g. 'adj_250.0', 'adj_1000.5').
        """
        self._check_fitted()
        X_df, _ = self._validate_inputs(X, None)
        X_np = X_df.values if isinstance(X_df, pd.DataFrame) else X_df

        # Baseline bin probabilities
        baseline_cdf = self.baseline.predict_cdf(X_df, self._cutpoints)
        baseline_probs = np.diff(baseline_cdf, axis=1)
        baseline_probs = np.clip(baseline_probs, 1e-10, 1.0)

        # DRN bin probabilities
        X_t = torch.tensor(X_np, dtype=torch.float32, device=self.device)
        self._network.eval()
        with torch.no_grad():
            log_adj = self._network(X_t)
            drn_logits = (
                torch.log(torch.tensor(baseline_probs, dtype=torch.float32, device=self.device))
                + log_adj
            )
            drn_pmf = torch.softmax(drn_logits, dim=1).cpu().numpy()

        # Adjustment factors
        a_k = drn_pmf / baseline_probs  # (n, K)

        # Column names: midpoint of each bin
        midpoints = 0.5 * (self._cutpoints[:-1] + self._cutpoints[1:])
        col_names = [f"adj_{m:.1f}" for m in midpoints]

        return pl.DataFrame({name: a_k[:, k] for k, name in enumerate(col_names)})

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Save the fitted DRN to disk.

        Saves network weights, cutpoints, and hyperparameters.
        The baseline model is NOT saved — you must provide it on load.

        Parameters
        ----------
        path : str or Path
            File path (e.g. 'models/drn_motor_severity.pt').
        """
        self._check_fitted()
        state = {
            "cutpoints": self._cutpoints,
            "network_state_dict": self._network.state_dict(),
            "network_config": {
                "n_features": self._n_features,
                "n_bins": len(self._cutpoints) - 1,
                "hidden_size": self.hidden_size,
                "num_hidden_layers": self.num_hidden_layers,
                "dropout_rate": self.dropout_rate,
            },
            "feature_names": self._feature_names,
            "train_history": self._train_history,
            "hyperparams": {
                "proportion": self.proportion,
                "min_obs": self.min_obs,
                "loss": self.loss,
                "kl_alpha": self.kl_alpha,
                "mean_alpha": self.mean_alpha,
                "tv_alpha": self.tv_alpha,
                "dv_alpha": self.dv_alpha,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "max_epochs": self.max_epochs,
                "patience": self.patience,
            },
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str | Path, baseline: BaselineDistribution) -> "DRN":
        """
        Load a saved DRN from disk.

        Parameters
        ----------
        path : str or Path
        baseline : BaselineDistribution
            The baseline model used during training. Must be fitted.

        Returns
        -------
        DRN
        """
        state = torch.load(path, map_location="cpu", weights_only=False)
        config = state["network_config"]
        hp = state["hyperparams"]

        drn = cls(
            baseline=baseline,
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_hidden_layers"],
            dropout_rate=config["dropout_rate"],
            **{k: hp[k] for k in hp},
        )
        drn._cutpoints = state["cutpoints"]
        drn._feature_names = state.get("feature_names")
        drn._n_features = config["n_features"]
        drn._train_history = state.get("train_history", {})

        drn._network = DRNNetwork(
            n_features=config["n_features"],
            n_bins=config["n_bins"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_hidden_layers"],
            dropout_rate=config["dropout_rate"],
        )
        drn._network.load_state_dict(state["network_state_dict"])
        drn._network.eval()
        drn._is_fitted = True
        return drn

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "DRN is not fitted. Call fit() before making predictions."
            )

    def _validate_inputs(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | None,
    ) -> tuple[pd.DataFrame, np.ndarray | None]:
        """Convert X to DataFrame, y to ndarray."""
        if isinstance(X, np.ndarray):
            cols = self._feature_names or [f"x{i}" for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=cols)
        elif isinstance(X, pd.DataFrame):
            X_df = X.reset_index(drop=True)
        else:
            raise TypeError(f"X must be pd.DataFrame or np.ndarray, got {type(X)}")

        y_arr = np.asarray(y, dtype=np.float64) if y is not None else None
        return X_df, y_arr

    def _train_val_split(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        exposure: np.ndarray | None,
        val_fraction: float,
    ) -> tuple:
        n = len(y)
        n_val = max(1, int(n * val_fraction))
        n_train = n - n_val

        if self.random_state is not None:
            rng = np.random.default_rng(self.random_state)
        else:
            rng = np.random.default_rng()

        idx = rng.permutation(n)
        train_idx, val_idx = idx[:n_train], idx[n_train:]

        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_val = X.iloc[val_idx].reset_index(drop=True)
        y_train = y[train_idx]
        y_val = y[val_idx]
        exp_train = exposure[train_idx] if exposure is not None else None
        exp_val = exposure[val_idx] if exposure is not None else None

        return X_train, X_val, y_train, y_val, exp_train, exp_val

    @staticmethod
    def _compute_y_indicators(y: np.ndarray, cutpoints: np.ndarray) -> np.ndarray:
        """
        Binary indicators: y_indicators[i, k] = 1 if y[i] <= cutpoints[k+1].

        For JBCE loss we evaluate at interior cutpoints c_1, ..., c_{K-1}.
        Returns (n, K-1) binary matrix.
        """
        # Interior cutpoints: exclude c_0 (all y > c_0 by construction) and c_K
        # Actually we evaluate at cumulative bins c_1 through c_{K-1}
        interior = cutpoints[1:-1]  # (K-1,)
        # y[:, None] <= interior[None, :] gives (n, K-1) bool
        return (y[:, np.newaxis] <= interior[np.newaxis, :]).astype(np.float32)

    def _get_bin_indices(self, y: torch.Tensor, bin_widths: torch.Tensor) -> torch.Tensor:
        """Get bin index for each observation (for NLL loss).

        Uses self._cutpoints interior boundaries [c_1, ..., c_{K-1}].
        torch.bucketize returns 0 for y <= c_1, k for c_k < y <= c_{k+1}, K-1 for y > c_{K-1}.
        """
        boundaries = torch.tensor(
            self._cutpoints[1:-1], dtype=torch.float32, device=y.device
        )
        idx = torch.bucketize(y, boundaries)  # shape (n,), values in [0, K-1]
        return idx.clamp(0, len(boundaries))

    @staticmethod
    def _get_bin_indices_from_cutpoints(
        y: torch.Tensor, cutpoints: np.ndarray
    ) -> torch.Tensor:
        """Static variant that takes cutpoints explicitly (used in validation)."""
        boundaries = torch.tensor(
            cutpoints[1:-1], dtype=torch.float32, device=y.device
        )
        idx = torch.bucketize(y, boundaries)
        return idx.clamp(0, len(boundaries))

    def _score_nll(self, dist: ExtendedHistogramBatch, y: np.ndarray) -> float:
        """NLL using histogram density."""
        bin_idx = np.searchsorted(self._cutpoints, y, side="right") - 1
        bin_idx = np.clip(bin_idx, 0, dist.K - 1)
        bin_widths = np.diff(self._cutpoints)
        p_k = dist.bin_probs[np.arange(len(y)), bin_idx]
        w_k = bin_widths[bin_idx]
        eps = 1e-10
        return float(-np.mean(np.log(p_k / w_k + eps)))

    @property
    def cutpoints(self) -> np.ndarray | None:
        """Histogram cutpoints. None before fitting."""
        return self._cutpoints

    @property
    def n_bins(self) -> int | None:
        """Number of histogram bins K. None before fitting."""
        if self._cutpoints is not None:
            return len(self._cutpoints) - 1
        return None

    @property
    def training_history(self) -> dict[str, list[float]]:
        """Train/val loss history per epoch."""
        return self._train_history

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"DRN({status}, "
            f"hidden_size={self.hidden_size}, "
            f"num_hidden_layers={self.num_hidden_layers}, "
            f"n_bins={self.n_bins}, "
            f"family={self.baseline.distribution_family!r})"
        )
