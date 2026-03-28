"""
MDN: Mixture Density Network for insurance severity.

Main estimator class following the sklearn-like fit/predict_mean/predict_distribution
pattern established by DRN. Models the conditional severity distribution as a
K-component lognormal mixture — Gaussian components on log(y).

References
----------
Bishop, C.M. (1994). 'Mixture Density Networks.'
    Technical Report NCRG/94/004, Aston University.
Delong, L., Lindholm, M., Wüthrich, M.V. (2021). 'Gamma Mixture Density
    Networks and their application to modelling insurance claim amounts.'
    Insurance: Mathematics and Economics.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from insurance_severity.mdn.distribution import MDNMixture
from insurance_severity.mdn.loss import mdn_nll_loss, mixture_mean
from insurance_severity.mdn.network import MDNNetwork


class MDN:
    """
    Mixture Density Network for insurance severity distributions.

    Models the conditional severity distribution as a K-component lognormal
    mixture: Gaussian components in log-space. The network outputs per-
    observation mixing weights π_k(x), log-space component means μ_k(x), and
    log-space component standard deviations σ_k(x).

    The lognormal mixture is a natural choice for insurance severity:

    - Positive support — no probability mass on negative claims.
    - Log-space Gaussian components are numerically well-behaved.
    - Multimodal distributions (e.g. EoW strip-out vs reinstatement peaks) are
      naturally captured with K=3.
    - Familiarity: actuaries already think in log-space for severity.

    Training uses the negative log-likelihood of the mixture with the
    log-sum-exp trick for numerical stability, gradient clipping (norm=1.0),
    and early stopping on held-out NLL.

    Parameters
    ----------
    n_components : int
        Number of Gaussian mixture components K. Default: 3.
        Rule of thumb: 3 for bimodal insurance severity (e.g. EoW), 5 for
        unknown distribution shape.
    hidden_size : int
        Width of each hidden layer. Default: 64.
    num_hidden_layers : int
        Number of hidden layers. Default: 2.
    dropout_rate : float
        Dropout probability during training. 0 disables dropout. Default: 0.1.
    lr : float
        Adam learning rate. Default: 1e-3.
    batch_size : int
        Training batch size. Default: 256.
    max_epochs : int
        Maximum number of training epochs. Default: 500.
    patience : int
        Early stopping patience: training stops if validation NLL does not
        improve for this many consecutive epochs. Default: 30.
    sigma_floor : float
        Lower bound on component standard deviations to prevent numerical
        collapse. Default: 1e-4.
    grad_clip_norm : float
        Maximum gradient norm for clipping. Default: 1.0.
    val_fraction : float
        Fraction of training data used as validation set (when no explicit
        val set is provided). Default: 0.2.
    device : str
        PyTorch device string: 'cpu', 'cuda', or 'mps'. Default: 'cpu'.
    random_state : int | None
        Random seed for reproducibility. Default: None.

    Examples
    --------
    Fit to escape-of-water claims:

    >>> mdn = MDN(n_components=3, hidden_size=64, max_epochs=200)
    >>> mdn.fit(X_train, y_train)
    >>> dist = mdn.predict_distribution(X_test)
    >>> dist.mean()            # shape (n,), expected severity
    >>> dist.quantile(0.995)   # shape (n,), SCR-level quantile
    >>> dist.ilf(limit=50_000, basic_limit=10_000)

    Predict expected value only:

    >>> mdn.predict_mean(X_test)   # shape (n,)

    Assess calibration:

    >>> pit = dist.pit_samples(y_test)   # should be uniform on [0, 1]

    Notes
    -----
    **Log transformation**: ``y`` must be strictly positive. The MDN applies
    ``log(y)`` internally; do not pass log-transformed targets.

    **Scaling**: Feature inputs are not automatically standardised. For best
    training stability, standardise continuous features before fitting.

    **Component count**: Use held-out NLL to select K. More components reduce
    NLL on training data but risk overfitting on small samples. K=3 is the
    default and works well for standard insurance severity datasets.

    **Mode collapse**: If training stagnates at a high NLL, try reducing
    ``lr``, increasing ``batch_size``, or setting ``random_state`` to try a
    different random initialisation.
    """

    def __init__(
        self,
        n_components: int = 3,
        hidden_size: int = 64,
        num_hidden_layers: int = 2,
        dropout_rate: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 256,
        max_epochs: int = 500,
        patience: int = 30,
        sigma_floor: float = 1e-4,
        grad_clip_norm: float = 1.0,
        val_fraction: float = 0.2,
        device: str = "cpu",
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.sigma_floor = sigma_floor
        self.grad_clip_norm = grad_clip_norm
        self.val_fraction = val_fraction
        self.device = torch.device(device)
        self.random_state = random_state

        # Set after fit()
        self._network: MDNNetwork | None = None
        self._n_features: int | None = None
        self._feature_names: list[str] | None = None
        self._train_history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        verbose: bool = True,
    ) -> "MDN":
        """
        Fit the MDN to training data.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray, shape (n, p)
            Feature matrix. Continuous features should be standardised before
            fitting for best results.
        y : np.ndarray, shape (n,)
            Observed severities. Must be strictly positive.
        X_val : pd.DataFrame or np.ndarray, optional
            Validation features. If None, a random ``val_fraction`` of
            training data is held out.
        y_val : np.ndarray, optional
            Validation severities. Required if X_val is provided.
        verbose : bool
            Print training progress. Default: True.

        Returns
        -------
        self
        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        X_arr, y_arr = self._validate_inputs(X, y)

        # Train/val split
        if X_val is None:
            X_arr, X_val_arr, y_arr, y_val_arr = self._train_val_split(
                X_arr, y_arr, self.val_fraction
            )
        else:
            X_val_arr, _ = self._validate_inputs(X_val, y_val)
            y_val_arr = np.asarray(y_val, dtype=np.float64)

        self._n_features = X_arr.shape[1]

        # Build network
        self._network = MDNNetwork(
            n_features=self._n_features,
            n_components=self.n_components,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            dropout_rate=self.dropout_rate,
        ).to(self.device)

        # Initialise component means from data distribution
        log_y_init = torch.tensor(
            np.log(np.clip(y_arr, 1e-8, None)),
            dtype=torch.float32,
            device=self.device,
        )
        self._network.init_from_data(log_y_init)

        # Tensors
        X_t = torch.tensor(X_arr, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y_arr, dtype=torch.float32, device=self.device)
        X_val_t = torch.tensor(X_val_arr, dtype=torch.float32, device=self.device)
        y_val_t = torch.tensor(y_val_arr, dtype=torch.float32, device=self.device)

        optimizer = optim.Adam(self._network.parameters(), lr=self.lr)

        best_val_loss = float("inf")
        best_weights = copy.deepcopy(self._network.state_dict())
        epochs_no_improve = 0

        for epoch in range(self.max_epochs):
            self._network.train()

            # Shuffle
            perm = torch.randperm(len(X_t), device=self.device)
            train_loss_sum = 0.0
            n_batches = 0

            for start in range(0, len(X_t), self.batch_size):
                idx = perm[start: start + self.batch_size]
                x_b = X_t[idx]
                y_b = y_t[idx]

                pi, mu, sigma = self._network(x_b)
                loss = mdn_nll_loss(pi, mu, sigma, y_b, sigma_floor=self.sigma_floor)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._network.parameters(), max_norm=self.grad_clip_norm
                )
                optimizer.step()

                train_loss_sum += float(loss.item())
                n_batches += 1

            train_loss_epoch = train_loss_sum / max(n_batches, 1)

            # Validation loss
            self._network.eval()
            with torch.no_grad():
                pi_v, mu_v, sigma_v = self._network(X_val_t)
                val_loss = float(
                    mdn_nll_loss(pi_v, mu_v, sigma_v, y_val_t, sigma_floor=self.sigma_floor).item()
                )

            self._train_history["train_loss"].append(train_loss_epoch)
            self._train_history["val_loss"].append(val_loss)

            if val_loss < best_val_loss - 1e-8:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(self._network.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if verbose and (epoch % 50 == 0 or epoch < 5):
                print(
                    f"Epoch {epoch:4d} | train_nll={train_loss_epoch:.4f}"
                    f" | val_nll={val_loss:.4f}"
                    + (" *" if epochs_no_improve == 0 else "")
                )

            if epochs_no_improve >= self.patience:
                if verbose:
                    print(
                        f"Early stopping at epoch {epoch} (patience={self.patience})"
                    )
                break

        self._network.load_state_dict(best_weights)
        self._network.eval()
        self._is_fitted = True

        if verbose:
            print(f"Training complete. Best val_nll={best_val_loss:.4f}")

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_params(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Raw mixture parameters for each observation.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray, shape (n, p)

        Returns
        -------
        pi : np.ndarray, shape (n, K) — mixing weights (sum to 1).
        mu : np.ndarray, shape (n, K) — component means in log-space.
        sigma : np.ndarray, shape (n, K) — component standard deviations.
        """
        self._check_fitted()
        X_arr, _ = self._validate_inputs(X, None)
        X_t = torch.tensor(X_arr, dtype=torch.float32, device=self.device)
        self._network.eval()
        with torch.no_grad():
            pi, mu, sigma = self._network(X_t)
        return (
            pi.cpu().numpy(),
            mu.cpu().numpy(),
            sigma.cpu().numpy(),
        )

    def predict_distribution(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> MDNMixture:
        """
        Full predictive distribution for each observation.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray, shape (n, p)

        Returns
        -------
        MDNMixture
            Contains pi, mu, sigma arrays and methods:
            ``.mean()``, ``.quantile(q)``, ``.cdf(y_grid)``,
            ``.pdf(y_grid)``, ``.pit_samples(y)``, ``.ilf(L, b)``.
        """
        pi, mu, sigma = self.predict_params(X)
        return MDNMixture(pi=pi, mu=mu, sigma=sigma)

    def predict_mean(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """
        Expected severity for each observation.

        E[Y | x] = Σ_k π_k(x) · exp(μ_k(x) + σ_k(x)²/2)

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray, shape (n, p)

        Returns
        -------
        np.ndarray, shape (n,)
        """
        return self.predict_distribution(X).mean()

    def score(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        metric: str = "nll",
    ) -> float:
        """
        Evaluate the MDN on held-out data.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
        y : np.ndarray, shape (n,)
        metric : str
            'nll' — mean negative log-likelihood (primary metric).
            'rmse' — root mean squared error of the point prediction.
            'mae' — mean absolute error.
            'crps' — continuous ranked probability score (simulation-based,
                     2000 samples per observation).

        Returns
        -------
        float
        """
        y_arr = np.asarray(y, dtype=np.float64)
        dist = self.predict_distribution(X)

        if metric == "nll":
            return float(-dist.log_prob(y_arr).mean())
        elif metric == "rmse":
            return float(np.sqrt(np.mean((dist.mean() - y_arr) ** 2)))
        elif metric == "mae":
            return float(np.mean(np.abs(dist.mean() - y_arr)))
        elif metric == "crps":
            return float(self._crps_score(dist, y_arr))
        else:
            raise ValueError(
                f"Unknown metric {metric!r}. Choose: 'nll', 'rmse', 'mae', 'crps'."
            )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Save the fitted MDN to disk.

        Saves network weights and all hyperparameters. Load with
        ``MDN.load(path)``.

        Parameters
        ----------
        path : str or Path
            File path (e.g. ``'models/mdn_eow_severity.pt'``).
        """
        self._check_fitted()
        state = {
            "network_state_dict": self._network.state_dict(),
            "network_config": {
                "n_features": self._n_features,
                "n_components": self.n_components,
                "hidden_size": self.hidden_size,
                "num_hidden_layers": self.num_hidden_layers,
                "dropout_rate": self.dropout_rate,
            },
            "feature_names": self._feature_names,
            "train_history": self._train_history,
            "hyperparams": {
                "lr": self.lr,
                "batch_size": self.batch_size,
                "max_epochs": self.max_epochs,
                "patience": self.patience,
                "sigma_floor": self.sigma_floor,
                "grad_clip_norm": self.grad_clip_norm,
                "val_fraction": self.val_fraction,
                "random_state": self.random_state,
            },
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str | Path) -> "MDN":
        """
        Load a saved MDN from disk.

        Parameters
        ----------
        path : str or Path

        Returns
        -------
        MDN
        """
        state = torch.load(path, map_location="cpu", weights_only=False)
        config = state["network_config"]
        hp = state["hyperparams"]

        mdn = cls(
            n_components=config["n_components"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_hidden_layers"],
            dropout_rate=config["dropout_rate"],
            **{k: hp[k] for k in hp},
        )
        mdn._n_features = config["n_features"]
        mdn._feature_names = state.get("feature_names")
        mdn._train_history = state.get("train_history", {})

        mdn._network = MDNNetwork(
            n_features=config["n_features"],
            n_components=config["n_components"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_hidden_layers"],
            dropout_rate=config["dropout_rate"],
        )
        mdn._network.load_state_dict(state["network_state_dict"])
        mdn._network.eval()
        mdn._is_fitted = True
        return mdn

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def training_history(self) -> dict[str, list[float]]:
        """Train and validation NLL per epoch."""
        return self._train_history

    @property
    def feature_names(self) -> list[str] | None:
        """Feature names from training DataFrame. None if ndarray was passed."""
        return self._feature_names

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"MDN({status}, "
            f"n_components={self.n_components}, "
            f"hidden_size={self.hidden_size}, "
            f"num_hidden_layers={self.num_hidden_layers})"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "MDN is not fitted. Call fit() before making predictions."
            )

    def _validate_inputs(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Coerce X to float32 ndarray; store feature names if DataFrame."""
        if isinstance(X, pd.DataFrame):
            if not self._is_fitted:
                self._feature_names = list(X.columns)
            X_arr = X.values.astype(np.float32)
        elif isinstance(X, np.ndarray):
            X_arr = X.astype(np.float32)
        else:
            raise TypeError(f"X must be pd.DataFrame or np.ndarray, got {type(X)}")

        y_arr = np.asarray(y, dtype=np.float64) if y is not None else None

        if y_arr is not None and np.any(y_arr <= 0):
            raise ValueError(
                "All severities must be strictly positive (y > 0). "
                "Remove or impute zero/negative claims before fitting."
            )

        return X_arr, y_arr

    def _train_val_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_fraction: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(y)
        n_val = max(1, int(n * val_fraction))
        rng = np.random.default_rng(self.random_state)
        idx = rng.permutation(n)
        train_idx = idx[n_val:]
        val_idx = idx[:n_val]
        return X[train_idx], X[val_idx], y[train_idx], y[val_idx]

    @staticmethod
    def _crps_score(dist: MDNMixture, y: np.ndarray, n_samples: int = 2000) -> float:
        """
        Energy form of CRPS: CRPS(F, y) = E|Y - y| - 0.5 E|Y - Y'|.

        Estimated by sampling from the mixture. For n observations with
        n_samples draws each, this is O(n * n_samples) which is approximate
        but fast.

        Parameters
        ----------
        dist : MDNMixture
        y : np.ndarray, shape (n,)
        n_samples : int
            Monte Carlo samples per observation.

        Returns
        -------
        float — mean CRPS (lower is better).
        """
        n = dist.n
        rng = np.random.default_rng(0)

        # Sample component indices per observation: (n, n_samples)
        comp_idx = np.array([
            rng.choice(dist.K, size=n_samples, p=dist.pi[i])
            for i in range(n)
        ])  # (n, n_samples)

        # Sample from chosen components: lognormal(mu, sigma)
        mu_s = dist.mu[np.arange(n)[:, np.newaxis], comp_idx]    # (n, n_samples)
        sigma_s = dist.sigma[np.arange(n)[:, np.newaxis], comp_idx]
        eps = rng.standard_normal(mu_s.shape)
        samples = np.exp(mu_s + sigma_s * eps)  # (n, n_samples)

        # E|Y - y_obs|
        e1 = np.mean(np.abs(samples - y[:, np.newaxis]), axis=1)  # (n,)

        # E|Y - Y'| (draw two independent samples)
        eps2 = rng.standard_normal(mu_s.shape)
        samples2 = np.exp(mu_s + sigma_s * eps2)
        e2 = 0.5 * np.mean(np.abs(samples - samples2), axis=1)   # (n,)

        return float(np.mean(e1 - e2))
