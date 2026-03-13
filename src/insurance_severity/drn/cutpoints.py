"""
Cutpoint selection for the DRN histogram.

The DRN partitions the response space into K bins using K+1 cutpoints.
This module provides ``drn_cutpoints``, the standard selection algorithm.

Key design decision: cutpoints on the original (not log) scale. The histogram
bins are uniform in probability mass under the training distribution, which
means denser bins in the body of the distribution and sparser at the tails.
For Solvency II SCR applications, set c_K above the 99.7th percentile so that
the 99.5th VaR lies within the histogram region where DRN has full control.
"""

from __future__ import annotations

import numpy as np


def drn_cutpoints(
    y: np.ndarray,
    proportion: float = 0.1,
    min_obs: int = 1,
    c_0: float | None = None,
    c_K: float | None = None,
    scr_aware: bool = False,
) -> np.ndarray:
    """
    Compute DRN histogram cutpoints from training observations.

    The default algorithm places cutpoints at empirical quantiles, spaced so
    that approximately ``proportion * n`` observations fall between each
    adjacent pair. Bins with fewer than ``min_obs`` observations are merged.

    Parameters
    ----------
    y : np.ndarray, shape (n,)
        Observed response values (positive). Must not contain NaN.
    proportion : float
        Target fraction of observations per bin. 0.1 means approximately
        n/10 observations per bin (so ~10 bins for any dataset size).
        Smaller values give more bins and finer distributional resolution.
        Paper default: 0.1.
    min_obs : int
        Minimum observations required per bin. Bins with fewer observations
        are merged with their neighbour. Prevents degenerate empty bins.
    c_0 : float, optional
        Lower cutpoint. Defaults to 0 (claims are non-negative). Set
        slightly below zero only if your data contains zeros.
    c_K : float, optional
        Upper cutpoint. Defaults to 1.05 * max(y). For SCR applications,
        use ``scr_aware=True`` or set explicitly to the 99.7th percentile.
    scr_aware : bool
        If True, set c_K to 1.1 * 99.7th percentile of y. This ensures
        the Solvency II SCR (99.5th VaR) falls within the histogram region.
        Overrides c_K if both are given.

    Returns
    -------
    np.ndarray, shape (K+1,)
        Sorted cutpoints c_0 < c_1 < ... < c_K.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> y = rng.gamma(2.0, 1000.0, size=10_000)
    >>> cuts = drn_cutpoints(y, proportion=0.1)
    >>> len(cuts)  # 11 cutpoints = 10 bins
    11
    """
    y = np.asarray(y, dtype=np.float64)
    if np.any(y <= 0):
        raise ValueError(
            "drn_cutpoints requires positive y values. "
            "If your data contains zeros (zero-inflation), model the "
            "positive claims separately and use a frequency model for zeros."
        )
    if np.any(np.isnan(y)):
        raise ValueError("y contains NaN values.")

    n = len(y)

    # Determine lower and upper bounds
    if c_0 is None:
        c_0 = 0.0
    if scr_aware:
        c_K = float(np.percentile(y, 99.7)) * 1.1
    elif c_K is None:
        c_K = float(np.max(y)) * 1.05

    # Number of interior cutpoints based on proportion
    n_bins_target = max(2, int(round(1.0 / proportion)))
    # Quantile positions (excluding 0 and 1 — those become c_0 and c_K)
    q_positions = np.linspace(0.0, 1.0, n_bins_target + 1)

    # Clip to the range [c_0, c_K] — use only observations within range
    y_clipped = y[(y > c_0) & (y < c_K)]
    if len(y_clipped) == 0:
        # Fallback: linspace between c_0 and c_K
        return np.linspace(c_0, c_K, n_bins_target + 1)

    # Compute interior quantile cutpoints
    interior = np.quantile(y_clipped, q_positions[1:-1])

    # Build full cutpoint array
    cutpoints = np.concatenate([[c_0], interior, [c_K]])
    cutpoints = np.unique(cutpoints)  # remove duplicates from quantile collisions

    # Merge bins with fewer than min_obs observations
    if min_obs > 1:
        cutpoints = _merge_sparse_bins(y, cutpoints, min_obs)

    return cutpoints


def _merge_sparse_bins(y: np.ndarray, cutpoints: np.ndarray, min_obs: int) -> np.ndarray:
    """
    Iteratively merge adjacent bins until all bins have >= min_obs observations.

    Merging strategy: merge the bin with the fewest observations with its
    smaller neighbour. Repeat until all bins satisfy the constraint.
    Preserves c_0 and c_K (first and last cutpoints).
    """
    cuts = list(cutpoints)

    while len(cuts) > 2:
        # Count observations in each bin
        counts = []
        for i in range(len(cuts) - 1):
            mask = (y >= cuts[i]) & (y < cuts[i + 1])
            counts.append(int(np.sum(mask)))

        if min(counts) >= min_obs:
            break

        # Find the bin with the fewest observations (excluding boundary bins)
        # Prefer interior merges; always keep cuts[0] and cuts[-1]
        min_count = min(counts)
        min_idx = counts.index(min_count)

        # Remove the interior cutpoint that merges the sparse bin
        if min_idx == 0:
            # Merge bin 0 with bin 1: remove cuts[1]
            del cuts[1]
        elif min_idx == len(counts) - 1:
            # Merge last bin with second-to-last: remove cuts[-2]
            del cuts[-2]
        else:
            # Merge with the smaller neighbour
            if counts[min_idx - 1] <= counts[min_idx + 1]:
                del cuts[min_idx]
            else:
                del cuts[min_idx + 1]

    return np.array(cuts)
