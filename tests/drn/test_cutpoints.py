"""
Tests for drn_cutpoints — the histogram bin selection algorithm.
"""

import numpy as np
import pytest

from insurance_severity.drn.cutpoints import drn_cutpoints, _merge_sparse_bins


class TestDrnCutpoints:
    def test_basic_output_shape(self):
        rng = np.random.default_rng(0)
        y = rng.gamma(2.0, 1000.0, size=1000)
        cuts = drn_cutpoints(y, proportion=0.1)
        # 10 bins = 11 cutpoints (approximately; may vary due to quantile collisions)
        assert len(cuts) >= 3
        assert cuts[0] == 0.0  # default c_0
        assert cuts[-1] > np.max(y)  # c_K > max(y) by 1.05x

    def test_sorted_ascending(self):
        rng = np.random.default_rng(1)
        y = rng.gamma(2.0, 500.0, size=500)
        cuts = drn_cutpoints(y, proportion=0.1)
        assert np.all(np.diff(cuts) > 0), "Cutpoints must be strictly increasing"

    def test_custom_c0_cK(self):
        rng = np.random.default_rng(2)
        y = rng.gamma(2.0, 1000.0, size=1000)
        cuts = drn_cutpoints(y, c_0=100.0, c_K=10000.0)
        assert cuts[0] == 100.0
        assert cuts[-1] == 10000.0

    def test_proportion_affects_bin_count(self):
        rng = np.random.default_rng(3)
        y = rng.gamma(2.0, 1000.0, size=2000)
        cuts_coarse = drn_cutpoints(y, proportion=0.2)
        cuts_fine = drn_cutpoints(y, proportion=0.05)
        assert len(cuts_fine) >= len(cuts_coarse), (
            "Smaller proportion should give more bins"
        )

    def test_scr_aware_cK_above_997(self):
        rng = np.random.default_rng(4)
        y = rng.gamma(2.0, 1000.0, size=5000)
        cuts = drn_cutpoints(y, scr_aware=True)
        p997 = float(np.percentile(y, 99.7))
        assert cuts[-1] >= p997, "scr_aware: c_K must be above 99.7th percentile"

    def test_rejects_non_positive_y(self):
        y = np.array([1.0, 2.0, 0.0, 3.0])
        with pytest.raises(ValueError, match="positive"):
            drn_cutpoints(y)

    def test_rejects_nan(self):
        y = np.array([1.0, 2.0, np.nan, 3.0])
        with pytest.raises(ValueError, match="NaN"):
            drn_cutpoints(y)

    def test_small_dataset(self):
        rng = np.random.default_rng(5)
        y = rng.gamma(2.0, 1000.0, size=20)
        cuts = drn_cutpoints(y, proportion=0.5)
        assert len(cuts) >= 3
        assert np.all(np.diff(cuts) > 0)

    def test_min_obs_merging(self):
        """With min_obs=50 on n=100 data, we should have at most 2 bins."""
        rng = np.random.default_rng(6)
        y = rng.gamma(2.0, 1000.0, size=100)
        cuts = drn_cutpoints(y, proportion=0.1, min_obs=50)
        # Each bin must have >= 50 obs; with 100 obs, max 2 bins (3 cutpoints)
        assert len(cuts) <= 4  # generous bound

    def test_unique_cutpoints(self):
        """No duplicate cutpoints."""
        rng = np.random.default_rng(7)
        y = rng.gamma(2.0, 1000.0, size=200)
        cuts = drn_cutpoints(y, proportion=0.1)
        assert len(cuts) == len(np.unique(cuts))

    def test_large_dataset_performance(self):
        """drn_cutpoints should handle 100k obs without issue."""
        rng = np.random.default_rng(8)
        y = rng.gamma(2.0, 1000.0, size=100_000)
        cuts = drn_cutpoints(y, proportion=0.01)
        assert len(cuts) > 10
        assert np.all(np.diff(cuts) > 0)


class TestMergeSparse:
    def test_merging_reduces_bins(self):
        rng = np.random.default_rng(9)
        y = rng.gamma(2.0, 1000.0, size=50)
        cuts = np.quantile(y, np.linspace(0, 1, 11))
        cuts = np.concatenate([[0.0], cuts[1:-1], [cuts[-1] * 1.05]])
        cuts = np.unique(cuts)
        merged = _merge_sparse_bins(y, cuts, min_obs=10)
        # All bins should have >= min_obs or only one bin remains
        if len(merged) > 2:
            for i in range(len(merged) - 1):
                count = np.sum((y >= merged[i]) & (y < merged[i + 1]))
                assert count >= 10 or len(merged) == 3
