"""
Tests for diagnostic functions.

We test behavior without matplotlib (skip plots if not installed).
Numeric functions (quantile_residuals, mean_excess_plot data) are always tested.
"""

import numpy as np
import pytest
import warnings

from insurance_severity.composite.models import LognormalGPDComposite, LognormalBurrComposite
from insurance_severity.composite.diagnostics import quantile_residuals


# ---------------------------------------------------------------------------
# Quantile residuals
# ---------------------------------------------------------------------------


class TestQuantileResiduals:

    def test_shape(self, lognormal_gpd_data):
        model = LognormalGPDComposite(threshold=50000.0, threshold_method="fixed")
        model.fit(lognormal_gpd_data)
        resid = quantile_residuals(model, lognormal_gpd_data)
        assert len(resid) == len(lognormal_gpd_data)

    def test_approximately_normal(self, lognormal_gpd_data):
        """Under correct model, residuals should be roughly N(0,1)."""
        model = LognormalGPDComposite(threshold=50000.0, threshold_method="fixed")
        model.fit(lognormal_gpd_data)
        resid = quantile_residuals(model, lognormal_gpd_data)
        # Mean should be close to 0, std close to 1
        assert abs(np.mean(resid)) < 1.5
        assert 0.3 < np.std(resid) < 3.0

    def test_finite_values(self, lognormal_gpd_data):
        model = LognormalGPDComposite(threshold=50000.0, threshold_method="fixed")
        model.fit(lognormal_gpd_data)
        resid = quantile_residuals(model, lognormal_gpd_data)
        assert np.all(np.isfinite(resid))

    def test_mode_matching_residuals(self, lognormal_burr_data):
        model = LognormalBurrComposite(threshold_method="mode_matching", n_starts=3)
        model.fit(lognormal_burr_data)
        resid = quantile_residuals(model, lognormal_burr_data)
        assert len(resid) == len(lognormal_burr_data)
        assert np.all(np.isfinite(resid))


# ---------------------------------------------------------------------------
# Plot functions (skip if matplotlib not available)
# ---------------------------------------------------------------------------


def _matplotlib_available():
    try:
        import matplotlib
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _matplotlib_available(), reason="matplotlib not installed")
class TestPlots:

    def test_density_overlay_runs(self, lognormal_gpd_data):
        from insurance_severity.composite.diagnostics import density_overlay_plot
        import matplotlib
        matplotlib.use("Agg")
        model = LognormalGPDComposite(threshold=50000.0, threshold_method="fixed")
        model.fit(lognormal_gpd_data)
        ax = density_overlay_plot(model, lognormal_gpd_data)
        assert ax is not None

    def test_qq_plot_runs(self, lognormal_gpd_data):
        from insurance_severity.composite.diagnostics import qq_plot
        import matplotlib
        matplotlib.use("Agg")
        model = LognormalGPDComposite(threshold=50000.0, threshold_method="fixed")
        model.fit(lognormal_gpd_data)
        ax = qq_plot(model, lognormal_gpd_data)
        assert ax is not None

    def test_mean_excess_plot_runs(self, lognormal_gpd_data):
        from insurance_severity.composite.diagnostics import mean_excess_plot
        import matplotlib
        matplotlib.use("Agg")
        ax = mean_excess_plot(lognormal_gpd_data)
        assert ax is not None

    def test_model_plot_fit_runs(self, lognormal_gpd_data):
        import matplotlib
        matplotlib.use("Agg")
        model = LognormalGPDComposite(threshold=50000.0, threshold_method="fixed")
        model.fit(lognormal_gpd_data)
        ax = model.plot_fit(lognormal_gpd_data)
        assert ax is not None
