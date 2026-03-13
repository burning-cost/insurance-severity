"""
insurance_severity.composite — composite (spliced) severity models.

Composite severity models split the claim distribution at a threshold into
a body distribution (moderate claims) and a tail distribution (large claims).
Covariate-dependent thresholds are supported via mode-matching regression.

Quick start
-----------
>>> from insurance_severity.composite import LognormalBurrComposite
>>> model = LognormalBurrComposite(threshold_method="mode_matching")
>>> model.fit(claim_amounts)
>>> model.threshold_
>>> model.ilf(limit=500_000, basic_limit=250_000)

With covariates:

>>> from insurance_severity.composite import CompositeSeverityRegressor
>>> reg = CompositeSeverityRegressor(
...     composite=LognormalBurrComposite(threshold_method="mode_matching"),
...     feature_cols=["vehicle_age", "driver_age", "region"],
... )
>>> reg.fit(X_train, y_train)
>>> reg.predict_thresholds(X_test)
"""

from insurance_severity.composite.distributions import (
    LognormalBody,
    GammaBody,
    GPDTail,
    ParetoTail,
    BurrTail,
)
from insurance_severity.composite.models import (
    LognormalBurrComposite,
    LognormalGPDComposite,
    GammaGPDComposite,
    CompositeSeverityModel,
)
from insurance_severity.composite.regression import CompositeSeverityRegressor
from insurance_severity.composite.diagnostics import (
    quantile_residuals,
    mean_excess_plot,
    density_overlay_plot,
    qq_plot,
)

__all__ = [
    # Distribution building blocks
    "LognormalBody",
    "GammaBody",
    "GPDTail",
    "ParetoTail",
    "BurrTail",
    # Composite models
    "CompositeSeverityModel",
    "LognormalBurrComposite",
    "LognormalGPDComposite",
    "GammaGPDComposite",
    # Regression
    "CompositeSeverityRegressor",
    # Diagnostics
    "quantile_residuals",
    "mean_excess_plot",
    "density_overlay_plot",
    "qq_plot",
]
