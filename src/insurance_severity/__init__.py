"""
insurance-severity: comprehensive severity modelling for insurance pricing.

Combines two complementary approaches:

1. **Composite (spliced) models** — insurance_severity.composite
   Body/tail splice with covariate-dependent thresholds, mode-matching,
   ILF/TVaR, and diagnostic tools. For when you know your claim distribution
   has a structural break between attritional and large losses.

2. **Distributional Refinement Network** — insurance_severity.drn
   Full predictive distributions by refining a GLM/GBM baseline with a
   neural network. For when you want the actuarial calibration of a GLM
   but the distributional flexibility of a neural network.

Quick start — composite:

>>> from insurance_severity import LognormalBurrComposite, CompositeSeverityRegressor
>>> model = LognormalBurrComposite(threshold_method="mode_matching")
>>> model.fit(y_train)
>>> model.ilf(limit=500_000, basic_limit=250_000)

Quick start — DRN:

>>> from insurance_severity import GLMBaseline, DRN
>>> baseline = GLMBaseline(glm_result)
>>> drn = DRN(baseline, hidden_size=64)
>>> drn.fit(X_train, y_train)
>>> dist = drn.predict_distribution(X_test)
>>> dist.quantile(0.995)  # SCR quantile
"""

# Composite subpackage
from insurance_severity.composite import (
    LognormalBody,
    GammaBody,
    GPDTail,
    ParetoTail,
    BurrTail,
    CompositeSeverityModel,
    LognormalBurrComposite,
    LognormalGPDComposite,
    GammaGPDComposite,
    CompositeSeverityRegressor,
    quantile_residuals,
    mean_excess_plot,
    density_overlay_plot,
    qq_plot,
)

# DRN subpackage
from insurance_severity.drn import (
    BaselineDistribution,
    GLMBaseline,
    CatBoostBaseline,
    ExtendedHistogramBatch,
    DRNNetwork,
    DRN,
    DRNDiagnostics,
    drn_cutpoints,
    jbce_loss,
    drn_regularisation,
)

__version__ = "0.1.0"

__all__ = [
    # Composite
    "LognormalBody",
    "GammaBody",
    "GPDTail",
    "ParetoTail",
    "BurrTail",
    "CompositeSeverityModel",
    "LognormalBurrComposite",
    "LognormalGPDComposite",
    "GammaGPDComposite",
    "CompositeSeverityRegressor",
    "quantile_residuals",
    "mean_excess_plot",
    "density_overlay_plot",
    "qq_plot",
    # DRN
    "BaselineDistribution",
    "GLMBaseline",
    "CatBoostBaseline",
    "ExtendedHistogramBatch",
    "DRNNetwork",
    "DRN",
    "DRNDiagnostics",
    "drn_cutpoints",
    "jbce_loss",
    "drn_regularisation",
]
