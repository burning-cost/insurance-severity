"""
insurance-severity: comprehensive severity modelling for insurance pricing.

Combines three complementary approaches:

1. **Composite (spliced) models** — insurance_severity.composite
   Body/tail splice with covariate-dependent thresholds, mode-matching,
   ILF/TVaR, and diagnostic tools. For when you know your claim distribution
   has a structural break between attritional and large losses.

2. **Distributional Refinement Network** — insurance_severity.drn
   Full predictive distributions by refining a GLM/GBM baseline with a
   neural network. For when you want the actuarial calibration of a GLM
   but the distributional flexibility of a neural network.

3. **EVT classes** — insurance_severity.evt
   Truncated GPD MLE (policy limits), censoring-corrected Hill estimator
   (IBNR), and Weibull-tempered Pareto (bounded tails). For when standard
   GPD gives biased estimates because your data has truncation or censoring.

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

Quick start — EVT:

>>> from insurance_severity import TruncatedGPD, CensoredHillEstimator, WeibullTemperedPareto
>>> gpd = TruncatedGPD(threshold=10_000)
>>> gpd.fit(exceedances, limits)  # limits = per-policy caps
>>> gpd.summary()
>>> hill = CensoredHillEstimator()
>>> hill.fit(claims, censored=ibnr_flag)
>>> hill.xi, hill.ci
>>> wtp = WeibullTemperedPareto(threshold=10_000)
>>> wtp.fit(exceedances)
>>> wtp.isf(0.001)  # 99.9th percentile of excess
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

# DRN subpackage — lazy import because torch is an optional dependency.
# Use: pip install insurance-severity[drn]
# Then: from insurance_severity.drn import DRN, GLMBaseline
try:
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
except ImportError:
    # torch not installed; DRN classes not available at top level.
    # Install with: pip install insurance-severity[drn]
    pass

# EVT subpackage
from insurance_severity.evt import (
    TruncatedGPD,
    CensoredHillEstimator,
    WeibullTemperedPareto,
)

__version__ = "0.2.0"

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
    # EVT
    "TruncatedGPD",
    "CensoredHillEstimator",
    "WeibullTemperedPareto",
]
