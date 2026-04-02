"""
insurance-severity: comprehensive severity modelling for insurance pricing.

Combines seven complementary approaches:

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

4. **Mixture Density Network** — insurance_severity.mdn
   Lognormal mixture model with neural network-parameterised mixing weights,
   means, and scales. For when the severity distribution is genuinely
   multimodal — the case for escape-of-water and other multi-phase perils.
   Requires PyTorch: ``pip install insurance-severity[mdn]``.

5. **SPQRx** — insurance_severity.spqrx
   Semi-parametric M-spline bulk density blended with a covariate-conditional
   GPD tail (Majumder & Richards, arXiv:2504.19994). For when you need ILF
   curves and extreme quantiles (Q99+) beyond the training data range, with
   a threshold that automatically varies by risk characteristics.
   Requires PyTorch: pip install insurance-severity[spqrx].

6. **Tail scoring** — insurance_severity.tail_scoring
   Allen et al. (2025, JASA) tail calibration diagnostics (all EVT domains)
   and Bladt & Øhlenschlæger (2026) tail log-score for Pareto-family model
   ranking (Fréchet domain). For when you need to know whether your model is
   calibrated in the tail, and which of several large-loss models fits best.

7. **Projection-to-Ultimate** — insurance_severity.projection
   One-shot PtU factor estimation for RBNS claims (Richman & Wüthrich,
   arXiv:2603.11660). OLS/Ridge regression of log(ultimate/paid) on
   development features. Empirically outperforms neural networks on
   realistic reserving triangle sizes.

8. **CMRS Allocator** — insurance_severity.cmrs
   Conditional Mean Risk Sharing via Laplace-Stieltjes transforms (Blier-Wong 2026).
   Computes h_i(s) = E[X_i | S = s] for independent losses X_1, ..., X_n with
   aggregate S. Supports exponential, gamma, and lognormal marginals with Euler
   summation inversion (Abate-Whitt 1995) and exponential tilting for tail values.

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

Quick start — tail variable importance:

>>> from insurance_severity import TailVariableImportance
>>> tvi = TailVariableImportance(threshold_quantile=0.90, alpha=0.1)
>>> tvi.fit(X_train, y_train, feature_names=feature_cols)
>>> tvi.importances          # dict: feature -> tail importance
>>> tvi.summary()            # threshold, n_selected, etc.
>>> tvi.plot(top_k=10)       # horizontal bar chart

Quick start — MDN:

>>> from insurance_severity import MDN
>>> mdn = MDN(n_components=3, hidden_size=64, max_epochs=200)
>>> mdn.fit(X_train, y_train)
>>> dist = mdn.predict_distribution(X_test)
>>> dist.mean()              # expected severity per observation
>>> dist.quantile(0.995)     # tail quantile
>>> dist.ilf(50_000, 10_000) # increased limits factors

Quick start — tail calibration and scoring:

>>> from insurance_severity import TailCalibration, BladtTailScore, pareto_qq
>>> # Confirm Fréchet domain:
>>> r2 = pareto_qq(y, ax=ax)  # R² > 0.95 suggests Fréchet
>>> # Check model calibration in the tail:
>>> tc = TailCalibration(cdf_func=my_model.cdf, n_obs=len(y))
>>> tc.fit(y)
>>> tc.summary_table(t_levels=np.quantile(y, [0.90, 0.95, 0.99]))
>>> # Rank competing tail index specifications:
>>> bs = BladtTailScore()
>>> bs.rank(y, gamma_candidates=[0.5, 0.8, 1.0, 1.3], k_grid=np.arange(20, 200))

Quick start — Projection-to-Ultimate:

>>> from insurance_severity import ProjectionToUltimate
>>> ptu = ProjectionToUltimate(
...     development_features=["dev_month", "log_paid", "claim_age"],
...     method="ols",
... )
>>> ptu.fit(train_df, paid_col="paid_to_date", ultimate_col="ultimate_cost")
>>> preds = ptu.predict(open_claims_df)
>>> ptu.summary()  # R², coefficients, residual diagnostics

Quick start — CMRS allocation:

>>> from insurance_severity import CMRSAllocator
>>> alloc = CMRSAllocator(distribution='gamma')
>>> alloc.fit_gamma(
...     alphas=np.array([2.0, 3.0, 1.5, 4.0, 2.5]),
...     betas=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
... )
>>> h = alloc.allocate(12_000_000.0)   # fair shares of £12M aggregate loss
>>> h_scr = alloc.allocate_quantile(np.array([0.995]))  # allocation at SCR level
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

# MDN subpackage — lazy import, also requires torch.
# Use: pip install insurance-severity[mdn]
# Then: from insurance_severity.mdn import MDN
try:
    from insurance_severity.mdn import (
        MDN,
        MDNMixture,
        MDNNetwork,
        mdn_nll_loss,
        mdn_log_prob,
        mixture_mean,
        mixture_quantile,
        mixture_cdf,
    )
except ImportError:
    # torch not installed; MDN classes not available at top level.
    # Install with: pip install insurance-severity[mdn]
    pass


# SPQRx subpackage — lazy import, also requires torch.
# Use: pip install insurance-severity[spqrx]
# Then: from insurance_severity.spqrx import SPQRxSeverity
try:
    from insurance_severity.spqrx import (
        SPQRxSeverity,
        SPQRxDistribution,
        SPQRxNetwork,
        make_mspline_basis,
        solve_bgpd_params,
    )
except ImportError:
    # torch not installed; SPQRx classes not available at top level.
    # Install with: pip install insurance-severity[spqrx]
    pass

# EVT subpackage
from insurance_severity.evt import (
    TruncatedGPD,
    CensoredHillEstimator,
    WeibullTemperedPareto,
    TailVariableImportance,
)

# Tail scoring
from insurance_severity.tail_scoring import (
    TailCalibration,
    BladtTailScore,
    pareto_qq,
)

# Projection-to-Ultimate
from insurance_severity.projection import ProjectionToUltimate

# CMRS allocation (pure scipy/numpy — no optional dependencies)
from insurance_severity.cmrs import CMRSAllocator

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("insurance-severity")
except PackageNotFoundError:
    __version__ = "0.0.0"  # not installed

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
    # MDN
    "MDN",
    "MDNMixture",
    "MDNNetwork",
    "mdn_nll_loss",
    "mdn_log_prob",
    "mixture_mean",
    "mixture_quantile",
    "mixture_cdf",
    # EVT
    "TruncatedGPD",
    "CensoredHillEstimator",
    "WeibullTemperedPareto",
    "TailVariableImportance",
    # Tail scoring
    "TailCalibration",
    "BladtTailScore",
    "pareto_qq",
    # SPQRx
    "SPQRxSeverity",
    "SPQRxDistribution",
    "SPQRxNetwork",
    "make_mspline_basis",
    "solve_bgpd_params",
    # Projection-to-Ultimate
    "ProjectionToUltimate",
    # CMRS allocation
    "CMRSAllocator",
]
