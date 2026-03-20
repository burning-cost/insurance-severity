"""
insurance_severity.evt — Extreme Value Theory for insurance severity.

Implements EVT methods corrected for two systematic data quality issues
in commercial lines portfolios:

1. **Policy limit truncation** (TruncatedGPD): claims are never observed
   above the policy limit T. Standard GPD MLE treats this as if the
   support ends at max(data), producing upward-biased xi. TruncatedGPD
   includes the normalisation log F(T - u) in the log-likelihood.

2. **IBNR right-censoring** (CensoredHillEstimator): open/developing
   claims are observed at development value, not ultimate. Standard Hill
   treats them as fully observed, producing downward-biased xi. The
   censored Hill divides by the empirical uncensored fraction p_k.

Also provides WeibullTemperedPareto for physically bounded tails (property,
D&O) where pure Pareto overpredicts at extremes.

All three based on Albrecher, Beirlant & Teugels (2025), arXiv:2511.22272.

Quick start
-----------
>>> from insurance_severity import evt
>>> model = evt.TruncatedGPD()
>>> model.fit_mle(claims, threshold=100_000, limits=policy_limits)
>>> model.return_level(T_years=100, n_per_year=500, threshold=100_000,
...                   alpha_threshold=0.05)

>>> hill = evt.CensoredHillEstimator(method="simple")
>>> hill.fit(z=claims, delta=is_settled)
>>> hill.xi_hat_   # corrected tail index
"""

from insurance_severity.evt.truncated_gpd import TruncatedGPD
from insurance_severity.evt.censored_hill import CensoredHillEstimator
from insurance_severity.evt.tempered_pareto import WeibullTemperedPareto
from insurance_severity.evt.diagnostics import (
    hill_plot,
    mean_excess_censored,
    threshold_stability_plot,
    pareto_qq_plot,
)

__all__ = [
    "TruncatedGPD",
    "CensoredHillEstimator",
    "WeibullTemperedPareto",
    "hill_plot",
    "mean_excess_censored",
    "threshold_stability_plot",
    "pareto_qq_plot",
]
