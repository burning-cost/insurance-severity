"""
insurance_severity.spqrx — SPQRx severity modelling.

Semi-parametric quantile regression with blended GPD tail (Majumder &
Richards, arXiv:2504.19994). Implements a joint bulk + tail conditional
density model:

- **Bulk**: M-spline mixture with covariate-dependent softmax weights (MLP).
  The CDF is an I-spline mixture — monotone by construction.
- **Tail**: Blended Generalised Pareto (bGPD) above quantile levels pa/pb.
  The GPD threshold ũ(x) is derived analytically from the bulk quantiles —
  no threshold stability plot required.
- **xi(x)**: Covariate-dependent tail shape from a second MLP head. Different
  risk segments get different tail heaviness.

Requires PyTorch: ``pip install insurance-severity[spqrx]``

Quick start::

    from insurance_severity.spqrx import SPQRxSeverity

    spqrx = SPQRxSeverity(n_splines=25, pa=0.85, pb=0.95)
    spqrx.fit(X_train, y_train)

    # Extreme quantile (Q99 for ILF/XL pricing)
    q99 = spqrx.predict_quantile(X_test, tau=0.99)

    # Full conditional distribution
    dist = spqrx.predict_distribution(X_test)
    dist.quantile(0.995)
    dist.ilf(limit=1_000_000, basic_limit=250_000)

    # Tail diagnostics (for actuarial sign-off)
    params = spqrx.tail_params(X_test)
    params['xi']         # GPD shape per observation
    params['u_tilde']    # effective threshold per observation
    params['sigma_tilde']  # GPD scale per observation

References
----------
Majumder, S. & Richards, J. (2025). 'Semi-parametric bulk and tail regression
    using spline-based neural networks.' arXiv:2504.19994.
"""

from insurance_severity.spqrx.spqrx import SPQRxSeverity
from insurance_severity.spqrx.distribution import SPQRxDistribution
from insurance_severity.spqrx.network import (
    SPQRxNetwork,
    make_mspline_basis,
    solve_bgpd_params,
)

__all__ = [
    "SPQRxSeverity",
    "SPQRxDistribution",
    "SPQRxNetwork",
    "make_mspline_basis",
    "solve_bgpd_params",
]
