"""
insurance_severity.mdn — Mixture Density Networks for insurance severity.

Implements Bishop (1994) MDN with lognormal mixture components: Gaussian
components fitted in log-space, naturally modelling positive, right-skewed
insurance claim amounts. Three components (K=3) captures the characteristic
multimodal structure of escape-of-water severity (drying peak, reinstatement
peak, major loss tail) that a Gamma GLM cannot represent.

Requires PyTorch: ``pip install insurance-severity[mdn]``
(or ``insurance-severity[drn]`` — both add ``torch>=2.0.0``).

Quick start::

    from insurance_severity.mdn import MDN

    mdn = MDN(n_components=3, hidden_size=64, max_epochs=200, random_state=42)
    mdn.fit(X_train, y_train)

    # Point prediction
    expected_severity = mdn.predict_mean(X_test)

    # Full conditional distribution
    dist = mdn.predict_distribution(X_test)
    dist.mean()                            # E[Y | x], shape (n,)
    dist.quantile(0.995)                   # SCR-level quantile
    dist.ilf(limit=50_000, basic_limit=10_000)  # increased limits factors
    dist.pit_samples(y_test)               # calibration check

    # Raw mixture parameters
    pi, mu, sigma = mdn.predict_params(X_test)

References
----------
Bishop, C.M. (1994). 'Mixture Density Networks.'
    Technical Report NCRG/94/004, Aston University.
Delong, L., Lindholm, M., Wüthrich, M.V. (2021). 'Gamma Mixture Density
    Networks and their application to modelling insurance claim amounts.'
    Insurance: Mathematics and Economics.
"""

from insurance_severity.mdn.mdn import MDN
from insurance_severity.mdn.distribution import MDNMixture
from insurance_severity.mdn.network import MDNNetwork
from insurance_severity.mdn.loss import (
    mdn_nll_loss,
    mdn_log_prob,
    mixture_mean,
    mixture_quantile,
    mixture_cdf,
)

__all__ = [
    "MDN",
    "MDNMixture",
    "MDNNetwork",
    "mdn_nll_loss",
    "mdn_log_prob",
    "mixture_mean",
    "mixture_quantile",
    "mixture_cdf",
]
