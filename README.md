# insurance-severity

[![PyPI](https://img.shields.io/pypi/v/insurance-severity)](https://pypi.org/project/insurance-severity/)
[![Python](https://img.shields.io/pypi/pyversions/insurance-severity)](https://pypi.org/project/insurance-severity/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-BSD--3-blue)]()

Comprehensive severity modelling for UK insurance pricing. Two complementary approaches in one package.

**Blog post:** [Spliced Severity Distributions: When One Distribution Isn't Enough](https://burning-cost.github.io/2027/01/15/spliced-severity-distributions-when-one-distribution-isnt-enough/)

## The problem

Claim severity distributions don't behave like textbook Gamma distributions. You have a body of attritional losses and a heavy tail of large losses, and these two populations have different drivers. Standard GLMs smooth over this structure. This package gives you two principled ways to deal with it.

## Quick Start

```bash
uv add insurance-severity
```

> 💬 Questions or feedback? Start a [Discussion](https://github.com/burning-cost/insurance-severity/discussions). Found it useful? A ⭐ helps others find it.

```python
import numpy as np
from insurance_severity import LognormalBurrComposite

rng = np.random.default_rng(42)

# Synthetic severity: lognormal attritional body + heavy Pareto-like tail
attritional = rng.lognormal(mean=7.5, sigma=1.0, size=850)    # ~85% of claims
large_loss  = rng.pareto(a=2.5, size=150) * 40_000 + 8_000    # ~15% large losses
claims = np.concatenate([attritional, large_loss])
rng.shuffle(claims)

# Fit the composite model — profile likelihood selects the threshold automatically
model = LognormalBurrComposite(threshold_method="mode_matching")
model.fit(claims)

print(f"Threshold:       £{model.threshold_:,.0f}")
# body_params_ = [mu, sigma] for the lognormal; tail_params_ = [alpha, delta, beta] for Burr XII
mu, sigma = model.body_params_
alpha, delta, beta = model.tail_params_
print(f"Body lognormal:  mu={mu:.3f}, sigma={sigma:.3f}  (log-scale)")
print(f"Tail Burr XII:   alpha={alpha:.3f}, delta={delta:.3f}, beta={beta:,.0f}")
print(f"Body weight pi:  {model.pi_:.3f}  ({model.pi_*100:.1f}% of claims are attritional)")

# ILF: expected loss in layer (250k xs 250k) relative to basic limit
ilf = model.ilf(limit=500_000, basic_limit=250_000)
print(f"ILF at £500k limit / £250k basic: {ilf:.4f}")

# Tail Value at Risk at 99.5th percentile (Solvency II capital proxy)
tvar = model.tvar(alpha=0.995)
print(f"TVaR 99.5%: £{tvar:,.0f}")
```

## What's in the package

### `insurance_severity.composite` — spliced severity models

Composite (spliced) models divide the claim distribution at a threshold into a body distribution (moderate claims) and a tail distribution (large losses). Each component is fitted separately and joined at the threshold.

What this package adds that R doesn't have: **covariate-dependent thresholds**. For a motor book, the threshold between attritional and large loss isn't the same for a HGV fleet and a private motor policy. With mode-matching regression, the threshold varies by policyholder.

Supported combinations:
- Lognormal body + Burr XII tail (mode-matching supported)
- Lognormal body + GPD tail
- Gamma body + GPD tail

Features:
- Fixed threshold, profile likelihood, and mode-matching threshold methods
- Covariate-dependent tail scale regression (`CompositeSeverityRegressor`)
- ILF computation per policyholder
- TVaR (Tail Value at Risk)
- Quantile residuals, mean excess plots, Q-Q plots

```python
import numpy as np
from insurance_severity import LognormalBurrComposite, CompositeSeverityRegressor

rng = np.random.default_rng(42)
n = 1000

# Synthetic severity data: lognormal attritional body + Pareto-like large losses
attritional = rng.lognormal(mean=7.5, sigma=1.2, size=int(n * 0.85))  # ~£1,800 median
large_losses = rng.pareto(a=2.5, size=int(n * 0.15)) * 50_000 + 10_000
claim_amounts = np.concatenate([attritional, large_losses])
rng.shuffle(claim_amounts)

# Rating factors for regression example
vehicle_age = rng.integers(0, 15, n).astype(float)
driver_age = rng.integers(18, 75, n).astype(float)
ncd_years = rng.integers(0, 5, n).astype(float)
X = np.column_stack([vehicle_age, driver_age, ncd_years])
n_train = int(0.8 * n)
X_train, X_test = X[:n_train], X[n_train:]
y_train = claim_amounts[:n_train]

# No covariates -- mode-matching threshold
model = LognormalBurrComposite(threshold_method="mode_matching")
model.fit(claim_amounts)
print(model.threshold_)
print(model.ilf(limit=500_000, basic_limit=250_000))

# With covariates -- threshold varies by policyholder
reg = CompositeSeverityRegressor(
    composite=LognormalBurrComposite(threshold_method="mode_matching"),
)
reg.fit(X_train, y_train)
thresholds = reg.predict_thresholds(X_test)   # per-policyholder thresholds
ilf_matrix = reg.compute_ilf(X_test, limits=[100_000, 250_000, 500_000, 1_000_000])
```

### `insurance_severity.drn` — Distributional Refinement Network

The DRN (Avanzi, Dong, Laub, Wong 2024, arXiv:2406.00998) starts from a frozen GLM or GBM baseline and refines it into a full predictive distribution using a neural network. The network outputs bin-probability adjustments to a histogram representation of the distribution, not the mean.

The practical payoff: you keep the actuarial calibration of your existing GLM pricing model, and the neural network fills in the distributional shape that the GLM can't capture — skewness, heteroskedastic dispersion, tail behaviour by segment.

**Note:** the DRN requires PyTorch. Install it before using this subpackage:

```bash
uv add torch --index-url https://download.pytorch.org/whl/cpu
uv add "insurance-severity[glm]"
```

```python
import numpy as np
from insurance_severity import GLMBaseline, DRN
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pandas as pd

rng = np.random.default_rng(42)
n = 1000

# Synthetic severity data with covariates
vehicle_age = rng.integers(0, 15, n).astype(float)
driver_age = rng.integers(18, 75, n).astype(float)
ncd_years = rng.integers(0, 5, n).astype(float)
region = rng.choice(["London", "SE", "NW", "Midlands", "Scotland"], n)

# Lognormal claim amounts with some large losses
log_mu = 7.5 + 0.03 * vehicle_age - 0.005 * driver_age - 0.05 * ncd_years
claim_amounts = rng.lognormal(mean=log_mu, sigma=1.1 + 0.02 * vehicle_age)

n_train = int(0.8 * n)
df = pd.DataFrame({
    "claims": claim_amounts,
    "age": driver_age,
    "vehicle_age": vehicle_age,
    "region": region,
})
df_train = df.iloc[:n_train]
df_test = df.iloc[n_train:]
X_train = np.column_stack([vehicle_age[:n_train], driver_age[:n_train], ncd_years[:n_train]])
X_test = np.column_stack([vehicle_age[n_train:], driver_age[n_train:], ncd_years[n_train:]])
y_train = claim_amounts[:n_train]
y_test = claim_amounts[n_train:]

# Fit your existing GLM
glm = smf.glm(
    "claims ~ age + C(region) + vehicle_age",
    data=df_train,
    family=sm.families.Gamma(sm.families.links.Log()),
).fit()

# Wrap it as a baseline
baseline = GLMBaseline(glm)

# Refine with DRN (scr_aware=True enables SCR-aware bin cutpoints at 99.5th percentile)
drn = DRN(baseline, hidden_size=64, max_epochs=300, scr_aware=True)
drn.fit(X_train, y_train, verbose=True)

# Full predictive distribution per policyholder
dist = drn.predict_distribution(X_test)
print(dist.mean())           # (n,) expected claim
print(dist.quantile(0.995))  # 99.5th percentile -- Solvency II SCR
print(dist.crps(y_test))     # CRPS scoring
```

### `insurance_severity.evt` — Extreme Value Theory for censored and truncated claims

Standard composite models assume you have clean, ground-up severity data. In practice you don't: policy limits truncate the right tail, IBNR means large claims are systematically under-reported at valuation, and the data you observe is not the data you need to model. The EVT module handles these distortions explicitly.

The three classes are based on Albrecher et al. (2025) — tail estimation under policy limits, IBNR censoring, and physically bounded tails.

**`TruncatedGPD`** — Generalised Pareto Distribution fitted to exceedances above a threshold, with optional right-truncation at a policy limit. If you naively fit a GPD to capped claims without accounting for the cap, you underestimate the tail index. This corrects that via MLE over the truncated likelihood.

**`CensoredHillEstimator`** — Hill estimator corrected for right-censoring. The standard Hill estimator applied to censored order statistics is biased; this uses the formula `sum(log x_j) - k * log(boundary)` to recover the true Pareto tail index from the top-k observations.

**`WeibullTemperedPareto`** — Models raw severity (x > 0) with a Pareto body and exponential tempering in the extreme tail. Useful when physical constraints bound the maximum possible loss — a nuclear power plant has a finite replacement cost, even if an unconstrained Pareto fit would extrapolate beyond it. Fitted via MLE.

```python
from insurance_severity.evt import TruncatedGPD, CensoredHillEstimator, WeibullTemperedPareto

# GPD fitted to exceedances above £10k, truncated at £100k policy limit
gpd = TruncatedGPD()
gpd.fit(large_claims, threshold=10_000, upper=100_000)
print(gpd.summary())

# Hill estimator corrected for censoring at £50k
hill = CensoredHillEstimator()
hill.fit(claims, boundary=50_000)
print(f"Tail index: {hill.xi:.3f}")

# Tempered Pareto — Pareto body with bounded extreme tail
wtp = WeibullTemperedPareto()
wtp.fit(claims)
print(wtp.summary())
```

All three classes expose `pdf`, `cdf`, `ppf`, `rvs` where appropriate, a `xi` property for the tail index, and a `summary()` method. `TruncatedGPD` additionally provides `mean_excess()` for mean excess plot diagnostics.

**When to use:** XL reinsurance where the cession data is limit-censored; reserving triangles where large claims are IBNR-censored; any context where naively fitting a GPD to the observed data would produce a biased tail index because not all large losses are visible in the data at valuation.

## Installation

```bash
uv add insurance-severity
```

With GLM support (statsmodels):

```bash
uv add "insurance-severity[glm]"
```

## Subpackages

Access subpackages directly if you only need one approach:

```python
from insurance_severity.composite import LognormalBurrComposite
from insurance_severity.drn import DRN
from insurance_severity.evt import TruncatedGPD, CensoredHillEstimator, WeibullTemperedPareto
```

---


## Databricks Notebook

A ready-to-run Databricks notebook benchmarking this library against standard approaches is available in [burning-cost-examples](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_severity_demo.py).

## Performance

Benchmarked against a **single Lognormal** on 2,500 train / 500 test synthetic claims with a known heavy-tailed DGP — Lognormal body (85%) below the splice point, Pareto tail (alpha=2.0, 15%) above. Post-Phase-98 fix numbers (composite predict() now returns the full mixture, pi fitted from data). Full script: `benchmarks/benchmark.py`.

DGP true quantiles: 95th=12,087 | 99th=24,701 | 99.5th=33,435.

| Metric | Single Lognormal | LognormalGPDComposite |
|--------|-----------------|----------------------|
| 95th percentile | 13,178 (err 1,091) | 11,586 (err 501) |
| 99th percentile | 27,702 (err 3,001) | 22,551 (err 2,150) |
| 99.5th percentile | 36,360 (err 2,925) | 29,465 (err 3,970) |
| Total tail error (sum of 3) | 7,017 | 6,621 |
| Test log-likelihood | -4,613.8 | -4,613.6 |
| Fit time | <1s | 2–5s (profile-likelihood grid) |

Overall tail quantile improvement: **5.6%** reduction in absolute error across the three quantile levels. The composite model fits a GPD tail above the profile-likelihood threshold (fitted ~5,022) with body weight pi=0.73.

The improvement is modest but directionally correct. On 3,000 observations with moderate tail heaviness (Pareto alpha=2.0), the profile-likelihood threshold selection has limited data in the tail — the Pareto 99.5th estimate overshoots (error 3,970 vs lognormal 2,925) because the GPD is trying to fit from only ~375 tail observations. At larger sample sizes (20k+) the tail fit stabilises and the composite advantage grows.

**When to use:** XL reinsurance pricing (where the expected loss in a layer depends entirely on tail behaviour), ILF computation at high policy limits, and large loss loading in ground-up pricing where the true severity distribution is Pareto-like. Concrete situations: motor bodily injury, liability lines, property CAT perils.

**When NOT to use:** When claims are capped (loss-limited data). Also when the portfolio has fewer than a few thousand claims — the profile-likelihood threshold selection is unstable with sparse data and the composite model may overfit the tail. For homogeneous attritional loss books where a Gamma fits well (small commercial property), the added complexity is not warranted.


## Source repos

- `insurance-composite` — archived, merged into this package
- `insurance-drn` — archived, merged into this package

## Related Libraries

| Library | What it does |
|---------|-------------|
| [insurance-frequency-severity](https://github.com/burning-cost/insurance-frequency-severity) | Joint frequency-severity models — combines this library's severity component with frequency in a Sarmanov copula |
| [insurance-quantile](https://github.com/burning-cost/insurance-quantile) | Quantile GBM for tail risk — non-parametric alternative when parametric severity assumptions are not tenable |
| [insurance-dynamics](https://github.com/burning-cost/insurance-dynamics) | Loss development models — severity projections are a key input to dynamic reserve models |
