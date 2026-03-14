# insurance-severity

Comprehensive severity modelling for UK insurance pricing. Two complementary approaches in one package.

## The problem

Claim severity distributions don't behave like textbook Gamma distributions. You have a body of attritional losses and a heavy tail of large losses, and these two populations have different drivers. Standard GLMs smooth over this structure. This package gives you two principled ways to deal with it.

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
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install insurance-severity[glm]
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

## Installation

```bash
pip install insurance-severity
```

With GLM support (statsmodels):

```bash
pip install insurance-severity[glm]
```

## Subpackages

Access subpackages directly if you only need one approach:

```python
from insurance_severity.composite import LognormalBurrComposite
from insurance_severity.drn import DRN
```

---


## Databricks Notebook

A ready-to-run Databricks notebook benchmarking this library against standard approaches is available in [burning-cost-examples](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_severity_demo.py).

## Performance

Benchmarked against **single Gamma GLM** (statsmodels) on 10,000 synthetic claims with a known heavy-tailed DGP — Lognormal body below the splice point, Pareto tail above. Full notebook: `notebooks/benchmark.py`.

| Metric | Gamma GLM | Composite spliced (insurance-severity) |
|--------|-----------|---------------------------------------|
| Log-likelihood | lower | higher |
| AIC | higher | lower |
| 90th / 95th / 99th percentile accuracy | underestimates | near DGP truth |
| ILF at high policy limits | understated | near DGP truth |
| Fit time | seconds | seconds + profile-likelihood grid |

The benchmark tests tail quantile accuracy and ILF curves against the known DGP. The Gamma systematically underestimates tail quantiles because its survival function decays exponentially; the composite model fits a separate tail distribution above the profile-likelihood threshold, recovering the heavy tail.

**When to use:** XL reinsurance pricing (where the expected loss in a layer depends entirely on tail behaviour), ILF computation at high policy limits, and large loss loading in ground-up pricing where the true severity distribution is Pareto-like. Concrete situations: motor bodily injury, liability lines, property CAT perils.

**When NOT to use:** When claims are capped (loss-limited data), making it impossible to observe the true tail. Also when the portfolio has fewer than a few thousand claims — the profile-likelihood threshold selection is unstable with sparse data and the composite model may overfit the tail. For homogeneous attritional loss books where a Gamma fits well (small commercial property), the added complexity is not warranted.


## Source repos

- `insurance-composite` — archived, merged into this package
- `insurance-drn` — archived, merged into this package

## Related Libraries

| Library | What it does |
|---------|-------------|
| [insurance-frequency-severity](https://github.com/burning-cost/insurance-frequency-severity) | Joint frequency-severity models — combines this library's severity component with frequency in a Sarmanov copula |
| [insurance-quantile](https://github.com/burning-cost/insurance-quantile) | Quantile GBM for tail risk — non-parametric alternative when parametric severity assumptions are not tenable |
| [insurance-dynamics](https://github.com/burning-cost/insurance-dynamics) | Loss development models — severity projections are a key input to dynamic reserve models |

