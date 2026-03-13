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
from insurance_severity import LognormalBurrComposite, CompositeSeverityRegressor

# No covariates — mode-matching threshold
model = LognormalBurrComposite(threshold_method="mode_matching")
model.fit(claim_amounts)
print(model.threshold_)
print(model.ilf(limit=500_000, basic_limit=250_000))

# With covariates — threshold varies by policyholder
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

```python
from insurance_severity import GLMBaseline, DRN
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Fit your existing GLM
glm = smf.glm(
    "claims ~ age + C(region) + vehicle_age",
    data=df_train,
    family=sm.families.Gamma(sm.families.links.Log()),
).fit()

# Wrap it as a baseline
baseline = GLMBaseline(glm)

# Refine with DRN
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
