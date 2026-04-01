# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-severity: SPQRxSeverity demo
# MAGIC
# MAGIC This notebook demonstrates SPQRxSeverity — Semi-parametric Quantile Regression
# MAGIC with blended GPD tail (Majumder & Richards, arXiv:2504.19994) — for UK large-loss
# MAGIC severity modelling.
# MAGIC
# MAGIC **The problem it solves**: You need ILF curves and Q99 estimates for an XL
# MAGIC reinsurance pricing exercise. Your data has 1,200 large losses above £100k from
# MAGIC ten years of UK motor TPBI. A TruncatedGPD gives you a single xi for all risks.
# MAGIC SPQRxSeverity gives you xi(x) — covariate-conditional tail heaviness — and
# MAGIC derives a covariate-conditional threshold ũ(x) automatically.
# MAGIC
# MAGIC **Key advantages over TruncatedGPD / EQRN**:
# MAGIC - No threshold stability plot required. The blend interval [pa, pb] replaces u.
# MAGIC - ũ(x) varies by observation — high-value risks blend into GPD at higher amounts.
# MAGIC - Single-stage training: bulk density and GPD tail fitted jointly.
# MAGIC - EVT-compliant extrapolation for tau >= pb via closed-form GPD formula.
# MAGIC
# MAGIC **Reference**: Majumder, S. & Richards, J. (2025). arXiv:2504.19994.

# COMMAND ----------

import subprocess
import sys

subprocess.run([
    sys.executable, "-m", "pip", "install", "--quiet",
    "git+https://github.com/burning-cost/insurance-severity.git[spqrx]",
], check=True)

# COMMAND ----------

import numpy as np
import pandas as pd
from scipy.stats import pareto

np.random.seed(42)
print("Libraries loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Synthetic UK motor TPBI large-loss dataset
# MAGIC
# MAGIC We generate synthetic data with the following structure:
# MAGIC
# MAGIC - Three covariates: `vehicle_value_std`, `injury_severity_idx`, `liability_pct`
# MAGIC - True tail shape xi(x) that varies with injury severity (higher xi for more
# MAGIC   severe injuries — the tail is heavier where catastrophic outcomes are likely)
# MAGIC - Pareto-tailed severity with covariate-dependent scale
# MAGIC
# MAGIC In practice, xi for UK TPBI is in the range 0.3–0.8. The catastrophic injury
# MAGIC segment (spinal, brain) is at the upper end; minor soft tissue at the lower end.

# COMMAND ----------

rng = np.random.default_rng(42)
n = 1200  # typical large-loss dataset size for mid-size UK insurer

# Covariates
vehicle_value_std = rng.standard_normal(n).astype(np.float32)
injury_severity_idx = rng.uniform(0, 1, n).astype(np.float32)  # 0=minor, 1=catastrophic
liability_pct = rng.beta(3, 2, n).astype(np.float32)  # proportion of liability admitted

X = np.column_stack([vehicle_value_std, injury_severity_idx, liability_pct])
df_X = pd.DataFrame(X, columns=["vehicle_value_std", "injury_severity_idx", "liability_pct"])

# True xi varies with injury severity: 0.3 for minor, 0.7 for catastrophic
xi_true = 0.3 + 0.4 * injury_severity_idx  # xi in [0.3, 0.7]

# True scale: higher for high-value vehicles
scale_true = 50_000 * (1 + 0.3 * vehicle_value_std.clip(-2, 2)) * liability_pct

# Generate Pareto losses: F(y) = 1 - (1 + y/scale)^{-1/xi}
# y = scale * (U^{-xi} - 1) where U ~ Uniform(0,1)
U = rng.uniform(0.01, 0.99, n)
y = scale_true * (U ** (-xi_true) - 1) + 50_000 * liability_pct
y = np.maximum(y, 5_000.0)

print(f"Dataset: n={n}, p=3")
print(f"y percentiles: P50={np.percentile(y, 50):,.0f}  P90={np.percentile(y, 90):,.0f}  P99={np.percentile(y, 99):,.0f}")
print(f"True xi range: [{xi_true.min():.3f}, {xi_true.max():.3f}]")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Fit SPQRxSeverity
# MAGIC
# MAGIC Default settings: 25 M-spline basis functions, blend interval [0.85, 0.95].
# MAGIC
# MAGIC The key parameters:
# MAGIC - `pa=0.85`: below this quantile, the bulk M-spline model is used exactly.
# MAGIC - `pb=0.95`: above this quantile, the GPD extrapolation formula is used exactly.
# MAGIC - `xi_l1=0.01`: L1 regularisation prevents xi(x) from collapsing to zero or
# MAGIC   exploding, stabilising estimation in the tail.

# COMMAND ----------

from insurance_severity.spqrx import SPQRxSeverity

model = SPQRxSeverity(
    n_splines=25,
    hidden_size=32,
    num_hidden_layers=2,
    pa=0.85,
    pb=0.95,
    xi_l1=0.01,
    max_epochs=200,   # reduce for demo speed
    patience=20,
    random_state=42,
)

# Train/test split
train_idx = rng.choice(n, size=900, replace=False)
test_idx = np.setdiff1d(np.arange(n), train_idx)
X_train, y_train = df_X.iloc[train_idx], y[train_idx]
X_test, y_test = df_X.iloc[test_idx], y[test_idx]

model.fit(X_train, y_train, verbose=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Tail parameter diagnostics
# MAGIC
# MAGIC The most important actuarial output. For each observation in the test set,
# MAGIC SPQRxSeverity provides:
# MAGIC - `xi(x)`: estimated GPD tail shape. For UK TPBI, expect 0.3–0.8.
# MAGIC - `u_tilde(x)`: effective GPD threshold (varies by risk).
# MAGIC - `sigma_tilde(x)`: GPD scale.
# MAGIC - `a(x)`, `b(x)`: lower and upper blend boundaries.

# COMMAND ----------

params = model.tail_params(X_test.values)

print("Tail parameter summary (test set):")
print(f"  xi(x):          mean={params['xi'].mean():.3f}  std={params['xi'].std():.3f}  "
      f"min={params['xi'].min():.3f}  max={params['xi'].max():.3f}")
print(f"  u_tilde(x):     mean={params['u_tilde'].mean():,.0f}  "
      f"min={params['u_tilde'].min():,.0f}  max={params['u_tilde'].max():,.0f}")
print(f"  sigma_tilde(x): mean={params['sigma_tilde'].mean():,.0f}")
print(f"  a(x) [Q85]:     mean={params['a'].mean():,.0f}")
print(f"  b(x) [Q95]:     mean={params['b'].mean():,.0f}")

# Compare fitted xi to true xi for the test set
print(f"\nTrue xi (test): mean={xi_true[test_idx].mean():.3f}")
print(f"Fitted xi:      mean={params['xi'].mean():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Extreme quantile prediction
# MAGIC
# MAGIC Three regimes:
# MAGIC - tau=0.75 (bulk spline inversion)
# MAGIC - tau=0.90 (blend region)
# MAGIC - tau=0.99 (GPD closed-form formula — EVT-compliant extrapolation)

# COMMAND ----------

for tau in [0.75, 0.85, 0.90, 0.95, 0.99, 0.999]:
    q = model.predict_quantile(X_test.values[:20], tau=tau)
    regime = "bulk" if tau < 0.85 else "blend" if tau < 0.95 else "GPD tail"
    print(f"Q({tau:.3f}) [{regime:9s}]: mean={q.mean():>12,.0f}  min={q.min():>10,.0f}  max={q.max():>12,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. ILF curves
# MAGIC
# MAGIC Increased Limits Factors for different policy limits.
# MAGIC
# MAGIC ILF(L, b) = E[min(Y, L)] / E[min(Y, b)]
# MAGIC
# MAGIC We use basic_limit=250,000 (typical UK motor TPBI basic limit for XL pricing).
# MAGIC The covariate-conditional ILFs mean each risk class gets its own factor.

# COMMAND ----------

dist = model.predict_distribution(X_test.values[:20])

basic_limit = 250_000
limits = [500_000, 1_000_000, 2_000_000, 5_000_000]

print(f"ILF curves (basic limit = £{basic_limit:,.0f}):")
print(f"{'Limit':>12}  {'Mean ILF':>10}  {'Min ILF':>10}  {'Max ILF':>10}")
for L in limits:
    ilf = dist.ilf(limit=L, basic_limit=basic_limit)
    print(f"£{L:>10,.0f}  {ilf.mean():>10.4f}  {ilf.min():>10.4f}  {ilf.max():>10.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Monotonicity check
# MAGIC
# MAGIC Quantiles must be non-decreasing in tau. A simple sanity check for any
# MAGIC probabilistic severity model.

# COMMAND ----------

X_check = X_test.values[:10]
taus = [0.10, 0.25, 0.50, 0.75, 0.85, 0.90, 0.95, 0.99]
quantiles = np.column_stack([model.predict_quantile(X_check, t) for t in taus])

violations = np.sum(np.diff(quantiles, axis=1) < -1)
print(f"Quantile monotonicity check: {violations} violations (should be 0)")
print(f"Q50 range: [{quantiles[:, 2].min():,.0f}, {quantiles[:, 2].max():,.0f}]")
print(f"Q99 range: [{quantiles[:, 7].min():,.0f}, {quantiles[:, 7].max():,.0f}]")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Threshold sensitivity analysis
# MAGIC
# MAGIC How sensitive is Q99 to the choice of pa and pb? If the result is stable
# MAGIC across a range of blending intervals, you have a robust estimate. If it
# MAGIC varies by >20%, investigate the data quality or increase sample size.

# COMMAND ----------

sensitivity = model.pa_pb_sensitivity(
    X_test.values[:30],
    tau=0.99,
    pa_range=[0.80, 0.85, 0.87],
    pb_range=[0.92, 0.95, 0.97],
)

print("pa/pb sensitivity for Q99:")
print(f"{'pa':>5}  {'pb':>5}  {'median_Q99':>12}  {'IQR':>10}")
for r in sensitivity["results"]:
    print(f"{r['pa']:>5.2f}  {r['pb']:>5.2f}  {r['median_q']:>12,.0f}  {r['iqr_q']:>10,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Comparison: SPQRx vs empirical quantile
# MAGIC
# MAGIC On the test set, compare SPQRx Q90 to the empirical Q90 of the test y values.
# MAGIC Note: SPQRx Q90 is a conditional quantile (varies by x), while empirical is
# MAGIC unconditional. The average of conditional quantiles should be near the
# MAGIC unconditional quantile if the model is calibrated.

# COMMAND ----------

q90_fitted = model.predict_quantile(X_test.values, tau=0.90)
q90_empirical = float(np.percentile(y_test, 90))
q99_fitted = model.predict_quantile(X_test.values, tau=0.99)
q99_empirical = float(np.percentile(y_test, 99))

print(f"Q90 — SPQRx mean: {q90_fitted.mean():,.0f}  |  Empirical: {q90_empirical:,.0f}")
print(f"Q99 — SPQRx mean: {q99_fitted.mean():,.0f}  |  Empirical: {q99_empirical:,.0f}")
print()
print("SPQRx Q99 > empirical Q99 is expected: the GPD formula extrapolates beyond")
print("the training data, which is exactly the point for XL reinsurance pricing.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC SPQRxSeverity provides:
# MAGIC
# MAGIC 1. **No threshold selection**: pa/pb are quantile levels, not absolute amounts.
# MAGIC    Interpretable for actuaries; no stability plot needed.
# MAGIC
# MAGIC 2. **Covariate-conditional xi**: catastrophic injury segment gets higher xi than
# MAGIC    soft tissue — actuarially correct and directly auditable via `tail_params()`.
# MAGIC
# MAGIC 3. **EVT-compliant extrapolation**: for tau >= pb, the GPD formula is exact.
# MAGIC    ILF curves beyond the observed data range are defensible.
# MAGIC
# MAGIC 4. **Standard API**: same `.fit()`, `.predict_distribution()`, `.ilf()` pattern
# MAGIC    as MDN and DRN — drop-in replacement for existing workflows.
# MAGIC
# MAGIC **When NOT to use SPQRx**:
# MAGIC - Data with policy limits (heterogeneous truncation): use TruncatedGPD instead.
# MAGIC - Very small n (< 300): use TruncatedGPD or WeibullTemperedPareto.
# MAGIC - Need regulatory sign-off on xi: consider the interpretable SPQRx-lite
# MAGIC   wrapper (Phase 46 Item 2) which gives a single global xi.
print("Demo complete.")
