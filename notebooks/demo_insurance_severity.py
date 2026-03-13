# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-severity: composite models + DRN demo
# MAGIC
# MAGIC This notebook demonstrates the full workflow for both subpackages:
# MAGIC - `insurance_severity.composite`: spliced severity models with covariate-dependent thresholds
# MAGIC - `insurance_severity.drn`: Distributional Refinement Network for full predictive distributions
# MAGIC
# MAGIC Run this on Databricks serverless compute.

# COMMAND ----------

import subprocess
import sys

subprocess.run([
    sys.executable, "-m", "pip", "install", "--quiet",
    "git+https://github.com/burning-cost/insurance-severity.git",
    "statsmodels>=0.14",
], check=True)

# COMMAND ----------

import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)

# ============================================================
# PART 1: Composite severity models
# ============================================================

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: Composite (spliced) severity models
# MAGIC
# MAGIC Composite models split claim severity at a threshold:
# MAGIC - Below threshold: body distribution (Lognormal or Gamma)
# MAGIC - Above threshold: tail distribution (Burr XII or GPD)
# MAGIC
# MAGIC The key feature: **covariate-dependent thresholds** via mode-matching.
# MAGIC Different policyholders get different thresholds based on their risk profile.

# COMMAND ----------

# --- Synthetic data: LognormalBurr composite ---
# Known parameters: sigma=1.2, alpha=2.5, delta=2.0, beta=10000
# Threshold ~ 4082 (Burr XII mode)

from insurance_severity.composite import LognormalBurrComposite

rng = np.random.default_rng(42)
alpha, delta, beta = 2.5, 2.0, 10_000.0
ratio = (delta - 1.0) / (alpha * delta + 1.0)
threshold_true = beta * ratio ** (1.0 / delta)
sigma = 1.2
mu = sigma**2 + np.log(threshold_true)
pi = 0.75

print(f"True threshold: {threshold_true:.1f}")
print(f"True pi (body weight): {pi}")

# Generate 1000 observations
n = 1000
n_body = int(n * pi)
n_tail = n - n_body

y_body = []
while len(y_body) < n_body:
    batch = stats.lognorm.rvs(s=sigma, scale=np.exp(mu), size=n_body * 3, random_state=rng)
    batch = batch[batch <= threshold_true]
    y_body.extend(batch[:n_body - len(y_body)])
y_body = np.array(y_body[:n_body])

y_tail = []
while len(y_tail) < n_tail:
    u = rng.uniform(size=n_tail * 3)
    x = beta * (u ** (-1.0/alpha) - 1.0) ** (1.0/delta)
    x = x[x > threshold_true]
    y_tail.extend(x[:n_tail - len(y_tail)])
y_tail = np.array(y_tail[:n_tail])

y = np.concatenate([y_body, y_tail])
rng.shuffle(y)

print(f"\nGenerated {len(y)} observations")
print(f"  Mean: {y.mean():.1f}")
print(f"  Median: {np.median(y):.1f}")
print(f"  99th pctile: {np.percentile(y, 99):.1f}")

# COMMAND ----------

# --- Fit LognormalBurr with mode-matching ---
model = LognormalBurrComposite(threshold_method="mode_matching")
model.fit(y)

print("Fitted model:")
print(f"  Estimated threshold: {model.threshold_:.1f}  (true: {threshold_true:.1f})")
print(f"  Body weight (pi): {model.pi_:.4f}  (true: {pi})")
print(f"  Body params (mu, sigma): {model.body_params_}")
print(f"  Tail params (alpha, delta, beta): {model.tail_params_}")
print(f"  Log-likelihood: {model.loglik_:.2f}")

# COMMAND ----------

# --- Risk measures ---
var_99 = model.var(0.99)
tvar_99 = model.tvar(0.99)
ilf_500k = model.ilf(limit=500_000, basic_limit=100_000)

print(f"99th percentile VaR: {var_99:,.0f}")
print(f"99th percentile TVaR (Expected Shortfall): {tvar_99:,.0f}")
print(f"ILF at 500k (basic limit 100k): {ilf_500k:.4f}")
print(f"  Meaning: a 500k limit costs {ilf_500k:.1%} more than 100k limit")

# COMMAND ----------

# --- ILF schedule ---
limits = [50_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000]
basic = 100_000
print(f"\nILF schedule (basic limit = {basic:,}):")
print(f"{'Limit':>12}  {'ILF':>8}")
for lim in limits:
    ilf = model.ilf(limit=lim, basic_limit=basic)
    print(f"  {lim:>10,}  {ilf:>8.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Composite regression: covariate-dependent thresholds
# MAGIC
# MAGIC Each policyholder gets their own threshold based on their risk profile.
# MAGIC The tail scale beta varies as: log(beta_i) = w0 + w1 * x1i + ...

# COMMAND ----------

from insurance_severity.composite import CompositeSeverityRegressor

# Generate data with a covariate that shifts the threshold
rng2 = np.random.default_rng(123)
n_reg = 400
x = rng2.normal(0, 1, n_reg)
log_beta = 8.5 + 0.3 * x  # true coefficient = 0.3
beta_arr = np.exp(log_beta)

y_reg = np.zeros(n_reg)
for i in range(n_reg):
    b_i = beta_arr[i]
    r_i = (delta - 1.0) / (alpha * delta + 1.0)
    t_i = b_i * r_i ** (1.0/delta)
    mu_i = sigma**2 + np.log(t_i)
    if rng2.random() < 0.75:
        while True:
            v = stats.lognorm.rvs(s=sigma, scale=np.exp(mu_i), random_state=rng2)
            if v <= t_i:
                y_reg[i] = v
                break
    else:
        u = rng2.random()
        S_t = (1.0 + (t_i / b_i) ** delta) ** (-alpha)
        S_x = u * S_t
        inner = max(S_x ** (-1.0/alpha) - 1.0, 1e-10)
        y_reg[i] = b_i * inner ** (1.0/delta)

X_reg = x.reshape(-1, 1)
print(f"Regression dataset: n={n_reg}, 1 covariate")
print(f"True coefficient on log(beta): 0.3")

# COMMAND ----------

reg = CompositeSeverityRegressor(
    composite=LognormalBurrComposite(threshold_method="mode_matching"),
    n_starts=2,
)
reg.fit(X_reg, y_reg)

print("Fitted regression:")
print(f"  Intercept: {reg.intercept_:.4f}  (true: 8.5)")
print(f"  Coefficient: {reg.coef_[0]:.4f}  (true: 0.3)")
print(f"  Shape params (alpha, delta): {reg.shape_params_}")
print(f"  Log-likelihood: {reg.loglik_:.2f}")

# COMMAND ----------

# Per-policyholder thresholds
X_test_reg = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
thresholds = reg.predict_thresholds(X_test_reg)
print("\nPer-policyholder thresholds:")
print(f"{'Covariate':>10}  {'Threshold':>12}  {'True threshold':>14}")
for xi, t_hat in zip(X_test_reg.ravel(), thresholds):
    b_true = np.exp(8.5 + 0.3 * xi)
    t_true = b_true * ratio ** (1.0/delta)
    print(f"  {xi:>8.1f}  {t_hat:>12,.1f}  {t_true:>14,.1f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Distributional Refinement Network (DRN)
# MAGIC
# MAGIC The DRN refines a GLM baseline into a full predictive distribution.
# MAGIC We use a mock GLM baseline here to avoid statsmodels dependency in the demo.

# COMMAND ----------

from insurance_severity.drn import DRN, ExtendedHistogramBatch, drn_cutpoints

# --- Synthetic Gamma severity data ---
n_drn = 2000
X_drn = pd.DataFrame({
    "age": np.random.uniform(20, 70, n_drn),
    "vehicle_age": np.random.uniform(0, 15, n_drn),
    "region": np.random.randint(0, 5, n_drn).astype(float),
})
log_mu = 7.5 + 0.005 * (X_drn["age"] - 40) - 0.02 * X_drn["vehicle_age"]
mu_true = np.exp(log_mu)
shape = 2.0
y_drn = np.random.gamma(shape=shape, scale=mu_true/shape)

print(f"DRN dataset: n={n_drn}")
print(f"  Mean severity: {y_drn.mean():.1f}")
print(f"  Std severity: {y_drn.std():.1f}")
print(f"  99th pctile: {np.percentile(y_drn, 99):.1f}")

# COMMAND ----------

# Mock GLM baseline — in production, use GLMBaseline(fitted_glm)
class MockGLMBaseline:
    distribution_family = "gamma"
    def __init__(self, mu_val, disp=0.5):
        self._mu = mu_val
        self._disp = disp
    def predict_params(self, X):
        return {"mu": np.full(len(X), self._mu), "dispersion": self._disp}
    def predict_cdf(self, X, cutpoints):
        params = self.predict_params(X)
        mu_2d = params["mu"][:, np.newaxis]
        alpha = 1.0 / self._disp
        scale = mu_2d * self._disp
        return stats.gamma.cdf(cutpoints[np.newaxis, :], a=alpha, scale=scale)

baseline = MockGLMBaseline(mu_val=float(mu_true.mean()))

# COMMAND ----------

# Fit DRN
drn = DRN(
    baseline=baseline,
    hidden_size=32,
    num_hidden_layers=2,
    max_epochs=15,   # short for demo — use 200+ in production
    patience=10,
    random_state=42,
    verbose=True,
)
drn.fit(X_drn, y_drn)

print(f"\nFitted DRN: {drn.n_bins} bins, cutpoints from {drn.cutpoints[0]:.1f} to {drn.cutpoints[-1]:.1f}")

# COMMAND ----------

# --- Predictions ---
X_test_drn = X_drn.head(5)

dist = drn.predict_distribution(X_test_drn)
print("Predictive distribution for 5 test observations:")
print(dist.summary())

# COMMAND ----------

# --- Key risk measures ---
means = drn.predict_mean(X_test_drn)
q995 = drn.predict_quantile(X_test_drn, quantiles=0.995)
es_995 = dist.expected_shortfall(alpha=0.995)

print("\nRisk measures for 5 obs:")
print(f"{'Obs':>4}  {'Mean':>10}  {'VaR 99.5%':>12}  {'ES 99.5%':>12}")
for i in range(5):
    print(f"  {i:>2}  {means[i]:>10,.1f}  {q995[i]:>12,.1f}  {es_995[i]:>12,.1f}")

# COMMAND ----------

# --- CRPS score ---
y_test_drn = y_drn[:5]
crps = dist.crps(y_test_drn)
print(f"\nCRPS for 5 obs: {crps}")
print(f"Mean CRPS: {crps.mean():.2f}")

# COMMAND ----------

# --- Adjustment factors (interpretability) ---
adj = drn.adjustment_factors(X_test_drn)
print(f"\nAdjustment factors shape: {adj.shape}")
print("First obs (bins where DRN > baseline have adj > 1):")
first_row = adj.head(1)
# Show which bins have largest adjustments
adj_np = adj.to_numpy()[0]
midpoints = 0.5 * (drn.cutpoints[:-1] + drn.cutpoints[1:])
top_idx = np.argsort(np.abs(adj_np - 1.0))[-5:][::-1]
print("  Top 5 bins by deviation from baseline:")
for idx in top_idx:
    print(f"    bin midpoint {midpoints[idx]:.1f}: adj={adj_np[idx]:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC Both subpackages demonstrated:
# MAGIC
# MAGIC **Composite:**
# MAGIC - `LognormalBurrComposite` with mode-matching threshold selection
# MAGIC - `CompositeSeverityRegressor` with covariate-dependent thresholds
# MAGIC - VaR, TVaR, ILF computation
# MAGIC
# MAGIC **DRN:**
# MAGIC - `DRN` fit, predict_distribution, predict_mean, predict_quantile
# MAGIC - `ExtendedHistogramBatch` for batch predictive distributions
# MAGIC - CRPS scoring, expected shortfall, adjustment factors

print("\n=== Demo complete ===")
print(f"Composite: LognormalBurrComposite threshold={model.threshold_:.0f}, ILF(500k)={ilf_500k:.3f}")
print(f"DRN: {drn.n_bins} bins, mean CRPS={crps.mean():.2f}")
