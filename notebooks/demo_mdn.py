# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-severity: Mixture Density Network (MDN) demo
# MAGIC
# MAGIC This notebook demonstrates the MDN subpackage for modelling multimodal
# MAGIC insurance severity distributions. The motivating example is escape-of-water
# MAGIC (EoW) claim severity, which has a characteristic three-mode structure:
# MAGIC
# MAGIC - Mode 1 (drying and minor strip-out): £1,500–£5,000
# MAGIC - Mode 2 (full reinstatement): £8,000–£20,000
# MAGIC - Mode 3 (major loss / alternative accommodation): £50,000+
# MAGIC
# MAGIC A Gamma GLM fits a single right-skewed mode and misses this structure
# MAGIC entirely. An MDN with K=3 lognormal components captures it directly.
# MAGIC
# MAGIC **Reference**: Bishop, C.M. (1994). 'Mixture Density Networks.'
# MAGIC Technical Report NCRG/94/004, Aston University.

# COMMAND ----------

import subprocess
import sys

subprocess.run([
    sys.executable, "-m", "pip", "install", "--quiet",
    "git+https://github.com/burning-cost/insurance-severity.git[mdn]",
], check=True)

# COMMAND ----------

import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)
print("Libraries loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Synthetic EoW severity dataset
# MAGIC
# MAGIC Generate 5,000 observations from a known 3-component lognormal mixture
# MAGIC where the mixing weights and component means depend on three covariates:
# MAGIC
# MAGIC - `sum_insured_std`: standardised sum insured (proxy for property value)
# MAGIC - `property_age_std`: standardised property age
# MAGIC - `is_listed`: binary flag for listed buildings
# MAGIC
# MAGIC The tail component weight (π_3) is driven primarily by `is_listed` and
# MAGIC `property_age_std` — something a Gamma GLM cannot capture because it has
# MAGIC no mechanism for covariate-dependent distribution shape.

# COMMAND ----------

def generate_eow_data(n: int = 5000, seed: int = 42) -> tuple:
    """
    Generate synthetic EoW severity data from a 3-component lognormal mixture.

    Returns
    -------
    X : pd.DataFrame, shape (n, 3)
    y : np.ndarray, shape (n,)
    pi_true : np.ndarray, shape (n, 3) — ground-truth mixing weights
    """
    rng = np.random.default_rng(seed)

    # Features
    sum_insured_std = rng.standard_normal(n)
    property_age_std = rng.standard_normal(n)
    is_listed = (rng.random(n) < 0.05).astype(float)   # 5% listed buildings

    X = pd.DataFrame({
        "sum_insured_std": sum_insured_std,
        "property_age_std": property_age_std,
        "is_listed": is_listed,
    })

    # Mixing weights: softmax of linear scores
    # Tail component (k=2) driven by listed and age
    scores = np.column_stack([
        0.5 * sum_insured_std,             # k=0: drying peak
        -0.2 * sum_insured_std,            # k=1: reinstatement peak
        -1.5 + 0.8 * property_age_std + 2.0 * is_listed,  # k=2: major loss
    ])
    exp_s = np.exp(scores - scores.max(axis=1, keepdims=True))
    pi_true = exp_s / exp_s.sum(axis=1, keepdims=True)

    # Component parameters (log-space)
    # Phase 1: emergency drying, £1,500-5,000
    mu0 = 7.8 + 0.15 * sum_insured_std    # log-mean ~ log(2,400)
    sigma0 = 0.5

    # Phase 2: full strip-out and reinstatement, £8,000-20,000
    mu1 = 9.4 + 0.25 * sum_insured_std    # log-mean ~ log(12,000)
    sigma1 = 0.6

    # Phase 3: major loss / total loss, £50,000+
    mu2 = 11.0 + 0.4 * property_age_std   # log-mean ~ log(60,000)
    sigma2 = 0.9

    mu = np.column_stack([mu0, mu1, mu2])
    sigma = np.array([[sigma0, sigma1, sigma2]] * n)

    # Sample component and then severity
    comp = np.array([rng.choice(3, p=pi_true[i]) for i in range(n)])
    log_y = mu[np.arange(n), comp] + sigma[np.arange(n), comp] * rng.standard_normal(n)
    y = np.exp(log_y)

    return X, y, pi_true


X, y, pi_true = generate_eow_data(n=5000, seed=42)

print(f"Dataset: n={len(y):,}, features={list(X.columns)}")
print(f"Severity summary:")
print(f"  Mean:   £{y.mean():,.0f}")
print(f"  Median: £{np.median(y):,.0f}")
print(f"  P75:    £{np.percentile(y, 75):,.0f}")
print(f"  P90:    £{np.percentile(y, 90):,.0f}")
print(f"  P99:    £{np.percentile(y, 99):,.0f}")
print(f"  Max:    £{y.max():,.0f}")
print(f"\nTrue mixing weight summary:")
print(f"  π_1 (drying):         mean={pi_true[:, 0].mean():.3f}")
print(f"  π_2 (reinstatement):  mean={pi_true[:, 1].mean():.3f}")
print(f"  π_3 (major loss):     mean={pi_true[:, 2].mean():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Fit the MDN

# COMMAND ----------

from insurance_severity.mdn import MDN

# K=3 maps naturally to the three EoW phases.
# hidden_size=64, num_hidden_layers=2 is a reasonable default for tabular data.
# max_epochs=200 with early stopping (patience=30) handles convergence.

mdn = MDN(
    n_components=3,
    hidden_size=64,
    num_hidden_layers=2,
    dropout_rate=0.1,
    lr=1e-3,
    batch_size=256,
    max_epochs=200,
    patience=30,
    random_state=42,
)

# Hold out 20% for evaluation
n_train = int(0.8 * len(X))
X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
y_train, y_test = y[:n_train], y[n_train:]
pi_test = pi_true[n_train:]

mdn.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Point predictions

# COMMAND ----------

y_pred = mdn.predict_mean(X_test)

# Compare predicted vs observed means
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
mae = np.mean(np.abs(y_pred - y_test))
print(f"Point prediction on test set (n={len(y_test):,}):")
print(f"  RMSE: £{rmse:,.0f}")
print(f"  MAE:  £{mae:,.0f}")
print(f"  True mean severity:      £{y_test.mean():,.0f}")
print(f"  Predicted mean severity: £{y_pred.mean():,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Full predictive distributions

# COMMAND ----------

dist = mdn.predict_distribution(X_test)

# Quantiles across the test set
q50 = dist.quantile(0.50)
q75 = dist.quantile(0.75)
q90 = dist.quantile(0.90)
q99 = dist.quantile(0.99)

print("Predictive distribution quantiles (test set):")
print(f"  Median (Q50): £{q50.mean():,.0f} (±{q50.std():,.0f})")
print(f"  Q75:          £{q75.mean():,.0f}")
print(f"  Q90:          £{q90.mean():,.0f}")
print(f"  Q99:          £{q99.mean():,.0f}")

# Empirical check
emp_q90 = np.percentile(y_test, 90)
mdn_q90_mean = q90.mean()
print(f"\n  Empirical Q90: £{emp_q90:,.0f}")
print(f"  MDN Q90 (mean across obs): £{mdn_q90_mean:,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Distributional scores

# COMMAND ----------

nll = mdn.score(X_test, y_test, metric="nll")
crps = mdn.score(X_test, y_test, metric="crps")

print(f"Distributional metrics (test set):")
print(f"  NLL:  {nll:.4f}")
print(f"  CRPS: £{crps:,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Recovered mixture parameters
# MAGIC
# MAGIC The key MDN advantage: the mixing weights π_k(x) are themselves functions
# MAGIC of features. We can check how well the model recovers the true mixing
# MAGIC weights on the test set.

# COMMAND ----------

pi_pred, mu_pred, sigma_pred = mdn.predict_params(X_test)

print("Recovered mixing weights vs ground truth (test set):")
for k in range(3):
    labels = ["drying", "reinstatement", "major loss"]
    print(f"\n  Component {k+1} ({labels[k]}):")
    print(f"    True π mean:      {pi_test[:, k].mean():.3f}")
    print(f"    Predicted π mean: {pi_pred[:, k].mean():.3f}")

# Correlation between true and predicted mixing weights
for k in range(3):
    labels = ["drying", "reinstatement", "major loss"]
    corr = np.corrcoef(pi_test[:, k], pi_pred[:, k])[0, 1]
    print(f"  Component {k+1} ({labels[k]}): correlation(true π, pred π) = {corr:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Listed buildings: tail weight comparison
# MAGIC
# MAGIC The true data-generating process has a higher tail weight (π_3) for listed
# MAGIC buildings. The MDN should recover this pattern.

# COMMAND ----------

is_listed_test = X_test["is_listed"].values.astype(bool)

print("Tail component weight (π_3) by building type:")
print(f"\n  True π_3:")
print(f"    Non-listed: {pi_test[~is_listed_test, 2].mean():.3f}")
print(f"    Listed:     {pi_test[is_listed_test, 2].mean():.3f}")
print(f"\n  Predicted π_3:")
print(f"    Non-listed: {pi_pred[~is_listed_test, 2].mean():.3f}")
print(f"    Listed:     {pi_pred[is_listed_test, 2].mean():.3f}")

if is_listed_test.sum() > 0:
    lift = (pi_pred[is_listed_test, 2].mean() /
            max(pi_pred[~is_listed_test, 2].mean(), 1e-8))
    print(f"\n  MDN-implied lift for listed buildings on π_3: {lift:.2f}x")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Increased limits factors (ILF)
# MAGIC
# MAGIC ILF(L, b) = E[min(Y, L)] / E[min(Y, b)].
# MAGIC The MDN computes these analytically from the mixture, enabling covariate-
# MAGIC dependent ILFs — something a Gamma GLM cannot produce.

# COMMAND ----------

ilf_10k_to_5k = dist.ilf(limit=10_000, basic_limit=5_000)
ilf_50k_to_5k = dist.ilf(limit=50_000, basic_limit=5_000)
ilf_250k_to_5k = dist.ilf(limit=250_000, basic_limit=5_000)

print("Covariate-dependent ILFs (test set):")
print(f"  ILF(£10k, £5k):  mean={ilf_10k_to_5k.mean():.3f}, range=[{ilf_10k_to_5k.min():.3f}, {ilf_10k_to_5k.max():.3f}]")
print(f"  ILF(£50k, £5k):  mean={ilf_50k_to_5k.mean():.3f}, range=[{ilf_50k_to_5k.min():.3f}, {ilf_50k_to_5k.max():.3f}]")
print(f"  ILF(£250k, £5k): mean={ilf_250k_to_5k.mean():.3f}, range=[{ilf_250k_to_5k.min():.3f}, {ilf_250k_to_5k.max():.3f}]")

# For listed buildings specifically
if is_listed_test.sum() > 0:
    print(f"\n  ILF(£250k, £5k) for listed buildings: {ilf_250k_to_5k[is_listed_test].mean():.3f}")
    print(f"  ILF(£250k, £5k) for non-listed:        {ilf_250k_to_5k[~is_listed_test].mean():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Calibration (PIT histogram)
# MAGIC
# MAGIC Probability integral transform values should be uniform on [0, 1] if the
# MAGIC model is well-calibrated. We print the mean and std as a quick check.

# COMMAND ----------

pit = dist.pit_samples(y_test)

print("PIT calibration check:")
print(f"  Mean PIT: {pit.mean():.3f} (ideal: 0.500)")
print(f"  Std PIT:  {pit.std():.3f} (ideal: {1/np.sqrt(12):.3f} for U[0,1])")

# Bucket into deciles and check uniformity
decile_counts = np.histogram(pit, bins=10, range=(0, 1))[0]
expected = len(pit) / 10
max_deviation = float(np.max(np.abs(decile_counts - expected)) / expected)
print(f"  Max decile deviation from uniform: {max_deviation*100:.1f}%")
print(f"  (Values under 20% indicate reasonable calibration for 5 training epochs)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Save and reload the model

# COMMAND ----------

import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "mdn_eow.pt"
    mdn.save(path)

    # Reload and verify predictions match
    from insurance_severity.mdn import MDN
    mdn_loaded = MDN.load(path)
    y_pred_loaded = mdn_loaded.predict_mean(X_test)
    max_diff = float(np.max(np.abs(y_pred - y_pred_loaded)))
    print(f"Save/load roundtrip max prediction difference: {max_diff:.2e}")
    print("Roundtrip OK" if max_diff < 1e-2 else "WARNING: predictions differ after reload")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Comparison: MDN vs Gamma GLM (point predictions)
# MAGIC
# MAGIC A simple Gamma GLM cannot capture the mixing weight structure.
# MAGIC On point predictions it may be competitive; the MDN advantage shows up
# MAGIC in distributional metrics (NLL, CRPS) and tail quantities (ILF, VaR).

# COMMAND ----------

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    df_train = X_train.copy()
    df_train["y"] = y_train

    # Fit Gamma GLM (log link)
    glm = smf.glm(
        "y ~ sum_insured_std + property_age_std + is_listed",
        data=df_train,
        family=sm.families.Gamma(sm.families.links.Log())
    ).fit(disp=False)

    df_test = X_test.copy()
    glm_pred = glm.predict(df_test)

    glm_rmse = float(np.sqrt(np.mean((glm_pred.values - y_test) ** 2)))
    glm_mae = float(np.mean(np.abs(glm_pred.values - y_test)))

    # GLM NLL (Gamma family)
    from scipy.stats import gamma as gamma_dist
    phi = 1.0 / glm.scale  # dispersion
    glm_shape = phi
    glm_scale = glm_pred.values / glm_shape
    glm_nll = float(-gamma_dist.logpdf(y_test, a=glm_shape, scale=glm_scale).mean())

    print("Point prediction comparison (test set):")
    print(f"  {'Metric':<10} {'Gamma GLM':>12} {'MDN (K=3)':>12}")
    print(f"  {'RMSE':.<10} £{glm_rmse:>10,.0f} £{rmse:>10,.0f}")
    print(f"  {'MAE':.<10} £{glm_mae:>10,.0f} £{mae:>10,.0f}")
    print(f"  {'NLL':.<10} {glm_nll:>12.4f} {nll:>12.4f}")

except ImportError:
    print("statsmodels not installed — skipping Gamma GLM comparison.")

# COMMAND ----------

print("\nMDN demo complete.")
print(f"Model: {mdn!r}")
print("Key outputs:")
print("  mdn.predict_mean(X)         -> point predictions")
print("  mdn.predict_distribution(X) -> MDNMixture with .mean()/.quantile()/.ilf()/...")
print("  mdn.predict_params(X)       -> raw (pi, mu, sigma) arrays")
print("  mdn.score(X, y, 'nll')      -> held-out NLL")
