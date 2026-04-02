# Databricks notebook source
# MAGIC %md
# MAGIC # PopulationSamplingReserve: AIPW doubly-robust IBNR reserving
# MAGIC
# MAGIC This notebook demonstrates the full AIPW reserving workflow from
# MAGIC Calcetero, Badescu, Lin (2025), arXiv:2502.15598.
# MAGIC
# MAGIC **The problem**: IBNR claim reserving using aggregate triangles ignores
# MAGIC per-claim information and assumes every accident period develops the same
# MAGIC way. Micro-level models do better but are biased: reported claims are a
# MAGIC non-representative sample because larger claims are reported faster. If you
# MAGIC fit a severity GLM on reported claims and project it to IBNR, you'll
# MAGIC overestimate reserves (your training data oversamples high-severity claims).
# MAGIC
# MAGIC **The solution**: AIPW (Augmented Inverse Probability Weighting) corrects
# MAGIC for this sampling bias using a doubly-robust estimator. You need either the
# MAGIC inclusion probability model or the severity model to be correct — not both.
# MAGIC
# MAGIC Run on Databricks serverless compute.

# COMMAND ----------

import subprocess
import sys

subprocess.run([
    sys.executable, "-m", "pip", "install", "--quiet",
    "git+https://github.com/burning-cost/insurance-severity.git",
], check=True)

# COMMAND ----------

import numpy as np
import pandas as pd
from scipy import stats

rng = np.random.default_rng(42)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Generate synthetic motor claims data
# MAGIC
# MAGIC We generate a realistic portfolio: 3 accident years, Weibull reporting
# MAGIC delays correlated with claim severity (large claims reported faster),
# MAGIC evaluated at a valuation date part-way through development.

# COMMAND ----------

def generate_claims(n_total: int, valuation_month: float, seed: int = 42):
    """
    Generate synthetic motor claims with known IBNR ground truth.

    Key feature: claim severity is negatively correlated with reporting delay
    (larger claims are reported faster). This is the sampling bias that AIPW
    corrects for.
    """
    rng = np.random.default_rng(seed)
    tau = valuation_month

    # Accident times spread over 3 years (0 to 36 months)
    t_acc = rng.uniform(0, tau * 0.85, n_total)

    # True claim severities: lognormal with covariates
    vehicle_age = rng.integers(0, 12, n_total).astype(float)
    coverage_type = rng.choice([0, 1, 2], n_total)  # 0=third party, 1=comp, 2=premium

    log_mu = (
        7.5  # ~£1,800 mean
        - 0.03 * vehicle_age
        + 0.4 * coverage_type
    )
    severity = rng.lognormal(mean=log_mu, sigma=0.8)

    # Reporting delay: Weibull with scale dependent on severity
    # Large claims → shorter delays (shape=1.5, scale decreases with severity)
    log_scale = 2.5 - 0.3 * np.log(severity / 2000.0)  # ~12 months baseline
    scale_delay = np.exp(log_scale)
    delay = rng.weibull(a=1.5, size=n_total) * scale_delay

    report_time = t_acc + delay

    # Reported = delay fits within observation window
    reported_mask = report_time <= tau

    df_all = pd.DataFrame({
        "accident_month": t_acc,
        "report_month": report_time,
        "severity": severity,
        "vehicle_age": vehicle_age,
        "coverage_type": coverage_type.astype(float),
        "reported": reported_mask,
    })

    df_reported = df_all[reported_mask].copy()
    true_ibnr = float(df_all.loc[~reported_mask, "severity"].sum())
    true_ultimate = float(df_all["severity"].sum())

    return df_reported, true_ibnr, true_ultimate, df_all


df_reported, true_ibnr, true_ultimate, df_all = generate_claims(
    n_total=2000, valuation_month=24.0
)

print(f"Total claims:     {len(df_all):,}")
print(f"Reported:         {len(df_reported):,} ({100*len(df_reported)/len(df_all):.1f}%)")
print(f"IBNR (unreported): {len(df_all) - len(df_reported):,}")
print(f"")
print(f"True reported losses: £{df_reported['severity'].sum():,.0f}")
print(f"True IBNR:            £{true_ibnr:,.0f}")
print(f"True ultimate:        £{true_ultimate:,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Demonstrate the sampling bias problem
# MAGIC
# MAGIC The reported claim distribution differs from the unreported one because
# MAGIC larger claims are reported faster.

# COMMAND ----------

reported_mean = df_all.loc[df_all["reported"], "severity"].mean()
unreported_mean = df_all.loc[~df_all["reported"], "severity"].mean()
all_mean = df_all["severity"].mean()

print("Mean severity comparison:")
print(f"  All claims:        £{all_mean:,.0f}")
print(f"  Reported claims:   £{reported_mean:,.0f}  ← biased upward")
print(f"  Unreported claims: £{unreported_mean:,.0f}")
print(f"  Ratio (rep/unrep): {reported_mean/unreported_mean:.2f}x")
print()
print("If you fit a model on reported claims and use it to predict IBNR,")
print(f"you'll overestimate IBNR severity by ~{100*(reported_mean/unreported_mean - 1):.0f}%.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Chain-ladder as baseline
# MAGIC
# MAGIC Chain-ladder using development factors is a special case of the IPW
# MAGIC estimator with π̂_i = 1/f_k.

# COMMAND ----------

from insurance_severity import PopulationSamplingReserve

# Compute empirical development factors by accident quarter
df_reported["accident_quarter"] = (df_reported["accident_month"] // 3).astype(int)

# For each quarter, compute f_k = ultimate / reported
# (in reality estimated from triangle; here we use actual data to set known f_k)
quarter_factors = {}
for q in sorted(df_reported["accident_quarter"].unique()):
    df_q_all = df_all[
        (df_all["accident_month"] // 3).astype(int) == q
    ]
    reported_q = df_q_all[df_q_all["reported"]]["severity"].sum()
    ultimate_q = df_q_all["severity"].sum()
    if reported_q > 0:
        quarter_factors[q] = float(ultimate_q / reported_q)

print("Development factors by accident quarter:")
for q, f in sorted(quarter_factors.items()):
    print(f"  Q{q}: f = {f:.3f}")

psr_cl = PopulationSamplingReserve(method="chain_ladder")
psr_cl.fit(
    df_reported,
    accident_time_col="accident_month",
    report_time_col="report_month",
    development_factors=quarter_factors,
    valuation_time=24.0,
)

print(f"\nChain-ladder IBNR:    £{psr_cl.estimate_ibnr():,.0f}")
print(f"True IBNR:            £{true_ibnr:,.0f}")
print(f"CL error:             {100*(psr_cl.estimate_ibnr()/true_ibnr - 1):+.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. IPW-only estimator
# MAGIC
# MAGIC Fits a Weibull reporting delay model automatically, then applies the
# MAGIC Horvitz-Thompson IPW correction.

# COMMAND ----------

psr_ipw = PopulationSamplingReserve(method="ipw")
psr_ipw.fit(
    df_reported,
    accident_time_col="accident_month",
    report_time_col="report_month",
    valuation_time=24.0,
)

print(f"IPW IBNR estimate:  £{psr_ipw.estimate_ibnr():,.0f}")
print(f"True IBNR:          £{true_ibnr:,.0f}")
print(f"IPW error:          {100*(psr_ipw.estimate_ibnr()/true_ibnr - 1):+.1f}%")
print()
diag = psr_ipw.diagnostics()
print(f"Estimated IBNR claim count: {diag['n_ibnr_estimated']:.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Full AIPW estimator
# MAGIC
# MAGIC Uses both inclusion probability model and a (biased) severity model.
# MAGIC The augmentation term corrects for the severity model's bias.
# MAGIC
# MAGIC We deliberately use a biased severity model (grand mean) to show that
# MAGIC AIPW corrects for it automatically.

# COMMAND ----------

# Biased severity model: predict the mean of reported claims for all
# (this will overestimate IBNR severity because reported mean > true IBNR mean)
class NaiveSeverityModel:
    """Predicts the mean of reported claims — biased upward."""
    def fit(self, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        n = len(X) if X is not None else 1
        return np.full(n, self.mean_)


naive_model = NaiveSeverityModel()
naive_model.fit(df_reported["severity"].to_numpy())
print(f"Naive model prediction (reported mean): £{naive_model.mean_:,.0f}")
print(f"True unreported mean:                   £{unreported_mean:,.0f}")
print(f"Bias: {100*(naive_model.mean_/unreported_mean - 1):+.1f}%")

psr_aipw = PopulationSamplingReserve(
    severity_model=naive_model,
    method="aipw",
)
psr_aipw.fit(
    df_reported,
    accident_time_col="accident_month",
    report_time_col="report_month",
    valuation_time=24.0,
)

diag = psr_aipw.diagnostics()
print()
print(f"AIPW IBNR estimate:     £{psr_aipw.estimate_ibnr():,.0f}")
print(f"True IBNR:              £{true_ibnr:,.0f}")
print(f"AIPW error:             {100*(psr_aipw.estimate_ibnr()/true_ibnr - 1):+.1f}%")
print()
print(f"Augmentation term:      £{diag['augmentation_term']:,.0f}  (bias correction)")
print(f"Weighted balance ratio: {diag['weighted_balance_ratio']:.3f} (1.0 = WBP satisfied)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Weighted balance property adjustment
# MAGIC
# MAGIC Instead of the full AIPW, apply a scalar b to the severity predictions.
# MAGIC One number corrects the mean bias in the severity model.

# COMMAND ----------

# Apply WBP scalar to the reported predictions
adjusted_preds = psr_aipw.weighted_balance_adjust(
    np.full(len(df_reported), naive_model.mean_)
)
b = diag["weighted_balance_ratio"]
print(f"Scalar b = {b:.4f}")
print(f"Original prediction: £{naive_model.mean_:,.0f}")
print(f"Adjusted prediction: £{adjusted_preds[0]:,.0f}  (= b × original)")
print()

# IBNR with WBP-adjusted severity
n_ibnr_est = diag["n_ibnr_estimated"]
ibnr_wbp = n_ibnr_est * float(adjusted_preds[0])
print(f"WBP-adjusted IBNR:  £{ibnr_wbp:,.0f}")
print(f"True IBNR:          £{true_ibnr:,.0f}")
print(f"WBP error:          {100*(ibnr_wbp/true_ibnr - 1):+.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Comparison summary

# COMMAND ----------

results = {
    "Naive micro (reported mean)": (n_ibnr_est * naive_model.mean_, None),
    "Chain-ladder": (psr_cl.estimate_ibnr(), None),
    "IPW only": (psr_ipw.estimate_ibnr(), None),
    "AIPW (doubly robust)": (psr_aipw.estimate_ibnr(), None),
    "WBP adjusted": (ibnr_wbp, None),
    "True IBNR": (true_ibnr, None),
}

print(f"{'Method':<30} {'IBNR £':>12} {'Error':>8}")
print("-" * 55)
for name, (est, _) in results.items():
    if name == "True IBNR":
        print(f"{'True IBNR':<30} {'£'+f'{est:,.0f}':>12} {'—':>8}")
    else:
        err = 100 * (est / true_ibnr - 1)
        print(f"{name:<30} {'£'+f'{est:,.0f}':>12} {err:>+7.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Using with sklearn severity model
# MAGIC
# MAGIC The inclusion_model and severity_model accept any sklearn-compatible
# MAGIC object. Here we use a GradientBoostingRegressor as the severity model
# MAGIC with per-claim features.

# COMMAND ----------

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import warnings

# Fit a GBM severity model on reported claims
feature_cols = ["vehicle_age", "coverage_type"]
X_rep = df_reported[feature_cols].to_numpy()
y_rep = df_reported["severity"].to_numpy()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    gbm = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )
    gbm.fit(X_rep, np.log(y_rep))  # fit on log-severity


# Wrapper to convert back from log scale
class LogGBM:
    def __init__(self, gbm):
        self.gbm = gbm

    def predict(self, X):
        return np.exp(self.gbm.predict(X))


psr_aipw_gbm = PopulationSamplingReserve(
    severity_model=LogGBM(gbm),
    method="aipw",
)
psr_aipw_gbm.fit(
    df_reported,
    accident_time_col="accident_month",
    report_time_col="report_month",
    severity_col="severity",
    feature_cols=feature_cols,
    valuation_time=24.0,
)

diag_gbm = psr_aipw_gbm.diagnostics()
print(f"AIPW + GBM severity IBNR:  £{psr_aipw_gbm.estimate_ibnr():,.0f}")
print(f"True IBNR:                 £{true_ibnr:,.0f}")
print(f"Error:                     {100*(psr_aipw_gbm.estimate_ibnr()/true_ibnr - 1):+.1f}%")
print()
print(f"Augmentation term:         £{diag_gbm['augmentation_term']:,.0f}")
print(f"Weighted balance ratio:    {diag_gbm['weighted_balance_ratio']:.3f}")
print()
print("The GBM is less biased than the naive mean, so the augmentation term")
print("is smaller — the GBM is closer to satisfying the weighted balance property.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC Key takeaways from this demonstration:
# MAGIC
# MAGIC 1. **Sampling bias is real**: reported claims have ~30% higher mean severity
# MAGIC    than unreported ones in this simulation. Any model fitted on reported
# MAGIC    claims alone will overestimate IBNR.
# MAGIC
# MAGIC 2. **Double robustness works**: AIPW with a biased severity model still
# MAGIC    produces a good IBNR estimate because the augmentation term corrects
# MAGIC    for the severity bias.
# MAGIC
# MAGIC 3. **Chain-ladder is IPW**: the CL estimate is recovered as a special case
# MAGIC    by supplying development factors as inclusion probabilities.
# MAGIC
# MAGIC 4. **WBP scalar is a useful shortcut**: the scalar b = Σ[(1−π)/π × Y] /
# MAGIC    Σ[(1−π)/π × Ŷ] adjusts any severity model to satisfy the weighted
# MAGIC    balance property. One number, large practical improvement.
# MAGIC
# MAGIC 5. **Plug in any sklearn model**: both the inclusion and severity models
# MAGIC    accept sklearn-compatible objects or plain callables.
