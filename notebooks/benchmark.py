# Databricks notebook source
# This file uses Databricks notebook format:
#   # COMMAND ----------  separates cells
#   # MAGIC %md           starts a markdown cell line
#
# Run end-to-end on Databricks. Do not run locally.

# COMMAND ----------

# MAGIC %md
# MAGIC # Benchmark: insurance-severity composite spliced model vs single Gamma GLM
# MAGIC
# MAGIC **Library:** `insurance-severity` — composite (spliced) severity models with separate
# MAGIC body and tail distributions, profile-likelihood threshold selection, and ILF computation
# MAGIC
# MAGIC **Baseline:** Single Gamma GLM (statsmodels) — the standard severity model used by
# MAGIC pricing actuaries. One distribution, one shape parameter, finite variance throughout.
# MAGIC
# MAGIC **Dataset:** 10,000 synthetic claims drawn from a known heavy-tailed DGP —
# MAGIC Lognormal body below the splice point, Pareto tail above it. We know the true
# MAGIC quantiles, so we can measure tail accuracy against ground truth.
# MAGIC
# MAGIC **Date:** 2026-03-13
# MAGIC
# MAGIC **Library version:** see pip output below
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC The standard severity model in UK general insurance is the Gamma GLM. It is
# MAGIC well-understood, easy to implement, and integrates naturally with Tweedie GLM
# MAGIC frameworks. Its limitation is structural: the Gamma is light-tailed. Its survival
# MAGIC function decays exponentially, which means it systematically underestimates the
# MAGIC probability of large claims.
# MAGIC
# MAGIC This matters for three specific applications:
# MAGIC
# MAGIC 1. **Increased Limit Factors (ILFs).** The ILF at a high limit is determined almost
# MAGIC    entirely by tail behaviour. A Gamma-based ILF is too low at high limits, which
# MAGIC    means you underprice excess layers or load large-loss provisions inadequately.
# MAGIC
# MAGIC 2. **Excess of loss (XL) reinsurance pricing.** The expected loss in a layer
# MAGIC    [attachment, limit] depends on the tail of the ground-up severity distribution.
# MAGIC    A light-tailed Gamma underestimates expected loss in layers with high attachments.
# MAGIC
# MAGIC 3. **Large loss loading in ground-up pricing.** If you model severity as Gamma and
# MAGIC    the true tail is Pareto-like, your expected cost at the 99th percentile is wrong
# MAGIC    by a material factor — not just a few percent.
# MAGIC
# MAGIC Composite spliced models address this directly: fit a standard body distribution
# MAGIC (Lognormal or Gamma) for the bulk of claims, and a heavy-tailed distribution (GPD,
# MAGIC Burr XII) above a data-driven threshold. The threshold is chosen by profile
# MAGIC likelihood — the value that maximises the composite log-likelihood over a grid.
# MAGIC
# MAGIC **Problem type:** ground-up severity modelling
# MAGIC
# MAGIC **Key metrics:** log-likelihood, AIC, tail quantile accuracy (90th / 95th / 99th),
# MAGIC ILF accuracy at policy limits, QQ plot visual, mean absolute error

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

# Install the library under test
%pip install git+https://github.com/burning-cost/insurance-severity.git

# Baseline dependency (Gamma GLM)
%pip install statsmodels

# Data, utilities, and visualisation
%pip install matplotlib seaborn pandas numpy scipy

# COMMAND ----------

# Restart Python after pip installs (required on Databricks)
dbutils.library.restartPython()

# COMMAND ----------

import time
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

# Library under test
from insurance_severity.composite import (
    LognormalGPDComposite,
    LognormalBurrComposite,
    GammaGPDComposite,
    qq_plot,
    density_overlay_plot,
    mean_excess_plot,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

print(f"Benchmark run at: {datetime.utcnow().isoformat()}Z")
print("Libraries loaded successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data: synthetic heavy-tailed claim severity

# COMMAND ----------

# MAGIC %md
# MAGIC We generate 10,000 claim amounts from a known mixture DGP:
# MAGIC
# MAGIC - **Body:** Lognormal(mu=9.5, sigma=1.2) — truncated below a splice threshold
# MAGIC - **Tail:** Generalised Pareto (xi=0.35, sigma=120,000) — above the splice threshold
# MAGIC - **Splice threshold:** 80th percentile of the pooled distribution (approximately £50,000)
# MAGIC - **Body weight:** pi = 0.80 (80% of claims below threshold)
# MAGIC
# MAGIC The tail shape xi=0.35 is typical of UK motor bodily injury and commercial
# MAGIC liability severity. It produces a finite mean and variance, but the distribution
# MAGIC is substantially heavier than Gamma or Lognormal throughout the upper tail.
# MAGIC
# MAGIC Because we know the true DGP, we can compute exact true quantiles and ILFs
# MAGIC and measure how far each model deviates from them.

# COMMAND ----------

RNG_SEED  = 42
N_CLAIMS  = 10_000
TRAIN_FRAC = 0.70

# --- True DGP parameters ---
TRUE_PI    = 0.80      # fraction of claims in body
TRUE_MU    = 9.5       # lognormal mu (log scale)
TRUE_SIGMA = 1.2       # lognormal sigma
TRUE_XI    = 0.35      # GPD tail shape (heavy tail)
TRUE_GPD_SIGMA = 120_000.0   # GPD scale

# We define the threshold as the 80th quantile of the lognormal body
# extended across the full range. In practice we set it explicitly to
# the lognormal 80th percentile (since pi = 0.80 means 80% of mass is body).
TRUE_THRESHOLD = np.exp(TRUE_MU + TRUE_SIGMA * stats.norm.ppf(TRUE_PI))
print(f"True splice threshold: £{TRUE_THRESHOLD:,.0f}")


def sample_dgp(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample from the composite Lognormal-GPD DGP.

    For each observation: with probability pi, draw from the truncated
    Lognormal(mu, sigma) body; with probability 1-pi, draw from the
    GPD(xi, sigma) tail shifted to the threshold.
    """
    n_body = rng.binomial(n, TRUE_PI)
    n_tail = n - n_body

    # Body: truncated lognormal at threshold
    # Draw from lognormal until within [0, threshold]
    body_samples = []
    while len(body_samples) < n_body:
        candidates = rng.lognormal(TRUE_MU, TRUE_SIGMA, n_body * 3)
        valid = candidates[candidates <= TRUE_THRESHOLD]
        body_samples.extend(valid.tolist())
    body_samples = np.array(body_samples[:n_body])

    # Tail: GPD exceedances above threshold
    tail_exceedances = stats.genpareto.rvs(
        c=TRUE_XI, scale=TRUE_GPD_SIGMA, size=n_tail, random_state=int(rng.integers(1e6))
    )
    tail_samples = TRUE_THRESHOLD + tail_exceedances

    return np.concatenate([body_samples, tail_samples])


rng = np.random.default_rng(RNG_SEED)
claims = sample_dgp(N_CLAIMS, rng)
claims = claims[claims > 0]   # defensive positive-only check

print(f"\nSynthetic dataset: {len(claims):,} claims")
print(f"  Mean:   £{np.mean(claims):,.0f}")
print(f"  Median: £{np.median(claims):,.0f}")
print(f"  95th:   £{np.percentile(claims, 95):,.0f}")
print(f"  99th:   £{np.percentile(claims, 99):,.0f}")
print(f"  Max:    £{np.max(claims):,.0f}")
print(f"  Fraction above threshold: {np.mean(claims > TRUE_THRESHOLD):.3f}  (true: {1-TRUE_PI:.3f})")

# COMMAND ----------

# Train / test split (random, since claims are iid — no temporal structure here)
n = len(claims)
idx = rng.permutation(n)
n_train = int(n * TRAIN_FRAC)

claims_train = claims[idx[:n_train]]
claims_test  = claims[idx[n_train:]]

print(f"Train: {len(claims_train):,} claims")
print(f"Test:  {len(claims_test):,} claims")

# COMMAND ----------

# MAGIC %md
# MAGIC ### True quantiles from the DGP
# MAGIC
# MAGIC We compute exact theoretical quantiles from the true composite distribution.
# MAGIC These are the targets each model is measured against. No simulation error.

# COMMAND ----------

def true_cdf(y: np.ndarray) -> np.ndarray:
    """CDF of the true composite Lognormal-GPD DGP."""
    y = np.asarray(y, dtype=float)
    result = np.zeros(len(y))
    in_body = y <= TRUE_THRESHOLD
    if np.any(in_body):
        # CDF of truncated lognormal: F_LN(y) / F_LN(threshold)
        F_t = stats.lognorm.cdf(TRUE_THRESHOLD, s=TRUE_SIGMA, scale=np.exp(TRUE_MU))
        result[in_body] = TRUE_PI * stats.lognorm.cdf(
            y[in_body], s=TRUE_SIGMA, scale=np.exp(TRUE_MU)
        ) / F_t
    if np.any(~in_body):
        # CDF of GPD tail shifted to threshold
        tail_cdf = stats.genpareto.cdf(
            y[~in_body] - TRUE_THRESHOLD, c=TRUE_XI, scale=TRUE_GPD_SIGMA
        )
        result[~in_body] = TRUE_PI + (1.0 - TRUE_PI) * tail_cdf
    return result


def true_ppf(q: float) -> float:
    """Quantile function of the true DGP — solved numerically."""
    from scipy.optimize import brentq
    if q <= TRUE_PI:
        # In body
        F_t = stats.lognorm.cdf(TRUE_THRESHOLD, s=TRUE_SIGMA, scale=np.exp(TRUE_MU))
        q_body = q / TRUE_PI
        return stats.lognorm.ppf(q_body * F_t, s=TRUE_SIGMA, scale=np.exp(TRUE_MU))
    else:
        # In tail
        q_tail = (q - TRUE_PI) / (1.0 - TRUE_PI)
        return TRUE_THRESHOLD + stats.genpareto.ppf(q_tail, c=TRUE_XI, scale=TRUE_GPD_SIGMA)


def true_lev(limit: float) -> float:
    """
    True Limited Expected Value: E[min(X, limit)].

    Integral representation: LEV(L) = integral_0^L S(x) dx
    where S(x) = 1 - F(x).
    """
    from scipy.integrate import quad
    def sf(x):
        return 1.0 - float(true_cdf(np.array([x]))[0])
    result, _ = quad(sf, 0.0, limit, limit=200)
    return result


def true_ilf(limit: float, basic_limit: float) -> float:
    """True ILF: LEV(limit) / LEV(basic_limit)."""
    return true_lev(limit) / true_lev(basic_limit)


# Compute reference quantiles
q_targets = [0.90, 0.95, 0.99]
true_quantiles = {q: true_ppf(q) for q in q_targets}
print("True DGP quantiles:")
for q, v in true_quantiles.items():
    print(f"  Q{q*100:.0f}: £{v:,.0f}")

# ILF reference limits (in £)
BASIC_LIMIT = 100_000
ILF_LIMITS  = [250_000, 500_000, 1_000_000, 2_000_000, 5_000_000]
true_ilfs   = {lim: true_ilf(lim, BASIC_LIMIT) for lim in ILF_LIMITS}

print(f"\nTrue ILFs (basic limit £{BASIC_LIMIT:,}):")
for lim, ilf in true_ilfs.items():
    print(f"  £{lim:>10,.0f}:  {ilf:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Baseline: Gamma GLM (statsmodels)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Baseline: Gamma GLM with log link
# MAGIC
# MAGIC We fit a Gamma GLM with log link and an intercept-only design matrix.
# MAGIC This is equivalent to estimating shape and scale parameters by MLE under the
# MAGIC Gamma family — the standard severity model in UK pricing practice.
# MAGIC
# MAGIC We use statsmodels rather than scipy.stats.gamma.fit() to stay consistent with
# MAGIC how this model actually appears in production pricing systems, where it forms
# MAGIC part of a GLM pipeline with rating factors. The intercept-only version here
# MAGIC isolates distributional fit from covariate modelling.
# MAGIC
# MAGIC The Gamma is fully characterised by two parameters: shape (alpha) and scale (theta).
# MAGIC Its survival function decays as approximately exp(-x/theta) for large x —
# MAGIC exponentially fast, not the power-law decay of heavy-tailed distributions.
# MAGIC For claims data with a Pareto tail, this means the Gamma cannot capture the
# MAGIC behaviour that matters most for reinsurance and large-loss loading.

# COMMAND ----------

t0_baseline = time.perf_counter()

# Intercept-only Gamma GLM: equivalent to global MLE of shape and rate
X_glm = np.ones((len(claims_train), 1))
gamma_glm = sm.GLM(
    claims_train,
    X_glm,
    family=sm.families.Gamma(link=sm.families.links.Log()),
)
gamma_result = gamma_glm.fit()

baseline_fit_time = time.perf_counter() - t0_baseline

# Extract fitted shape and scale
# statsmodels GLM: dispersion phi = 1/shape, fitted mean = exp(X @ beta) = exp(intercept)
phi_hat   = gamma_result.scale          # dispersion = 1 / shape
shape_hat = 1.0 / phi_hat              # Gamma shape alpha
mean_hat  = np.exp(gamma_result.params[0])  # fitted mean
scale_hat = mean_hat / shape_hat       # Gamma scale theta

print(f"Gamma GLM fit time: {baseline_fit_time:.3f}s")
print(f"  Fitted shape (alpha): {shape_hat:.4f}")
print(f"  Fitted scale (theta): £{scale_hat:,.2f}")
print(f"  Fitted mean:          £{mean_hat:,.0f}")
print(f"  Log-likelihood:       {gamma_result.llf:.2f}")
print(f"  AIC:                  {gamma_result.aic:.2f}")
print(f"  BIC:                  {gamma_result.bic:.2f}")

# scipy.stats.gamma object for easy CDF/PPF evaluation
baseline_dist = stats.gamma(a=shape_hat, scale=scale_hat)

# Sanity check: compare true mean with sample mean
print(f"\n  Sample mean (train):  £{claims_train.mean():,.0f}")
print(f"  Fitted mean:          £{mean_hat:,.0f}")

# COMMAND ----------

# Log-likelihood computed manually on the test set for fair comparison
def gamma_loglik(y: np.ndarray, shape: float, scale: float) -> float:
    """Total log-likelihood under Gamma(shape, scale)."""
    return float(np.sum(stats.gamma.logpdf(y, a=shape, scale=scale)))


# Also compute on train set to detect any overfitting pattern
ll_baseline_train = gamma_loglik(claims_train, shape_hat, scale_hat)
ll_baseline_test  = gamma_loglik(claims_test,  shape_hat, scale_hat)

print(f"Gamma GLM log-likelihood:")
print(f"  Train: {ll_baseline_train:.2f}  ({ll_baseline_train/len(claims_train):.4f} per obs)")
print(f"  Test:  {ll_baseline_test:.2f}  ({ll_baseline_test/len(claims_test):.4f} per obs)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Library: composite spliced model

# COMMAND ----------

# MAGIC %md
# MAGIC ### Library: LognormalGPDComposite with profile-likelihood threshold
# MAGIC
# MAGIC We fit a composite Lognormal-GPD model using `threshold_method="profile_likelihood"`.
# MAGIC The library searches over a grid of candidate thresholds (quantiles 70th–95th of
# MAGIC the training data) and selects the one that maximises the composite log-likelihood.
# MAGIC
# MAGIC Model structure:
# MAGIC - Below threshold: Lognormal(mu, sigma) truncated above at threshold
# MAGIC - Above threshold: Generalised Pareto(xi, sigma_gpd) on exceedances
# MAGIC - Mixing weight pi: estimated as n_body / n (MLE given other parameters)
# MAGIC
# MAGIC The profile likelihood approach is honest: it selects the threshold that best fits
# MAGIC the data, without requiring the user to specify it. For comparison we also fit
# MAGIC LognormalBurrComposite with mode-matching, which gives a C1-continuous splice
# MAGIC at the cost of a more complex optimisation.
# MAGIC
# MAGIC We deliberately do not use covariates in this benchmark. The benchmark question
# MAGIC is: does the distributional family matter? That is answered by a univariate fit.
# MAGIC Covariate-dependent thresholds are demonstrated separately via CompositeSeverityRegressor.

# COMMAND ----------

# --- Variant A: Lognormal-GPD, profile likelihood threshold ---
t0_ln_gpd = time.perf_counter()

model_ln_gpd = LognormalGPDComposite(
    threshold_method="profile_likelihood",
    n_threshold_grid=60,
    threshold_quantile_range=(0.70, 0.95),
)
model_ln_gpd.fit(claims_train)

ln_gpd_fit_time = time.perf_counter() - t0_ln_gpd

print("LognormalGPDComposite (profile likelihood)")
print(model_ln_gpd.summary(claims_train))
print(f"  Fit time: {ln_gpd_fit_time:.2f}s")

# COMMAND ----------

# --- Variant B: Lognormal-Burr, mode-matching threshold ---
t0_ln_burr = time.perf_counter()

model_ln_burr = LognormalBurrComposite(
    threshold_method="mode_matching",
    n_starts=5,
)
model_ln_burr.fit(claims_train)

ln_burr_fit_time = time.perf_counter() - t0_ln_burr

print("LognormalBurrComposite (mode-matching)")
print(model_ln_burr.summary(claims_train))
print(f"  Fit time: {ln_burr_fit_time:.2f}s")

# COMMAND ----------

# --- Variant C: Gamma-GPD, profile likelihood threshold ---
# This is the natural hybrid: body behaves like a Gamma GLM, tail follows GPD.
# If the body data genuinely follows Gamma, this will outperform Lognormal body.
# It gives us a fair comparison of "GLM family for body" vs the baseline.
t0_gamma_gpd = time.perf_counter()

model_gamma_gpd = GammaGPDComposite(
    threshold_method="profile_likelihood",
    n_threshold_grid=60,
    threshold_quantile_range=(0.70, 0.95),
)
model_gamma_gpd.fit(claims_train)

gamma_gpd_fit_time = time.perf_counter() - t0_gamma_gpd

print("GammaGPDComposite (profile likelihood)")
print(model_gamma_gpd.summary(claims_train))
print(f"  Fit time: {gamma_gpd_fit_time:.2f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Log-likelihood and information criteria

# COMMAND ----------

def composite_loglik_on_data(model, y: np.ndarray) -> float:
    """Sum of log-densities under the fitted composite model."""
    return float(np.sum(model.logpdf(y)))


ll_ln_gpd_train   = composite_loglik_on_data(model_ln_gpd,   claims_train)
ll_ln_gpd_test    = composite_loglik_on_data(model_ln_gpd,   claims_test)

ll_ln_burr_train  = composite_loglik_on_data(model_ln_burr,  claims_train)
ll_ln_burr_test   = composite_loglik_on_data(model_ln_burr,  claims_test)

ll_gamma_gpd_train = composite_loglik_on_data(model_gamma_gpd, claims_train)
ll_gamma_gpd_test  = composite_loglik_on_data(model_gamma_gpd, claims_test)

# AIC and BIC use the training log-likelihood (as defined by the model)
aic_baseline  = gamma_result.aic
bic_baseline  = gamma_result.bic
aic_ln_gpd    = model_ln_gpd.aic(claims_train)
bic_ln_gpd    = model_ln_gpd.bic(claims_train)
aic_ln_burr   = model_ln_burr.aic(claims_train)
bic_ln_burr   = model_ln_burr.bic(claims_train)
aic_gamma_gpd = model_gamma_gpd.aic(claims_train)
bic_gamma_gpd = model_gamma_gpd.bic(claims_train)

print("Log-likelihood comparison (per-observation, test set):")
print(f"  Baseline (Gamma GLM):   {ll_baseline_test/len(claims_test):>10.4f}")
print(f"  LognormalGPD:           {ll_ln_gpd_test/len(claims_test):>10.4f}")
print(f"  LognormalBurr:          {ll_ln_burr_test/len(claims_test):>10.4f}")
print(f"  GammaGPD:               {ll_gamma_gpd_test/len(claims_test):>10.4f}")

print("\nAIC (train, lower is better):")
print(f"  Baseline (Gamma GLM):   {aic_baseline:>12.2f}  (2 params)")
print(f"  LognormalGPD:           {aic_ln_gpd:>12.2f}  (5 params)")
print(f"  LognormalBurr:          {aic_ln_burr:>12.2f}  (6 params)")
print(f"  GammaGPD:               {aic_gamma_gpd:>12.2f}  (5 params)")

print("\nBIC (train, lower is better):")
print(f"  Baseline (Gamma GLM):   {bic_baseline:>12.2f}")
print(f"  LognormalGPD:           {bic_ln_gpd:>12.2f}")
print(f"  LognormalBurr:          {bic_ln_burr:>12.2f}")
print(f"  GammaGPD:               {bic_gamma_gpd:>12.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Tail quantile accuracy

# COMMAND ----------

# MAGIC %md
# MAGIC ### Measuring tail accuracy against the known DGP
# MAGIC
# MAGIC Each model predicts a quantile at the 90th, 95th, and 99th percentile.
# MAGIC We compare against the true theoretical quantiles from the DGP.
# MAGIC
# MAGIC The metric is **relative error**: (model_quantile - true_quantile) / true_quantile.
# MAGIC Negative means the model underestimates severity at that percentile.
# MAGIC For reinsurance and large-loss loading, systematic underestimation is the
# MAGIC dangerous direction — you are pricing insufficiently for events that do occur.
# MAGIC
# MAGIC We also compute the **empirical quantile** from the test set as a sanity check.
# MAGIC The empirical quantile should be close to the true quantile given n=3,000 test claims.

# COMMAND ----------

def relative_error(predicted: float, true: float) -> float:
    """(predicted - true) / true, as a percentage."""
    if true == 0:
        return float("nan")
    return (predicted - true) / abs(true) * 100.0


quantile_rows = []
for q in q_targets:
    v_true     = true_quantiles[q]
    v_empirical = float(np.quantile(claims_test, q))
    v_baseline  = float(baseline_dist.ppf(q))
    v_ln_gpd    = float(model_ln_gpd.ppf(np.array([q]))[0])
    v_ln_burr   = float(model_ln_burr.ppf(np.array([q]))[0])
    v_gamma_gpd = float(model_gamma_gpd.ppf(np.array([q]))[0])

    quantile_rows.append({
        "Quantile":        f"Q{q*100:.0f}",
        "True (DGP)":      f"£{v_true:,.0f}",
        "Empirical":       f"£{v_empirical:,.0f}",
        "Gamma GLM":       f"£{v_baseline:,.0f}  ({relative_error(v_baseline, v_true):+.1f}%)",
        "LognormalGPD":    f"£{v_ln_gpd:,.0f}  ({relative_error(v_ln_gpd, v_true):+.1f}%)",
        "LognormalBurr":   f"£{v_ln_burr:,.0f}  ({relative_error(v_ln_burr, v_true):+.1f}%)",
        "GammaGPD":        f"£{v_gamma_gpd:,.0f}  ({relative_error(v_gamma_gpd, v_true):+.1f}%)",
    })

quantile_df = pd.DataFrame(quantile_rows).set_index("Quantile")
print("Tail quantile accuracy (relative error vs true DGP):")
print(quantile_df.to_string())

# COMMAND ----------

# Numeric-only table for downstream use
q_numeric_rows = []
for q in q_targets:
    v_true      = true_quantiles[q]
    v_baseline  = float(baseline_dist.ppf(q))
    v_ln_gpd    = float(model_ln_gpd.ppf(np.array([q]))[0])
    v_ln_burr   = float(model_ln_burr.ppf(np.array([q]))[0])
    v_gamma_gpd = float(model_gamma_gpd.ppf(np.array([q]))[0])
    q_numeric_rows.append({
        "q":            q,
        "true":         v_true,
        "baseline":     v_baseline,
        "ln_gpd":       v_ln_gpd,
        "ln_burr":      v_ln_burr,
        "gamma_gpd":    v_gamma_gpd,
        "err_baseline_pct":  relative_error(v_baseline, v_true),
        "err_ln_gpd_pct":    relative_error(v_ln_gpd, v_true),
        "err_ln_burr_pct":   relative_error(v_ln_burr, v_true),
        "err_gamma_gpd_pct": relative_error(v_gamma_gpd, v_true),
    })
q_num = pd.DataFrame(q_numeric_rows).set_index("q")

worst_baseline = q_num["err_baseline_pct"].abs().max()
worst_ln_gpd   = q_num["err_ln_gpd_pct"].abs().max()
worst_ln_burr  = q_num["err_ln_burr_pct"].abs().max()
worst_gamma_gpd = q_num["err_gamma_gpd_pct"].abs().max()

print(f"\nMax absolute tail quantile error across Q90/Q95/Q99:")
print(f"  Gamma GLM:     {worst_baseline:.1f}%")
print(f"  LognormalGPD:  {worst_ln_gpd:.1f}%")
print(f"  LognormalBurr: {worst_ln_burr:.1f}%")
print(f"  GammaGPD:      {worst_gamma_gpd:.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. ILF accuracy

# COMMAND ----------

# MAGIC %md
# MAGIC ### Increased Limit Factors
# MAGIC
# MAGIC The ILF is the ratio of the Limited Expected Value at a given limit to the
# MAGIC LEV at the basic limit:
# MAGIC
# MAGIC     ILF(L) = E[min(X, L)] / E[min(X, basic_limit)]
# MAGIC
# MAGIC ILFs determine excess-of-basic layer premiums and are used directly in
# MAGIC XL reinsurance pricing. A model that underestimates the tail will produce
# MAGIC ILFs that are too low at high limits — systematically underpricing high-limit
# MAGIC business relative to basic-limit business.
# MAGIC
# MAGIC We compare model ILFs against the true DGP ILFs at limits from £250k to £5m,
# MAGIC with a basic limit of £100k. These are realistic values for UK commercial liability
# MAGIC and motor BI pricing.

# COMMAND ----------

ilf_rows = []

for lim in ILF_LIMITS:
    v_true       = true_ilfs[lim]
    v_baseline   = float(baseline_dist.expect(lambda x: np.minimum(x, lim))) / \
                   float(baseline_dist.expect(lambda x: np.minimum(x, BASIC_LIMIT)))
    v_ln_gpd     = float(model_ln_gpd.ilf(lim, BASIC_LIMIT))
    v_ln_burr    = float(model_ln_burr.ilf(lim, BASIC_LIMIT))
    v_gamma_gpd  = float(model_gamma_gpd.ilf(lim, BASIC_LIMIT))

    ilf_rows.append({
        "Limit":         f"£{lim:>12,.0f}",
        "True ILF":      f"{v_true:.4f}",
        "Gamma GLM":     f"{v_baseline:.4f}  ({relative_error(v_baseline, v_true):+.1f}%)",
        "LognormalGPD":  f"{v_ln_gpd:.4f}  ({relative_error(v_ln_gpd, v_true):+.1f}%)",
        "LognormalBurr": f"{v_ln_burr:.4f}  ({relative_error(v_ln_burr, v_true):+.1f}%)",
        "GammaGPD":      f"{v_gamma_gpd:.4f}  ({relative_error(v_gamma_gpd, v_true):+.1f}%)",
    })

    ilf_numeric_row = {
        "limit": lim,
        "true": v_true,
        "baseline": v_baseline,
        "ln_gpd": v_ln_gpd,
        "ln_burr": v_ln_burr,
        "gamma_gpd": v_gamma_gpd,
    }
    if lim == ILF_LIMITS[0]:
        ilf_numeric = []
    ilf_numeric.append(ilf_numeric_row)

ilf_df = pd.DataFrame(ilf_rows).set_index("Limit")
print(f"ILF comparison (basic limit £{BASIC_LIMIT:,}, relative error vs true DGP):")
print(ilf_df.to_string())

ilf_num = pd.DataFrame(ilf_numeric)

# Max ILF error at the highest limit (£5m) — most sensitive to tail
v_true_5m = true_ilfs[5_000_000]
print(f"\nILF error at £5m limit:")
print(f"  True ILF:       {v_true_5m:.4f}")
print(f"  Gamma GLM:      {ilf_num.loc[ilf_num['limit']==5_000_000, 'baseline'].values[0]:.4f}  ({relative_error(ilf_num.loc[ilf_num['limit']==5_000_000, 'baseline'].values[0], v_true_5m):+.1f}%)")
print(f"  LognormalGPD:   {ilf_num.loc[ilf_num['limit']==5_000_000, 'ln_gpd'].values[0]:.4f}  ({relative_error(ilf_num.loc[ilf_num['limit']==5_000_000, 'ln_gpd'].values[0], v_true_5m):+.1f}%)")
print(f"  GammaGPD:       {ilf_num.loc[ilf_num['limit']==5_000_000, 'gamma_gpd'].values[0]:.4f}  ({relative_error(ilf_num.loc[ilf_num['limit']==5_000_000, 'gamma_gpd'].values[0], v_true_5m):+.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Summary metrics table

# COMMAND ----------

def winner_label(values: dict, lower_is_better: bool = True) -> str:
    """Return the key with the best value."""
    if lower_is_better:
        return min(values, key=values.get)
    else:
        return max(values, key=values.get)


summary_rows = [
    {
        "Metric":        "Test log-lik per obs",
        "Baseline":      f"{ll_baseline_test/len(claims_test):.4f}",
        "LognormalGPD":  f"{ll_ln_gpd_test/len(claims_test):.4f}",
        "LognormalBurr": f"{ll_ln_burr_test/len(claims_test):.4f}",
        "GammaGPD":      f"{ll_gamma_gpd_test/len(claims_test):.4f}",
        "Winner":        winner_label({
            "Baseline":      ll_baseline_test/len(claims_test),
            "LognormalGPD":  ll_ln_gpd_test/len(claims_test),
            "LognormalBurr": ll_ln_burr_test/len(claims_test),
            "GammaGPD":      ll_gamma_gpd_test/len(claims_test),
        }, lower_is_better=False),
        "Note": "Higher is better",
    },
    {
        "Metric":        "AIC (train)",
        "Baseline":      f"{aic_baseline:.1f}",
        "LognormalGPD":  f"{aic_ln_gpd:.1f}",
        "LognormalBurr": f"{aic_ln_burr:.1f}",
        "GammaGPD":      f"{aic_gamma_gpd:.1f}",
        "Winner":        winner_label({
            "Baseline":      aic_baseline,
            "LognormalGPD":  aic_ln_gpd,
            "LognormalBurr": aic_ln_burr,
            "GammaGPD":      aic_gamma_gpd,
        }),
        "Note": "Lower is better",
    },
    {
        "Metric":        "Max tail error (Q90-Q99, %)",
        "Baseline":      f"{worst_baseline:.1f}%",
        "LognormalGPD":  f"{worst_ln_gpd:.1f}%",
        "LognormalBurr": f"{worst_ln_burr:.1f}%",
        "GammaGPD":      f"{worst_gamma_gpd:.1f}%",
        "Winner":        winner_label({
            "Baseline":      worst_baseline,
            "LognormalGPD":  worst_ln_gpd,
            "LognormalBurr": worst_ln_burr,
            "GammaGPD":      worst_gamma_gpd,
        }),
        "Note": "Lower is better",
    },
    {
        "Metric":        "Fit time (s)",
        "Baseline":      f"{baseline_fit_time:.3f}",
        "LognormalGPD":  f"{ln_gpd_fit_time:.2f}",
        "LognormalBurr": f"{ln_burr_fit_time:.2f}",
        "GammaGPD":      f"{gamma_gpd_fit_time:.2f}",
        "Winner":        "Baseline",
        "Note": "GLM solves in milliseconds; composite optimizes by grid",
    },
    {
        "Metric":        "No. parameters",
        "Baseline":      "2",
        "LognormalGPD":  "5",
        "LognormalBurr": "6",
        "GammaGPD":      "5",
        "Winner":        "Baseline",
        "Note": "Composite adds threshold + tail shape parameters",
    },
]

summary_df = pd.DataFrame(summary_rows).set_index("Metric")
print("Head-to-head summary:")
print(summary_df[["Baseline", "LognormalGPD", "LognormalBurr", "GammaGPD", "Winner"]].to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Visualisation

# COMMAND ----------

# MAGIC %md
# MAGIC Four diagnostic plots:
# MAGIC
# MAGIC 1. **Density overlay (log scale):** fitted densities overlaid on the empirical histogram.
# MAGIC    Log scale makes tail differences visible — if the fitted density bends away from the
# MAGIC    data in the upper tail, the model is underspecified there.
# MAGIC
# MAGIC 2. **Tail survival comparison:** log S(x) vs x for each model and the true DGP.
# MAGIC    A Gamma has a parabolic log-survival (normal on log scale); a GPD tail has a
# MAGIC    linear log-survival on a log-log scale. The curvature of the Gamma survival
# MAGIC    function at high claims is the structural failure mode.
# MAGIC
# MAGIC 3. **QQ plot:** theoretical vs empirical quantiles. Points below the diagonal
# MAGIC    in the upper tail mean the model underestimates large claims.
# MAGIC
# MAGIC 4. **ILF comparison:** model ILF schedules vs the true DGP ILF at limits up to £5m.
# MAGIC    Divergence at high limits shows where Gamma-based pricing goes wrong for
# MAGIC    high-limit accounts or XL layers.

# COMMAND ----------

fig = plt.figure(figsize=(18, 16))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

ax1 = fig.add_subplot(gs[0, 0])   # Density overlay
ax2 = fig.add_subplot(gs[0, 1])   # Tail survival functions
ax3 = fig.add_subplot(gs[1, 0])   # QQ plot
ax4 = fig.add_subplot(gs[1, 1])   # ILF comparison


# ── Plot 1: Density overlay on test claims ────────────────────────────────
x_lo = np.percentile(claims_test, 0.5)
x_hi = np.percentile(claims_test, 99.2)
x_plot = np.linspace(x_lo, x_hi, 600)

ax1.hist(claims_test, bins=60, density=True, alpha=0.35, color="steelblue", label="Test data")

# True DGP density
true_dens = np.gradient(true_cdf(x_plot), x_plot)
true_dens = np.maximum(true_dens, 1e-15)
ax1.plot(x_plot, true_dens, "k-", lw=2.5, label="True DGP", zorder=5)

# Baseline
ax1.plot(x_plot, baseline_dist.pdf(x_plot), "b--", lw=1.8, label="Gamma GLM", alpha=0.9)

# LognormalGPD
ax1.plot(x_plot, model_ln_gpd.pdf(x_plot), "r-",  lw=1.8, label="LognormalGPD", alpha=0.9)

# GammaGPD
ax1.plot(x_plot, model_gamma_gpd.pdf(x_plot), "g-.", lw=1.8, label="GammaGPD", alpha=0.9)

ax1.axvline(model_ln_gpd.threshold_, ls=":", color="red", alpha=0.5, lw=1,
            label=f"LN-GPD threshold £{model_ln_gpd.threshold_:,.0f}")
ax1.set_yscale("log")
ax1.set_xlabel("Claim amount (£)")
ax1.set_ylabel("Density (log scale)")
ax1.set_title("Density overlay — log scale\nGamma bends away from data in upper tail", fontsize=10)
ax1.legend(fontsize=7, loc="upper right")
ax1.grid(True, alpha=0.2)


# ── Plot 2: Log-survival function comparison ─────────────────────────────
# Range: from threshold to 99.5th percentile of test claims
x_tail = np.linspace(TRUE_THRESHOLD, np.percentile(claims_test, 99.5), 400)

sf_true     = 1.0 - true_cdf(x_tail)
sf_baseline = baseline_dist.sf(x_tail)
sf_ln_gpd   = model_ln_gpd.sf(x_tail)
sf_ln_burr  = model_ln_burr.sf(x_tail)
sf_gamma_gpd = model_gamma_gpd.sf(x_tail)

# Guard against -inf in log
eps = 1e-20

ax2.semilogy(x_tail, np.maximum(sf_true,      eps), "k-",  lw=2.5, label="True DGP")
ax2.semilogy(x_tail, np.maximum(sf_baseline,  eps), "b--", lw=1.8, label="Gamma GLM", alpha=0.9)
ax2.semilogy(x_tail, np.maximum(sf_ln_gpd,    eps), "r-",  lw=1.8, label="LognormalGPD", alpha=0.9)
ax2.semilogy(x_tail, np.maximum(sf_ln_burr,   eps), "m-.", lw=1.5, label="LognormalBurr", alpha=0.8)
ax2.semilogy(x_tail, np.maximum(sf_gamma_gpd, eps), "g-.", lw=1.5, label="GammaGPD", alpha=0.8)

ax2.set_xlabel("Claim amount (£)")
ax2.set_ylabel("Survival function S(x)  [log scale]")
ax2.set_title("Tail survival: Gamma decays too fast\nComposite tracks the heavy tail", fontsize=10)
ax2.legend(fontsize=7)
ax2.grid(True, alpha=0.2)


# ── Plot 3: QQ plot — composite vs baseline against empirical ─────────────
y_sorted = np.sort(claims_test)
n_test   = len(y_sorted)
probs    = (np.arange(1, n_test + 1) - 0.5) / n_test

# Subsample for readability
idx_q = np.round(np.linspace(0, n_test - 1, 300)).astype(int)
y_sub    = y_sorted[idx_q]
probs_sub = probs[idx_q]

q_baseline  = baseline_dist.ppf(probs_sub)
q_ln_gpd    = model_ln_gpd.ppf(probs_sub)
q_gamma_gpd = model_gamma_gpd.ppf(probs_sub)

ax3.scatter(q_baseline,  y_sub, s=8,  color="blue",  alpha=0.5, label="Gamma GLM")
ax3.scatter(q_ln_gpd,    y_sub, s=8,  color="red",   alpha=0.5, label="LognormalGPD")
ax3.scatter(q_gamma_gpd, y_sub, s=8,  color="green", alpha=0.5, label="GammaGPD")

lim_max = max(y_sub.max(), q_ln_gpd.max(), q_baseline.max(), q_gamma_gpd.max()) * 1.05
lim_min = 0
ax3.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=1.5, label="y = x (perfect)")
ax3.set_xlabel("Theoretical quantile (model)")
ax3.set_ylabel("Empirical quantile (test data)")
ax3.set_title("Q-Q plot\nPoints below y=x: model underestimates severity", fontsize=10)
ax3.legend(fontsize=7)
ax3.grid(True, alpha=0.2)
# Focus on the upper range where differences are visible
upper_cutoff = np.percentile(claims_test, 98)
ax3.set_xlim(0, upper_cutoff * 1.1)
ax3.set_ylim(0, upper_cutoff * 1.1)


# ── Plot 4: ILF comparison ────────────────────────────────────────────────
lim_values = np.array(ILF_LIMITS, dtype=float)
ilf_true_arr     = np.array([true_ilfs[lim] for lim in ILF_LIMITS])
ilf_baseline_arr = np.array([
    float(baseline_dist.expect(lambda x, L=lim: np.minimum(x, L))) /
    float(baseline_dist.expect(lambda x: np.minimum(x, BASIC_LIMIT)))
    for lim in ILF_LIMITS
])
ilf_ln_gpd_arr    = np.array([float(model_ln_gpd.ilf(lim, BASIC_LIMIT))    for lim in ILF_LIMITS])
ilf_gamma_gpd_arr = np.array([float(model_gamma_gpd.ilf(lim, BASIC_LIMIT)) for lim in ILF_LIMITS])

ax4.plot(lim_values / 1e6, ilf_true_arr,     "k-o",  lw=2.5, ms=7, label="True DGP", zorder=5)
ax4.plot(lim_values / 1e6, ilf_baseline_arr, "b--^", lw=1.8, ms=6, label="Gamma GLM", alpha=0.9)
ax4.plot(lim_values / 1e6, ilf_ln_gpd_arr,   "r-s",  lw=1.8, ms=6, label="LognormalGPD", alpha=0.9)
ax4.plot(lim_values / 1e6, ilf_gamma_gpd_arr,"g-D",  lw=1.8, ms=6, label="GammaGPD", alpha=0.9)

ax4.set_xlabel("Policy limit (£m)")
ax4.set_ylabel("ILF")
ax4.set_title(f"ILF comparison (basic limit £{BASIC_LIMIT:,})\nGamma ILFs collapse at high limits", fontsize=10)
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.2)
ax4.axhline(1.0, ls=":", color="gray", alpha=0.5)


plt.suptitle(
    "insurance-severity: Composite spliced model vs Gamma GLM\n"
    "10,000 synthetic claims, Lognormal-Pareto DGP (xi=0.35), profile-likelihood threshold",
    fontsize=13,
    fontweight="bold",
    y=1.01,
)
plt.savefig("/tmp/benchmark_severity.png", dpi=120, bbox_inches="tight")
plt.show()
print("Plot saved to /tmp/benchmark_severity.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Mean excess plot and threshold diagnostics

# COMMAND ----------

# MAGIC %md
# MAGIC The mean excess plot is the standard diagnostic for choosing a threshold when
# MAGIC fitting a GPD tail. Above the true threshold, the mean excess function should be
# MAGIC approximately linear (for GPD). Below it, the body distribution dominates and the
# MAGIC function may behave differently.
# MAGIC
# MAGIC We overlay the profile-likelihood selected threshold to verify it falls in a
# MAGIC sensible region. A well-chosen threshold should sit at the start of the linear
# MAGIC regime in the mean excess plot.

# COMMAND ----------

fig_mex, axes_mex = plt.subplots(1, 2, figsize=(16, 5))

# Mean excess plot for the full training data
ax_mex = mean_excess_plot(claims_train, ax=axes_mex[0])
ax_mex.axvline(
    model_ln_gpd.threshold_,
    color="red", lw=2, ls="--",
    label=f"LN-GPD threshold (profile LL): £{model_ln_gpd.threshold_:,.0f}"
)
ax_mex.axvline(
    TRUE_THRESHOLD,
    color="black", lw=2, ls="-",
    label=f"True threshold: £{TRUE_THRESHOLD:,.0f}"
)
ax_mex.legend(fontsize=8)
ax_mex.set_title("Mean excess plot with profile-likelihood threshold\nLinear regime above true threshold confirms GPD tail", fontsize=9)

# Profile likelihood surface: log-likelihood vs threshold grid
# Recompute manually to display the profile LL curve
from insurance_severity.composite.distributions import LognormalBody, GPDTail

q_lo, q_hi = 0.70, 0.95
t_grid = np.quantile(claims_train, np.linspace(q_lo, q_hi, 60))
t_grid = np.unique(t_grid)

profile_lls = []
for t in t_grid:
    y_body = claims_train[claims_train <= t]
    y_tail = claims_train[claims_train > t]
    if len(y_body) < 10 or len(y_tail) < 30:
        profile_lls.append(np.nan)
        continue
    try:
        body = LognormalBody()
        tail = GPDTail()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            body.fit_mle(y_body, t)
            tail.fit_mle(y_tail, t)
        pi = len(y_body) / len(claims_train)
        in_body = claims_train <= t
        ll = 0.0
        ll += np.sum(np.log(pi) + body.logpdf(claims_train[in_body], t))
        ll += np.sum(np.log(1.0 - pi) + tail.logpdf(claims_train[~in_body], t))
        profile_lls.append(ll if np.isfinite(ll) else np.nan)
    except Exception:
        profile_lls.append(np.nan)

profile_lls = np.array(profile_lls, dtype=float)

axes_mex[1].plot(t_grid / 1000, profile_lls, "b-", lw=2)
axes_mex[1].axvline(
    model_ln_gpd.threshold_ / 1000,
    color="red", lw=2, ls="--",
    label=f"Selected: £{model_ln_gpd.threshold_:,.0f}"
)
axes_mex[1].axvline(
    TRUE_THRESHOLD / 1000,
    color="black", lw=2, ls="-",
    label=f"True: £{TRUE_THRESHOLD:,.0f}"
)
axes_mex[1].set_xlabel("Threshold (£ thousands)")
axes_mex[1].set_ylabel("Profile log-likelihood")
axes_mex[1].set_title("Profile likelihood over threshold grid\nSelected threshold maximises composite LL", fontsize=9)
axes_mex[1].legend(fontsize=8)
axes_mex[1].grid(True, alpha=0.3)

plt.suptitle("Threshold diagnostics: LognormalGPD composite", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("/tmp/benchmark_severity_threshold.png", dpi=120, bbox_inches="tight")
plt.show()
print("Threshold diagnostics saved to /tmp/benchmark_severity_threshold.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Verdict

# COMMAND ----------

# MAGIC %md
# MAGIC ### When to use composite spliced models over a single Gamma GLM
# MAGIC
# MAGIC **The composite model wins when:**
# MAGIC
# MAGIC - **You price high-limit accounts or excess layers.** A Gamma GLM will
# MAGIC   systematically underprice accounts with policy limits above £500k because
# MAGIC   its ILF curve flattens too quickly. The ILF error at £5m limits is typically
# MAGIC   15–40% under a Pareto tail, depending on the tail index. That gap is real
# MAGIC   money in a reinsurance tender or a large-account submission.
# MAGIC
# MAGIC - **Your book has a recognisable body-tail transition.** Most commercial and
# MAGIC   motor BI books have a Lognormal-like bulk and a fat tail that is clearly
# MAGIC   heavier than Gamma. The mean excess plot will show a rising slope above
# MAGIC   some threshold. This is the structural signal that a composite model is
# MAGIC   appropriate.
# MAGIC
# MAGIC - **You need provably conservative tail quantiles.** For Solvency II SCR
# MAGIC   calibration, internal model validation, or Lloyd's SBF stress tests, the
# MAGIC   quantile error of the Gamma at the 99th percentile is a material concern.
# MAGIC   Composite models with GPD tails are better calibrated at extreme quantiles.
# MAGIC
# MAGIC - **You compute large loss loadings from frequency-severity models.** If severity
# MAGIC   is Gamma and the true tail is Pareto, your expected cost conditional on
# MAGIC   X > 250k is materially underestimated. This flows directly into the large-loss
# MAGIC   loading on ground-up rates.
# MAGIC
# MAGIC **The Gamma GLM remains appropriate when:**
# MAGIC
# MAGIC - **The book is predominantly attritional, with no meaningful excess of loss
# MAGIC   exposure.** For personal lines property with low limits and claims that are
# MAGIC   nearly all below £50k, the tail region of the distribution is either not priced
# MAGIC   or reinsured away. The Gamma GLM is adequate and far simpler to implement.
# MAGIC
# MAGIC - **You need covariate-heavy models with rating factors across 50+ categories.**
# MAGIC   The Gamma GLM runs in milliseconds and integrates with standard GLM tooling.
# MAGIC   A composite model with covariate-dependent thresholds (CompositeSeverityRegressor)
# MAGIC   requires numerical optimisation per fitting cycle. The computational cost is
# MAGIC   manageable but not trivial.
# MAGIC
# MAGIC - **Regulatory or actuarial sign-off requires interpretability over fit quality.**
# MAGIC   "Gamma with log link" is a one-sentence model description. A composite
# MAGIC   Lognormal-GPD requires explaining threshold selection, mixing weights, and
# MAGIC   GPD tail shape to a non-technical reviewer. Sometimes the simpler model
# MAGIC   wins the governance argument even if it loses the statistical one.
# MAGIC
# MAGIC **Expected results from this benchmark (10k claims, true xi=0.35):**
# MAGIC
# MAGIC | Metric                      | Gamma GLM          | Composite (LN-GPD)  | Composite (Gamma-GPD) |
# MAGIC |-----------------------------|---------------------|---------------------|----------------------|
# MAGIC | Test log-lik per obs        | < composite         | Best (or near-best) | Similar to LN-GPD    |
# MAGIC | AIC                         | Lowest params       | Better overall fit  | Similar              |
# MAGIC | Q99 relative error          | Often -20% to -40%  | Within 5-10%        | Within 5-10%         |
# MAGIC | ILF at £5m vs truth         | -15% to -40%        | Within 5-15%        | Within 5-15%         |
# MAGIC | Threshold identification    | N/A                 | Within 10-20% of true | Within 10-20%     |
# MAGIC | Fit time                    | < 0.01s             | 5-30s               | 5-30s                |
# MAGIC
# MAGIC **Which composite variant to use:**
# MAGIC
# MAGIC - `LognormalGPDComposite(threshold_method="profile_likelihood")` is the default
# MAGIC   choice. GPD is the canonical EVT tail; profile likelihood is theoretically
# MAGIC   motivated and data-driven. Use this for new books or when you have no prior
# MAGIC   view on the threshold.
# MAGIC
# MAGIC - `GammaGPDComposite` is the natural upgrade path from a Gamma GLM. If your
# MAGIC   team already uses Gamma for the bulk and wants to add a heavy tail, this
# MAGIC   is the least disruptive change.
# MAGIC
# MAGIC - `LognormalBurrComposite(threshold_method="mode_matching")` is the right choice
# MAGIC   when you want a C1-continuous splice (smooth density at the threshold) and the
# MAGIC   tail has a recognisable mode structure. More parameter-heavy, more expensive to
# MAGIC   fit, but the smoothest distributional model.

# COMMAND ----------

# Structured verdict print
print("=" * 72)
print("VERDICT: insurance-severity composite vs Gamma GLM")
print("=" * 72)
print()
print(f"  True DGP tail index (xi): {TRUE_XI}  (heavy tail, finite mean and variance)")
print(f"  True splice threshold:    £{TRUE_THRESHOLD:,.0f}")
print(f"  Training claims:          {len(claims_train):,}")
print(f"  Test claims:              {len(claims_test):,}")
print()
print("  -- Log-likelihood (test set) --")
print(f"     Gamma GLM:       {ll_baseline_test/len(claims_test):>8.4f} per obs")
print(f"     LognormalGPD:    {ll_ln_gpd_test/len(claims_test):>8.4f} per obs  ({'+' if ll_ln_gpd_test > ll_baseline_test else ''}{(ll_ln_gpd_test - ll_baseline_test)/len(claims_test):+.4f})")
print(f"     LognormalBurr:   {ll_ln_burr_test/len(claims_test):>8.4f} per obs  ({'+' if ll_ln_burr_test > ll_baseline_test else ''}{(ll_ln_burr_test - ll_baseline_test)/len(claims_test):+.4f})")
print(f"     GammaGPD:        {ll_gamma_gpd_test/len(claims_test):>8.4f} per obs  ({'+' if ll_gamma_gpd_test > ll_baseline_test else ''}{(ll_gamma_gpd_test - ll_baseline_test)/len(claims_test):+.4f})")
print()
print("  -- Tail quantile accuracy --")
for _, row in q_num.iterrows():
    q_lbl = f"Q{row.name*100:.0f}"
    print(f"     {q_lbl}:  Gamma {row['err_baseline_pct']:+.1f}%  |  LN-GPD {row['err_ln_gpd_pct']:+.1f}%  |  G-GPD {row['err_gamma_gpd_pct']:+.1f}%")
print()
print("  -- ILF at £5m limit --")
v_5m_base = ilf_num.loc[ilf_num['limit']==5_000_000, 'baseline'].values[0]
v_5m_lgpd = ilf_num.loc[ilf_num['limit']==5_000_000, 'ln_gpd'].values[0]
v_5m_ggpd = ilf_num.loc[ilf_num['limit']==5_000_000, 'gamma_gpd'].values[0]
print(f"     True:         {v_true_5m:.4f}")
print(f"     Gamma GLM:    {v_5m_base:.4f}  ({relative_error(v_5m_base, v_true_5m):+.1f}%)")
print(f"     LognormalGPD: {v_5m_lgpd:.4f}  ({relative_error(v_5m_lgpd, v_true_5m):+.1f}%)")
print(f"     GammaGPD:     {v_5m_ggpd:.4f}  ({relative_error(v_5m_ggpd, v_true_5m):+.1f}%)")
print()
print("  -- Selected thresholds --")
print(f"     True:              £{TRUE_THRESHOLD:,.0f}")
print(f"     LognormalGPD:      £{model_ln_gpd.threshold_:,.0f}  ({relative_error(model_ln_gpd.threshold_, TRUE_THRESHOLD):+.1f}% from true)")
print(f"     GammaGPD:          £{model_gamma_gpd.threshold_:,.0f}  ({relative_error(model_gamma_gpd.threshold_, TRUE_THRESHOLD):+.1f}% from true)")
print()
print("  Bottom line:")
print("  A single Gamma distribution is light-tailed by construction. For any book")
print("  with a genuinely heavy tail (xi > 0.15), the Gamma underestimates upper")
print("  quantiles and ILFs at high limits by a material margin. Composite spliced")
print("  models capture the body-tail transition and produce materially better ILF")
print("  curves and large-loss loadings, at a computational cost of a few seconds")
print("  per fit.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. README performance snippet

# COMMAND ----------

# Auto-generate the Performance section for the library README.
# Copy-paste this output directly into the README.md Performance section.

v_5m_base = ilf_num.loc[ilf_num['limit']==5_000_000, 'baseline'].values[0]
v_5m_lgpd = ilf_num.loc[ilf_num['limit']==5_000_000, 'ln_gpd'].values[0]
v_5m_ggpd = ilf_num.loc[ilf_num['limit']==5_000_000, 'gamma_gpd'].values[0]

q90_row = q_num.loc[0.90]
q99_row = q_num.loc[0.99]

readme_snippet = f"""
## Performance

Benchmarked against a **single Gamma GLM** (statsmodels, log link) on 10,000 synthetic
claims drawn from a known Lognormal-Pareto DGP (tail shape xi={TRUE_XI}, threshold £{TRUE_THRESHOLD:,.0f}).
See `notebooks/benchmark.py` for full methodology.

**True DGP tail index:** xi={TRUE_XI}  (typical UK motor BI / commercial liability)

| Metric                          | Gamma GLM (baseline)   | LognormalGPD (composite) | GammaGPD (composite)   |
|---------------------------------|------------------------|--------------------------|------------------------|
| Test log-lik per obs            | {ll_baseline_test/len(claims_test):.4f}             | {ll_ln_gpd_test/len(claims_test):.4f}                 | {ll_gamma_gpd_test/len(claims_test):.4f}              |
| AIC                             | {aic_baseline:.0f}                | {aic_ln_gpd:.0f}                    | {aic_gamma_gpd:.0f}                  |
| Q90 quantile error vs truth     | {q90_row['err_baseline_pct']:+.1f}%               | {q90_row['err_ln_gpd_pct']:+.1f}%                  | {q90_row['err_gamma_gpd_pct']:+.1f}%                |
| Q99 quantile error vs truth     | {q99_row['err_baseline_pct']:+.1f}%               | {q99_row['err_ln_gpd_pct']:+.1f}%                  | {q99_row['err_gamma_gpd_pct']:+.1f}%                |
| ILF at £5m vs truth             | {relative_error(v_5m_base, v_true_5m):+.1f}%               | {relative_error(v_5m_lgpd, v_true_5m):+.1f}%                  | {relative_error(v_5m_ggpd, v_true_5m):+.1f}%                |
| Fit time                        | {baseline_fit_time:.3f}s                | {ln_gpd_fit_time:.2f}s                     | {gamma_gpd_fit_time:.2f}s                    |
| Parameters                      | 2                      | 5                        | 5                      |

The Gamma GLM is light-tailed by construction. For books with a heavy tail
(xi > 0.15), it systematically underestimates upper quantiles and ILF values
at high policy limits. The composite models capture the body-tail transition
using profile-likelihood threshold selection and produce materially better ILF
schedules for XL pricing and large-loss loading.
"""

print(readme_snippet)
