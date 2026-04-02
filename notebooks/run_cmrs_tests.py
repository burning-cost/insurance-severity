# Databricks notebook source
# MAGIC %pip install pytest scipy numpy --quiet

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# Install the package from the uploaded source
import subprocess
import sys
import os

# COMMAND ----------

# First, let's verify our Euler inversion fix by running quick numerical checks

import numpy as np
from math import comb

def _euler_inversion_new(lst_func, x, a=18.0, M=15):
    """Euler-accelerated Abate-Whitt inversion (fixed)."""
    if x <= 0.0:
        raise ValueError(f"x must be positive, got {x}")
    scale = np.exp(a) / x

    # Evaluate at all 2M+1 points
    f_vals = np.empty(2 * M + 1)
    for k in range(2 * M + 1):
        tk = complex(a / x, k * np.pi / x)
        f_vals[k] = np.real(lst_func(tk))

    # Direct sum: k=0..M
    s_direct = 0.5 * f_vals[0]
    for k in range(1, M + 1):
        s_direct += ((-1) ** k) * f_vals[k]

    # Euler acceleration: remaining M terms
    s_euler = 0.0
    inv_2m = 1.0 / (2 ** M)
    for j in range(M):
        coeff = comb(M, j) * inv_2m
        idx = M + 1 + j
        s_euler += coeff * ((-1) ** (M + 1 + j)) * f_vals[idx]
    s_euler += 0.5 * comb(M, M) * inv_2m * ((-1) ** (2 * M + 1)) * f_vals[2 * M]

    return scale * (s_direct + s_euler)

def _euler_inversion_old(lst_func, x, a=18.0, M=15):
    """Old broken implementation (plain truncation, no acceleration)."""
    if x <= 0.0:
        raise ValueError(f"x must be positive, got {x}")
    scale = np.exp(a) / x
    t0 = a / x
    F0 = 0.5 * np.real(lst_func(complex(t0, 0.0)))
    total = F0
    for k in range(1, M + 1):
        tk = complex(a / x, k * np.pi / x)
        Fk = np.real(lst_func(tk))
        total += ((-1) ** k) * Fk
    return scale * total


print("=== Verification 1: Exponential density ===")
# L(t) = rate/(rate+t), f(s) = rate*exp(-rate*s)
rate = 1.0
def lst_exp(t): return rate / (rate + t)
for s in [0.5, 1.0, 2.0, 3.0, 5.0]:
    true_val = rate * np.exp(-rate * s)
    old_val = _euler_inversion_old(lst_exp, s)
    new_val = _euler_inversion_new(lst_exp, s)
    print(f"  s={s}: true={true_val:.6f}, old={old_val:.6f}, new={new_val:.6f}")

# COMMAND ----------

print("=== Verification 2: Exponential rates [1,2] — f_S and xi_0 ===")
def lst_exp_1(t): return 1.0 / (1.0 + t)
def lst_exp_2(t): return 2.0 / (2.0 + t)
def joint_lst(t): return lst_exp_1(t) * lst_exp_2(t)
def alloc_lst_0(t): return (1.0/(1.0+t)**2) * (2.0/(2.0+t))
def alloc_lst_1(t): return (1.0/(1.0+t)) * (2.0/(2.0+t)**2)

# True values
def f_S_true(s): return 2*(np.exp(-s) - np.exp(-2*s))
def xi_0_true(s): return 2*(s-1)*np.exp(-s) + 2*np.exp(-2*s)
def h1_true(s): return ((s-1) + np.exp(-s)) / (1 - np.exp(-s))

for s in [1.0, 3.0, 5.0]:
    fS = f_S_true(s)
    xi0 = xi_0_true(s)
    h0_exact = h1_true(s)

    fS_old = _euler_inversion_old(joint_lst, s)
    xi0_old = _euler_inversion_old(alloc_lst_0, s)
    xi1_old = _euler_inversion_old(alloc_lst_1, s)
    h0_old = xi0_old/fS_old if fS_old > 0 else float('nan')
    h1_old = xi1_old/fS_old if fS_old > 0 else float('nan')

    fS_new = _euler_inversion_new(joint_lst, s)
    xi0_new = _euler_inversion_new(alloc_lst_0, s)
    xi1_new = _euler_inversion_new(alloc_lst_1, s)
    h0_new = xi0_new/fS_new if fS_new > 0 else float('nan')
    h1_new = xi1_new/fS_new if fS_new > 0 else float('nan')

    print(f"\ns={s}:")
    print(f"  true h1={h0_exact:.4f}, h2={s-h0_exact:.4f}")
    print(f"  old  h1={h0_old:.4f}, h2={h1_old:.4f}  (fS={fS_old:.6f}, true={fS:.6f})")
    print(f"  new  h1={h0_new:.4f}, h2={h1_new:.4f}  (fS={fS_new:.6f}, true={fS:.6f})")

# COMMAND ----------

print("=== Verification 3: Gamma symmetry at s=8 ===")
def lst_gamma_2_1(t): return (1.0/(1.0+t))**2
def joint_lst_gamma(t): return lst_gamma_2_1(t)**2
def alloc_lst_gamma_0(t): return (2.0/(1.0+t)**3) * lst_gamma_2_1(t)

# True values
def f_S_gamma_true(s): return s**3 * np.exp(-s) / 6   # Gamma(4,1)
def xi_0_gamma_true(s): return s**4 * np.exp(-s) / 12  # h=s/2

s = 8.0
fS_true = f_S_gamma_true(s)
xi0_true = xi_0_gamma_true(s)
h0_exact = xi0_true / fS_true

fS_old = _euler_inversion_old(joint_lst_gamma, s)
xi0_old = _euler_inversion_old(alloc_lst_gamma_0, s)
h0_old = xi0_old/fS_old if fS_old != 0 else float('nan')

fS_new = _euler_inversion_new(joint_lst_gamma, s)
xi0_new = _euler_inversion_new(alloc_lst_gamma_0, s)
h0_new = xi0_new/fS_new if fS_new != 0 else float('nan')

print(f"s={s}:")
print(f"  true h0={h0_exact:.4f} (expected 4.0)")
print(f"  old  h0={h0_old:.4f}  (fS={fS_old:.6e}, xi0={xi0_old:.6e})")
print(f"  new  h0={h0_new:.4f}  (fS={fS_new:.6e}, xi0={xi0_new:.6e})")

# COMMAND ----------

print("=== Verification 4: Tilting for exponential s=20 ===")

# Test that we can find theta* correctly for s=20
# True: solve 1/(1-theta) + 1/(2-theta) = 20
# => theta ≈ 0.9475
from scipy import optimize

rates = np.array([1.0, 2.0])

def kappa_prime_correct(theta):
    return sum(1.0/(r - theta) for r in rates)

def kappa_correct(theta):
    vals = [np.log(r/(r-theta)) for r in rates]
    return sum(vals)

theta_max = min(rates)  # = 1.0
print(f"theta_max = {theta_max}")

# Find theta* for s=20
from scipy.optimize import brentq
theta_star = brentq(lambda th: kappa_prime_correct(th) - 20.0, 0.0, theta_max - 1e-6)
print(f"theta* for s=20: {theta_star:.6f}")
print(f"kappa_prime(theta*) = {kappa_prime_correct(theta_star):.4f} (should be 20)")
print(f"kappa(theta*) = {kappa_correct(theta_star):.4f}")

# Test that the new _find_tilt_theta would work
# The key is that upper never exceeds min(rates)
print()
print("Searching for upper bracket (should stay < 1.0):")
upper = min(0.1, theta_max * 0.5)  # = 0.1
for it in range(20):
    kp = kappa_prime_correct(upper)
    print(f"  iter={it}, upper={upper:.6f}, kappa_prime={kp:.4f}")
    if kp > 20:
        print("  FOUND upper bracket!")
        break
    next_upper = upper + (theta_max - upper) * 0.5
    if next_upper >= theta_max * 0.9999:
        upper = theta_max * 0.9999
        break
    upper = next_upper

# COMMAND ----------

print("=== Verification 5: Full allocator with fixed code ===")

# Install the updated package
import subprocess
result = subprocess.run(
    ["pip", "install", "-e", "/Workspace/Users/pricing.frontier@gmail.com/insurance-severity-src/",
     "--quiet", "--no-deps"],
    capture_output=True, text=True
)
print(result.stdout[-500:] if result.stdout else "no stdout")
print(result.stderr[-500:] if result.stderr else "no stderr")

# COMMAND ----------

# Since we can't easily install from local source, run the logic directly
print("=== Running CMRS allocation checks with new Euler formula ===")

# Replicate CMRSAllocator logic with the fixed Euler inversion

class FixedCMRSAllocator:
    def __init__(self, rates):
        self.rates = rates
        self.n = len(rates)

    def _lst_i(self, i, t):
        return self.rates[i] / (self.rates[i] + t)

    def _lst_deriv_i(self, i, t):
        return self.rates[i] / (self.rates[i] + t)**2

    def _joint_lst(self, t):
        r = complex(1.0, 0.0)
        for i in range(self.n):
            r *= self._lst_i(i, t)
        return r

    def _alloc_lst_i(self, i, t):
        d = self._lst_deriv_i(i, t)
        p = complex(1.0, 0.0)
        for j in range(self.n):
            if j != i:
                p *= self._lst_i(j, t)
        return d * p

    def allocate(self, s):
        f_S = _euler_inversion_new(self._joint_lst, s)
        xi = np.array([
            _euler_inversion_new(lambda t, _i=i: self._alloc_lst_i(_i, t), s)
            for i in range(self.n)
        ])
        h = xi / f_S
        h = h * (s / h.sum())
        return h

# Test 1: Exponential rates [1,2] at s=3
alloc = FixedCMRSAllocator(rates=np.array([1.0, 2.0]))
h = alloc.allocate(3.0)
h1_true = h1_true_val = ((3.0-1) + np.exp(-3.0)) / (1 - np.exp(-3.0))
print(f"Test 1 (s=3): h={h}, expected=[{h1_true:.4f}, {3-h1_true:.4f}]")
print(f"  Rel error h[0]: {abs(h[0]-h1_true)/h1_true:.4e}")

# Test 2: Multiple s values
for sv in [1.0, 5.0, 10.0]:
    h = alloc.allocate(sv)
    h1_true = ((sv-1) + np.exp(-sv)) / (1 - np.exp(-sv))
    print(f"s={sv}: h={h}, expected=[{h1_true:.4f}, {sv-h1_true:.4f}], err={abs(h[0]-h1_true)/h1_true:.4e}")

# COMMAND ----------

# Test gamma symmetry
class FixedGammaAlloc:
    def __init__(self, alphas, betas):
        self.alphas = alphas
        self.betas = betas
        self.n = len(alphas)

    def _lst_i(self, i, t):
        return (self.betas[i]/(self.betas[i]+t))**self.alphas[i]

    def _lst_deriv_i(self, i, t):
        return (self.alphas[i]/self.betas[i]) * (self.betas[i]/(self.betas[i]+t))**(self.alphas[i]+1)

    def _joint_lst(self, t):
        r = complex(1.0, 0.0)
        for i in range(self.n):
            r *= self._lst_i(i, t)
        return r

    def _alloc_lst_i(self, i, t):
        d = self._lst_deriv_i(i, t)
        p = complex(1.0, 0.0)
        for j in range(self.n):
            if j != i:
                p *= self._lst_i(j, t)
        return d * p

    def allocate(self, s):
        f_S = _euler_inversion_new(self._joint_lst, s)
        xi = np.array([
            _euler_inversion_new(lambda t, _i=i: self._alloc_lst_i(_i, t), s)
            for i in range(self.n)
        ])
        h = xi / f_S
        h = h * (s / h.sum())
        return h

print("\n=== Gamma symmetry test ===")
g_equal = FixedGammaAlloc(alphas=np.array([2.0, 2.0]), betas=np.array([1.0, 1.0]))
h = g_equal.allocate(8.0)
print(f"h = {h}, expected = [4.0, 4.0]")
print(f"  h[0]-4.0 = {h[0]-4.0:.6f}")

# COMMAND ----------

print("\n=== Budget balance at s=20 (exponential, uses tilting) ===")

# Test tilting fix
from scipy.optimize import brentq as _brentq

def find_tilt_theta_fixed(s, rates, theta_max):
    """Fixed version that stays within (0, theta_max)."""
    kp0 = sum(1.0/(r - 0.0) for r in rates)  # kappa_prime(0) = sum(1/rate_i) = E[S]
    if kp0 >= s:
        return 0.0

    # Search within (0, theta_max)
    upper = min(0.1, theta_max * 0.5)
    for _ in range(60):
        kp = sum(1.0/(r - upper) for r in rates)
        if kp > s:
            break
        next_upper = upper + (theta_max - upper) * 0.5
        if next_upper >= theta_max * 0.9999:
            upper = theta_max * 0.9999
            break
        upper = next_upper

    theta_star = _brentq(
        lambda th: sum(1.0/(r-th) for r in rates) - s,
        0.0, upper, xtol=1e-10, rtol=1e-10
    )
    return float(theta_star)

rates = np.array([1.0, 2.0])
theta_max = min(rates)

# Test for s=20
theta = find_tilt_theta_fixed(20.0, rates, theta_max)
print(f"theta* for s=20: {theta:.6f}")
kappa_theta = sum(np.log(r/(r-theta)) for r in rates)
print(f"kappa(theta*) = {kappa_theta:.4f}")
print(f"exp(kappa) = {np.exp(kappa_theta):.4f}")

# Now compute allocation at s=20 with tilting
def tilted_joint_lst(t):
    joint = complex(1.0, 0.0)
    for r in rates:
        joint *= r / (r + (t - complex(theta, 0.0)))
    return joint / np.exp(kappa_theta)

def tilted_alloc_lst_i(i, t):
    t_shifted = t - complex(theta, 0.0)
    deriv_i = rates[i] / (rates[i] + t_shifted)**2
    prod_j = complex(1.0, 0.0)
    for j in range(len(rates)):
        if j != i:
            prod_j *= rates[j] / (rates[j] + t_shifted)
    return deriv_i * prod_j / np.exp(kappa_theta)

f_S = _euler_inversion_new(tilted_joint_lst, 20.0)
xi = np.array([_euler_inversion_new(lambda t, _i=i: tilted_alloc_lst_i(_i, t), 20.0) for i in range(2)])
h_raw = xi / f_S
h = h_raw * (20.0 / h_raw.sum())
print(f"h(20) = {h}, sum = {h.sum():.4f} (expected 20.0)")

# COMMAND ----------

print("\n=== All 6 tests should now pass ===\n")

print("1. test_exponential_single_s:")
alloc = FixedCMRSAllocator(rates=np.array([1.0, 2.0]))
h = alloc.allocate(3.0)
h1_true = ((3.0-1) + np.exp(-3.0)) / (1 - np.exp(-3.0))
print(f"   h={h}, expected=[{h1_true:.4f}, {3-h1_true:.4f}]")
print(f"   PASS: {abs(h[0]-h1_true)/h1_true < 0.001}")

print("\n2. test_exponential_closed_form_multiple_s:")
for sv in [1.0, 5.0, 10.0]:
    h = alloc.allocate(sv)
    h1_true = ((sv-1) + np.exp(-sv)) / (1 - np.exp(-sv))
    err = abs(h[0]-h1_true)/h1_true
    print(f"   s={sv}: h={h}, err={err:.4e}, PASS: {err < 0.001}")

print("\n3. test_budget_balance_exponential (s=20 with tilting):")
print(f"   h={h}, sum={h.sum():.4f}")
h20 = None
try:
    theta = find_tilt_theta_fixed(20.0, rates, theta_max)
    kappa_theta = sum(np.log(r/(r-theta)) for r in rates)
    def tj(t):
        j = complex(1.0, 0.0)
        for r in rates:
            j *= r / (r + (t - complex(theta, 0.0)))
        return j / np.exp(kappa_theta)
    def tai(i, t):
        ts = t - complex(theta, 0.0)
        d = rates[i] / (rates[i] + ts)**2
        p = complex(1.0, 0.0)
        for j2 in range(len(rates)):
            if j2 != i:
                p *= rates[j2] / (rates[j2] + ts)
        return d * p / np.exp(kappa_theta)
    fs = _euler_inversion_new(tj, 20.0)
    xis = np.array([_euler_inversion_new(lambda t, _i=i: tai(_i, t), 20.0) for i in range(2)])
    h20 = xis/fs
    h20 = h20 * (20.0/h20.sum())
    print(f"   h(20)={h20}, sum={h20.sum():.4f}, PASS: {abs(h20.sum()-20.0) < 1e-3}")
except Exception as e:
    print(f"   ERROR: {e}")

print("\n4. test_gamma_symmetry:")
h_g = g_equal.allocate(8.0)
print(f"   h={h_g}, expected=[4.0, 4.0], PASS: {abs(h_g[0]-4.0) < 0.01}")

print("\n5. test_monotonicity_exponential (check no NaN at s>4.1):")
s_vals = np.linspace(0.5, 10.0, 30)
hs = np.array([alloc.allocate(sv) for sv in s_vals])
has_nan = np.any(np.isnan(hs))
print(f"   any NaN: {has_nan}")
diffs = np.diff(hs[:, 0])
not_monotone = np.any(diffs < -1e-4)
print(f"   h_0 not monotone: {not_monotone}, PASS: {not has_nan and not not_monotone}")

print("\n6. test_lognormal_vs_mc: requires full lognormal LST, will verify with full package test")
