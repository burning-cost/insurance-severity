# Databricks notebook source
# MAGIC %pip install insurance-severity --quiet

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import numpy as np
from scipy import stats

# Reproduce the Euler inversion function exactly as in cmrs.py
def _euler_inversion(lst_func, x, a=18.0, M=15):
    if x <= 0.0:
        raise ValueError(f"x must be positive for Euler inversion, got {x}")
    scale = np.exp(a) / x
    t0 = a / x
    F0 = 0.5 * np.real(lst_func(complex(t0, 0.0)))
    total = F0
    for k in range(1, M + 1):
        tk = complex(a / x, k * np.pi / x)
        Fk = np.real(lst_func(tk))
        total += ((-1) ** k) * Fk
    return scale * total

def _lst_exponential(t, rate):
    return rate / (rate + t)

def _lst_deriv_exponential(t, rate):
    return rate / (rate + t) ** 2

def _lst_gamma(t, alpha, beta):
    return (beta / (beta + t)) ** alpha

def _lst_deriv_gamma(t, alpha, beta):
    return (alpha / beta) * (beta / (beta + t)) ** (alpha + 1)

print("=== Test 1: Exponential rates [1, 2] at s=3 ===")
print()

# Compute analytically
s = 3.0
# True E[X1|S=3] for Exp(1) + Exp(2)
# f_S(s) = 2*(e^{-s} - e^{-2s})
f_S_true = 2*(np.exp(-s) - np.exp(-2*s))
print(f"True f_S(3) = {f_S_true:.6f}")

# E[X1|S=3] = ((s-1)*e^{-s} + e^{-2s}) / (e^{-s} - e^{-2s})
# Wait, let me recompute
# xi_0(s) = 2*(s-1)*e^{-s} + 2*e^{-2s}
xi_0_true = 2*(s-1)*np.exp(-s) + 2*np.exp(-2*s)
print(f"True xi_0(3) = {xi_0_true:.6f}")
print(f"True E[X1|S=3] = {xi_0_true/f_S_true:.6f}")
xi_1_true = s*f_S_true - xi_0_true
print(f"True E[X2|S=3] = {xi_1_true/f_S_true:.6f}")
print()

# Compute via Euler inversion
rates = np.array([1.0, 2.0])

def joint_lst(t):
    return _lst_exponential(t, rates[0]) * _lst_exponential(t, rates[1])

def alloc_lst_0(t):
    return _lst_deriv_exponential(t, rates[0]) * _lst_exponential(t, rates[1])

def alloc_lst_1(t):
    return _lst_exponential(t, rates[0]) * _lst_deriv_exponential(t, rates[1])

f_S_euler = _euler_inversion(joint_lst, s)
xi_0_euler = _euler_inversion(alloc_lst_0, s)
xi_1_euler = _euler_inversion(alloc_lst_1, s)

print(f"Euler f_S(3) = {f_S_euler:.6f} (true: {f_S_true:.6f})")
print(f"Euler xi_0(3) = {xi_0_euler:.6f} (true: {xi_0_true:.6f})")
print(f"Euler xi_1(3) = {xi_1_true:.6f}")
print(f"Euler h_0(3) = {xi_0_euler/f_S_euler:.6f} (true: {xi_0_true/f_S_true:.6f})")
print(f"Euler h_1(3) = {xi_1_euler/f_S_euler:.6f}")

# COMMAND ----------

print("=== Test 2: Individual terms in Euler sum for alloc_lst_0 at s=3 ===")
x = 3.0
a = 18.0
M = 15
scale = np.exp(a) / x
print(f"scale = {scale:.4e}")
print()

total = 0.0
for k in range(0, M+1):
    if k == 0:
        tk = complex(a/x, 0.0)
        val = 0.5 * np.real(alloc_lst_0(tk))
    else:
        tk = complex(a/x, k * np.pi / x)
        val = ((-1)**k) * np.real(alloc_lst_0(tk))
    total += val
    print(f"k={k:2d}: t={tk}, F={np.real(alloc_lst_0(complex(a/x, k*np.pi/x if k>0 else 0))):.6e}, term={val:.6e}, running={total:.6e}")

print(f"\nFinal: scale*total = {scale*total:.6f}")

# COMMAND ----------

print("=== Test 3: Gamma symmetry at s=8 ===")
print()
s = 8.0
# Two identical Gamma(2,1)
# L_S(t) = (1/(1+t))^4
# xi_0(s) = 2/(1+t)^5 -> L^{-1} = 2*s^4*exp(-s)/24 = s^4*exp(-s)/12
xi_0_gamma_true = s**4 * np.exp(-s) / 12
f_S_gamma_true = s**3 * np.exp(-s) / 6  # Gamma(4,1)
print(f"True f_S(8) = {f_S_gamma_true:.6f}")
print(f"True xi_0(8) = {xi_0_gamma_true:.6f}")
print(f"True E[X1|S=8] = {xi_0_gamma_true/f_S_gamma_true:.6f} (expected s/2 = 4)")

def joint_lst_gamma(t):
    return _lst_gamma(t, 2.0, 1.0) * _lst_gamma(t, 2.0, 1.0)

def alloc_lst_gamma_0(t):
    return _lst_deriv_gamma(t, 2.0, 1.0) * _lst_gamma(t, 2.0, 1.0)

f_S_g_euler = _euler_inversion(joint_lst_gamma, s)
xi_0_g_euler = _euler_inversion(alloc_lst_gamma_0, s)

print(f"\nEuler f_S(8) = {f_S_g_euler:.6f} (true: {f_S_gamma_true:.6f})")
print(f"Euler xi_0(8) = {xi_0_g_euler:.6f} (true: {xi_0_gamma_true:.6f})")
print(f"Euler h_0(8) = {xi_0_g_euler/f_S_g_euler:.6f} (expected 4.0)")

# COMMAND ----------

print("=== Test 4: Investigate the Euler terms for gamma at s=8 ===")
x = 8.0
a = 18.0
M = 15
scale = np.exp(a) / x
print(f"scale = {scale:.4e}")
print()

print("--- joint_lst_gamma (L_S) ---")
total = 0.0
for k in range(0, M+1):
    tk = complex(a/x, k * np.pi / x)
    raw = np.real(joint_lst_gamma(tk))
    if k == 0:
        val = 0.5 * raw
    else:
        val = ((-1)**k) * raw
    total += val
    print(f"k={k:2d}: Re[L(t_k)]={raw:.6e}, term={val:.6e}, running_sum={total:.6e}")
print(f"scale*total = {scale*total:.6e}")

print()
print("--- alloc_lst_gamma_0 (xi_0) ---")
total2 = 0.0
for k in range(0, M+1):
    tk = complex(a/x, k * np.pi / x)
    raw = np.real(alloc_lst_gamma_0(tk))
    if k == 0:
        val = 0.5 * raw
    else:
        val = ((-1)**k) * raw
    total2 += val
    print(f"k={k:2d}: Re[L(t_k)]={raw:.6e}, term={val:.6e}, running_sum={total2:.6e}")
print(f"scale*total = {scale*total2:.6e}")

# COMMAND ----------

print("=== Test 5: Use more terms (M=50) to see if convergence improves ===")
def euler_inv_large(lst_func, x, a=18.0, M=50):
    scale = np.exp(a) / x
    t0 = a / x
    total = 0.5 * np.real(lst_func(complex(t0, 0.0)))
    for k in range(1, M + 1):
        tk = complex(a / x, k * np.pi / x)
        total += ((-1) ** k) * np.real(lst_func(tk))
    return scale * total

print("Exponential s=3:")
print(f"  M=15: f_S={_euler_inversion(joint_lst, 3.0):.6f}, xi_0={_euler_inversion(alloc_lst_0, 3.0):.6f}")
print(f"  M=50: f_S={euler_inv_large(joint_lst, 3.0, M=50):.6f}, xi_0={euler_inv_large(alloc_lst_0, 3.0, M=50):.6f}")
print(f"  M=100: f_S={euler_inv_large(joint_lst, 3.0, M=100):.6f}, xi_0={euler_inv_large(alloc_lst_0, 3.0, M=100):.6f}")

print("Gamma s=8:")
print(f"  M=15: f_S={_euler_inversion(joint_lst_gamma, 8.0):.6f}, xi_0={_euler_inversion(alloc_lst_gamma_0, 8.0):.6f}")
print(f"  M=50: f_S={euler_inv_large(joint_lst_gamma, 8.0, M=50):.6f}, xi_0={euler_inv_large(alloc_lst_gamma_0, 8.0, M=50):.6f}")
print(f"  M=100: f_S={euler_inv_large(joint_lst_gamma, 8.0, M=100):.6f}, xi_0={euler_inv_large(alloc_lst_gamma_0, 8.0, M=100):.6f}")
print(f"  M=200: f_S={euler_inv_large(joint_lst_gamma, 8.0, M=200):.6f}, xi_0={euler_inv_large(alloc_lst_gamma_0, 8.0, M=200):.6f}")

# COMMAND ----------

# Try Euler acceleration (standard Euler-Abate-Whitt with proper acceleration)
def euler_inv_accelerated(lst_func, x, a=18.0, n1=15, n2=15):
    """Euler-accelerated Laplace inversion (Abate & Whitt 1992).

    Uses n1 "reliable" terms plus n2 Euler-accelerated terms.
    Total terms: n1 + n2.
    """
    scale = np.exp(a) / x

    # Compute all terms a_k = (-1)^k Re[F((a+ik*pi)/x)]  for k=0..n1+n2
    terms = []
    for k in range(n1 + n2 + 1):
        tk = complex(a / x, k * np.pi / x)
        val = np.real(lst_func(tk))
        terms.append(val)

    # First n1 terms: direct alternating sum
    # S = sum_{k=0}^{n1} (-1)^k a_k but with a_0 halved
    S = 0.5 * terms[0]
    for k in range(1, n1 + 1):
        S += ((-1)**k) * terms[k]

    # Euler acceleration of remaining n2 terms
    # Using binomial weights: sum_{j=0}^{n2} C(n2,j)/2^n2 * (-1)^{n1+1+j} a_{n1+1+j}
    from math import comb
    euler_tail = 0.0
    for j in range(n2 + 1):
        coeff = comb(n2, j) / (2**n2)
        idx = n1 + 1 + j
        if idx < len(terms):
            euler_tail += coeff * ((-1)**(n1 + 1 + j)) * terms[idx]

    return scale * (S + euler_tail)

print("=== Test 6: Euler-accelerated inversion ===")
print()
print("Exponential s=3 (true h=[2.157, 0.843]):")
f_s = euler_inv_accelerated(joint_lst, 3.0)
xi0 = euler_inv_accelerated(alloc_lst_0, 3.0)
xi1 = euler_inv_accelerated(alloc_lst_1, 3.0)
print(f"  f_S={f_s:.6f}, xi_0={xi0:.6f}, xi_1={xi1:.6f}")
print(f"  h_0={xi0/f_s:.4f}, h_1={xi1/f_s:.4f}")

print()
print("Gamma s=8 (true h=[4, 4]):")
f_s_g = euler_inv_accelerated(joint_lst_gamma, 8.0)
xi0_g = euler_inv_accelerated(alloc_lst_gamma_0, 8.0)
print(f"  f_S={f_s_g:.6f}, xi_0={xi0_g:.6f}")
print(f"  h_0={xi0_g/f_s_g:.4f}")

# COMMAND ----------

# Also try with larger a
def euler_inv_large_a(lst_func, x, a=24.0, M=25):
    scale = np.exp(a) / x
    t0 = a / x
    total = 0.5 * np.real(lst_func(complex(t0, 0.0)))
    for k in range(1, M + 1):
        tk = complex(a / x, k * np.pi / x)
        total += ((-1) ** k) * np.real(lst_func(tk))
    return scale * total

print("=== Test 7: Larger a parameter ===")
print()
print("Gamma s=8 with different (a, M):")
for a_val, M_val in [(18, 15), (18, 30), (18, 50), (24, 30), (30, 35)]:
    def inv(f, x):
        return euler_inv_large_a(f, x, a=a_val, M=M_val)
    f_s_g = inv(joint_lst_gamma, 8.0)
    xi0_g = inv(alloc_lst_gamma_0, 8.0)
    h = xi0_g/f_s_g if f_s_g > 0 else float('nan')
    print(f"  a={a_val}, M={M_val}: f_S={f_s_g:.4e}, xi_0={xi0_g:.4e}, h_0={h:.4f}")

# COMMAND ----------

# Check what (a, M) gives accurate results for exponential
print("=== Test 8: Exponential convergence with (a, M) ===")
print()
print("Exponential rates [1,2], s=3 (true h_0=2.157, h_1=0.843):")
for a_val, M_val in [(18, 15), (18, 30), (18, 50), (24, 25), (20, 20)]:
    def inv(f, x):
        return euler_inv_large_a(f, x, a=a_val, M=M_val)
    f_s_e = inv(joint_lst, 3.0)
    xi0_e = inv(alloc_lst_0, 3.0)
    xi1_e = inv(alloc_lst_1, 3.0)
    h0 = xi0_e/f_s_e if f_s_e > 0 else float('nan')
    h1 = xi1_e/f_s_e if f_s_e > 0 else float('nan')
    print(f"  a={a_val}, M={M_val}: f_S={f_s_e:.4e}, h_0={h0:.4f}, h_1={h1:.4f}")

# COMMAND ----------

# Test the full CMRSAllocator from installed package
# First install from local files

# COMMAND ----------

print("=== Test 9: Check installed CMRSAllocator ===")
try:
    from insurance_severity.cmrs import CMRSAllocator
    alloc = CMRSAllocator(distribution="exponential", n_euler_terms=15, euler_a=18.0)
    alloc.fit_exponential(rates=np.array([1.0, 2.0]))
    h = alloc.allocate(3.0)
    print(f"Installed version: h(3) = {h}")
except Exception as e:
    print(f"Error: {e}")

# COMMAND ----------

# Test tilting for s=20
print("=== Test 10: Tilting for s=20 ===")
print()
# Manually trace _kappa_prime for exponential rates [1, 2]
def kappa_prime_exp(theta, rates):
    total = 0.0
    for rate in rates:
        t = complex(-theta, 0.0)
        l_i = np.real(rate / (rate + t))  # = rate/(rate-theta)
        dl_i = np.real(rate / (rate + t)**2)  # = rate/(rate-theta)^2
        total += dl_i / (l_i + 1e-300)
    return total

rates = np.array([1.0, 2.0])
for theta in [0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0, 1.1, 1.5, 2.0]:
    try:
        kp = kappa_prime_exp(theta, rates)
        print(f"  theta={theta:.3f}: kappa_prime={kp:.4f}")
    except Exception as e:
        print(f"  theta={theta:.3f}: ERROR {e}")

print()
print("For s=20, need kappa_prime(theta) = 20")
print("True solution: 1/(1-theta) + 1/(2-theta) = 20 => theta ~ 0.9475")

# Check the upper bracket search
upper = 0.1
for iteration in range(20):
    kp = kappa_prime_exp(upper, rates)
    print(f"  iter={iteration}, upper={upper:.4f}, kappa_prime={kp:.4f}")
    if kp > 20:
        print(f"  FOUND: upper={upper} gives kappa_prime > 20")
        break
    upper *= 2.0
else:
    print(f"  NEVER found: final upper={upper:.4e}")

# COMMAND ----------

print("=== Test 11: The correct tilting fix ===")
print()
# For exponentials, kappa_prime(theta) = sum_i 1/(rate_i - theta)
# This is monotone increasing in theta on (0, min_rate)
# The search must stay within (0, min_rate)

def kappa_prime_exp_corrected(theta, rates):
    # Works correctly only for theta < min(rates)
    return sum(1.0/(rate - theta) for rate in rates)

min_rate = min(rates)
print(f"Min rate = {min_rate}, need theta < {min_rate}")
print()

# Search within (0, min_rate)
from scipy import optimize

# Find theta such that kappa_prime(theta) = 20
def f(theta):
    return kappa_prime_exp_corrected(theta, rates) - 20.0

# Check bounds
print(f"kappa_prime(0) = {kappa_prime_exp_corrected(0, rates):.4f}")
print(f"kappa_prime(0.9) = {kappa_prime_exp_corrected(0.9, rates):.4f}")
print(f"kappa_prime(0.94) = {kappa_prime_exp_corrected(0.94, rates):.4f}")
print(f"kappa_prime(0.95) = {kappa_prime_exp_corrected(0.95, rates):.4f}")

theta_star = optimize.brentq(f, 0.0, min_rate - 1e-6)
print(f"\ntheta* = {theta_star:.6f}")
print(f"kappa_prime(theta*) = {kappa_prime_exp_corrected(theta_star, rates):.4f}")

# COMMAND ----------

# The real fix: the upper bound in _find_tilt_theta must be < min(rate_i)
# The current code lets upper grow unboundedly (doubling), passing through
# the singularity at rate_1=1, after which kappa_prime becomes negative,
# and the loop never finds kappa_prime > 20 in 40 iterations.

print("=== Root cause summary ===")
print()
print("BUG 1 (NaN at s=20):")
print("  _find_tilt_theta lets 'upper' grow past min(rate_i) = 1.0")
print("  Once theta > rate_i, kappa_prime returns wrong (negative) values")
print("  After 40 doublings, upper=0.1*2^40 ~ 1e11, and upper/2 is returned as theta_star")
print("  kappa(1e11) = log(negative) = NaN")
print("  This NaN propagates through all subsequent calculations -> NaN output")
print()
print("BUG 2 (wrong values at moderate s):")
print("  The Euler inversion with M=15 terms may give inaccurate results")
print("  Need to check with more terms and/or Euler acceleration")
print()

# Check if M=15 is actually accurate for these transforms
print("Accuracy check with M=15:")
print(f"  Exp s=3: h_0 computed={xi0_e:.4f}, true=2.157")

# COMMAND ----------

# Test with full Euler acceleration
print("=== Test 12: Euler acceleration results ===")
print()
for (test_name, f_lst, xi_lst, x, true_vals) in [
    ("Exp s=3", joint_lst, [alloc_lst_0, alloc_lst_1], 3.0, [2.157, 0.843]),
    ("Gamma s=8", joint_lst_gamma, [alloc_lst_gamma_0, alloc_lst_gamma_0], 8.0, [4.0, 4.0]),
]:
    print(f"\n{test_name}:")
    f_s = euler_inv_accelerated(f_lst, x)
    xis = [euler_inv_accelerated(xi, x) for xi in xi_lst]
    hs = [xi/f_s for xi in xis]
    print(f"  h = {[f'{h:.4f}' for h in hs]}, expected {true_vals}")
    print(f"  f_S = {f_s:.6e}")

    # With no acceleration (current code)
    f_s2 = _euler_inversion(f_lst, x)
    xis2 = [_euler_inversion(xi, x) for xi in xi_lst]
    hs2 = [xi/f_s2 for xi in xis2]
    print(f"  h (no accel) = {[f'{h:.4f}' for h in hs2]}")
