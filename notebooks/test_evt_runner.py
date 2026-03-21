# Databricks notebook source
# MAGIC %md
# MAGIC # EVT Module Test Runner
# MAGIC
# MAGIC Runs all tests for `insurance_severity.evt` (TruncatedGPD, CensoredHillEstimator,
# MAGIC WeibullTemperedPareto) on Databricks serverless compute.

# COMMAND ----------

# MAGIC %pip install --quiet scipy numpy matplotlib

# COMMAND ----------

import subprocess
import sys
import os

# Install the package from the uploaded source
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", "/Workspace/insurance-severity-src", "--quiet"],
    capture_output=True, text=True
)
print(result.stdout[-2000:] if result.stdout else "")
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:])

# COMMAND ----------

# Run just the EVT tests
result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/Workspace/insurance-severity-src/tests/test_evt.py",
     "-v", "--tb=short", "--no-header"],
    capture_output=True, text=True,
    cwd="/Workspace/insurance-severity-src"
)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-1000:])
print(f"\nReturn code: {result.returncode}")

# COMMAND ----------

# Also run the full test suite to check no regressions
result2 = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/Workspace/insurance-severity-src/tests/",
     "-v", "--tb=short", "--no-header",
     "--ignore=/Workspace/insurance-severity-src/tests/test_evt.py"],
    capture_output=True, text=True,
    cwd="/Workspace/insurance-severity-src"
)
print("=== Regression tests ===")
print(result2.stdout[-3000:])
print(f"\nReturn code: {result2.returncode}")
