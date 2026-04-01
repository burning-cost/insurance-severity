# Databricks notebook source
# MAGIC %md
# MAGIC # ProjectionToUltimate Test Runner
# MAGIC
# MAGIC Runs all tests for `insurance_severity.projection` (ProjectionToUltimate)
# MAGIC on Databricks serverless compute.
# MAGIC
# MAGIC Reference: Richman & Wüthrich (arXiv:2603.11660, March 2026)

# COMMAND ----------

# MAGIC %pip install --quiet scipy numpy polars

# COMMAND ----------

import subprocess
import sys

# Install the package from the uploaded source
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", "/Workspace/insurance-severity-src", "--quiet"],
    capture_output=True, text=True
)
print(result.stdout[-2000:] if result.stdout else "")
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:])

# COMMAND ----------

# Run projection tests
result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/Workspace/insurance-severity-src/tests/test_projection.py",
     "-v", "--tb=short", "--no-header"],
    capture_output=True, text=True,
    cwd="/Workspace/insurance-severity-src"
)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-1000:])
print(f"\nReturn code: {result.returncode}")

# COMMAND ----------

# Full regression test suite (skip torch-dependent tests if not installed)
result2 = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/Workspace/insurance-severity-src/tests/",
     "-v", "--tb=short", "--no-header",
     "--ignore=/Workspace/insurance-severity-src/tests/drn",
     "--ignore=/Workspace/insurance-severity-src/tests/test_mdn.py"],
    capture_output=True, text=True,
    cwd="/Workspace/insurance-severity-src"
)
print("=== Full regression suite ===")
print(result2.stdout[-5000:])
print(f"\nReturn code: {result2.returncode}")

assert result.returncode == 0, "Projection tests FAILED"
assert result2.returncode == 0, "Regression tests FAILED"
print("\nALL TESTS PASSED")
