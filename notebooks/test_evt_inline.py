# Databricks notebook source
# MAGIC %md
# MAGIC # EVT Module Inline Tests
# MAGIC
# MAGIC Copies sources to /tmp (read-write) and runs pytest from there.

# COMMAND ----------

import sys
import subprocess
import os
import shutil

# Copy source files to /tmp where __pycache__ writes work
src_dst = "/tmp/insurance_severity_test"
shutil.rmtree(src_dst, ignore_errors=True)
shutil.copytree("/Workspace/insurance-severity-src", src_dst)
print(f"Copied to {src_dst}")
print("Contents:", os.listdir(src_dst))

# Install test dependencies
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", "pytest", "scipy", "numpy", "matplotlib"],
    capture_output=True, text=True
)
if result.returncode != 0:
    print("pip stderr:", result.stderr[-500:])
else:
    print("Dependencies installed ok")

# COMMAND ----------

# Verify import works from the copied path
sys.path.insert(0, f"{src_dst}/src")

import insurance_severity
print(f"insurance_severity version: {insurance_severity.__version__}")

from insurance_severity.evt import TruncatedGPD, CensoredHillEstimator, WeibullTemperedPareto
print("EVT classes imported successfully")

# COMMAND ----------

# Run tests
env = {**os.environ, "PYTHONPATH": f"{src_dst}/src"}
result = subprocess.run(
    [sys.executable, "-m", "pytest",
     f"{src_dst}/tests/test_evt.py",
     "-v", "--tb=long", "--no-header", "-p", "no:cacheprovider"],
    capture_output=True, text=True,
    cwd="/tmp",
    env=env,
)
output = result.stdout + "\n" + result.stderr
print(output[:8000])
print(f"\n=== Return code: {result.returncode} ===")

# COMMAND ----------

if result.returncode != 0:
    dbutils.notebook.exit(f"FAILED\n{output[:5000]}")
else:
    # Count passed tests
    import re
    m = re.search(r"(\d+) passed", output)
    passed = m.group(0) if m else "unknown"
    dbutils.notebook.exit(f"PASSED — {passed}\n{output[:3000]}")
