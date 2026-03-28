# Databricks notebook source
# MAGIC %md
# MAGIC # Run MDN tests on Databricks serverless compute
# MAGIC
# MAGIC Installs insurance-severity from the GitHub repo and runs the MDN test suite.

# COMMAND ----------

# MAGIC %pip install git+https://github.com/burning-cost/insurance-severity.git[mdn] pytest

# COMMAND ----------

import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "--pyargs", "insurance_severity.mdn",
     "-v", "--tb=short", "--no-header",
     "-x"],
    capture_output=True, text=True
)

# Print full output
output = result.stdout + ("\nSTDERR:\n" + result.stderr if result.stderr.strip() else "")
print(output[-10000:] if len(output) > 10000 else output)
print(f"\nReturn code: {result.returncode}")

assert result.returncode == 0, "MDN tests failed — check output above"
