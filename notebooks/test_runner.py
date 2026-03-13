# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-severity test runner
# MAGIC Installs the package from GitHub and runs all tests.

# COMMAND ----------

# MAGIC %pip install git+https://github.com/burning-cost/insurance-severity.git[glm] pytest pytest-cov

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "--tb=short", "-v",
     "--import-mode=importlib",
     "/Workspace/insurance-severity/tests/"],
    capture_output=True,
    text=True
)

print(result.stdout[-10000:] if len(result.stdout) > 10000 else result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-3000:])

assert result.returncode == 0, f"Tests failed with return code {result.returncode}"
print("ALL TESTS PASSED")
