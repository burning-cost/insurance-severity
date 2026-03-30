# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-severity tail scoring tests
# MAGIC Runs tests/test_tail_scoring.py on Databricks.

# COMMAND ----------

# MAGIC %pip install matplotlib scipy pandas numpy

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "--tb=short", "-v",
     "--import-mode=importlib",
     "/Workspace/insurance-severity/tests/test_tail_scoring.py"],
    capture_output=True,
    text=True
)

print(result.stdout[-15000:] if len(result.stdout) > 15000 else result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-3000:])

assert result.returncode == 0, f"Tests failed with return code {result.returncode}"
print("ALL TAIL SCORING TESTS PASSED")
