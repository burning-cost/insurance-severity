"""
Submit and wait for EVT test run on Databricks via Jobs API.
Run from local machine: python notebooks/run_evt_tests_job.py
"""
import os
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import NotebookTask, Task, Source


def main():
    # Reads DATABRICKS_HOST and DATABRICKS_TOKEN from environment or ~/.databrickscfg
    w = WorkspaceClient()

    run = w.jobs.submit(
        run_name="evt-tests-insurance-severity",
        tasks=[
            Task(
                task_key="run-evt-tests",
                notebook_task=NotebookTask(
                    notebook_path="/Workspace/insurance-severity-src/notebooks/test_evt_runner",
                    source=Source.WORKSPACE,
                ),
                new_cluster={
                    "spark_version": "15.4.x-scala2.12",
                    "node_type_id": "i3.xlarge",
                    "num_workers": 0,
                    "spark_conf": {"spark.master": "local[*]"},
                },
            )
        ],
    ).result()

    print(f"Run completed: {run}")


if __name__ == "__main__":
    main()
