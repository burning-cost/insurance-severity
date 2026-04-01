"""
Upload source and submit Databricks job to run ProjectionToUltimate tests.
Run from local machine: python notebooks/run_projection_tests_job.py
"""
import os
import sys
import subprocess
import time

# Load Databricks credentials
env_path = os.path.expanduser("~/.config/burning-cost/databricks.env")
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k] = v

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import NotebookTask, Task, Source
from databricks.sdk.service import workspace


WORKSPACE_PATH = "/Workspace/insurance-severity-src"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def upload_sources(w: WorkspaceClient) -> None:
    """Upload src/, tests/, and notebooks/ to Databricks workspace."""
    print(f"Uploading sources to {WORKSPACE_PATH} ...")
    result = subprocess.run(
        [
            "databricks", "workspace", "import-dir",
            REPO_ROOT,
            WORKSPACE_PATH,
            "--overwrite",
            "--exclude-artifact",
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        # Fallback: use SDK file upload for key files
        print(f"import-dir failed ({result.stderr[:200]}), using SDK upload...")
        _sdk_upload(w)
    else:
        print("Upload complete.")


def _sdk_upload(w: WorkspaceClient) -> None:
    """Fallback: upload key files individually via SDK."""
    import base64

    def upload_file(local_path: str, remote_path: str) -> None:
        with open(local_path, "rb") as f:
            content = f.read()
        w.workspace.upload(
            path=remote_path,
            content=content,
            overwrite=True,
            format=workspace.ImportFormat.AUTO,
        )

    files_to_upload = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        # Skip hidden dirs, __pycache__, .git, dist, build
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".") and d not in ("__pycache__", "dist", "build", ".eggs")
        ]
        for fname in filenames:
            if fname.endswith((".py", ".toml", ".txt", ".md")):
                local = os.path.join(dirpath, fname)
                rel = os.path.relpath(local, REPO_ROOT)
                remote = f"{WORKSPACE_PATH}/{rel}"
                files_to_upload.append((local, remote))

    for local, remote in files_to_upload:
        remote_dir = "/".join(remote.split("/")[:-1])
        try:
            w.workspace.mkdirs(path=remote_dir)
        except Exception:
            pass
        try:
            upload_file(local, remote)
            print(f"  Uploaded: {remote}")
        except Exception as e:
            print(f"  WARN: could not upload {remote}: {e}")


def main() -> None:
    w = WorkspaceClient()
    upload_sources(w)

    print("Submitting test job...")
    run = w.jobs.submit(
        run_name="projection-tests-insurance-severity",
        tasks=[
            Task(
                task_key="run-projection-tests",
                notebook_task=NotebookTask(
                    notebook_path=f"{WORKSPACE_PATH}/notebooks/test_projection_runner",
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

    print(f"\nJob run result: {run.state}")
    if run.state.result_state.value != "SUCCESS":
        sys.exit(1)
    print("ALL TESTS PASSED on Databricks.")


if __name__ == "__main__":
    main()
