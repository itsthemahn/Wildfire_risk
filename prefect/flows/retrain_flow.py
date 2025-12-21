import json
import subprocess
from pathlib import Path
from typing import Optional, List

from prefect import flow, task, get_run_logger

# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATUS_PATH = PROJECT_ROOT / "monitoring" / "drift" / "status.json"


# -------------------------------------------------------------------
# Tasks
# -------------------------------------------------------------------
@task
def read_drift_status() -> Optional[dict]:
    logger = get_run_logger()

    if not STATUS_PATH.exists():
        logger.warning("status.json not found — skipping drift-based retrain")
        return None

    with open(STATUS_PATH) as f:
        status = json.load(f)

    logger.info(f"Drift status loaded: {status}")
    return status


@task
def retrain_model(models: List[str]):
    logger = get_run_logger()
    logger.info(f"Starting model retraining for models: {models}")

    cmd = [
        "python",
        str(PROJECT_ROOT / "train.py"),
        "--models",
        *models,
    ]

    logger.info(f"Running command: {' '.join(cmd)}")

    subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        check=True,
    )

    logger.info("Model retraining completed")


@task
def register_model():
    logger = get_run_logger()
    logger.info("Registering model in MLflow...")

    subprocess.run(
        ["python", str(PROJECT_ROOT / "register_model.py")],
        cwd=PROJECT_ROOT,
        check=True,
    )

    logger.info("Model registered in MLflow")


@task
def dvc_version_data():
    logger = get_run_logger()
    logger.info("Versioning data & artifacts with DVC...")

    subprocess.run(["dvc", "add", "Data"], cwd=PROJECT_ROOT, check=False)
    subprocess.run(["dvc", "add", "models"], cwd=PROJECT_ROOT, check=False)
    subprocess.run(
        ["dvc", "commit", "-m", "Retrain after drift detected"],
        cwd=PROJECT_ROOT,
        check=False,
    )

    logger.info("DVC versioning complete")


# -------------------------------------------------------------------
# Flow
# -------------------------------------------------------------------
@flow(name="wildfire-retraining-pipeline")
def retrain_pipeline(
    force_retrain: bool = False,
    models: List[str] = ["RandomForest", "XGBoost"],
):
    """
    force_retrain:
        - False: retrain only if drift trigger is true
        - True : retrain regardless of drift status

    models:
        Models to retrain (safe default excludes DL)
    """
    logger = get_run_logger()

    if force_retrain:
        logger.warning("Manual retrain triggered (force_retrain=True)")
        retrain_model(models)
        register_model()
        dvc_version_data()
        return

    status = read_drift_status()

    if not status:
        logger.info("No drift status available — exiting pipeline")
        return

    if status.get("trigger_retrain"):
        logger.info("Drift threshold exceeded — retraining triggered")
        retrain_model(models)
        register_model()
        dvc_version_data()
    else:
        logger.info("No drift detected — skipping retraining")


# -------------------------------------------------------------------
# Manual execution
# -------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    force = "--force" in sys.argv

    retrain_pipeline(
        force_retrain=force,
        models=["RandomForest", "XGBoost"],  # SAFE DEFAULT
    )
