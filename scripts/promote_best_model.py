"""
Sets the "champion" alias on whichever registered model version has the
best (lowest) val_mae among its source runs. src/model_deployment/app.py
always serves whatever this alias points to -- run this after any new
run registers a version to promote it if it's actually better.

Usage:
    uv run python scripts/promote_best_model.py
"""

from pathlib import Path

import mlflow
import yaml

MODEL_NAME = "cv_logistics_bin_count"
ALIAS = "champion"

REPO_ROOT = Path(__file__).resolve().parents[1]
with open(REPO_ROOT / "config" / "mlflow_config.yaml") as f:
    _mlflow_cfg = yaml.safe_load(f)["mlflow"]


def main() -> None:
    mlflow.set_tracking_uri(_mlflow_cfg["tracking_uri"])
    client = mlflow.tracking.MlflowClient()

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        print(f"No versions registered for '{MODEL_NAME}' yet.")
        return

    scored = []
    for v in versions:
        run = client.get_run(v.run_id)
        val_mae = run.data.metrics.get("val_mae")
        if val_mae is not None:
            scored.append((val_mae, v))

    if not scored:
        print("No versions have a val_mae metric.")
        return

    best_mae, best_version = min(scored, key=lambda x: x[0])
    client.set_registered_model_alias(MODEL_NAME, ALIAS, best_version.version)

    print(f"'{ALIAS}' -> v{best_version.version} (val_mae={best_mae:.4f})")
    for mae, v in sorted(scored):
        marker = " <- champion" if v.version == best_version.version else ""
        print(f"  v{v.version}: val_mae={mae:.4f}{marker}")


if __name__ == "__main__":
    main()
