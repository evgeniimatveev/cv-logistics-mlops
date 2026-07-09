"""
Unattended weekly automation, meant to be launched by Windows Task
Scheduler (see scripts/weekly_task.ps1 + scripts/register_task.ps1):

1. Make sure the local MLflow tracking server is up (starts it if not).
2. Pick the next not-yet-run config from run_benchmarks.CONFIGS; once
   all four are done, fall back to one trial of the W&B sweep instead
   (sweep id cached in config/sweep_id.txt after first creation), so
   the automation keeps producing new comparison points indefinitely.
3. Train it, append a row to results/history.json (the append-only
   JSON log), regenerate BENCHMARKS.md, and git commit + push.

Everything here is local-only by design -- there is no cloud runner
that can reach this machine's Postgres/MLflow, so this has to run on
the machine itself (Task Scheduler), not GitHub Actions.

Usage:
    uv run python scripts/weekly_pipeline.py
"""

import json
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import wandb
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src.model_training.train import train_one_run  # noqa: E402
from scripts.run_benchmarks import BASE, CONFIGS, WANDB_PROJECT  # noqa: E402

HISTORY_PATH = REPO_ROOT / "results" / "history.json"
SWEEP_ID_PATH = REPO_ROOT / "config" / "sweep_id.txt"
MLFLOW_LOG = REPO_ROOT / "mlflow_server.log"

with open(REPO_ROOT / "config" / "mlflow_config.yaml") as f:
    _mlflow_cfg = yaml.safe_load(f)["mlflow"]


def ensure_mlflow_server_running() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.5)
        if sock.connect_ex(("127.0.0.1", 5000)) == 0:
            print("MLflow server already up.")
            return

    print("MLflow server not running -- starting it.")
    log_file = open(MLFLOW_LOG, "a")
    subprocess.Popen(
        [
            "uv", "run", "mlflow", "server",
            "--backend-store-uri", _mlflow_cfg["backend_store_uri"],
            "--default-artifact-root", "./mlflow_artifacts_v1",
            "--host", "127.0.0.1", "--port", "5000",
        ],
        cwd=REPO_ROOT, stdout=log_file, stderr=log_file,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    for _ in range(30):
        time.sleep(2)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.5)
            if sock.connect_ex(("127.0.0.1", 5000)) == 0:
                print("MLflow server is up.")
                return
    raise RuntimeError("MLflow server did not come up within 60s")


def already_run_names() -> set[str]:
    import mlflow

    mlflow.set_tracking_uri(_mlflow_cfg["tracking_uri"])
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(_mlflow_cfg["experiment_name"])
    if exp is None:
        return set()
    runs = client.search_runs([exp.experiment_id], filter_string="status = 'FINISHED'")
    return {r.data.tags.get("mlflow.runName") for r in runs}


def next_fixed_config() -> dict | None:
    done = already_run_names()
    for overrides in CONFIGS:
        if overrides["run_name"] not in done:
            return {**BASE, **overrides}
    return None


def run_sweep_trial() -> dict:
    if SWEEP_ID_PATH.exists():
        sweep_id = SWEEP_ID_PATH.read_text().strip()
    else:
        with open(REPO_ROOT / "sweeps" / "sweep_config.yaml") as f:
            sweep_config = yaml.safe_load(f)
        sweep_id = wandb.sweep(sweep_config, project=WANDB_PROJECT)
        SWEEP_ID_PATH.write_text(sweep_id)
        print(f"Created new sweep: {sweep_id}")

    result_holder = {}

    def _train():
        wandb.init()
        config = dict(wandb.config)
        config.setdefault("manifest", BASE["manifest"])
        result = train_one_run(config)
        result_holder["config"] = config
        result_holder["result"] = result
        wandb.finish()

    wandb.agent(sweep_id, function=_train, count=1, project=WANDB_PROJECT)
    return {**result_holder.get("config", {}), **result_holder.get("result", {})}


def append_history(entry: dict) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    history = json.loads(HISTORY_PATH.read_text()) if HISTORY_PATH.exists() else []
    history.append(entry)
    HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")


def git(*args: str) -> None:
    subprocess.run(["git", *args], cwd=REPO_ROOT, check=True)


def main() -> None:
    ensure_mlflow_server_running()

    config = next_fixed_config()
    if config is not None:
        print(f"=== WEEKLY RUN: {config['run_name']} (fixed comparison config) ===")
        run_name = config["run_name"]
        wandb.init(project=WANDB_PROJECT, config=config, name=run_name, reinit=True)
        result = train_one_run(config)
        wandb.finish()
        entry = {**config, **result}
    else:
        print("=== WEEKLY RUN: all fixed configs done -- running one sweep trial ===")
        entry = run_sweep_trial()
        run_name = entry.get("run_name", "sweep_trial")

    entry["run_name"] = run_name
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    append_history(entry)
    print(f"Logged to {HISTORY_PATH}")

    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "generate_benchmarks_md.py")],
        check=True,
    )

    git("add", "BENCHMARKS.md", "results/history.json")
    diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO_ROOT)
    if diff.returncode == 0:
        print("No changes to commit.")
        return

    git(
        "commit", "-m",
        f"Weekly auto-benchmark: {run_name}\n\nAutomated run via scripts/weekly_pipeline.py (Task Scheduler).",
    )
    git("push")
    print("Pushed to GitHub.")


if __name__ == "__main__":
    main()
