# CV Logistics — Bin Item-Count · MLflow · W&B · PostgreSQL + DuckDB

![CV](https://img.shields.io/badge/Computer_Vision-Logistics-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange?logo=mlflow)
![W&B](https://img.shields.io/badge/W%26B-Sweeps-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-Transfer_Learning-red?logo=pytorch)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Experiments-blue?logo=postgresql)
![DuckDB](https://img.shields.io/badge/DuckDB-Local_Analytics-yellow?logo=duckdb)
![Status](https://img.shields.io/badge/Status-In_Progress-yellow)

## What This Project Does

A warehouse robot photographs a bin and the model predicts how many items are inside it — the same counting problem Amazon Robotics uses to sanity-check inventory without a human opening every bin. Companion project to [mlops_project](https://github.com/evgeniimatveev/mlops_project) (XGBoost/tabular); this one swaps the model for a CNN and the dataset for images, but reuses the exact same tracking stack.

**Dataset:** [Amazon Bin Image Dataset](https://registry.opendata.aws/amazon-bin-imagery/) — a curated ~10,441-image subset (bin counts 1-5), same subset used in the Udacity "Inventory Monitoring at Distribution Centers" capstone. Full ABID is 500k+ images / 50+ GB; this subset is ~450MB, chosen to fit comfortably on this machine's disk and train on CPU or a 2GB-VRAM GPU.

**Task:** 5-class classification (predict count 1-5) via transfer learning on a frozen/fine-tuned MobileNetV2 or ResNet18 backbone.

---

## Architecture

```mermaid
flowchart LR
    subgraph Data
        S3[("S3: aft-vbi-pds\n~10,441 images")] --> DL["download_data"]
        DL --> CD["clean_data\nmanifest + train/val/test split"]
    end

    subgraph Training
        CD --> MT["model_training\nMobileNetV2 / ResNet18"]
    end

    subgraph Tracking
        MT --> MLF[("MLflow\nPostgres backend")]
        MT --> WB[("W&B\ncloud")]
    end

    subgraph Analytics
        MLF --> SQL["Postgres SQL"]
        MLF --> DDB[("DuckDB\nlocal mirror")]
    end

    MT --> API["FastAPI\ninference"]
```

---

## Results

First baseline (`full_run_v1`, frozen MobileNetV2 backbone, 10 epochs): **32.2% val accuracy, 0.95 val MAE**. `train_loss`/`train_accuracy` improved steadily while `val_*` stayed flat and noisy — the classic signature of a linear-probe ceiling: a single trainable layer on top of frozen, generic ImageNet features can only get so far at a task ImageNet was never trained for (guessing item counts in visual clutter, not "what object is this").

![W&B single-run dashboard](docs/screenshots/wandb_full_run_v1.png)

The comparison benchmark (`scripts/run_benchmarks.py`, same 5 epochs each, only `unfreeze_layers` + a correspondingly lower learning rate change) confirms it — and the result is monotonic: the more of the backbone is trainable, the better every metric gets, all the way to full fine-tune.

![W&B run comparison overlay, all 4 configs](docs/screenshots/wandb_all_configs_comparison.png)

| Config | Epochs | `unfreeze_layers` | Val accuracy | Val MAE | Best val loss |
|---|---|---|---|---|---|
| **Full fine-tune** | 5 | -1 | **37.9%** | **0.827** | **1.330** |
| Partial fine-tune, 4 blocks | 5 | 4 | 34.2% | 0.884 | 1.385 |
| Partial fine-tune, 2 blocks | 5 | 2 | 31.9% | 0.914 | 1.413 |
| Frozen, 10 epochs (original baseline) | 10 | 0 | 32.2% | 0.949 | 1.465 |
| Frozen (5 epochs) | 5 | 0 | 31.4% | 1.009 | 1.475 |

Open question: full fine-tune's `val_loss` was still falling at epoch 5 (no sign yet of overfitting the 8,352-image train split), so this may not be the ceiling — worth another run with more epochs and/or early stopping to find out.

Best model (`bench_full`, v2) is registered in the MLflow Model Registry as `cv_logistics_bin_count`, promoted to the `champion` alias — `src/model_deployment/app.py` serves whatever version currently holds that alias, no redeploy needed when a better run comes along.

Live comparison generated straight from Postgres: [`BENCHMARKS.md`](BENCHMARKS.md) (refresh with `uv run python scripts/generate_benchmarks_md.py`), or query `sql_queries/run_comparison.sql` directly.

---

## Why Postgres *and* DuckDB

One MLflow tracking server (shared with `mlops_project`) writes every run to a single Postgres `mlflow_db` — that's the source of truth. `sql_queries/export_to_duckdb.py` pulls this experiment's runs/params/metrics out of Postgres into a local DuckDB file, so you can do fast ad-hoc SQL against experiment history without a live Postgres connection.

```bash
uv run python sql_queries/export_to_duckdb.py
duckdb analytics/mlflow_runs.duckdb -c "select * from runs limit 5"
```

### Best runs by validation MAE (Postgres)

```sql
SELECT r.run_uuid, r.name AS run_name, lm.value AS val_mae
FROM runs r
JOIN experiments     e  ON r.experiment_id = e.experiment_id
JOIN latest_metrics   lm ON r.run_uuid      = lm.run_uuid
WHERE lm.key = 'val_mae'
  AND e.name = 'cv_logistics_bin_count_v1'
  AND r.lifecycle_stage = 'active'
ORDER BY lm.value ASC
LIMIT 5;
```

(Uses `latest_metrics`, not `metrics` — the raw table has one row per epoch, so ordering by it directly mixes epochs across runs once more than one run exists.)

More in `sql_queries/` — hyperparameter impact, per-experiment run counts, epoch-level learning curves.

---

## Project Structure

```
cv-logistics-mlops/
├── src/
│   ├── download_data/        # pulls ABID subset from public S3
│   ├── clean_data/           # validates images, builds train/val/test manifest
│   ├── model_training/       # CNN training loop, MLflow + W&B logging
│   ├── model_deployment/     # FastAPI inference endpoint
│   └── utils/                # Dataset + model-factory shared by training & sweeps
├── sql_queries/               # Postgres analytics + Postgres->DuckDB export
├── sweeps/                    # W&B Bayesian sweep (backbone, lr, batch size, dropout)
├── scripts/                   # benchmark runner + BENCHMARKS.md generator
├── config/                    # mlflow_config.yaml, duckdb_config.yaml
├── models/                    # saved checkpoints (gitignored)
├── docs/screenshots/          # README images
└── .github/workflows/         # CI
```

---

## How to Run

```bash
# 1. Environment (uv-managed, Python 3.11)
uv sync

# 2. Start MLflow tracking server (shared Postgres backend — same DB mlops_project uses)
uv run mlflow server \
  --backend-store-uri "postgresql://mlflow_user:mlflow_pass123@localhost/mlflow_db" \
  --default-artifact-root "./mlflow_artifacts_v1" \
  --host 127.0.0.1 --port 5000
# Open http://localhost:5000

# 3. Pipeline
uv run python src/download_data/run.py          # ~10,441 images, ~450MB
uv run python src/clean_data/run.py              # builds data/processed/manifest.csv
uv run python -m src.model_training.run --epochs 8 --backbone mobilenet_v2

# 4. W&B hyperparameter sweep (10 Bayesian runs)
uv run python sweeps/sweep.py

# 4b. Or: fixed frozen/partial/full fine-tune comparison + results table
uv run python scripts/run_benchmarks.py
uv run python scripts/generate_benchmarks_md.py   # refreshes BENCHMARKS.md

# 5. Analytics
uv run python sql_queries/export_to_duckdb.py
# then run sql_queries/*.sql against Postgres (DBeaver) or DuckDB directly

# 6. Serve the trained model
uv run uvicorn src.model_deployment.app:app --port 8000
```

---

## Stack

| Layer | Technology |
|-------|-----------|
| Model | PyTorch (MobileNetV2 / ResNet18 transfer learning) |
| Experiment Tracking | MLflow (Postgres backend, shared server) + W&B (cloud) |
| Hyperparameter Tuning | W&B Sweeps (Bayesian) |
| Experiment Analytics | PostgreSQL (source of truth) + DuckDB (local fast queries) |
| Serving | FastAPI |
| CI/CD | GitHub Actions |
| Package management | uv |

---

## Connect

- GitHub: [evgeniimatveev](https://github.com/evgeniimatveev)
- Portfolio: [datascienceportfol.io/evgeniimatveevusa](https://www.datascienceportfol.io/evgeniimatveevusa)
- LinkedIn: [Evgenii Matveev](https://www.linkedin.com/in/evgenii-matveev-510926276/)
