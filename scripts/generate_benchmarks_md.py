"""
Pulls this experiment's finished runs out of Postgres and writes a
markdown comparison table to BENCHMARKS.md at the repo root, so
results are visible on GitHub without opening MLflow or W&B.

Usage:
    uv run python scripts/generate_benchmarks_md.py
"""

from pathlib import Path

import psycopg2
import psycopg2.extras
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
with open(REPO_ROOT / "config" / "mlflow_config.yaml") as f:
    _mlflow_cfg = yaml.safe_load(f)["mlflow"]

QUERY = """
SELECT
    r.name AS run_name,
    r.start_time,
    MAX(CASE WHEN p.key = 'unfreeze_layers' THEN p.value
             WHEN p.key = 'freeze_backbone' THEN p.value END) AS backbone_setting,
    MAX(CASE WHEN p.key = 'learning_rate' THEN p.value END)   AS learning_rate,
    MAX(CASE WHEN lm.key = 'val_accuracy' THEN lm.value END)  AS val_accuracy,
    MAX(CASE WHEN lm.key = 'val_mae' THEN lm.value END)       AS val_mae,
    MAX(CASE WHEN lm.key = 'best_val_loss' THEN lm.value END) AS best_val_loss
FROM runs r
JOIN experiments e ON r.experiment_id = e.experiment_id
LEFT JOIN params         p ON r.run_uuid = p.run_uuid
LEFT JOIN latest_metrics lm ON r.run_uuid = lm.run_uuid
WHERE e.name = %(experiment_name)s
  AND r.lifecycle_stage = 'active'
  AND r.status = 'FINISHED'
GROUP BY r.name, r.run_uuid, r.start_time
ORDER BY val_mae ASC NULLS LAST
"""


def main() -> None:
    conn = psycopg2.connect(_mlflow_cfg["backend_store_uri"])
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(QUERY, {"experiment_name": _mlflow_cfg["experiment_name"]})
        rows = cur.fetchall()
    conn.close()

    lines = [
        "# Benchmarks — cv-logistics-bin-count",
        "",
        "Bin item-count classification (5 classes), MobileNetV2 transfer learning. "
        "Auto-generated from the MLflow/Postgres backend -- run "
        "`uv run python scripts/generate_benchmarks_md.py` after new experiments to refresh.",
        "",
        "| Run | Backbone setting | Learning rate | Val accuracy | Val MAE | Best val loss |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        val_acc = f"{float(r['val_accuracy']):.3f}" if r["val_accuracy"] is not None else "-"
        val_mae = f"{float(r['val_mae']):.3f}" if r["val_mae"] is not None else "-"
        best_loss = f"{float(r['best_val_loss']):.4f}" if r["best_val_loss"] is not None else "-"
        lines.append(
            f"| {r['run_name'] or '-'} | {r['backbone_setting'] or '-'} "
            f"| {r['learning_rate'] or '-'} | {val_acc} | {val_mae} | {best_loss} |"
        )

    lines.append("")
    lines.append(
        "`unfreeze_layers`: 0 = fully frozen backbone, N = last N blocks unfrozen, "
        "-1 = full fine-tune. `freeze_backbone`: legacy param name (True/False) "
        "logged by runs from before the unfreeze_layers refactor."
    )

    out_path = REPO_ROOT / "BENCHMARKS.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
