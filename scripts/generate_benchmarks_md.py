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
    -- best_val_mae (val_mae at the best-val_loss epoch) is what the
    -- registered model artifact actually corresponds to; val_mae alone
    -- is just whatever epoch happened to log last, which is the same
    -- thing only if the run never started overfitting before it ended
    COALESCE(
        MAX(CASE WHEN lm.key = 'best_val_mae' THEN lm.value END),
        MAX(CASE WHEN lm.key = 'val_mae' THEN lm.value END)
    ) AS val_mae,
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
        "Auto-generated from the MLflow/Postgres backend -- run "
        "`uv run python scripts/generate_benchmarks_md.py` after new experiments to refresh. "
        "Don't hand-edit this file; edit the generator instead.",
        "",
        "## Setup",
        "",
        "**Task:** predict how many items (1-5) are in a warehouse bin photo -- 5-class "
        "classification on a 10,441-image subset of the Amazon Bin Image Dataset "
        "(8,352 train / 1,044 val / 1,045 test, stratified split).",
        "",
        "**Model:** MobileNetV2 or ResNet18 (ImageNet-pretrained), transfer learning "
        "with a `Dropout -> Linear(5)` head. `unfreeze_layers` controls how much of "
        "the backbone trains alongside the head: `0` = frozen (linear probe), `N` = "
        "last `N` backbone blocks unfrozen, `-1` = full fine-tune. Block granularity "
        "differs by architecture -- MobileNetV2 has 19 blocks, so `N=2` unfreezes "
        "~40% of its parameters; ResNet18 only has 5 stage-groups, so the same `N=2` "
        "already covers ~94% of its parameters (`layer3`+`layer4`, where most of a "
        "ResNet's weights live). \"Partial\" isn't directly comparable across the two.",
        "",
        "**Two sources of runs here:**",
        "- `bench_*` / `full_run_v1` -- `scripts/run_benchmarks.py`, a controlled "
        "comparison: fixed MobileNetV2, same 5 epochs/batch size/dropout, only "
        "`unfreeze_layers` varies, with a hand-picked lower learning rate for the "
        "more-unfrozen configs (a flat lr across all of them would blow up full "
        "fine-tune -- see `misty-sweep-8`/`gallant-sweep-6` below for what that "
        "looks like when the sweep didn't know to scale it down).",
        "- `*-sweep-N` -- `sweeps/sweep.py`, a 10-trial W&B Bayesian sweep over "
        "backbone, `unfreeze_layers`, learning rate, batch size, and dropout "
        "jointly, 4 epochs each. Wider net, less controlled.",
        "",
        "## Results",
        "",
        "| Run | Backbone setting | Learning rate | Val accuracy | Val MAE | Best val loss |",
        "|---|---|---|---|---|---|",
    ]
    best_row = None
    for r in rows:
        val_acc = f"{float(r['val_accuracy']):.3f}" if r["val_accuracy"] is not None else "-"
        val_mae = f"{float(r['val_mae']):.3f}" if r["val_mae"] is not None else "-"
        best_loss = f"{float(r['best_val_loss']):.4f}" if r["best_val_loss"] is not None else "-"

        # rows are pre-sorted by val_mae ASC -- the first one with a real
        # value is the best; bold it so it stands out in the table
        is_best = best_row is None and r["val_mae"] is not None
        if is_best:
            best_row = r
        run_name = f"**{r['run_name']}**" if is_best else (r["run_name"] or "-")

        lines.append(
            f"| {run_name} | {r['backbone_setting'] or '-'} "
            f"| {r['learning_rate'] or '-'} | {val_acc} | {val_mae} | {best_loss} |"
        )

    lines.append("")

    if best_row is not None:
        lines.append("## Key finding")
        lines.append("")
        lines.append(
            f"As of this refresh, **{best_row['run_name']}** leads "
            f"(`unfreeze_layers={best_row['backbone_setting']}`, "
            f"val MAE {float(best_row['val_mae']):.3f}). In the controlled "
            "comparison (`bench_*`), accuracy/MAE improved monotonically with more "
            "of the backbone unfrozen, all the way to full fine-tune -- no frozen "
            "or lightly-unfrozen config won there. Every run here trained only 4-5 "
            "epochs, so rankings among them are relative signal, not converged "
            "final answers."
        )
        lines.append("")
        lines.append(
            "**Resolved:** `champion_extended` re-ran `classic-sweep-4`'s exact "
            "hyperparameters for 15 epochs instead of 4 to check whether it "
            "would keep improving. It didn't -- `val_loss` bottoms out around "
            "epoch 5 (1.30) then climbs steadily to 2.17 by epoch 15 while "
            "`train_loss` keeps falling the whole time (1.50 -> 0.62): textbook "
            "overfitting on the 8,352-image train split, not a config that "
            "just needed more epochs. The original 4-epoch result was close to "
            "the real optimum, not a lucky early stop."
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
