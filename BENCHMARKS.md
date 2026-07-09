# Benchmarks — cv-logistics-bin-count

Bin item-count classification (5 classes), MobileNetV2 transfer learning. Auto-generated from the MLflow/Postgres backend -- run `uv run python scripts/generate_benchmarks_md.py` after new experiments to refresh.

| Run | Backbone setting | Learning rate | Val accuracy | Val MAE | Best val loss |
|---|---|---|---|---|---|
| full_run_v1 | True | 0.001 | 0.322 | 0.949 | 1.4652 |

`unfreeze_layers`: 0 = fully frozen backbone, N = last N blocks unfrozen, -1 = full fine-tune. `freeze_backbone`: legacy param name (True/False) logged by runs from before the unfreeze_layers refactor.
