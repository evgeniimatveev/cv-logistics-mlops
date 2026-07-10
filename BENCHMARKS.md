# Benchmarks — cv-logistics-bin-count

Auto-generated from the MLflow/Postgres backend -- run `uv run python scripts/generate_benchmarks_md.py` after new experiments to refresh. Don't hand-edit this file; edit the generator instead.

## Setup

**Task:** predict how many items (1-5) are in a warehouse bin photo -- 5-class classification on a 10,441-image subset of the Amazon Bin Image Dataset (8,352 train / 1,044 val / 1,045 test, stratified split).

**Model:** MobileNetV2 (ImageNet-pretrained), transfer learning with a `Dropout -> Linear(5)` head. `unfreeze_layers` controls how much of the backbone trains alongside the head: `0` = frozen (linear probe), `N` = last `N` of its 19 inverted-residual blocks unfrozen, `-1` = full fine-tune.

**Comparison method:** `scripts/run_benchmarks.py` trains all four `unfreeze_layers` settings back to back -- same data split, same 5 epochs, same batch size/dropout -- varying only `unfreeze_layers` and a correspondingly lower learning rate for the more-unfrozen configs (higher lr on a pretrained backbone with more trainable layers destroys the pretrained weights). `full_run_v1` is an earlier, differently-scoped run (frozen, 10 epochs) kept for reference, not part of the controlled comparison.

## Results

| Run | Backbone setting | Learning rate | Val accuracy | Val MAE | Best val loss |
|---|---|---|---|---|---|
| **bench_full** | -1 | 5e-05 | 0.379 | 0.827 | 1.3301 |
| bench_partial4 | 4 | 0.0001 | 0.342 | 0.884 | 1.3849 |
| bench_partial2 | 2 | 0.0003 | 0.319 | 0.914 | 1.4133 |
| full_run_v1 | True | 0.001 | 0.322 | 0.949 | 1.4652 |
| bench_frozen | 0 | 0.001 | 0.314 | 1.009 | 1.4752 |

## Key finding

As of this refresh, **bench_full** leads (`unfreeze_layers=-1`, val MAE 0.827). Across the controlled comparison, accuracy/MAE improved monotonically with more of the backbone unfrozen -- no diminishing returns yet within the 5-epoch budget tested. If the leader is the full fine-tune config, note its val_loss may still have been falling at the last epoch (not yet converged) -- worth a longer run to find its real ceiling.

`unfreeze_layers`: 0 = fully frozen backbone, N = last N blocks unfrozen, -1 = full fine-tune. `freeze_backbone`: legacy param name (True/False) logged by runs from before the unfreeze_layers refactor.
