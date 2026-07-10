# Benchmarks — cv-logistics-bin-count

Auto-generated from the MLflow/Postgres backend -- run `uv run python scripts/generate_benchmarks_md.py` after new experiments to refresh. Don't hand-edit this file; edit the generator instead.

## Setup

**Task:** predict how many items (1-5) are in a warehouse bin photo -- 5-class classification on a 10,441-image subset of the Amazon Bin Image Dataset (8,352 train / 1,044 val / 1,045 test, stratified split).

**Model:** MobileNetV2 or ResNet18 (ImageNet-pretrained), transfer learning with a `Dropout -> Linear(5)` head. `unfreeze_layers` controls how much of the backbone trains alongside the head: `0` = frozen (linear probe), `N` = last `N` backbone blocks unfrozen, `-1` = full fine-tune. Block granularity differs by architecture -- MobileNetV2 has 19 blocks, so `N=2` unfreezes ~40% of its parameters; ResNet18 only has 5 stage-groups, so the same `N=2` already covers ~94% of its parameters (`layer3`+`layer4`, where most of a ResNet's weights live). "Partial" isn't directly comparable across the two.

**Two sources of runs here:**
- `bench_*` / `full_run_v1` -- `scripts/run_benchmarks.py`, a controlled comparison: fixed MobileNetV2, same 5 epochs/batch size/dropout, only `unfreeze_layers` varies, with a hand-picked lower learning rate for the more-unfrozen configs (a flat lr across all of them would blow up full fine-tune -- see `misty-sweep-8`/`gallant-sweep-6` below for what that looks like when the sweep didn't know to scale it down).
- `*-sweep-N` -- `sweeps/sweep.py`, a 10-trial W&B Bayesian sweep over backbone, `unfreeze_layers`, learning rate, batch size, and dropout jointly, 4 epochs each. Wider net, less controlled.

## Results

| Run | Backbone setting | Learning rate | Val accuracy | Val MAE | Best val loss |
|---|---|---|---|---|---|
| **classic-sweep-4** | 2 | 0.0022685055152514267 | 0.414 | 0.755 | 1.3074 |
| champion_extended | 2 | 0.0022685055152514267 | 0.375 | 0.789 | 1.3005 |
| bench_full | -1 | 5e-05 | 0.379 | 0.827 | 1.3301 |
| eternal-sweep-6 | 2 | 0.0022366334139440284 | 0.399 | 0.844 | 1.3040 |
| bench_partial4 | 4 | 0.0001 | 0.342 | 0.884 | 1.3849 |
| dazzling-sweep-4 | 2 | 0.0020741135290263846 | 0.352 | 0.910 | 1.3737 |
| bench_partial2 | 2 | 0.0003 | 0.319 | 0.914 | 1.4133 |
| fancy-sweep-10 | 4 | 0.0029777404875780894 | 0.341 | 0.935 | 1.4400 |
| genial-sweep-7 | 4 | 0.004827951814035521 | 0.327 | 0.938 | 1.4457 |
| likely-sweep-9 | 4 | 0.0019986749099878223 | 0.339 | 0.944 | 1.4123 |
| full_run_v1 | True | 0.001 | 0.322 | 0.949 | 1.4652 |
| distinctive-sweep-5 | 2 | 0.000626198961943601 | 0.338 | 0.960 | 1.4021 |
| misty-sweep-8 | -1 | 0.0024941294234340707 | 0.301 | 0.966 | 1.4691 |
| denim-sweep-5 | 4 | 0.00249601374771328 | 0.332 | 0.998 | 1.3952 |
| gallant-sweep-6 | -1 | 0.0046866802375005 | 0.315 | 1.003 | 1.4407 |
| bench_frozen | 0 | 0.001 | 0.314 | 1.009 | 1.4752 |

## Key finding

As of this refresh, **classic-sweep-4** leads (`unfreeze_layers=2`, val MAE 0.755). In the controlled comparison (`bench_*`), accuracy/MAE improved monotonically with more of the backbone unfrozen, all the way to full fine-tune -- no frozen or lightly-unfrozen config won there. Every run here trained only 4-5 epochs, so rankings among them are relative signal, not converged final answers.

**Resolved:** `champion_extended` re-ran `classic-sweep-4`'s exact hyperparameters for 15 epochs instead of 4 to check whether it would keep improving. It didn't -- `val_loss` bottoms out around epoch 5 (1.30) then climbs steadily to 2.17 by epoch 15 while `train_loss` keeps falling the whole time (1.50 -> 0.62): textbook overfitting on the 8,352-image train split, not a config that just needed more epochs. The original 4-epoch result was close to the real optimum, not a lucky early stop.

`unfreeze_layers`: 0 = fully frozen backbone, N = last N blocks unfrozen, -1 = full fine-tune. `freeze_backbone`: legacy param name (True/False) logged by runs from before the unfreeze_layers refactor.
