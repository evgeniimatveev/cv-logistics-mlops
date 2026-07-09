-- Side-by-side comparison of all finished runs: one row per run with its
-- key hyperparams and final metrics, for eyeballing results directly in
-- DBeaver/pgAdmin without generating BENCHMARKS.md. Same logic as
-- scripts/generate_benchmarks_md.py, in pure SQL.
SELECT
    r.name AS run_name,
    MAX(CASE WHEN p.key = 'unfreeze_layers' THEN p.value
             WHEN p.key = 'freeze_backbone' THEN p.value END) AS backbone_setting,
    MAX(CASE WHEN p.key = 'learning_rate' THEN p.value END)   AS learning_rate,
    MAX(CASE WHEN lm.key = 'val_accuracy' THEN lm.value END)  AS val_accuracy,
    MAX(CASE WHEN lm.key = 'val_mae' THEN lm.value END)       AS val_mae,
    MAX(CASE WHEN lm.key = 'best_val_loss' THEN lm.value END) AS best_val_loss
FROM runs r
JOIN experiments      e  ON r.experiment_id = e.experiment_id
LEFT JOIN params       p  ON r.run_uuid = p.run_uuid
LEFT JOIN latest_metrics lm ON r.run_uuid = lm.run_uuid
WHERE e.name = 'cv_logistics_bin_count_v1'
  AND r.lifecycle_stage = 'active'
  AND r.status = 'FINISHED'
GROUP BY r.run_uuid, r.name
ORDER BY val_mae ASC NULLS LAST;
