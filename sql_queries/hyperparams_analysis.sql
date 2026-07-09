-- Average FINAL validation accuracy per hyperparameter value, across all runs.
-- Uses latest_metrics (final epoch per run) -- averaging the raw metrics
-- table would average over every epoch of every run, diluting the result.
SELECT
    p.key            AS hyperparam,
    p.value          AS param_value,
    COUNT(DISTINCT r.run_uuid) AS n_runs,
    AVG(lm.value)    AS avg_val_accuracy
FROM runs           r
JOIN params          p  ON r.run_uuid = p.run_uuid
JOIN latest_metrics   lm ON r.run_uuid = lm.run_uuid
JOIN experiments      e  ON r.experiment_id = e.experiment_id
WHERE lm.key = 'val_accuracy'
  AND e.name = 'cv_logistics_bin_count_v1'
  AND r.lifecycle_stage = 'active'  -- exclude soft-deleted runs
  AND p.key IN ('backbone', 'unfreeze_layers', 'freeze_backbone', 'dropout', 'learning_rate', 'batch_size')
GROUP BY p.key, p.value
ORDER BY p.key, avg_val_accuracy DESC;
