-- Epoch-by-epoch val_loss curve for a given run (swap in a run_uuid from
-- best_models.sql). MLflow keeps one metrics row per (run, key, step), so
-- this reconstructs the training curve straight from Postgres.
SELECT
    step,
    value AS val_loss
FROM metrics
WHERE run_uuid = :run_uuid
  AND key = 'val_loss'
ORDER BY step;

-- Runs where FINAL accuracy is decent but FINAL MAE is still bad (model
-- became more "confidently wrong" on adjacent counts) -- worth a manual
-- look. Uses latest_metrics -- with the raw metrics table this would
-- match on the best accuracy epoch and worst MAE epoch independently,
-- which usually aren't even the same epoch.
SELECT
    r.run_uuid,
    r.name AS run_name,
    MAX(CASE WHEN lm.key = 'val_accuracy' THEN lm.value END) AS val_accuracy,
    MAX(CASE WHEN lm.key = 'val_mae' THEN lm.value END)      AS val_mae
FROM runs r
JOIN latest_metrics lm ON r.run_uuid = lm.run_uuid
JOIN experiments     e  ON r.experiment_id = e.experiment_id
WHERE e.name = 'cv_logistics_bin_count_v1'
  AND r.lifecycle_stage = 'active'  -- exclude soft-deleted runs
GROUP BY r.run_uuid, r.name
HAVING MAX(CASE WHEN lm.key = 'val_accuracy' THEN lm.value END) > 0.4
   AND MAX(CASE WHEN lm.key = 'val_mae' THEN lm.value END) > 1.0
ORDER BY val_mae DESC;
