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

-- Runs where accuracy improved but MAE got worse (model became more
-- "confidently wrong" on adjacent counts) -- worth a manual look.
SELECT
    r.run_uuid,
    MAX(CASE WHEN m.key = 'val_accuracy' THEN m.value END) AS val_accuracy,
    MAX(CASE WHEN m.key = 'val_mae' THEN m.value END)      AS val_mae
FROM runs r
JOIN metrics m ON r.run_uuid = m.run_uuid
JOIN experiments e ON r.experiment_id = e.experiment_id
WHERE e.name = 'cv_logistics_bin_count_v1'
  AND r.lifecycle_stage = 'active'  -- exclude soft-deleted runs
GROUP BY r.run_uuid
HAVING MAX(CASE WHEN m.key = 'val_accuracy' THEN m.value END) > 0.4
   AND MAX(CASE WHEN m.key = 'val_mae' THEN m.value END) > 1.0
ORDER BY val_mae DESC;
