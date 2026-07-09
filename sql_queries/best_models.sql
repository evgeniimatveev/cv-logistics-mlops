-- Best 5 runs by validation MAE (item-count prediction error, in units)
SELECT
    r.run_uuid,
    e.name          AS experiment_name,
    m.value         AS val_mae
FROM runs r
JOIN experiments e ON r.experiment_id = e.experiment_id
JOIN metrics     m ON r.run_uuid      = m.run_uuid
WHERE m.key = 'val_mae'
  AND e.name = 'cv_logistics_bin_count_v1'
  AND r.lifecycle_stage = 'active'  -- exclude soft-deleted runs
ORDER BY m.value ASC
LIMIT 5;
