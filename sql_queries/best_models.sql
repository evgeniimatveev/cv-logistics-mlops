-- Best 5 runs by validation MAE (item-count prediction error, in units).
-- Uses latest_metrics (one row per run+key = the final epoch's value) --
-- the raw metrics table has one row per epoch, so ORDER BY + LIMIT on it
-- picks the best epochs across runs, not the best runs.
SELECT
    r.run_uuid,
    r.name          AS run_name,
    e.name          AS experiment_name,
    lm.value        AS val_mae
FROM runs r
JOIN experiments     e  ON r.experiment_id = e.experiment_id
JOIN latest_metrics   lm ON r.run_uuid      = lm.run_uuid
WHERE lm.key = 'val_mae'
  AND e.name = 'cv_logistics_bin_count_v1'
  AND r.lifecycle_stage = 'active'  -- exclude soft-deleted runs
ORDER BY lm.value ASC
LIMIT 5;
