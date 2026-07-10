-- Best 5 runs by validation MAE (item-count prediction error, in units).
-- Uses latest_metrics (one row per run+key = the final epoch's value) --
-- the raw metrics table has one row per epoch, so ORDER BY + LIMIT on it
-- picks the best epochs across runs, not the best runs.
--
-- Ranks by best_val_mae (val_mae at the best-val_loss epoch) when
-- available, falling back to val_mae (final epoch) for older runs --
-- the registered model's weights are the best checkpoint, not
-- whatever the last epoch happened to be, so ranking should match.
SELECT
    r.run_uuid,
    r.name AS run_name,
    e.name AS experiment_name,
    COALESCE(
        MAX(CASE WHEN lm.key = 'best_val_mae' THEN lm.value END),
        MAX(CASE WHEN lm.key = 'val_mae' THEN lm.value END)
    ) AS val_mae
FROM runs r
JOIN experiments     e  ON r.experiment_id = e.experiment_id
JOIN latest_metrics   lm ON r.run_uuid      = lm.run_uuid
WHERE e.name = 'cv_logistics_bin_count_v1'
  AND r.lifecycle_stage = 'active'  -- exclude soft-deleted runs
GROUP BY r.run_uuid, r.name, e.name
ORDER BY val_mae ASC
LIMIT 5;
