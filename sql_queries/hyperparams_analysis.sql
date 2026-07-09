-- Average validation accuracy per hyperparameter value, across all runs.
-- Useful for spotting which backbone / freeze_backbone / dropout setting wins.
SELECT
    p.key            AS hyperparam,
    p.value          AS param_value,
    COUNT(DISTINCT r.run_uuid) AS n_runs,
    AVG(m.value)     AS avg_val_accuracy
FROM runs    r
JOIN params  p ON r.run_uuid = p.run_uuid
JOIN metrics m ON r.run_uuid = m.run_uuid
JOIN experiments e ON r.experiment_id = e.experiment_id
WHERE m.key = 'val_accuracy'
  AND e.name = 'cv_logistics_bin_count_v1'
  AND p.key IN ('backbone', 'freeze_backbone', 'dropout', 'learning_rate', 'batch_size')
GROUP BY p.key, p.value
ORDER BY p.key, avg_val_accuracy DESC;
