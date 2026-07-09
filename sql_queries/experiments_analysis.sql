-- Run counts and status breakdown per experiment (sanity check across
-- both this project and mlops_project, since they share one MLflow server).
SELECT
    e.experiment_id,
    e.name                                   AS experiment_name,
    COUNT(r.run_uuid)                        AS total_runs,
    COUNT(*) FILTER (WHERE r.status = 'FINISHED') AS finished_runs,
    COUNT(*) FILTER (WHERE r.status = 'FAILED')   AS failed_runs
FROM experiments e
LEFT JOIN runs r ON e.experiment_id = r.experiment_id
GROUP BY e.experiment_id, e.name
ORDER BY total_runs DESC;
