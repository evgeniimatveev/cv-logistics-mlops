"""
Hybrid analytics layer: pulls this experiment's runs/params/metrics out of
the shared MLflow Postgres backend and mirrors them into a local DuckDB
file. Postgres stays the source of truth (and is what the running MLflow
server writes to); DuckDB is a disposable, fast local copy for ad-hoc SQL
without needing Postgres up or a network round-trip.

Usage:
    uv run python sql_queries/export_to_duckdb.py
    duckdb analytics/mlflow_runs.duckdb -c "select * from runs limit 5"
"""

from pathlib import Path

import duckdb
import psycopg2
import psycopg2.extras
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
with open(REPO_ROOT / "config" / "mlflow_config.yaml") as f:
    _mlflow_cfg = yaml.safe_load(f)["mlflow"]

PG_DSN = _mlflow_cfg["backend_store_uri"]
EXPERIMENT_NAME = _mlflow_cfg["experiment_name"]
DUCKDB_PATH = REPO_ROOT / "analytics" / "mlflow_runs.duckdb"

TABLES = {
    "experiments": "SELECT experiment_id, name, artifact_location, lifecycle_stage FROM experiments",
    "runs": """
        SELECT r.run_uuid, r.experiment_id, r.status, r.start_time, r.end_time, r.name
        FROM runs r JOIN experiments e ON r.experiment_id = e.experiment_id
        WHERE e.name = %(experiment_name)s AND r.lifecycle_stage = 'active'
    """,
    "metrics": """
        SELECT m.run_uuid, m.key, m.value, m.step
        FROM metrics m
        JOIN runs r ON m.run_uuid = r.run_uuid
        JOIN experiments e ON r.experiment_id = e.experiment_id
        WHERE e.name = %(experiment_name)s AND r.lifecycle_stage = 'active'
    """,
    "params": """
        SELECT p.run_uuid, p.key, p.value
        FROM params p
        JOIN runs r ON p.run_uuid = r.run_uuid
        JOIN experiments e ON r.experiment_id = e.experiment_id
        WHERE e.name = %(experiment_name)s AND r.lifecycle_stage = 'active'
    """,
}


def export() -> None:
    DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)

    pg_conn = psycopg2.connect(PG_DSN)
    duck_conn = duckdb.connect(str(DUCKDB_PATH))

    for table, query in TABLES.items():
        with pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, {"experiment_name": EXPERIMENT_NAME})
            rows = cur.fetchall()

        duck_conn.execute(f"DROP TABLE IF EXISTS {table}")
        if not rows:
            print(f"{table}: 0 rows (skipped, no schema to infer)")
            continue

        cols = list(rows[0].keys())
        duck_conn.execute(
            f"CREATE TABLE {table} ({', '.join(f'{c} VARCHAR' for c in cols)})"
        )
        duck_conn.executemany(
            f"INSERT INTO {table} VALUES ({', '.join('?' for _ in cols)})",
            [[str(r[c]) if r[c] is not None else None for c in cols] for r in rows],
        )
        print(f"{table}: {len(rows)} rows -> {DUCKDB_PATH}")

    pg_conn.close()
    duck_conn.close()


if __name__ == "__main__":
    export()
