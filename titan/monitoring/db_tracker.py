import os
import json
import sqlite3
import time
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ExperimentDatabase:
    """
    SQLite-backed experiment tracking database for local and on-prem deployments.

    Tracks:
    - Training runs with hyperparameters and final metrics
    - Per-step loss / accuracy curves
    - Checkpoint metadata and file locations
    - Evaluation benchmark results across model versions

    This serves as a lightweight alternative to Azure ML when running
    experiments on local clusters or HPC environments. Data can later
    be synced to Azure for long-term archival.

    Schema:
        experiments:     top-level training runs
        metrics:         per-step scalar metrics
        checkpoints:     saved model snapshots
        evaluations:     benchmark results per checkpoint
    """

    def __init__(self, db_path: str = "titan_experiments.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Creates the database schema if it doesn't exist."""
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    config TEXT,
                    status TEXT DEFAULT 'running',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    finished_at TIMESTAMP,
                    tags TEXT
                );

                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                );

                CREATE TABLE IF NOT EXISTS checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    path TEXT NOT NULL,
                    size_mb REAL,
                    azure_blob_url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                );

                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    checkpoint_step INTEGER NOT NULL,
                    benchmark TEXT NOT NULL,
                    score REAL NOT NULL,
                    details TEXT,
                    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                );

                CREATE INDEX IF NOT EXISTS idx_metrics_exp ON metrics(experiment_id, step);
                CREATE INDEX IF NOT EXISTS idx_evals_exp ON evaluations(experiment_id, benchmark);
            """)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ── Experiment Lifecycle ──────────────────────────────────────────────

    def create_experiment(self, experiment_id: str, name: str, stage: str,
                          config: dict = None, tags: dict = None) -> str:
        """Registers a new training experiment."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO experiments (id, name, stage, config, tags) VALUES (?, ?, ?, ?, ?)",
                (experiment_id, name, stage, json.dumps(config or {}), json.dumps(tags or {})),
            )
        logger.info(f"Created experiment: {experiment_id} ({name})")
        return experiment_id

    def finish_experiment(self, experiment_id: str, status: str = "completed"):
        """Marks an experiment as finished."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE experiments SET status=?, finished_at=CURRENT_TIMESTAMP WHERE id=?",
                (status, experiment_id),
            )

    def get_experiment(self, experiment_id: str) -> Optional[dict]:
        """Retrieves experiment metadata."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE id=?", (experiment_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_experiments(self, stage: str = None, limit: int = 50) -> List[dict]:
        """Lists recent experiments, optionally filtered by stage."""
        with self._connect() as conn:
            if stage:
                rows = conn.execute(
                    "SELECT * FROM experiments WHERE stage=? ORDER BY created_at DESC LIMIT ?",
                    (stage, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]

    # ── Metric Logging ────────────────────────────────────────────────────

    def log_metric(self, experiment_id: str, step: int, name: str, value: float):
        """Logs a single scalar metric for a given step."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO metrics (experiment_id, step, metric_name, metric_value) VALUES (?, ?, ?, ?)",
                (experiment_id, step, name, value),
            )

    def log_metrics(self, experiment_id: str, step: int, metrics: Dict[str, float]):
        """Logs multiple metrics for a single step (batch insert)."""
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO metrics (experiment_id, step, metric_name, metric_value) VALUES (?, ?, ?, ?)",
                [(experiment_id, step, k, v) for k, v in metrics.items()],
            )

    def get_metric_history(self, experiment_id: str, metric_name: str) -> List[dict]:
        """Retrieves the full history of a metric across training steps."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT step, metric_value FROM metrics WHERE experiment_id=? AND metric_name=? ORDER BY step",
                (experiment_id, metric_name),
            ).fetchall()
            return [{"step": r["step"], "value": r["metric_value"]} for r in rows]

    # ── Checkpoint Tracking ───────────────────────────────────────────────

    def log_checkpoint(self, experiment_id: str, step: int, path: str,
                       size_mb: float = None, azure_url: str = None):
        """Records a saved checkpoint with its location."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO checkpoints (experiment_id, step, path, size_mb, azure_blob_url) VALUES (?, ?, ?, ?, ?)",
                (experiment_id, step, path, size_mb, azure_url),
            )

    def get_checkpoints(self, experiment_id: str) -> List[dict]:
        """Lists all checkpoints for an experiment."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM checkpoints WHERE experiment_id=? ORDER BY step",
                (experiment_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ── Evaluation Results ────────────────────────────────────────────────

    def log_evaluation(self, experiment_id: str, checkpoint_step: int,
                       benchmark: str, score: float, details: dict = None):
        """Records a benchmark evaluation result."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO evaluations (experiment_id, checkpoint_step, benchmark, score, details) VALUES (?, ?, ?, ?, ?)",
                (experiment_id, checkpoint_step, benchmark, score, json.dumps(details or {})),
            )

    def get_evaluations(self, experiment_id: str) -> List[dict]:
        """Gets all evaluation results for an experiment."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM evaluations WHERE experiment_id=? ORDER BY evaluated_at",
                (experiment_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def compare_benchmarks(self, benchmark: str, limit: int = 10) -> List[dict]:
        """Compares scores across experiments for a specific benchmark."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT e.name, e.stage, ev.checkpoint_step, ev.score, ev.evaluated_at
                FROM evaluations ev
                JOIN experiments e ON ev.experiment_id = e.id
                WHERE ev.benchmark = ?
                ORDER BY ev.score DESC
                LIMIT ?
            """, (benchmark, limit)).fetchall()
            return [dict(r) for r in rows]


class MetricsCollector:
    """
    Prometheus-style metrics collector for the inference serving layer.

    Tracks:
    - Request latency histograms
    - Tokens generated per second
    - GPU memory utilization
    - Active request counts
    - Error rates by status code

    Can be scraped by Prometheus or exported to Azure Monitor.
    """

    def __init__(self):
        self._request_count = 0
        self._error_count = 0
        self._total_latency_ms = 0.0
        self._total_tokens = 0
        self._latency_buckets = {50: 0, 100: 0, 250: 0, 500: 0, 1000: 0, 5000: 0, 10000: 0}
        self._start_time = time.time()

    def record_request(self, latency_ms: float, tokens_generated: int, success: bool = True):
        """Records a completed inference request."""
        self._request_count += 1
        self._total_latency_ms += latency_ms
        self._total_tokens += tokens_generated

        if not success:
            self._error_count += 1

        # Histogram bucketing
        for bucket in sorted(self._latency_buckets.keys()):
            if latency_ms <= bucket:
                self._latency_buckets[bucket] += 1
                break

    def get_metrics(self) -> Dict[str, Any]:
        """Returns current metrics snapshot for monitoring dashboards."""
        uptime = time.time() - self._start_time
        avg_latency = self._total_latency_ms / max(self._request_count, 1)
        rps = self._request_count / max(uptime, 1)

        metrics = {
            "uptime_seconds": round(uptime, 1),
            "total_requests": self._request_count,
            "total_errors": self._error_count,
            "error_rate": round(self._error_count / max(self._request_count, 1), 4),
            "avg_latency_ms": round(avg_latency, 2),
            "requests_per_second": round(rps, 2),
            "total_tokens_generated": self._total_tokens,
            "tokens_per_second": round(self._total_tokens / max(uptime, 1), 1),
            "latency_histogram": self._latency_buckets,
        }

        # GPU metrics (if available)
        try:
            import torch
            if torch.cuda.is_available():
                metrics["gpu_memory_used_mb"] = round(torch.cuda.memory_allocated() / 1e6, 1)
                metrics["gpu_memory_reserved_mb"] = round(torch.cuda.memory_reserved() / 1e6, 1)
                metrics["gpu_utilization_pct"] = "N/A (requires nvidia-smi)"
        except ImportError:
            pass

        return metrics

    def to_prometheus_format(self) -> str:
        """Exports metrics in Prometheus text exposition format."""
        m = self.get_metrics()
        lines = [
            f'# HELP titan_requests_total Total inference requests',
            f'# TYPE titan_requests_total counter',
            f'titan_requests_total {m["total_requests"]}',
            f'# HELP titan_errors_total Total failed requests',
            f'# TYPE titan_errors_total counter',
            f'titan_errors_total {m["total_errors"]}',
            f'# HELP titan_latency_avg_ms Average inference latency',
            f'# TYPE titan_latency_avg_ms gauge',
            f'titan_latency_avg_ms {m["avg_latency_ms"]}',
            f'# HELP titan_tokens_per_second Generation throughput',
            f'# TYPE titan_tokens_per_second gauge',
            f'titan_tokens_per_second {m["tokens_per_second"]}',
        ]
        return "\n".join(lines)
