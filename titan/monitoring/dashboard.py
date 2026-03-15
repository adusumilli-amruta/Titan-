"""
Health Dashboard Data Aggregator for Titan.

Consolidates system, inference, and experiment metrics into a unified
snapshot for monitoring dashboards (Grafana, custom UIs, or CLI tools).

Tracks:
- System resources: CPU, memory, disk usage
- GPU utilization and VRAM (via torch.cuda / nvidia-smi)
- Inference server metrics: latency, throughput, error rates
- Experiment status: running/completed experiments, latest checkpoints

Usage:
    dashboard = DashboardAggregator(
        metrics_collector=metrics,
        experiment_db=db,
    )
    snapshot = dashboard.get_dashboard_data()
"""

import os
import time
import logging
import subprocess
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DashboardAggregator:
    """
    Central aggregator that pulls metrics from multiple sources and
    returns a unified JSON-serializable snapshot for dashboards.

    Designed to be called periodically by a monitoring endpoint
    (e.g., the FastAPI /v1/dashboard route) or a Grafana data source.
    """

    def __init__(
        self,
        metrics_collector=None,
        experiment_db=None,
        gpu_monitoring: bool = True,
    ):
        """
        Args:
            metrics_collector: Instance of MetricsCollector for inference metrics.
            experiment_db: Instance of ExperimentDatabase for training history.
            gpu_monitoring: Whether to attempt GPU metric collection.
        """
        self.metrics_collector = metrics_collector
        self.experiment_db = experiment_db
        self.gpu_monitoring = gpu_monitoring
        self._start_time = time.time()

    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Collects system-level resource utilization.
        Uses psutil if available, with graceful fallback.
        """
        system = {
            "uptime_seconds": round(time.time() - self._start_time, 1),
            "pid": os.getpid(),
        }

        try:
            import psutil

            # CPU
            system["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            system["cpu_count"] = psutil.cpu_count()

            # Memory
            mem = psutil.virtual_memory()
            system["memory_total_gb"] = round(mem.total / (1024**3), 2)
            system["memory_used_gb"] = round(mem.used / (1024**3), 2)
            system["memory_percent"] = mem.percent

            # Disk
            disk = psutil.disk_usage("/")
            system["disk_total_gb"] = round(disk.total / (1024**3), 2)
            system["disk_used_gb"] = round(disk.used / (1024**3), 2)
            system["disk_percent"] = disk.percent

        except ImportError:
            system["note"] = "psutil not installed; limited system metrics"

        return system

    def get_gpu_metrics(self) -> List[Dict[str, Any]]:
        """
        Collects GPU metrics via torch.cuda and nvidia-smi.
        Returns a list of per-GPU metric dictionaries.
        """
        gpus = []

        if not self.gpu_monitoring:
            return gpus

        # Attempt torch.cuda metrics
        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_info = {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "memory_allocated_mb": round(
                            torch.cuda.memory_allocated(i) / 1e6, 1
                        ),
                        "memory_reserved_mb": round(
                            torch.cuda.memory_reserved(i) / 1e6, 1
                        ),
                        "memory_total_mb": round(
                            torch.cuda.get_device_properties(i).total_mem / 1e6, 1
                        ),
                    }
                    gpus.append(gpu_info)
        except (ImportError, RuntimeError):
            pass

        # Attempt nvidia-smi for utilization percentage
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 4:
                        idx = int(parts[0])
                        # Match to existing GPU entry or create new
                        if idx < len(gpus):
                            gpus[idx]["utilization_percent"] = float(parts[1])
                            gpus[idx]["temperature_c"] = float(parts[2])
                            gpus[idx]["power_watts"] = float(parts[3])
                        else:
                            gpus.append({
                                "index": idx,
                                "utilization_percent": float(parts[1]),
                                "temperature_c": float(parts[2]),
                                "power_watts": float(parts[3]),
                            })
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return gpus

    def get_inference_metrics(self) -> Optional[Dict[str, Any]]:
        """Pulls current inference metrics from the MetricsCollector."""
        if self.metrics_collector is None:
            return None
        return self.metrics_collector.get_metrics()

    def get_experiment_summary(self, limit: int = 5) -> Optional[Dict[str, Any]]:
        """
        Summarizes recent experiment status from the ExperimentDatabase.
        Returns counts by status and the most recent experiments.
        """
        if self.experiment_db is None:
            return None

        try:
            all_exps = self.experiment_db.list_experiments(limit=limit)

            # Count by status
            status_counts = {}
            for exp in all_exps:
                status = exp.get("status", "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1

            return {
                "total_recent": len(all_exps),
                "by_status": status_counts,
                "recent_experiments": [
                    {
                        "id": exp["id"],
                        "name": exp["name"],
                        "stage": exp["stage"],
                        "status": exp.get("status", "unknown"),
                    }
                    for exp in all_exps
                ],
            }
        except Exception as e:
            logger.warning(f"Failed to fetch experiment summary: {e}")
            return {"error": str(e)}

    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Returns a complete dashboard snapshot combining all metric sources.

        This is the primary method called by API endpoints or monitoring tools.

        Returns:
            A JSON-serializable dictionary with system, GPU, inference,
            and experiment metrics.
        """
        snapshot = {
            "timestamp": int(time.time()),
            "system": self.get_system_metrics(),
            "gpus": self.get_gpu_metrics(),
        }

        inference = self.get_inference_metrics()
        if inference is not None:
            snapshot["inference"] = inference

        experiments = self.get_experiment_summary()
        if experiments is not None:
            snapshot["experiments"] = experiments

        return snapshot

    def to_prometheus_format(self) -> str:
        """
        Exports system metrics in Prometheus text exposition format,
        complementing the MetricsCollector's inference metrics.
        """
        data = self.get_dashboard_data()
        lines = []

        # System metrics
        sys = data.get("system", {})
        if "cpu_percent" in sys:
            lines.extend([
                "# HELP titan_system_cpu_percent CPU utilization",
                "# TYPE titan_system_cpu_percent gauge",
                f'titan_system_cpu_percent {sys["cpu_percent"]}',
            ])
        if "memory_percent" in sys:
            lines.extend([
                "# HELP titan_system_memory_percent Memory utilization",
                "# TYPE titan_system_memory_percent gauge",
                f'titan_system_memory_percent {sys["memory_percent"]}',
            ])

        # GPU metrics
        for gpu in data.get("gpus", []):
            idx = gpu.get("index", 0)
            if "memory_allocated_mb" in gpu:
                lines.extend([
                    f"# HELP titan_gpu_memory_allocated_mb GPU VRAM allocated",
                    f"# TYPE titan_gpu_memory_allocated_mb gauge",
                    f'titan_gpu_memory_allocated_mb{{gpu="{idx}"}} {gpu["memory_allocated_mb"]}',
                ])
            if "utilization_percent" in gpu:
                lines.extend([
                    f"# HELP titan_gpu_utilization_percent GPU compute utilization",
                    f"# TYPE titan_gpu_utilization_percent gauge",
                    f'titan_gpu_utilization_percent{{gpu="{idx}"}} {gpu["utilization_percent"]}',
                ])

        return "\n".join(lines)
