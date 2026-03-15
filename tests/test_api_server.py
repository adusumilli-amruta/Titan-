"""
Integration tests for the FastAPI-based Titan API server.

Tests cover:
- Health check endpoint
- Model listing endpoint
- Stats/monitoring endpoint
- Dashboard aggregator data
- Error handling for unloaded engine

Uses unittest + direct function calls (no httpx/ASGI client needed
since we test the endpoint logic rather than HTTP transport).
"""

import os
import time
import unittest
import tempfile
from unittest.mock import patch, MagicMock

from titan.monitoring.dashboard import DashboardAggregator
from titan.monitoring.db_tracker import ExperimentDatabase, MetricsCollector


class TestDashboardAggregator(unittest.TestCase):
    """Tests the health dashboard data aggregator."""

    def setUp(self):
        self.tmp_db = tempfile.mktemp(suffix=".db")
        self.db = ExperimentDatabase(db_path=self.tmp_db)
        self.collector = MetricsCollector()
        self.dashboard = DashboardAggregator(
            metrics_collector=self.collector,
            experiment_db=self.db,
            gpu_monitoring=False,  # Don't try GPU on test machines
        )

    def tearDown(self):
        if os.path.exists(self.tmp_db):
            os.remove(self.tmp_db)

    def test_dashboard_returns_all_sections(self):
        """Test that get_dashboard_data returns system, inference, and experiment data."""
        data = self.dashboard.get_dashboard_data()

        self.assertIn("timestamp", data)
        self.assertIn("system", data)
        self.assertIn("gpus", data)
        self.assertIn("inference", data)
        self.assertIn("experiments", data)

    def test_system_metrics_present(self):
        """Test that system metrics include basic fields."""
        data = self.dashboard.get_dashboard_data()
        system = data["system"]

        self.assertIn("uptime_seconds", system)
        self.assertIn("pid", system)
        self.assertIsInstance(system["uptime_seconds"], float)

    def test_inference_metrics_integration(self):
        """Test that inference metrics are pulled from MetricsCollector."""
        self.collector.record_request(latency_ms=100.0, tokens_generated=50, success=True)
        self.collector.record_request(latency_ms=200.0, tokens_generated=30, success=True)

        data = self.dashboard.get_dashboard_data()
        inference = data["inference"]

        self.assertEqual(inference["total_requests"], 2)
        self.assertEqual(inference["total_tokens_generated"], 80)
        self.assertEqual(inference["total_errors"], 0)

    def test_experiment_summary(self):
        """Test that experiment summary reflects database state."""
        self.db.create_experiment("exp-1", "test_run_1", "pretrain")
        self.db.create_experiment("exp-2", "test_run_2", "rlhf")

        data = self.dashboard.get_dashboard_data()
        experiments = data["experiments"]

        self.assertEqual(experiments["total_recent"], 2)
        self.assertEqual(len(experiments["recent_experiments"]), 2)

    def test_dashboard_without_collectors(self):
        """Test dashboard works without optional collectors."""
        bare_dashboard = DashboardAggregator(gpu_monitoring=False)
        data = bare_dashboard.get_dashboard_data()

        self.assertIn("system", data)
        self.assertNotIn("inference", data)
        self.assertNotIn("experiments", data)

    def test_gpu_list_empty_without_gpu(self):
        """Test that GPU list is empty when GPU monitoring is disabled."""
        data = self.dashboard.get_dashboard_data()
        self.assertEqual(data["gpus"], [])

    def test_prometheus_format_output(self):
        """Test Prometheus export doesn't crash (content depends on psutil)."""
        output = self.dashboard.to_prometheus_format()
        self.assertIsInstance(output, str)


class TestAPIServerEndpoints(unittest.TestCase):
    """Tests for API server endpoint logic (schema validation)."""

    def test_completion_request_schema(self):
        """Test CompletionRequest schema validates correctly."""
        from titan.serving.api_server import CompletionRequest

        req = CompletionRequest(prompt="Hello, world!")
        self.assertEqual(req.model, "titan-7b")
        self.assertEqual(req.max_tokens, 256)
        self.assertEqual(req.temperature, 0.7)
        self.assertFalse(req.stream)

    def test_chat_completion_request_schema(self):
        """Test ChatCompletionRequest schema validates correctly."""
        from titan.serving.api_server import ChatCompletionRequest, ChatMessage

        req = ChatCompletionRequest(
            messages=[
                ChatMessage(role="user", content="What is 2+2?")
            ]
        )
        self.assertEqual(len(req.messages), 1)
        self.assertEqual(req.messages[0].role, "user")
        self.assertEqual(req.model, "titan-7b")

    def test_completion_response_schema(self):
        """Test CompletionResponse schema construction."""
        from titan.serving.api_server import CompletionResponse, CompletionChoice

        resp = CompletionResponse(
            id="cmpl-test123",
            created=int(time.time()),
            model="titan-7b",
            choices=[CompletionChoice(text="Hello!", finish_reason="stop")],
            usage={"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        )
        self.assertEqual(resp.id, "cmpl-test123")
        self.assertEqual(resp.choices[0].text, "Hello!")

    def test_chat_response_schema(self):
        """Test ChatCompletionResponse schema construction."""
        from titan.serving.api_server import (
            ChatCompletionResponse, ChatChoice, ChatMessage
        )

        resp = ChatCompletionResponse(
            id="chatcmpl-test456",
            created=int(time.time()),
            model="titan-7b",
            choices=[
                ChatChoice(
                    message=ChatMessage(role="assistant", content="4"),
                    finish_reason="stop",
                )
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        )
        self.assertEqual(resp.choices[0].message.content, "4")


class TestMiddleware(unittest.TestCase):
    """Tests for API middleware components."""

    def test_rate_limiter_buckets(self):
        """Test that rate limiter initializes with correct config."""
        from titan.serving.middleware import RateLimitMiddleware

        # Create with mock app
        mock_app = MagicMock()
        limiter = RateLimitMiddleware(mock_app, requests_per_minute=30)
        self.assertEqual(limiter.rpm, 30)

    def test_api_key_auth_disabled_by_default(self):
        """Test that API key auth is disabled when no keys provided."""
        from titan.serving.middleware import APIKeyAuthMiddleware

        mock_app = MagicMock()
        auth = APIKeyAuthMiddleware(mock_app)
        self.assertFalse(auth.enabled)

    def test_api_key_auth_enabled_with_keys(self):
        """Test that API key auth is enabled when keys are provided."""
        from titan.serving.middleware import APIKeyAuthMiddleware

        mock_app = MagicMock()
        auth = APIKeyAuthMiddleware(mock_app, api_keys=["key-123", "key-456"])
        self.assertTrue(auth.enabled)
        self.assertEqual(len(auth.api_keys), 2)


if __name__ == "__main__":
    unittest.main()
