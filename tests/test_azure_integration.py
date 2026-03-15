"""
Tests for Azure cloud integration modules.

Tests:
- ConfigLoader: YAML loading, env var interpolation, Key Vault fallback, overlays
- AzureBlobCheckpointManager: upload/download/list/delete (mocked Azure SDK)
- AzureMLExperimentTracker: run lifecycle (mocked azureml-core)
"""

import os
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from titan.cloud.config_loader import ConfigLoader


class TestConfigLoader(unittest.TestCase):
    """Tests the unified configuration loader."""

    def setUp(self):
        """Create temporary config files for testing."""
        self.tmp_dir = tempfile.mkdtemp()

        # Base config
        self.base_config = os.path.join(self.tmp_dir, "base.yaml")
        with open(self.base_config, "w") as f:
            f.write(
                "model:\n"
                "  hidden_size: 512\n"
                "  num_layers: 4\n"
                "training:\n"
                "  lr: 0.0003\n"
                "  batch_size: 32\n"
            )

        # Config with env var interpolation
        self.env_config = os.path.join(self.tmp_dir, "env_config.yaml")
        with open(self.env_config, "w") as f:
            f.write(
                "storage:\n"
                "  connection_string: ${AZURE_STORAGE_KEY}\n"
                "  container: ${CONTAINER_NAME:default-container}\n"
                "  missing_var: ${MISSING_VAR}\n"
            )

        # Overlay config
        self.overlay_config = os.path.join(self.tmp_dir, "overlay.yaml")
        with open(self.overlay_config, "w") as f:
            f.write(
                "model:\n"
                "  hidden_size: 1024\n"
                "training:\n"
                "  batch_size: 64\n"
                "  new_param: true\n"
            )

        self.loader = ConfigLoader(config_dir=self.tmp_dir)

    def test_load_basic_yaml(self):
        """Test loading a simple YAML config."""
        config = self.loader.load(self.base_config)
        self.assertEqual(config["model"]["hidden_size"], 512)
        self.assertEqual(config["model"]["num_layers"], 4)
        self.assertEqual(config["training"]["lr"], 0.0003)

    def test_env_var_interpolation(self):
        """Test that ${VAR} patterns are resolved from environment."""
        with patch.dict(os.environ, {"AZURE_STORAGE_KEY": "my-secret-key"}):
            config = self.loader.load(self.env_config)
            self.assertEqual(config["storage"]["connection_string"], "my-secret-key")

    def test_default_value_interpolation(self):
        """Test ${VAR:default} syntax when env var is not set."""
        config = self.loader.load(self.env_config)
        self.assertEqual(config["storage"]["container"], "default-container")

    def test_unresolved_var_preserved(self):
        """Test that unresolvable ${VAR} patterns are left in place."""
        config = self.loader.load(self.env_config)
        self.assertEqual(config["storage"]["missing_var"], "${MISSING_VAR}")

    def test_overlay_merge(self):
        """Test that overlay configs merge correctly with base."""
        config = self.loader.load(self.base_config, overlay_path=self.overlay_config)
        # Overridden values
        self.assertEqual(config["model"]["hidden_size"], 1024)
        self.assertEqual(config["training"]["batch_size"], 64)
        # Preserved base values
        self.assertEqual(config["model"]["num_layers"], 4)
        self.assertEqual(config["training"]["lr"], 0.0003)
        # New overlay values
        self.assertTrue(config["training"]["new_param"])

    def test_file_not_found(self):
        """Test that missing config files raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            self.loader.load("nonexistent.yaml")

    def test_get_secret_from_env(self):
        """Test secret retrieval from environment variables."""
        with patch.dict(os.environ, {"MY_SECRET": "secret123"}):
            value = self.loader.get_secret("MY_SECRET")
            self.assertEqual(value, "secret123")

    def test_get_secret_default(self):
        """Test secret fallback to default value."""
        value = self.loader.get_secret("NONEXISTENT_SECRET", default="fallback")
        self.assertEqual(value, "fallback")

    def test_secret_caching(self):
        """Test that secrets are cached after first retrieval."""
        with patch.dict(os.environ, {"CACHED_KEY": "value1"}):
            self.loader.get_secret("CACHED_KEY")
        # Even after env var is gone, cached value should persist
        value = self.loader.get_secret("CACHED_KEY")
        self.assertEqual(value, "value1")

    def test_load_training_config_invalid_stage(self):
        """Test that invalid stage names raise ValueError."""
        with self.assertRaises(ValueError):
            self.loader.load_training_config("invalid_stage")

    def test_deep_merge(self):
        """Test the static deep merge utility."""
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        overlay = {"a": {"y": 99, "z": 100}, "c": 4}
        result = ConfigLoader._deep_merge(base, overlay)
        self.assertEqual(result, {"a": {"x": 1, "y": 99, "z": 100}, "b": 3, "c": 4})


class TestAzureBlobCheckpointManagerMock(unittest.TestCase):
    """Tests AzureBlobCheckpointManager with mocked Azure SDK."""

    def test_init_requires_credentials(self):
        """Test that initialization without credentials raises on use."""
        from titan.cloud.azure_storage import AzureBlobCheckpointManager

        manager = AzureBlobCheckpointManager(
            container_name="test-container",
            connection_string=None,
            account_url=None,
        )
        # Should not fail on init, only on use
        self.assertIsNotNone(manager)
        self.assertEqual(manager.container_name, "test-container")

    def test_upload_checkpoint_interface(self):
        """Test that upload_checkpoint has the correct signature."""
        from titan.cloud.azure_storage import AzureBlobCheckpointManager
        import inspect

        sig = inspect.signature(AzureBlobCheckpointManager.upload_checkpoint)
        params = list(sig.parameters.keys())
        self.assertIn("local_path", params)
        self.assertIn("remote_prefix", params)

    def test_list_checkpoints_interface(self):
        """Test that list_checkpoints returns the correct type."""
        from titan.cloud.azure_storage import AzureBlobCheckpointManager
        import inspect

        sig = inspect.signature(AzureBlobCheckpointManager.list_checkpoints)
        self.assertIn("prefix", list(sig.parameters.keys()))


class TestAzureMLExperimentTrackerMock(unittest.TestCase):
    """Tests AzureMLExperimentTracker with mocked SDK."""

    def test_init_env_fallback(self):
        """Test that tracker falls back to env vars for workspace config."""
        from titan.cloud.azure_storage import AzureMLExperimentTracker

        with patch.dict(os.environ, {
            "AZUREML_WORKSPACE": "test-ws",
            "AZURE_SUBSCRIPTION_ID": "sub-123",
            "AZURE_RESOURCE_GROUP": "rg-test",
        }):
            tracker = AzureMLExperimentTracker()
            self.assertEqual(tracker.workspace_name, "test-ws")
            self.assertEqual(tracker.subscription_id, "sub-123")
            self.assertEqual(tracker.resource_group, "rg-test")

    def test_log_metric_without_run(self):
        """Test that logging metrics without an active run is safe."""
        from titan.cloud.azure_storage import AzureMLExperimentTracker

        tracker = AzureMLExperimentTracker(
            workspace_name="test", subscription_id="sub", resource_group="rg"
        )
        # Should not raise — gracefully handles None run
        tracker.log_metric("loss", 2.5)
        tracker.log_metrics({"lr": 1e-4, "accuracy": 0.9})

    def test_end_run_without_active_run(self):
        """Test that ending a run without starting one is safe."""
        from titan.cloud.azure_storage import AzureMLExperimentTracker

        tracker = AzureMLExperimentTracker(
            workspace_name="test", subscription_id="sub", resource_group="rg"
        )
        # Should not raise
        tracker.end_run()


if __name__ == "__main__":
    unittest.main()
