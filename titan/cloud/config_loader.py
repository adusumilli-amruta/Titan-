"""
Unified Configuration Loader with Azure Key Vault Integration.

Provides a single entry point for loading YAML-based configs with:
- Environment variable interpolation (${VAR_NAME} syntax)
- Azure Key Vault secret resolution for sensitive values
- Config overlay support (base + environment-specific overrides)
- Graceful fallback to environment variables when Key Vault is unavailable

Usage:
    loader = ConfigLoader(keyvault_url="https://my-vault.vault.azure.net")
    config = loader.load("configs/7b_pretrain.yaml")
    secret = loader.get_secret("AZURE_STORAGE_KEY")
"""

import os
import re
import logging
from typing import Any, Dict, Optional
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Production configuration manager for Titan training and serving.

    Supports three layers of configuration (in order of precedence):
    1. Environment variables (highest priority)
    2. Azure Key Vault secrets
    3. YAML config files (lowest priority)

    This design follows 12-factor app principles and integrates with
    Azure-native secrets management for enterprise deployments.
    """

    # Regex for ${VAR_NAME} or ${VAR_NAME:default_value} interpolation
    _ENV_VAR_PATTERN = re.compile(r"\$\{([^}:]+)(?::([^}]*))?\}")

    def __init__(
        self,
        keyvault_url: Optional[str] = None,
        config_dir: str = "configs",
    ):
        """
        Args:
            keyvault_url: Azure Key Vault URL (e.g., https://my-vault.vault.azure.net).
                          If None, falls back to AZURE_KEYVAULT_URL env var.
            config_dir: Default directory for YAML config files.
        """
        self.keyvault_url = keyvault_url or os.environ.get("AZURE_KEYVAULT_URL")
        self.config_dir = Path(config_dir)
        self._kv_client = None
        self._secret_cache: Dict[str, str] = {}

    def _get_kv_client(self):
        """Lazily initializes the Azure Key Vault SecretClient."""
        if self._kv_client is None and self.keyvault_url:
            try:
                from azure.keyvault.secrets import SecretClient
                from azure.identity import DefaultAzureCredential

                credential = DefaultAzureCredential()
                self._kv_client = SecretClient(
                    vault_url=self.keyvault_url, credential=credential
                )
                logger.info(f"Connected to Azure Key Vault: {self.keyvault_url}")
            except ImportError:
                logger.warning(
                    "azure-keyvault-secrets not installed. "
                    "Falling back to environment variables for secrets."
                )
            except Exception as e:
                logger.warning(
                    f"Failed to connect to Key Vault: {e}. "
                    "Falling back to environment variables."
                )
        return self._kv_client

    def get_secret(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """
        Retrieves a secret with the following precedence:
        1. Environment variable
        2. Azure Key Vault
        3. Default value

        Args:
            name: Secret name (used as both env var name and KV secret name).
            default: Fallback value if secret is not found anywhere.

        Returns:
            The secret value, or default if not found.
        """
        # Check cache first
        if name in self._secret_cache:
            return self._secret_cache[name]

        # 1. Environment variable (highest precedence)
        env_value = os.environ.get(name)
        if env_value is not None:
            self._secret_cache[name] = env_value
            return env_value

        # 2. Azure Key Vault
        kv_client = self._get_kv_client()
        if kv_client:
            try:
                # Key Vault doesn't allow underscores; convert to hyphens
                kv_name = name.replace("_", "-")
                secret = kv_client.get_secret(kv_name)
                self._secret_cache[name] = secret.value
                logger.debug(f"Retrieved secret '{name}' from Key Vault")
                return secret.value
            except Exception:
                logger.debug(f"Secret '{name}' not found in Key Vault")

        # 3. Default
        return default

    def _interpolate(self, value: Any) -> Any:
        """
        Recursively interpolates ${VAR} and ${VAR:default} patterns
        in strings, dicts, and lists.
        """
        if isinstance(value, str):
            def replacer(match):
                var_name = match.group(1)
                default_val = match.group(2)  # May be None
                resolved = self.get_secret(var_name, default=default_val)
                if resolved is None:
                    logger.warning(
                        f"Unresolved config variable: ${{{var_name}}}. "
                        "Set it as an env var or add to Key Vault."
                    )
                    return match.group(0)  # Leave unresolved
                return resolved

            return self._ENV_VAR_PATTERN.sub(replacer, value)

        elif isinstance(value, dict):
            return {k: self._interpolate(v) for k, v in value.items()}

        elif isinstance(value, list):
            return [self._interpolate(item) for item in value]

        return value

    def load(
        self,
        config_path: str,
        overlay_path: Optional[str] = None,
        interpolate: bool = True,
    ) -> Dict[str, Any]:
        """
        Loads a YAML configuration file with optional overlay and interpolation.

        Args:
            config_path: Path to the base YAML config file (relative to config_dir
                         or absolute).
            overlay_path: Optional path to an overlay YAML that merges on top of
                          the base config (e.g., production overrides).
            interpolate: If True, resolves ${VAR} patterns against env vars
                         and Key Vault.

        Returns:
            Merged configuration dictionary.
        """
        # Resolve path
        path = Path(config_path)
        if not path.is_absolute():
            path = self.config_dir / path

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            config = yaml.safe_load(f) or {}

        logger.info(f"Loaded config: {path}")

        # Apply overlay if provided
        if overlay_path:
            overlay_full = Path(overlay_path)
            if not overlay_full.is_absolute():
                overlay_full = self.config_dir / overlay_full

            if overlay_full.exists():
                with open(overlay_full, "r") as f:
                    overlay = yaml.safe_load(f) or {}
                config = self._deep_merge(config, overlay)
                logger.info(f"Applied overlay: {overlay_full}")
            else:
                logger.warning(f"Overlay file not found: {overlay_full}")

        # Interpolate environment variables and secrets
        if interpolate:
            config = self._interpolate(config)

        return config

    @staticmethod
    def _deep_merge(base: dict, overlay: dict) -> dict:
        """
        Recursively merges overlay into base dict.
        Overlay values take precedence over base values.
        """
        merged = base.copy()
        for key, value in overlay.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = ConfigLoader._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    def load_training_config(self, stage: str) -> Dict[str, Any]:
        """
        Convenience method to load a training config by stage name.

        Args:
            stage: One of 'pretrain', 'context_scaling', 'rlhf_ppo'.

        Returns:
            The resolved training configuration dict.
        """
        stage_files = {
            "pretrain": "7b_pretrain.yaml",
            "context_scaling": "context_scaling.yaml",
            "rlhf_ppo": "rlhf_ppo.yaml",
        }

        filename = stage_files.get(stage)
        if not filename:
            raise ValueError(
                f"Unknown stage '{stage}'. Choose from: {list(stage_files.keys())}"
            )

        return self.load(filename)
