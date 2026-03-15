import os
import io
import json
import logging
from typing import Optional, BinaryIO
from pathlib import Path

logger = logging.getLogger(__name__)


class AzureBlobCheckpointManager:
    """
    Manages model checkpoint upload/download to Azure Blob Storage.

    Enables cloud-native model lifecycle management:
    - Upload checkpoints after each training stage
    - Download checkpoints for inference serving
    - List available model versions
    - Atomic checkpoint swaps for zero-downtime upgrades

    Requires the 'azure-storage-blob' package and either:
    - AZURE_STORAGE_CONNECTION_STRING environment variable, or
    - Azure Managed Identity (when running on Azure VMs / AKS)
    """

    def __init__(
        self,
        container_name: str = "titan-checkpoints",
        connection_string: Optional[str] = None,
        account_url: Optional[str] = None,
    ):
        self.container_name = container_name
        self.connection_string = connection_string or os.environ.get(
            "AZURE_STORAGE_CONNECTION_STRING"
        )
        self.account_url = account_url

        self._client = None

    def _get_client(self):
        """Lazily initializes the Azure Blob Service client."""
        if self._client is None:
            try:
                from azure.storage.blob import BlobServiceClient

                if self.connection_string:
                    self._client = BlobServiceClient.from_connection_string(
                        self.connection_string
                    )
                elif self.account_url:
                    from azure.identity import DefaultAzureCredential
                    credential = DefaultAzureCredential()
                    self._client = BlobServiceClient(
                        account_url=self.account_url, credential=credential
                    )
                else:
                    raise ValueError(
                        "Either AZURE_STORAGE_CONNECTION_STRING or account_url is required."
                    )

                # Ensure container exists
                container = self._client.get_container_client(self.container_name)
                if not container.exists():
                    container.create_container()
                    logger.info(f"Created Azure Blob container: {self.container_name}")

            except ImportError:
                raise ImportError(
                    "azure-storage-blob is required. Install with: "
                    "pip install azure-storage-blob azure-identity"
                )
        return self._client

    def upload_checkpoint(self, local_path: str, remote_prefix: str):
        """
        Uploads an entire checkpoint directory to Azure Blob Storage.

        Args:
            local_path: Local directory containing checkpoint files
            remote_prefix: Blob path prefix (e.g., "models/7b-v1/pretrain")
        """
        client = self._get_client()
        container = client.get_container_client(self.container_name)
        local_dir = Path(local_path)

        uploaded = 0
        for file_path in local_dir.rglob("*"):
            if file_path.is_file():
                blob_name = f"{remote_prefix}/{file_path.relative_to(local_dir)}"
                blob_client = container.get_blob_client(blob_name)

                with open(file_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                    uploaded += 1
                    logger.info(f"  Uploaded: {blob_name}")

        logger.info(
            f"Checkpoint uploaded: {uploaded} files → "
            f"az://{self.container_name}/{remote_prefix}"
        )
        return uploaded

    def download_checkpoint(self, remote_prefix: str, local_path: str):
        """
        Downloads a checkpoint from Azure Blob Storage to local disk.

        Args:
            remote_prefix: Blob path prefix to download
            local_path: Local directory to save files into
        """
        client = self._get_client()
        container = client.get_container_client(self.container_name)
        local_dir = Path(local_path)
        local_dir.mkdir(parents=True, exist_ok=True)

        downloaded = 0
        blobs = container.list_blobs(name_starts_with=remote_prefix)

        for blob in blobs:
            relative = blob.name[len(remote_prefix):].lstrip("/")
            target = local_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)

            blob_client = container.get_blob_client(blob.name)
            with open(target, "wb") as f:
                download_stream = blob_client.download_blob()
                download_stream.readinto(f)
                downloaded += 1
                logger.info(f"  Downloaded: {blob.name} → {target}")

        logger.info(
            f"Checkpoint downloaded: {downloaded} files ← "
            f"az://{self.container_name}/{remote_prefix}"
        )
        return downloaded

    def list_checkpoints(self, prefix: str = "") -> list:
        """Lists all available checkpoint versions in the container."""
        client = self._get_client()
        container = client.get_container_client(self.container_name)

        # Group by top-level prefix to get checkpoint "versions"
        versions = set()
        for blob in container.list_blobs(name_starts_with=prefix):
            parts = blob.name.split("/")
            if len(parts) >= 2:
                versions.add("/".join(parts[:2]))

        return sorted(versions)

    def delete_checkpoint(self, remote_prefix: str):
        """Deletes a checkpoint version from Azure Blob Storage."""
        client = self._get_client()
        container = client.get_container_client(self.container_name)
        deleted = 0

        for blob in container.list_blobs(name_starts_with=remote_prefix):
            container.delete_blob(blob.name)
            deleted += 1

        logger.info(f"Deleted {deleted} blobs under {remote_prefix}")
        return deleted


class AzureMLExperimentTracker:
    """
    Integration with Azure Machine Learning for experiment tracking.

    Logs training metrics (loss, accuracy, learning rate) to Azure ML
    runs, enabling:
    - Experiment comparison across training stages
    - Hyperparameter search tracking
    - Model registry for versioning
    - Automated model deployment triggers

    Requires the 'azureml-core' package and an Azure ML workspace.
    """

    def __init__(
        self,
        workspace_name: Optional[str] = None,
        subscription_id: Optional[str] = None,
        resource_group: Optional[str] = None,
    ):
        self.workspace_name = workspace_name or os.environ.get("AZUREML_WORKSPACE")
        self.subscription_id = subscription_id or os.environ.get("AZURE_SUBSCRIPTION_ID")
        self.resource_group = resource_group or os.environ.get("AZURE_RESOURCE_GROUP")
        self._workspace = None
        self._run = None

    def _get_workspace(self):
        """Lazily connects to the Azure ML workspace."""
        if self._workspace is None:
            try:
                from azureml.core import Workspace

                self._workspace = Workspace(
                    subscription_id=self.subscription_id,
                    resource_group=self.resource_group,
                    workspace_name=self.workspace_name,
                )
                logger.info(f"Connected to Azure ML workspace: {self.workspace_name}")
            except ImportError:
                raise ImportError(
                    "azureml-core is required. Install with: pip install azureml-core"
                )
        return self._workspace

    def start_run(self, experiment_name: str, run_name: Optional[str] = None, tags: dict = None):
        """Starts a new Azure ML experiment run for tracking."""
        ws = self._get_workspace()
        from azureml.core import Experiment, Run

        experiment = Experiment(workspace=ws, name=experiment_name)
        self._run = experiment.start_logging(display_name=run_name, tags=tags or {})
        logger.info(f"Started Azure ML run: {self._run.id}")
        return self._run.id

    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Logs a scalar metric to the active run."""
        if self._run:
            self._run.log(name, value, step=step)

    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """Logs a dictionary of metrics in one call."""
        for name, value in metrics.items():
            self.log_metric(name, value, step)

    def log_hyperparameters(self, params: dict):
        """Logs hyperparameters for reproducibility."""
        if self._run:
            for k, v in params.items():
                self._run.tag(k, str(v))

    def register_model(self, model_name: str, model_path: str, tags: dict = None):
        """Registers a trained model in the Azure ML Model Registry."""
        if self._run:
            from azureml.core import Model
            model = self._run.register_model(
                model_name=model_name,
                model_path=model_path,
                tags=tags or {},
            )
            logger.info(f"Model registered: {model.name} v{model.version}")
            return model

    def end_run(self, status: str = "Completed"):
        """Marks the Azure ML run as complete."""
        if self._run:
            self._run.complete()
            logger.info(f"Azure ML run completed: {self._run.id}")
            self._run = None
