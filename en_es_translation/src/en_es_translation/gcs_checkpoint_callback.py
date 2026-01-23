"""
Callback to upload PyTorch Lightning checkpoints to GCS.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.callback import Callback

logger = logging.getLogger(__name__)

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False


class GCSCheckpointCallback(Callback):
    """
    Callback to upload checkpoints to GCS after they're saved locally.
    
    This callback monitors checkpoint saves and uploads them to GCS.
    It should be used alongside ModelCheckpoint callback.
    """
    
    def __init__(self, gcs_checkpoint_dir: str, local_checkpoint_dir: str = None):
        """
        Args:
            gcs_checkpoint_dir: GCS path for checkpoints (e.g., gs://bucket/checkpoints)
            local_checkpoint_dir: Local directory where checkpoints are saved (optional)
        """
        if not GCS_AVAILABLE:
            raise ImportError(
                "google-cloud-storage is required. Install with: pip install google-cloud-storage"
            )
        
        if not gcs_checkpoint_dir.startswith("gs://"):
            raise ValueError(f"Invalid GCS path: {gcs_checkpoint_dir}. Must start with 'gs://'")
        
        self.gcs_checkpoint_dir = gcs_checkpoint_dir
        self.local_checkpoint_dir = local_checkpoint_dir
        self.storage_client = storage.Client()
        
        # Parse bucket and prefix
        parts = gcs_checkpoint_dir[5:].split("/", 1)  # Remove 'gs://'
        self.bucket_name = parts[0]
        self.gcs_prefix = parts[1] if len(parts) > 1 else ""
        self.bucket = self.storage_client.bucket(self.bucket_name)
        
        logger.info(f"Initialized GCS checkpoint callback: {gcs_checkpoint_dir}")
    
    def _upload_to_gcs(self, local_file: Path, gcs_path: str):
        """Upload a file to GCS."""
        try:
            blob = self.bucket.blob(gcs_path)
            blob.upload_from_filename(str(local_file))
            logger.info(f"✓ Uploaded checkpoint to gs://{self.bucket_name}/{gcs_path}")
        except Exception as e:
            logger.error(f"Failed to upload checkpoint to GCS: {e}")
            raise
    
    def on_train_start(self, trainer, pl_module):
        """Called when training starts."""
        if self.local_checkpoint_dir is None:
            # Try to get checkpoint dir from ModelCheckpoint callback
            for callback in trainer.callbacks:
                if isinstance(callback, ModelCheckpoint):
                    self.local_checkpoint_dir = callback.dirpath
                    logger.info(f"Detected checkpoint directory: {self.local_checkpoint_dir}")
                    break
        
        if self.local_checkpoint_dir is None:
            logger.warning(
                "Could not detect local checkpoint directory. "
                "Checkpoints may not be uploaded to GCS."
            )
    
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Called after a checkpoint is saved."""
        if self.local_checkpoint_dir is None:
            return
        
        local_dir = Path(self.local_checkpoint_dir)
        if not local_dir.exists():
            return
        
        # Find the most recent checkpoint file
        checkpoint_files = list(local_dir.glob("*.ckpt"))
        if not checkpoint_files:
            return
        
        # Upload all checkpoint files
        for checkpoint_file in checkpoint_files:
            relative_path = checkpoint_file.name
            gcs_path = f"{self.gcs_prefix}/{relative_path}".strip("/")
            
            logger.info(f"Uploading checkpoint: {checkpoint_file.name}")
            self._upload_to_gcs(checkpoint_file, gcs_path)
    
    def on_train_end(self, trainer, pl_module):
        """Called when training ends - upload final checkpoint."""
        self.on_save_checkpoint(trainer, pl_module, None)
