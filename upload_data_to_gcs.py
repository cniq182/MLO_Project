#!/usr/bin/env python3
"""
Helper script to upload processed data to GCS for Vertex AI training.

Usage:
    python upload_data_to_gcs.py \
        --bucket YOUR_BUCKET_NAME \
        --local-data-dir en_es_translation/data/processed \
        --gcs-path data/processed
"""

import argparse
import logging
from pathlib import Path
from google.cloud import storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def upload_directory_to_gcs(
    bucket_name: str,
    local_dir: Path,
    gcs_prefix: str,
    project_id: str = None,
):
    """
    Upload a local directory to GCS.
    
    Args:
        bucket_name: GCS bucket name
        local_dir: Local directory path
        gcs_prefix: GCS path prefix (e.g., "data/processed")
        project_id: GCP project ID (optional)
    """
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    
    local_path = Path(local_dir)
    if not local_path.exists():
        raise FileNotFoundError(f"Local directory not found: {local_path}")
    
    # Find all files to upload
    files_to_upload = []
    for file_path in local_path.rglob("*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(local_path)
            gcs_path = f"{gcs_prefix}/{relative_path}".replace("\\", "/")
            files_to_upload.append((file_path, gcs_path))
    
    logger.info(f"Found {len(files_to_upload)} files to upload")
    
    # Upload files
    for local_file, gcs_path in files_to_upload:
        logger.info(f"Uploading {local_file.name} -> gs://{bucket_name}/{gcs_path}")
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(str(local_file))
        logger.info(f"✓ Uploaded {gcs_path}")
    
    logger.info(f"Successfully uploaded {len(files_to_upload)} files to gs://{bucket_name}/{gcs_prefix}/")


def main():
    parser = argparse.ArgumentParser(
        description="Upload processed data to GCS for Vertex AI training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--bucket",
        type=str,
        required=True,
        help="GCS bucket name",
    )
    parser.add_argument(
        "--local-data-dir",
        type=str,
        default="en_es_translation/data/processed",
        help="Local directory containing processed data (default: en_es_translation/data/processed)",
    )
    parser.add_argument(
        "--gcs-path",
        type=str,
        default="data/processed",
        help="GCS path prefix (default: data/processed)",
    )
    parser.add_argument(
        "--project-id",
        type=str,
        default=None,
        help="GCP project ID (optional, uses default credentials)",
    )
    
    args = parser.parse_args()
    
    upload_directory_to_gcs(
        bucket_name=args.bucket,
        local_dir=Path(args.local_data_dir),
        gcs_prefix=args.gcs_path,
        project_id=args.project_id,
    )


if __name__ == "__main__":
    main()
