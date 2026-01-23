#!/usr/bin/env python3
"""
Script to submit training jobs to Vertex AI.

Usage:
    python vertex_ai_train.py \
        --project-id YOUR_PROJECT_ID \
        --region us-central1 \
        --gcs-bucket YOUR_BUCKET_NAME \
        --image-uri gcr.io/YOUR_PROJECT_ID/train-en-es:latest
"""

import argparse
import os
from pathlib import Path
from google.cloud import aiplatform
from google.cloud.aiplatform import CustomTrainingJob


def submit_training_job(
    project_id: str,
    region: str,
    gcs_bucket: str,
    image_uri: str,
    machine_type: str = "n1-standard-4",
    accelerator_type: str = None,
    accelerator_count: int = 0,
    replica_count: int = 1,
    config_path: str = None,
    config_name: str = "config",
    wandb_api_key: str = None,
    gcs_data_path: str = None,
    gcs_checkpoint_path: str = None,
    job_display_name: str = None,
):
    """
    Submit a training job to Vertex AI.
    
    Args:
        project_id: GCP project ID
        region: GCP region (e.g., us-central1)
        gcs_bucket: GCS bucket name for data and checkpoints
        image_uri: Container image URI (e.g., gcr.io/PROJECT_ID/train-en-es:latest)
        machine_type: Machine type for training (default: n1-standard-4)
        accelerator_type: GPU accelerator type (e.g., NVIDIA_TESLA_T4, None for CPU)
        accelerator_count: Number of accelerators
        replica_count: Number of replicas
        config_path: Path to Hydra config directory (default: /app/en_es_translation/configs)
        config_name: Hydra config name (default: config)
        wandb_api_key: W&B API key (or set WANDB_API_KEY env var)
        gcs_data_path: GCS path to processed data (e.g., gs://bucket/data/processed)
        gcs_checkpoint_path: GCS path for checkpoints (e.g., gs://bucket/checkpoints)
        job_display_name: Display name for the job
    """
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)
    
    # Set default paths
    if config_path is None:
        config_path = "/app/en_es_translation/configs"
    
    if gcs_data_path is None:
        gcs_data_path = f"gs://{gcs_bucket}/data/processed"
    
    if gcs_checkpoint_path is None:
        gcs_checkpoint_path = f"gs://{gcs_bucket}/checkpoints"
    
    if job_display_name is None:
        job_display_name = f"en-es-translation-{aiplatform.utils.timestamped_unique_name()}"
    
    # Get W&B API key from argument or environment
    wandb_key = wandb_api_key or os.getenv("WANDB_API_KEY")
    if not wandb_key:
        print("Warning: WANDB_API_KEY not provided. W&B logging may fail.")
    
    # Build command arguments
    args = [
        f"paths.processed_data_dir={gcs_data_path}",
        f"paths.checkpoint_dir={gcs_checkpoint_path}",
    ]
    
    # Add Hydra config path if custom
    if config_path != "/app/en_es_translation/configs":
        args.append(f"--config-path={config_path}")
    
    if config_name != "config":
        args.append(f"--config-name={config_name}")
    
    # Set environment variables
    env_vars = {}
    if wandb_key:
        env_vars["WANDB_API_KEY"] = wandb_key
    
    # Configure accelerator if specified
    accelerator_spec = None
    if accelerator_type and accelerator_count > 0:
        accelerator_spec = {
            "type": accelerator_type,
            "count": accelerator_count,
        }
    
    # Create and run the custom training job
    print(f"Submitting training job: {job_display_name}")
    print(f"  Image: {image_uri}")
    print(f"  Machine type: {machine_type}")
    if accelerator_spec:
        print(f"  Accelerator: {accelerator_spec['count']}x {accelerator_spec['type']}")
    print(f"  Data path: {gcs_data_path}")
    print(f"  Checkpoint path: {gcs_checkpoint_path}")
    print(f"  Args: {' '.join(args)}")
    
    job = CustomTrainingJob(
        display_name=job_display_name,
        container_uri=image_uri,
        model_serving_container_image_uri=image_uri,
    )
    
    job.run(
        replica_count=replica_count,
        machine_type=machine_type,
        accelerator_type=accelerator_spec["type"] if accelerator_spec else None,
        accelerator_count=accelerator_spec["count"] if accelerator_spec else None,
        args=args,
        environment_variables=env_vars,
        sync=False,  # Don't wait for completion
    )
    
    print(f"\nJob submitted successfully!")
    print(f"Job name: {job_display_name}")
    print(f"Monitor your job at: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={project_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Submit training jobs to Vertex AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument(
        "--project-id",
        type=str,
        required=True,
        help="GCP project ID",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-central1",
        help="GCP region (default: us-central1)",
    )
    parser.add_argument(
        "--gcs-bucket",
        type=str,
        required=True,
        help="GCS bucket name for data and checkpoints",
    )
    parser.add_argument(
        "--image-uri",
        type=str,
        required=True,
        help="Container image URI (e.g., gcr.io/PROJECT_ID/train-en-es:latest)",
    )
    
    # Optional arguments
    parser.add_argument(
        "--machine-type",
        type=str,
        default="n1-standard-4",
        help="Machine type (default: n1-standard-4). For GPU: n1-standard-4",
    )
    parser.add_argument(
        "--accelerator-type",
        type=str,
        default=None,
        choices=["NVIDIA_TESLA_T4", "NVIDIA_TESLA_V100", "NVIDIA_TESLA_P100", "NVIDIA_A100"],
        help="GPU accelerator type (optional)",
    )
    parser.add_argument(
        "--accelerator-count",
        type=int,
        default=0,
        help="Number of accelerators (default: 0)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to Hydra config directory (default: /app/en_es_translation/configs)",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="config",
        help="Hydra config name (default: config)",
    )
    parser.add_argument(
        "--wandb-api-key",
        type=str,
        default=None,
        help="W&B API key (or set WANDB_API_KEY env var)",
    )
    parser.add_argument(
        "--gcs-data-path",
        type=str,
        default=None,
        help=f"GCS path to processed data (default: gs://BUCKET/data/processed)",
    )
    parser.add_argument(
        "--gcs-checkpoint-path",
        type=str,
        default=None,
        help=f"GCS path for checkpoints (default: gs://BUCKET/checkpoints)",
    )
    parser.add_argument(
        "--job-display-name",
        type=str,
        default=None,
        help="Display name for the job (default: auto-generated)",
    )
    
    args = parser.parse_args()
    
    submit_training_job(
        project_id=args.project_id,
        region=args.region,
        gcs_bucket=args.gcs_bucket,
        image_uri=args.image_uri,
        machine_type=args.machine_type,
        accelerator_type=args.accelerator_type,
        accelerator_count=args.accelerator_count,
        config_path=args.config_path,
        config_name=args.config_name,
        wandb_api_key=args.wandb_api_key,
        gcs_data_path=args.gcs_data_path,
        gcs_checkpoint_path=args.gcs_checkpoint_path,
        job_display_name=args.job_display_name,
    )


if __name__ == "__main__":
    main()
