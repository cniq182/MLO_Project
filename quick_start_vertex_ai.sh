#!/bin/bash
# Quick start script for Vertex AI training

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Vertex AI Training Quick Start ===${NC}\n"

# Check for required environment variables
if [ -z "$GCP_PROJECT_ID" ]; then
    echo -e "${RED}Error: GCP_PROJECT_ID not set${NC}"
    echo "Set it with: export GCP_PROJECT_ID=your-project-id"
    exit 1
fi

if [ -z "$GCS_BUCKET" ]; then
    echo -e "${RED}Error: GCS_BUCKET not set${NC}"
    echo "Set it with: export GCS_BUCKET=your-bucket-name"
    exit 1
fi

# Set defaults
REGION="${GCP_REGION:-us-central1}"
IMAGE_NAME="train-en-es"
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE_URI="gcr.io/${GCP_PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}"

echo -e "${YELLOW}Configuration:${NC}"
echo "  Project ID: ${GCP_PROJECT_ID}"
echo "  Region: ${REGION}"
echo "  Bucket: ${GCS_BUCKET}"
echo "  Image: ${IMAGE_URI}"
echo ""

# Step 1: Upload data (optional - skip if already uploaded)
read -p "Upload processed data to GCS? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}Step 1: Uploading data to GCS...${NC}"
    uv run python upload_data_to_gcs.py \
        --bucket "${GCS_BUCKET}" \
        --local-data-dir en_es_translation/data/processed \
        --gcs-path data/processed \
        --project-id "${GCP_PROJECT_ID}"
    echo ""
fi

# Step 2: Build and push image
echo -e "${GREEN}Step 2: Building and pushing Docker image...${NC}"
gcloud builds submit \
    --config cloudbuild.yaml \
    --substitutions _IMAGE_URI="${IMAGE_URI}" \
    --project "${GCP_PROJECT_ID}"
echo ""

# Step 3: Submit training job
echo -e "${GREEN}Step 3: Submitting training job...${NC}"
read -p "Use GPU? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ACCELERATOR_ARGS="--accelerator-type NVIDIA_TESLA_T4 --accelerator-count 1"
    MACHINE_TYPE="n1-standard-4"
else
    ACCELERATOR_ARGS=""
    MACHINE_TYPE="n1-standard-4"
fi

# Get W&B API key if available
WANDB_ARGS=""
if [ -n "$WANDB_API_KEY" ]; then
    WANDB_ARGS="--wandb-api-key ${WANDB_API_KEY}"
fi

uv run python vertex_ai_train.py \
    --project-id "${GCP_PROJECT_ID}" \
    --region "${REGION}" \
    --gcs-bucket "${GCS_BUCKET}" \
    --image-uri "${IMAGE_URI}" \
    --machine-type "${MACHINE_TYPE}" \
    ${ACCELERATOR_ARGS} \
    ${WANDB_ARGS}

echo ""
echo -e "${GREEN}✓ Training job submitted!${NC}"
echo ""
echo "Monitor your job at:"
echo "  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${GCP_PROJECT_ID}"
