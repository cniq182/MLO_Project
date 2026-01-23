#!/bin/bash
# Script to build and push Docker image for Vertex AI training

set -e

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
REGION="${GCP_REGION:-us-central1}"
IMAGE_NAME="train-en-es"
IMAGE_TAG="${IMAGE_TAG:-latest}"
REPOSITORY="${REPOSITORY:-gcr.io}"  # or us-central1-docker.pkg.dev for Artifact Registry

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building and pushing Docker image for Vertex AI training...${NC}"

# Check if PROJECT_ID is set
if [ "$PROJECT_ID" = "your-project-id" ]; then
    echo -e "${RED}Error: GCP_PROJECT_ID not set.${NC}"
    echo "Set it with: export GCP_PROJECT_ID=your-project-id"
    echo "Or pass it as: GCP_PROJECT_ID=your-project-id ./build_and_push.sh"
    exit 1
fi

# Set image URI
if [ "$REPOSITORY" = "gcr.io" ]; then
    IMAGE_URI="${REPOSITORY}/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}"
else
    # For Artifact Registry
    REPO_NAME="${ARTIFACT_REGISTRY_REPO:-ml-repo}"
    IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"
fi

echo -e "${YELLOW}Project ID: ${PROJECT_ID}${NC}"
echo -e "${YELLOW}Image URI: ${IMAGE_URI}${NC}"
echo ""

# Build and push using Cloud Build
echo -e "${GREEN}Building Docker image...${NC}"
gcloud builds submit \
    --config cloudbuild.yaml \
    --substitutions _IMAGE_URI="${IMAGE_URI}" \
    --project "${PROJECT_ID}"

echo ""
echo -e "${GREEN}✓ Image built and pushed successfully!${NC}"
echo -e "${GREEN}Image URI: ${IMAGE_URI}${NC}"
echo ""
echo "You can now submit a training job with:"
echo "  python vertex_ai_train.py \\"
echo "    --project-id ${PROJECT_ID} \\"
echo "    --region ${REGION} \\"
echo "    --gcs-bucket YOUR_BUCKET_NAME \\"
echo "    --image-uri ${IMAGE_URI}"
