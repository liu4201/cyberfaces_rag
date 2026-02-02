#!/bin/bash

# Cyberfaces RAG - Build, Push, and Deploy Script
# This script builds a multi-platform Docker image, pushes it to a registry, and deploys to Kubernetes

set -e  # Exit on error

# Configuration
IMAGE_NAME="cyberfaces-rag"
IMAGE_TAG="${IMAGE_TAG:-latest}"
REGISTRY="${REGISTRY:-}"  # Set this to your container registry (e.g., docker.io/username, gcr.io/project-id)
REGISTRY="registry.anvil.rcac.purdue.edu/cyberfaces"
NAMESPACE="cyberfaces-dev"
K8S_MANIFEST="k8s-deployment.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if REGISTRY is set
if [ -z "$REGISTRY" ]; then
    print_error "REGISTRY is not set. Please set the REGISTRY environment variable or edit this script."
    print_info "Example: export REGISTRY=docker.io/yourusername"
    print_info "         export REGISTRY=gcr.io/your-project-id"
    exit 1
fi

FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

print_info "Starting build, push, and deploy process..."
print_info "Image: ${FULL_IMAGE_NAME}"

# Step 1: Build the Docker image with buildx for linux/amd64
print_info "Building Docker image for linux/amd64 platform..."

# Check if buildx builder exists, create if not
if ! docker buildx inspect multiplatform-builder > /dev/null 2>&1; then
    print_info "Creating buildx builder instance..."
    docker buildx create --name multiplatform-builder --use
    docker buildx inspect --bootstrap
else
    print_info "Using existing buildx builder..."
    docker buildx use multiplatform-builder
fi

# Build and push the image
print_info "Building and pushing image..."
docker buildx build \
    --platform linux/amd64 \
    -t "${FULL_IMAGE_NAME}" \
    --push \
    .

if [ $? -eq 0 ]; then
    print_info "Docker image built and pushed successfully!"
else
    print_error "Docker build/push failed!"
    exit 1
fi

# Step 2: Update the Kubernetes manifest with the correct image
print_info "Updating Kubernetes manifest with image name..."
sed -i.bak "s|<REGISTRY_PLACEHOLDER>/cyberfaces-rag:latest|${FULL_IMAGE_NAME}|g" "${K8S_MANIFEST}"

if [ $? -eq 0 ]; then
    print_info "Kubernetes manifest updated successfully!"
else
    print_error "Failed to update Kubernetes manifest!"
    mv "${K8S_MANIFEST}.bak" "${K8S_MANIFEST}"
    exit 1
fi

# Step 3: Deploy to Kubernetes
print_info "Deploying to Kubernetes cluster (namespace: ${NAMESPACE})..."
kubectl apply -f "${K8S_MANIFEST}" -n "${NAMESPACE}"

if [ $? -eq 0 ]; then
    print_info "Deployment successful!"
else
    print_error "Kubernetes deployment failed!"
    # Restore the original manifest
    mv "${K8S_MANIFEST}.bak" "${K8S_MANIFEST}"
    exit 1
fi

# Clean up backup file
rm -f "${K8S_MANIFEST}.bak"

# Step 4: Check deployment status
print_info "Checking deployment status..."
kubectl rollout status deployment/cyberfaces-rag -n "${NAMESPACE}"

# Display service information
print_info "Service information:"
kubectl get service cyberfaces-rag-service -n "${NAMESPACE}"

print_info "Deployment complete!"
print_info "To check pod status: kubectl get pods -l app=cyberfaces-rag -n ${NAMESPACE}"
print_info "To view logs: kubectl logs -l app=cyberfaces-rag -f -n ${NAMESPACE}"
print_info "To access the API within the cluster: http://cyberfaces-rag-service.${NAMESPACE}:8000"
