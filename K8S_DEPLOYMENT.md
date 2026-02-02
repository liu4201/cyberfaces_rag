# Kubernetes Deployment Guide

Quick guide to deploy the Cyberfaces RAG application to the Anvil Kubernetes cluster.

## Prerequisites

- Docker with buildx support
- kubectl configured for Anvil cluster
- Access to `registry.anvil.rcac.purdue.edu/cyberfaces`
- Access to `cyberfaces-dev` namespace

## Deployment

Simply run the deployment script:

```bash
./deploy.sh
```

The script will automatically:
1. Build the Docker image for linux/amd64 platform
2. Push to `registry.anvil.rcac.purdue.edu/cyberfaces`
3. Deploy to the `cyberfaces-dev` namespace
4. Wait for the deployment to complete

## What Gets Deployed

The deployment creates the following resources in the `cyberfaces-dev` namespace:

- **Deployment**: FastAPI application with health checks
- **Service**: ClusterIP service on port 8000
- **ConfigMap**: Environment configuration
- **PersistentVolumeClaim**: 5Gi storage for ChromaDB (optional, since chromaDB is in the image)

## Accessing the API

Other services in the cluster can access the API at:

```
http://cyberfaces-rag-service.cyberfaces-dev:8000
```

### Available Endpoints

- `POST /search_semantic` - Semantic search
- `POST /search_lexical` - Lexical/keyword search
- `POST /search_RRF` - Reciprocal Rank Fusion search
- `POST /search_reranking_base` - Reranking-based search
- `GET /docs` - API documentation (Swagger UI)

## Verification

Check deployment status:

```bash
kubectl get pods -l app=cyberfaces-rag -n cyberfaces-dev
```

View logs:

```bash
kubectl logs -l app=cyberfaces-rag -f -n cyberfaces-dev
```

Test the API from within the cluster:

```bash
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -n cyberfaces-dev -- \
  curl http://cyberfaces-rag-service.cyberfaces-dev:8000/docs
```

## Updating

To deploy a new version, just run the script again:

```bash
./deploy.sh
```

## Troubleshooting

**View pod details:**
```bash
kubectl describe pod -l app=cyberfaces-rag -n cyberfaces-dev
```

**Check service:**
```bash
kubectl get service cyberfaces-rag-service -n cyberfaces-dev
```

**Check PVC:**
```bash
kubectl get pvc cyberfaces-chromadb-pvc -n cyberfaces-dev
```

## Cleanup

To remove the deployment:

```bash
kubectl delete -f k8s-deployment.yaml -n cyberfaces-dev
```
