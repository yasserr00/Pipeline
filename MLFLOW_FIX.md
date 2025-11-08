# MLflow Container Health Check Fix

## Problem
MLflow container fails health check because it takes time to install packages on first run.

## Solution Applied
Increased health check `start_period` from 60s to 300s (5 minutes) to allow time for:
- System package installation (apt-get)
- Python package installation (mlflow, numpy, pandas, scipy, matplotlib, etc.)
- Directory creation
- Server startup

## Current Status
- Health check: `start_period: 300s` (5 minutes)
- Retries: 10
- Interval: 30s

## Monitoring Progress

### Check if MLflow is installing:
```bash
docker-compose logs -f mlflow
```

Look for:
- "Collecting mlflow" - Installing MLflow
- "Downloading" - Downloading packages
- "mlflow server" - Server starting
- "Application startup complete" - Server ready

### Check container status:
```bash
docker-compose ps mlflow
```

Status should change from:
- `health: starting` → `healthy` (when ready)

### Check if server is running:
```bash
docker exec ml-pipeline-mlflow curl -f http://localhost:5000/health
```

## If Still Failing

### Option 1: Wait Longer
The first installation can take 5-10 minutes on slow networks. Be patient.

### Option 2: Use Pre-built MLflow Image
Instead of installing MLflow on-the-fly, use the official MLflow image:

```yaml
mlflow:
  image: ghcr.io/mlflow/mlflow:v2.8.1
  # ... rest of config
```

### Option 3: Build Custom Image
Create a Dockerfile for MLflow to cache installations:

```dockerfile
FROM python:3.10-slim
RUN apt-get update && apt-get install -y curl && \
    pip install --no-cache-dir mlflow && \
    mkdir -p /app/mlruns && chmod -R 777 /app/mlruns
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "file:///app/mlruns", \
     "--default-artifact-root", "/app/mlruns"]
```

Then in docker-compose.yml:
```yaml
mlflow:
  build:
    context: .
    dockerfile: Dockerfile.mlflow
  # ... rest of config
```

## Quick Fix Commands

```bash
# Restart MLflow with new health check settings
docker-compose down mlflow
docker-compose up -d mlflow

# Monitor installation
docker-compose logs -f mlflow

# Check status (wait 5 minutes)
docker-compose ps mlflow

# Once healthy, start all services
docker-compose up -d
```

