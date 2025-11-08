#!/bin/bash
# Sync configuration from dev.yml to all files
# This script loads dev.yml and updates:
# - docker-compose.yml (via environment variables)
# - docker-entrypoint.sh (via environment variables)
# - Any other files that need configuration

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Syncing configuration from dev.yml..."

# Check if dev.yml exists
if [ ! -f "dev.yml" ]; then
    echo "Error: dev.yml not found!"
    exit 1
fi

# Load configuration and export as environment variables
source scripts/load_config_env.sh

# Update docker-compose.yml using Python script
if [ -f "scripts/update_docker_compose.sh" ]; then
    echo "Updating docker-compose.yml..."
    bash scripts/update_docker_compose.sh
fi

echo "Configuration sync complete!"
echo ""
echo "Current configuration:"
echo "   PostgreSQL: ${POSTGRES_USER}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}"
echo "   Airflow: ${AIRFLOW_HOME} (port ${AIRFLOW_WEBSERVER_PORT})"
echo "   MLflow: ${MLFLOW_TRACKING_URI} (external port ${MLFLOW_EXTERNAL_PORT})"
echo ""
echo "To use these values, run: source scripts/load_config_env.sh"
echo "Or restart docker-compose: docker-compose up -d"

