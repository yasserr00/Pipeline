#!/bin/bash
# Docker entrypoint script for ML Pipeline

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting ML Pipeline Container...${NC}"

# Load configuration (use environment variables set by docker-compose)
POSTGRES_USER=${POSTGRES_USER:-airflow}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-airflow}
POSTGRES_DB=${POSTGRES_DB:-airflow}
POSTGRES_HOST=${POSTGRES_HOST:-postgres}
MLFLOW_HOST=${MLFLOW_HOST:-mlflow}
MLFLOW_PORT=${MLFLOW_PORT:-5000}

# Wait for PostgreSQL to be ready (only if using postgres)
if [ "$AIRFLOW__DATABASE__SQL_ALCHEMY_CONN" != "sqlite"* ]; then
    echo -e "${YELLOW}Waiting for PostgreSQL...${NC}"
    until PGPASSWORD=${POSTGRES_PASSWORD} psql -h ${POSTGRES_HOST} -U ${POSTGRES_USER} -d ${POSTGRES_DB} -c '\q' 2>/dev/null; do
        echo "PostgreSQL is unavailable - sleeping"
        sleep 1
    done
    echo -e "${GREEN}PostgreSQL is ready!${NC}"
fi

# Wait for MLflow to be ready (only if command is not mlflow itself)
if [ "$1" != "mlflow" ]; then
    echo -e "${YELLOW}Waiting for MLflow...${NC}"
    MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://${MLFLOW_HOST}:${MLFLOW_PORT}}
    until curl -f ${MLFLOW_TRACKING_URI}/health 2>/dev/null || [ "$1" = "mlflow" ]; do
        echo "MLflow is unavailable - sleeping"
        sleep 1
    done
    echo -e "${GREEN}MLflow is ready!${NC}"
fi

# Initialize Airflow database if needed
if [ ! -f "$AIRFLOW_HOME/airflow.db" ] && [ "$AIRFLOW__DATABASE__SQL_ALCHEMY_CONN" != "sqlite"* ]; then
    echo -e "${YELLOW}Initializing Airflow database...${NC}"
    airflow db init
    
    # Create admin user
    echo -e "${YELLOW}Creating Airflow admin user...${NC}"
    AIRFLOW_ADMIN_USERNAME=${AIRFLOW_ADMIN_USERNAME:-admin}
    AIRFLOW_ADMIN_PASSWORD=${AIRFLOW_ADMIN_PASSWORD:-admin}
    AIRFLOW_ADMIN_EMAIL=${AIRFLOW_ADMIN_EMAIL:-admin@example.com}
    AIRFLOW_ADMIN_FIRSTNAME=${AIRFLOW_ADMIN_FIRSTNAME:-Admin}
    AIRFLOW_ADMIN_LASTNAME=${AIRFLOW_ADMIN_LASTNAME:-User}
    
    airflow users create \
        --username ${AIRFLOW_ADMIN_USERNAME} \
        --firstname ${AIRFLOW_ADMIN_FIRSTNAME} \
        --lastname ${AIRFLOW_ADMIN_LASTNAME} \
        --role Admin \
        --email ${AIRFLOW_ADMIN_EMAIL} \
        --password ${AIRFLOW_ADMIN_PASSWORD} || echo "User might already exist"
    
    echo -e "${GREEN}Airflow database initialized!${NC}"
fi

# Create necessary directories
mkdir -p "$AIRFLOW_HOME/dags" "$AIRFLOW_HOME/logs" "$AIRFLOW_HOME/plugins"
mkdir -p /app/data /app/mlruns /app/models

# Set permissions
chmod -R 755 "$AIRFLOW_HOME"
chmod -R 755 /app/data /app/mlruns /app/models

# Function to handle shutdown gracefully
cleanup() {
    echo -e "\n${YELLOW}Shutting down gracefully...${NC}"
    kill $PID 2>/dev/null || true
    wait $PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGTERM SIGINT

# Start the requested service
case "$1" in
    webserver)
        echo -e "${GREEN}Starting Airflow Webserver...${NC}"
        exec airflow webserver --port 8080
        ;;
    scheduler)
        echo -e "${GREEN}Starting Airflow Scheduler...${NC}"
        exec airflow scheduler
        ;;
    mlflow)
        echo -e "${GREEN}Starting MLflow Tracking Server...${NC}"
        exec mlflow server --host 0.0.0.0 --port 5000 \
            --backend-store-uri sqlite:///mlruns/mlflow.db \
            --default-artifact-root /app/mlruns
        ;;
    api)
        echo -e "${GREEN}Starting Model Serving API...${NC}"
        API_EXPERIMENT=${API_EXPERIMENT:-House_Price_Prediction}
        API_PORT=${API_PORT:-5050}
        exec python main.py --experiment "${API_EXPERIMENT}" --port "${API_PORT}"
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo "Usage: $0 {webserver|scheduler|mlflow|api}"
        exit 1
        ;;
esac

