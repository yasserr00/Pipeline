# Dockerfile for ML Pipeline with Airflow and MLflow
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AIRFLOW_HOME=/app/airflow \
    AIRFLOW__CORE__EXECUTOR=LocalExecutor \
    AIRFLOW__CORE__LOAD_EXAMPLES=False \
    AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres:5432/airflow \
    AIRFLOW__CORE__FERNET_KEY='' \
    AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=True \
    MLFLOW_TRACKING_URI=http://mlflow:5000

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir apache-airflow==2.8.0 \
    apache-airflow-providers-postgres==5.7.0 \
    psycopg2-binary==2.9.9

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/airflow/dags \
    /app/airflow/logs \
    /app/airflow/plugins \
    /app/data \
    /app/mlruns \
    /app/models

# Copy entrypoint script
COPY docker-entrypoint.sh /app/docker-entrypoint.sh

# Copy DAG file to Airflow DAGs folder
RUN cp airflow/dags/ml_pipeline_dag.py /app/airflow/dags/ || true

# Make entrypoint script executable
RUN chmod +x /app/docker-entrypoint.sh

# Expose ports
EXPOSE 8080 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Set entrypoint
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default command (can be overridden in docker-compose)
CMD ["webserver"]

