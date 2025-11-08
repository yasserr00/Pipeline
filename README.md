# ML Model Training Pipeline with MLflow and Airflow

A complete, production-ready machine learning pipeline with MLflow experiment tracking and Airflow orchestration. This project supports both regression and classification problems with comprehensive EDA, feature engineering, model training, model serving, and automated pipeline orchestration.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Running the Full Pipeline](#running-the-full-pipeline)
- [Configuration](#configuration)
- [Accessing Services](#accessing-services)
- [Troubleshooting](#troubleshooting)

## Overview

This pipeline consists of three tiers:

1. **Tier 1 - ML Model Training**: Train and compare multiple ML models with MLflow tracking
2. **Tier 2 - Airflow Orchestration**: Automated pipeline execution with data validation, training, and evaluation
3. **Tier 3 - Docker Containerization**: Containerized deployment with all services

### Pipeline Components

- **Data Extraction**: Load and validate CSV data
- **Data Quality Validation**: Check for missing values, data types, and value ranges
- **Data Preprocessing**: Feature engineering, encoding, scaling, outlier handling
- **Model Training**: Train Linear Regression, Random Forest, and XGBoost models
- **Model Evaluation**: Compare models and promote best model to production
- **Model Serving**: Flask API for predictions with web interface
- **Orchestration**: Airflow DAG for automated pipeline execution

## Project Structure

```
Pipeline/
├── data/                      # Data directory
│   ├── housing.csv            # Training dataset
│   └── loan_approval.csv      # Alternative dataset
│
├── models/                     # Saved models and artifacts
│   ├── model_comparison.csv   # Model comparison table
│   └── feature_importance_*.png  # Feature importance plots
│
├── mlruns/                     # MLflow tracking data (auto-generated)
│
├── airflow/                    # Airflow configuration
│   ├── dags/                  # Airflow DAGs
│   │   └── ml_pipeline_dag.py # Main pipeline DAG
│   ├── logs/                  # Airflow logs
│   └── config/                # Airflow configuration
│
├── src/                        # Source code
│   └── utils/                 # Utility modules
│       ├── data_loader.py     # Data loading and validation
│       ├── feature_engineering.py  # Feature engineering
│       ├── feature_preprocessor.py # Prediction preprocessing
│       └── model_evaluation.py     # Model evaluation metrics
│
├── controller/                 # Flask API controllers
│   └── routes.py              # All Flask API endpoints
│
├── front/                      # Frontend directory
│   ├── templates/             # HTML templates
│   │   └── index.html        # Main web interface
│   └── static/                # Static assets (CSS, JS)
│
├── scripts/                    # Utility scripts
│   ├── load_config.py         # Configuration loader
│   ├── load_config_env.sh     # Bash config loader
│   ├── sync_config.sh         # Config sync script
│   └── update_docker_compose.sh  # Docker compose updater
│
├── train_model.py             # Main training script (Tier 1)
├── main.py                    # Model serving entry point (OOP)
├── mlflow_config.py           # MLflow configuration
├── dev.yml                    # Centralized configuration
├── docker-compose.yml         # Docker services
├── Dockerfile                 # Docker image definition
├── docker-entrypoint.sh       # Container entrypoint
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Prerequisites

### System Requirements

- Python 3.10 or higher
- Docker 20.10+ and Docker Compose 2.0+ (for containerized deployment)
- 4GB+ RAM recommended
- 10GB+ free disk space

### Python Dependencies

All dependencies are listed in `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Local Development (Without Docker)

#### Step 1: Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Step 2: Generate Sample Data (if needed)

```bash
python generate_sample_data.py
```

This creates sample datasets in `data/` directory.

#### Step 3: Train Models

```bash
python train_model.py
```

This will:
- Load data from `data/housing.csv`
- Perform EDA and preprocessing
- Train 3 models (Linear Regression, Random Forest, XGBoost)
- Log everything to MLflow
- Save best model and comparison table

#### Step 4: View MLflow UI

```bash
mlflow ui
```

Open http://localhost:5000 to view experiments.

#### Step 5: Serve Model (Optional)

```bash
python main.py --experiment "House_Price_Prediction" --port 5000
```

Access web interface at http://localhost:5000

### Option 2: Docker Deployment (Recommended)

#### Step 1: Configure (Optional)

Edit `dev.yml` to customize ports, passwords, etc.

#### Step 2: Start All Services

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check status
docker-compose ps
```

#### Step 3: Access Services

- **Airflow UI**: http://localhost:8080 (admin/admin)
- **MLflow UI**: http://localhost:5001
- **Model Serving API**: http://localhost:5050
  - Health check: http://localhost:5050/health
  - Web interface: http://localhost:5050

## Running the Full Pipeline

### Method 1: Via Airflow UI (Recommended)

1. Start Docker services:
   ```bash
   docker-compose up -d
   ```

2. Open Airflow UI: http://localhost:8080
   - Username: `admin`
   - Password: `admin`

3. Find DAG: `ml_housing_pipeline`

4. Toggle switch to "On" (if paused)

5. Click "Trigger DAG" button

6. Monitor progress in Graph View

### Method 2: Via Command Line

```bash
# Trigger DAG
docker exec ml-pipeline-scheduler airflow dags trigger ml_housing_pipeline

# Check DAG status
docker exec ml-pipeline-scheduler airflow dags state ml_housing_pipeline 2025-11-08

# View logs
docker-compose logs -f airflow-scheduler
```

### Method 3: Direct Training (Without Airflow)

```bash
# Activate virtual environment
source venv/bin/activate

# Run training
python train_model.py
```

## Configuration

### Centralized Configuration (dev.yml)

All configuration is centralized in `dev.yml`. Edit this file to change:

- Ports (Airflow, MLflow, PostgreSQL)
- Passwords and credentials
- Service settings
- Container names

After editing `dev.yml`, sync configuration:

```bash
bash scripts/sync_config.sh
docker-compose down
docker-compose up -d
```

### Key Configuration Values

**PostgreSQL**:
- User: `airflow`
- Password: `airflow` (change in production)
- Port: `5432`

**Airflow**:
- Web UI Port: `8080`
- Admin: `admin/admin` (change in production)
- Executor: `LocalExecutor`

**MLflow**:
- Tracking URI: `http://mlflow:5000` (internal)
- External Port: `5001` (host)
- Backend: SQLite (local) or PostgreSQL (production)

### Training Configuration

Edit `train_model.py` to change:

- Dataset path
- Target variable
- Problem type (regression/classification)
- Models to train
- Cross-validation folds
- Train/test split ratio

## Accessing Services

### Airflow Web UI

- URL: http://localhost:8080
- Username: `admin`
- Password: `admin`

**Features**:
- View DAGs and their status
- Trigger DAGs manually
- Monitor task execution
- View task logs
- Check DAG history

### MLflow UI

- URL: http://localhost:5001 (Docker) or http://localhost:5000 (local)

**Features**:
- View all experiments
- Compare model metrics
- Download model artifacts
- View feature importance
- Track hyperparameters

### Model Serving API

- URL: http://localhost:5050 (Docker) or http://localhost:5000 (local)

**Endpoints**:
- `GET /` - Web interface for predictions
- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /predict_batch` - Batch predictions

**Note**: The API service automatically starts when running `docker-compose up -d` and loads the best model from the specified experiment.

## Common Commands

### Docker Commands

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs -f [service-name]

# Restart service
docker-compose restart [service-name]

# Check status
docker-compose ps

# Execute command in container
docker exec ml-pipeline-scheduler airflow dags list
```

### Airflow Commands

```bash
# List DAGs
docker exec ml-pipeline-scheduler airflow dags list

# Trigger DAG
docker exec ml-pipeline-scheduler airflow dags trigger ml_housing_pipeline

# Unpause DAG
docker exec ml-pipeline-scheduler airflow dags unpause ml_housing_pipeline

# Check DAG status
docker exec ml-pipeline-scheduler airflow dags state ml_housing_pipeline 2025-11-08

# View import errors
docker exec ml-pipeline-scheduler airflow dags list-import-errors
```

### MLflow Commands

```bash
# Start MLflow UI (local)
mlflow ui

# View experiments
mlflow experiments list

# Compare runs
mlflow compare [run_id1] [run_id2]
```

### Training Commands

```bash
# Train models
python train_model.py

# Serve model (local)
python main.py --experiment "House_Price_Prediction" --port 5000

# Serve model (Docker - automatically runs on port 5050)
# No command needed - starts automatically with docker-compose up

# Generate sample data
python generate_sample_data.py
```

## Pipeline Workflow

The complete pipeline consists of 6 tasks:

1. **extract_data**: Load CSV file and validate schema
2. **validate_data_quality**: Check missing values, data types, value ranges
3. **preprocess_data**: Feature engineering, encoding, scaling, outlier handling
4. **train_model**: Train all models and log to MLflow
5. **evaluate_model**: Compare new model with production model
6. **send_alert**: Send success/failure notifications

### Task Dependencies

```
extract_data → validate_data_quality → preprocess_data → train_model → evaluate_model → send_alert
```

## Troubleshooting

### Services Not Starting

```bash
# Check logs
docker-compose logs -f [service-name]

# Check if ports are in use
lsof -i :8080  # Airflow
lsof -i :5001  # MLflow
lsof -i :5432  # PostgreSQL

# Restart services
docker-compose restart
```

### DAG Not Appearing

```bash
# Check for import errors
docker exec ml-pipeline-scheduler airflow dags list-import-errors

# Verify DAG file exists
docker exec ml-pipeline-scheduler ls -la /app/airflow/dags/

# Restart scheduler
docker-compose restart airflow-scheduler
```

### Database Issues

```bash
# Reset Airflow database
docker exec ml-pipeline-webserver airflow db reset

# Reinitialize
docker exec ml-pipeline-webserver airflow db init
```

### Port Conflicts

Edit `dev.yml` to change ports, then:

```bash
bash scripts/sync_config.sh
docker-compose down
docker-compose up -d
```

### Model Training Errors

- Verify data file exists: `data/housing.csv`
- Check data format and column names
- Ensure target variable exists in data
- Check Python dependencies: `pip install -r requirements.txt`

## Model Metrics

### Regression Metrics

- **MAE (Mean Absolute Error)**: Average absolute difference between predictions and actuals
- **RMSE (Root Mean Squared Error)**: Square root of average squared differences (penalizes large errors)
- **R² (R-squared)**: Proportion of variance explained (1.0 = perfect, 0.0 = baseline)

### Classification Metrics

- **Accuracy**: Percentage of correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall

## Production Deployment

### Security Considerations

1. Change default passwords in `dev.yml`
2. Generate Fernet key for Airflow:
   ```bash
   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
   ```
3. Use environment variables for sensitive data
4. Enable SSL/TLS for web services
5. Use PostgreSQL instead of SQLite for production

### Scaling

- Use `CeleryExecutor` for parallel task execution
- Add multiple Airflow workers
- Use object storage (S3, GCS) for MLflow artifacts
- Deploy on Kubernetes for better orchestration

## Support

For issues or questions:
1. Check logs: `docker-compose logs -f`
2. Verify configuration: `bash scripts/sync_config.sh`
3. Check service status: `docker-compose ps`

## License

This project is provided as-is for educational and development purposes.
