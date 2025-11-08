"""
Airflow DAG for ML Model Training Pipeline

This DAG orchestrates the complete ML pipeline:
1. Extract data from CSV
2. Validate data quality
3. Preprocess data
4. Train model
5. Evaluate model
6. Promote to production (if better)
7. Send alerts
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import os
import sys
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import existing utilities
from src.utils import (
    load_data, validate_data, split_features_target
)
from mlflow_config import setup_mlflow
from src.pipeline.ml_pipeline import MLPipeline

# ============================================================================
# CONFIGURATION VARIABLES
# ============================================================================

# Data Configuration
DATA_PATH = os.path.join(project_root, "data", "housing.csv")
TARGET_VARIABLE = "price"
PROBLEM_TYPE = "regression"

# Data Quality Thresholds
MISSING_VALUE_THRESHOLD = 0.05  # 5% maximum missing values
MIN_PRICE = 0
MAX_BEDROOMS = 10
MIN_BEDROOMS = 1

# MLflow Configuration
EXPERIMENT_NAME = "House_Price_Prediction"
METRIC_TO_COMPARE = "test_R2"  # For regression, use R2. For classification, use "test_Accuracy"
IMPROVEMENT_THRESHOLD = 0.01  # New model must be at least 1% better

# Alert Configuration (optional - configure if you have email/Slack)
SEND_ALERTS = False
ALERT_EMAIL = None  # Set to your email if you want email alerts

# ============================================================================
# TASK FUNCTIONS
# ============================================================================

def extract_data(**context):
    """
    Task 1: Extract data from CSV file using MLPipeline class.
    
    Returns:
        dict: Data extraction metadata
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"Extracting data from: {DATA_PATH}")
    
    # Create pipeline instance
    pipeline = MLPipeline()
    
    # Extract data
    result = pipeline.extract_data(DATA_PATH)
    
    # Store pipeline instance and metadata in XCom
    context['ti'].xcom_push(key='pipeline_df_shape', value=pipeline.df.shape)
    context['ti'].xcom_push(key='columns', value=list(pipeline.df.columns))
    
    logger.info(f"Data extracted successfully: {result['rows']} rows × {result['columns']} columns")
    
    return result


def validate_data_quality(**context):
    """
    Task 2: Validate data quality using MLPipeline class.
    
    Returns:
        dict: Validation results
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("Validating data quality...")
    
    # Create pipeline instance and load data
    pipeline = MLPipeline()
    pipeline.extract_data(DATA_PATH)
    
    # Validate data quality
    result = pipeline.validate_data_quality(MISSING_VALUE_THRESHOLD)
    
    # Additional custom validations
    issues = result.get('issues', [])
    
    # Check value ranges
    if TARGET_VARIABLE in pipeline.df.columns:
        if pipeline.df[TARGET_VARIABLE].min() < MIN_PRICE:
            issues.append(f"Target variable '{TARGET_VARIABLE}' has values below {MIN_PRICE}")
    
    if 'bedrooms' in pipeline.df.columns:
        if pipeline.df['bedrooms'].max() > MAX_BEDROOMS or pipeline.df['bedrooms'].min() < MIN_BEDROOMS:
            issues.append(f"Bedrooms out of range: {pipeline.df['bedrooms'].min()}-{pipeline.df['bedrooms'].max()} (expected: {MIN_BEDROOMS}-{MAX_BEDROOMS})")
    
    # Check for negative values where they shouldn't be
    if 'square_feet' in pipeline.df.columns:
        if (pipeline.df['square_feet'] < 0).any():
            issues.append("Negative values found in 'square_feet'")
    
    if 'age' in pipeline.df.columns:
        if (pipeline.df['age'] < 0).any():
            issues.append("Negative values found in 'age'")
    
    # Log results
    if issues:
        logger.warning(f"Found {len(issues)} data quality issues:")
        for issue in issues:
            logger.warning(f"   - {issue}")
        
        # Reject if too many critical issues
        if len(issues) > 5:
            raise ValueError(f"Data quality check failed with {len(issues)} issues. Pipeline stopped.")
    else:
        logger.info("Data quality validation passed!")
    
    result['issues'] = issues
    result['issues_count'] = len(issues)
    result['status'] = 'success' if len(issues) <= 5 else 'failed'
    
    context['ti'].xcom_push(key='validation_issues', value=issues)
    context['ti'].xcom_push(key='validation_passed', value=len(issues) <= 5)
    
    return result


def preprocess_data_task(**context):
    """
    Task 3: Preprocess data using MLPipeline class.
    
    Returns:
        dict: Preprocessing results
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("Preprocessing data...")
    
    # Create pipeline instance
    pipeline = MLPipeline()
    
    # Extract and preprocess data
    pipeline.extract_data(DATA_PATH)
    result = pipeline.preprocess_data()
    
    logger.info(f"Preprocessing complete! Final shape: {pipeline.X_train_processed.shape}")
    
    context['ti'].xcom_push(key='preprocessed_shape', value=pipeline.X_train_processed.shape)
    context['ti'].xcom_push(key='feature_count', value=pipeline.X_train_processed.shape[1])
    
    return result


def train_model_task(**context):
    """
    Task 4: Train the ML model using MLPipeline class.
    
    Returns:
        dict: Training results
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("Training ML model...")
    
    # Create pipeline instance
    pipeline = MLPipeline()
    
    # Run pipeline steps up to training
    pipeline.extract_data(DATA_PATH)
    pipeline.preprocess_data()
    
    # Train models using OOP method
    result = pipeline.train_models()
    
    logger.info("Model training completed!")
    logger.info(f"   Best model: {result['best_model']}")
    logger.info(f"   Best {METRIC_TO_COMPARE}: {result['best_metric']:.4f}")
    
    # Push to XCom
    context['ti'].xcom_push(key='best_model', value=result['best_model'])
    context['ti'].xcom_push(key='best_metric', value=result['best_metric'])
    context['ti'].xcom_push(key='best_run_id', value=result.get('best_run_id'))
    
    return result


def evaluate_model_task(**context):
    """
    Task 5: Evaluate new model and compare with production model.
    
    Returns:
        dict: Evaluation results
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("Evaluating model and comparing with production...")
    
    # Get best model info from previous task
    ti = context['ti']
    best_model = ti.xcom_pull(task_ids='train_model', key='best_model')
    best_metric = ti.xcom_pull(task_ids='train_model', key='best_metric')
    best_run_id = ti.xcom_pull(task_ids='train_model', key='best_run_id')
    
    # If XCom doesn't have data, try to get from MLflow directly
    if not best_model or best_model == 'unknown':
        logger.warning("   Could not retrieve from XCom, fetching from MLflow directly...")
        setup_mlflow(EXPERIMENT_NAME)
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        
        if experiment:
            try:
                runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=[f'metrics.{METRIC_TO_COMPARE} DESC'])
                if len(runs) > 0:
                    metric_col = f'metrics.{METRIC_TO_COMPARE}'
                    runs_with_metric = runs[runs[metric_col].notna()]
                    if len(runs_with_metric) > 0:
                        best_run = runs_with_metric.iloc[0]
                        best_model = str(best_run.get('tags.mlflow.runName', 'unknown'))
                        best_metric = float(best_run[metric_col])
                        best_run_id = str(best_run.get('run_id', ''))
                        logger.info(f"   Retrieved from MLflow: {best_model} (metric: {best_metric:.4f})")
            except Exception as e:
                logger.error(f"   Error fetching from MLflow: {e}")
    
    if not best_model or best_model == 'unknown':
        logger.warning("   Could not retrieve best model. Using defaults for comparison.")
        best_model = 'unknown'
        best_metric = 0.0
    
    logger.info(f"   New model: {best_model}")
    logger.info(f"   New model {METRIC_TO_COMPARE}: {best_metric:.4f}")
    
    # Get production model metric
    setup_mlflow(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    
    production_metric = None
    production_model = None
    
    if experiment:
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        # Try to find a model marked as production
        if 'tags.mlflow.runName' in runs.columns:
            # For now, get the previous best (second best if current is best)
            if len(runs) > 1:
                sorted_runs = runs.sort_values(f'metrics.{METRIC_TO_COMPARE}', ascending=False)
                # Get second best as "production"
                if len(sorted_runs) > 1:
                    prod_run = sorted_runs.iloc[1]
                    production_model = prod_run['tags.mlflow.runName']
                    production_metric = prod_run.get(f'metrics.{METRIC_TO_COMPARE}', None)
    
    # Use MLPipeline class to evaluate
    pipeline = MLPipeline()
    eval_result = pipeline.evaluate_model(best_metric, production_metric, IMPROVEMENT_THRESHOLD)
    
    promote_to_production = eval_result['promote_to_production']
    improvement = eval_result['improvement']
    
    # Log results
    if production_metric is not None:
        improvement_pct = (improvement / production_metric) * 100 if production_metric != 0 else 0
        logger.info(f"   Production model: {production_model}")
        logger.info(f"   Production {METRIC_TO_COMPARE}: {production_metric:.4f}")
        logger.info(f"   Improvement: {improvement:.4f} ({improvement_pct:.2f}%)")
        
        if promote_to_production:
            logger.info(f"New model is better! Promoting to production...")
        else:
            logger.info(f"New model is not significantly better. Keeping production model.")
    else:
        logger.info("No production model found. Promoting new model to production.")
    
    context['ti'].xcom_push(key='promote_to_production', value=promote_to_production)
    context['ti'].xcom_push(key='improvement', value=improvement)
    
    return eval_result


def send_alert(**context):
    """
    Task 6: Send alert on success or failure.
    
    Returns:
        dict: Alert status
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Get task status
    dag_run = context.get('dag_run')
    task_instance = context.get('ti')
    
    # Check if any task failed
    failed_tasks = []
    for task_id in ['extract_data', 'validate_data_quality', 'preprocess_data', 
                    'train_model', 'evaluate_model']:
        try:
            task_state = task_instance.get_task_instance(task_id).state
            if task_state == 'failed':
                failed_tasks.append(task_id)
        except:
            pass
    
    if failed_tasks:
        message = f"ML Pipeline FAILED!\n\nFailed tasks: {', '.join(failed_tasks)}"
        logger.error(message)
        
        if SEND_ALERTS and ALERT_EMAIL:
            # Here you would implement email/Slack sending
            # For now, just log
            logger.info(f"   Would send alert email to {ALERT_EMAIL}")
    else:
        # Get evaluation results
        promote = task_instance.xcom_pull(task_ids='evaluate_model', key='promote_to_production')
        best_model = task_instance.xcom_pull(task_ids='train_model', key='best_model')
        
        message = f"ML Pipeline SUCCESS!\n\nBest model: {best_model}\nPromoted to production: {promote}"
        logger.info(message)
        
        if SEND_ALERTS and ALERT_EMAIL:
            logger.info(f"   Would send success email to {ALERT_EMAIL}")
    
    return {
        'status': 'success',
        'alert_sent': SEND_ALERTS,
        'failed_tasks': failed_tasks
    }


# ============================================================================
# DAG DEFINITION
# ============================================================================

default_args = {
    'owner': 'ml_team',
    'depends_on_past': False,
    'email': [ALERT_EMAIL] if ALERT_EMAIL else [],
    'email_on_failure': SEND_ALERTS,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': days_ago(1),
}

dag = DAG(
    'ml_housing_pipeline',
    default_args=default_args,
    description='ML Model Training Pipeline with MLflow',
    schedule_interval='0 2 * * *',  # Daily at 2 AM
    catchup=False,
    tags=['ml', 'training', 'mlflow'],
    max_active_runs=1,
)

# ============================================================================
# TASK DEFINITIONS
# ============================================================================

# Task 1: Extract Data
extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag,
)

# Task 2: Validate Data Quality
validate_task = PythonOperator(
    task_id='validate_data_quality',
    python_callable=validate_data_quality,
    dag=dag,
)

# Task 3: Preprocess Data
preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data_task,
    dag=dag,
)

# Task 4: Train Model
train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model_task,
    dag=dag,
)

# Task 5: Evaluate Model
evaluate_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model_task,
    dag=dag,
)

# Task 6: Send Alert
alert_task = PythonOperator(
    task_id='send_alert',
    python_callable=send_alert,
    dag=dag,
    trigger_rule='all_done',  # Run regardless of previous task status
)

# ============================================================================
# TASK DEPENDENCIES
# ============================================================================

extract_task >> validate_task >> preprocess_task >> train_task >> evaluate_task >> alert_task

