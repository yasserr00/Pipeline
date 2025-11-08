"""
MLflow Configuration Module

This module handles MLflow experiment setup and tracking URI configuration.
It provides a centralized way to manage MLflow experiments and runs.
"""

import os
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime


def setup_mlflow(experiment_name: str = None, tracking_uri: str = None):
    """
    Set up MLflow experiment and tracking URI.
    
    Args:
        experiment_name (str): Name of the MLflow experiment. 
                              If None, uses default or creates new one.
        tracking_uri (str): MLflow tracking URI. 
                           If None, uses local file system (./mlruns)
    
    Returns:
        str: The experiment ID
    """
    # Set tracking URI (local file system by default)
    if tracking_uri is None:
        # Use file:// prefix for file-based storage
        tracking_uri = f"file://{os.path.join(os.getcwd(), 'mlruns')}"
    
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    
    # Set or create experiment
    if experiment_name is None:
        experiment_name = f"ML_Experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created new experiment: {experiment_name}")
    except Exception:
        # Experiment already exists
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment: {experiment_name}")
    
    mlflow.set_experiment(experiment_name)
    
    return experiment_id


def get_mlflow_client():
    """
    Get MLflow client instance.
    
    Returns:
        MlflowClient: MLflow client for advanced operations
    """
    return MlflowClient()


def log_model_info(model_name: str, model_type: str, target_variable: str):
    """
    Log general model information as tags.
    
    Args:
        model_name (str): Name of the model
        model_type (str): Type of problem (regression/classification)
        target_variable (str): Name of the target variable
    """
    mlflow.set_tag("model_name", model_name)
    mlflow.set_tag("problem_type", model_type)
    mlflow.set_tag("target_variable", target_variable)
    mlflow.set_tag("timestamp", datetime.now().isoformat())


def mark_model_as_production(run_id: str, model_name: str):
    """
    Mark a model as production-ready.
    
    Args:
        run_id (str): The MLflow run ID
        model_name (str): Name of the model to mark as production
    """
    client = get_mlflow_client()
    client.transition_model_version_stage(
        name=model_name,
        version=1,
        stage="Production"
    )
    print(f"Model '{model_name}' marked as Production")


if __name__ == "__main__":
    # Test the configuration
    setup_mlflow("Test_Experiment")
    print("MLflow configuration test completed!")

