#!/usr/bin/env python3
"""
Configuration Loader
Loads configuration from dev.yml and makes it available as environment variables
or as a Python dictionary.
"""

import yaml
import os
import sys
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_FILE = PROJECT_ROOT / "dev.yml"


def load_config():
    """Load configuration from dev.yml file."""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_FILE}")
    
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_config_value(key_path, default=None):
    """
    Get a configuration value using dot notation.
    
    Example:
        get_config_value('postgres.port') -> 5432
        get_config_value('airflow.webserver_port') -> 8080
    """
    config = load_config()
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


def export_env_vars():
    """Export configuration as environment variables."""
    config = load_config()
    
    # PostgreSQL
    os.environ['POSTGRES_USER'] = str(config['postgres']['user'])
    os.environ['POSTGRES_PASSWORD'] = str(config['postgres']['password'])
    os.environ['POSTGRES_DB'] = str(config['postgres']['database'])
    os.environ['POSTGRES_PORT'] = str(config['postgres']['port'])
    
    # Airflow
    os.environ['AIRFLOW_HOME'] = config['airflow']['home']
    os.environ['AIRFLOW__CORE__EXECUTOR'] = config['airflow']['executor']
    os.environ['AIRFLOW__CORE__LOAD_EXAMPLES'] = str(config['airflow']['load_examples']).lower()
    os.environ['AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION'] = str(config['airflow']['dags_paused_at_creation']).lower()
    os.environ['AIRFLOW_WEBSERVER_PORT'] = str(config['airflow']['webserver_port'])
    os.environ['AIRFLOW_ADMIN_USERNAME'] = config['airflow']['admin_username']
    os.environ['AIRFLOW_ADMIN_PASSWORD'] = config['airflow']['admin_password']
    
    # MLflow
    os.environ['MLFLOW_TRACKING_URI'] = config['mlflow']['tracking_uri']
    os.environ['MLFLOW_PORT'] = str(config['mlflow']['port'])
    os.environ['MLFLOW_EXTERNAL_PORT'] = str(config['mlflow']['external_port'])
    
    return config


if __name__ == "__main__":
    # Test loading config
    config = load_config()
    print("Configuration loaded successfully!")
    print(f"PostgreSQL Port: {config['postgres']['port']}")
    print(f"Airflow Web Port: {config['airflow']['webserver_port']}")
    print(f"MLflow Port: {config['mlflow']['port']}")
    print(f"MLflow External Port: {config['mlflow']['external_port']}")

