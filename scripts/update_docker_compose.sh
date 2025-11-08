#!/bin/bash
# Script to update docker-compose.yml with values from dev.yml
# This script uses Python to read dev.yml and generate docker-compose.yml

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required to update docker-compose.yml"
    exit 1
fi

# Check if PyYAML is installed
python3 -c "import yaml" 2>/dev/null || {
    echo "Installing PyYAML..."
    pip install pyyaml
}

# Run Python script to update docker-compose.yml
python3 << 'PYTHON_SCRIPT'
import yaml
import sys
from pathlib import Path

# Load dev.yml
with open('dev.yml', 'r') as f:
    config = yaml.safe_load(f)

# Load docker-compose.yml
with open('docker-compose.yml', 'r') as f:
    compose = yaml.safe_load(f)

# Helper function to update or add env var in list
def update_env_var(env_list, key, value):
    """Update or add environment variable in list format."""
    updated = False
    for i, env_var in enumerate(env_list):
        if isinstance(env_var, str) and env_var.startswith(f'{key}='):
            env_list[i] = f"{key}={value}"
            updated = True
            break
    if not updated:
        env_list.append(f"{key}={value}")

# Update PostgreSQL
if 'environment' not in compose['services']['postgres']:
    compose['services']['postgres']['environment'] = {}
if isinstance(compose['services']['postgres']['environment'], dict):
    compose['services']['postgres']['environment']['POSTGRES_USER'] = config['postgres']['user']
    compose['services']['postgres']['environment']['POSTGRES_PASSWORD'] = config['postgres']['password']
    compose['services']['postgres']['environment']['POSTGRES_DB'] = config['postgres']['database']
compose['services']['postgres']['ports'] = [f"{config['postgres']['external_port']}:{config['postgres']['port']}"]

# Update MLflow - handle environment as list
if 'environment' not in compose['services']['mlflow']:
    compose['services']['mlflow']['environment'] = []
if not isinstance(compose['services']['mlflow']['environment'], list):
    compose['services']['mlflow']['environment'] = [f"{k}={v}" for k, v in compose['services']['mlflow']['environment'].items()]

env_list = compose['services']['mlflow']['environment']
update_env_var(env_list, 'MLFLOW_TRACKING_URI', config['mlflow']['tracking_uri'])
compose['services']['mlflow']['ports'] = [f"{config['mlflow']['external_port']}:{config['mlflow']['port']}"]

# Update Airflow Webserver
if 'environment' not in compose['services']['airflow-webserver']:
    compose['services']['airflow-webserver']['environment'] = []
if not isinstance(compose['services']['airflow-webserver']['environment'], list):
    compose['services']['airflow-webserver']['environment'] = [f"{k}={v}" for k, v in compose['services']['airflow-webserver']['environment'].items()]

env_list = compose['services']['airflow-webserver']['environment']
update_env_var(env_list, 'AIRFLOW_HOME', config['airflow']['home'])
update_env_var(env_list, 'AIRFLOW__CORE__EXECUTOR', config['airflow']['executor'])
update_env_var(env_list, 'AIRFLOW__DATABASE__SQL_ALCHEMY_CONN', 
    f"postgresql+psycopg2://{config['postgres']['user']}:{config['postgres']['password']}@{config['postgres']['host']}:{config['postgres']['port']}/{config['postgres']['database']}")
update_env_var(env_list, 'AIRFLOW__CORE__LOAD_EXAMPLES', str(config['airflow']['load_examples']).lower())
update_env_var(env_list, 'AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION', str(config['airflow']['dags_paused_at_creation']).lower())
update_env_var(env_list, 'MLFLOW_TRACKING_URI', config['mlflow']['tracking_uri'])

compose['services']['airflow-webserver']['ports'] = [f"{config['airflow']['webserver_external_port']}:{config['airflow']['webserver_port']}"]

# Update Airflow Scheduler
if 'environment' not in compose['services']['airflow-scheduler']:
    compose['services']['airflow-scheduler']['environment'] = []
if not isinstance(compose['services']['airflow-scheduler']['environment'], list):
    compose['services']['airflow-scheduler']['environment'] = [f"{k}={v}" for k, v in compose['services']['airflow-scheduler']['environment'].items()]

env_list = compose['services']['airflow-scheduler']['environment']
update_env_var(env_list, 'AIRFLOW_HOME', config['airflow']['home'])
update_env_var(env_list, 'AIRFLOW__CORE__EXECUTOR', config['airflow']['executor'])
update_env_var(env_list, 'AIRFLOW__DATABASE__SQL_ALCHEMY_CONN', 
    f"postgresql+psycopg2://{config['postgres']['user']}:{config['postgres']['password']}@{config['postgres']['host']}:{config['postgres']['port']}/{config['postgres']['database']}")
update_env_var(env_list, 'AIRFLOW__CORE__LOAD_EXAMPLES', str(config['airflow']['load_examples']).lower())
update_env_var(env_list, 'AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION', str(config['airflow']['dags_paused_at_creation']).lower())
update_env_var(env_list, 'MLFLOW_TRACKING_URI', config['mlflow']['tracking_uri'])

# Add API service if it doesn't exist
if 'api' not in compose['services']:
    compose['services']['api'] = {
        'build': {'context': '.', 'dockerfile': 'Dockerfile'},
        'container_name': config['containers']['api'],
        'command': 'api',
        'environment': [],
        'volumes': [
            './data:/app/data',
            './mlruns:/app/mlruns',
            './models:/app/models',
            './src:/app/src',
            './mlflow_config.py:/app/mlflow_config.py',
            './train_model.py:/app/train_model.py',
            './main.py:/app/main.py',
            './controller:/app/controller',
            './front:/app/front'
        ],
        'ports': [f"{config['api']['external_port']}:{config['api']['port']}"],
        'networks': ['ml-pipeline-network'],
        'restart': 'unless-stopped',
        'depends_on': {
            'mlflow': {'condition': 'service_healthy'}
        },
        'healthcheck': {
            'test': ['CMD', 'curl', '-f', f"http://localhost:{config['api']['port']}/health"],
            'interval': '30s',
            'timeout': '10s',
            'retries': 5,
            'start_period': '60s'
        }
    }

# Update API service environment variables
if 'environment' not in compose['services']['api']:
    compose['services']['api']['environment'] = []
if not isinstance(compose['services']['api']['environment'], list):
    compose['services']['api']['environment'] = [f"{k}={v}" for k, v in compose['services']['api']['environment'].items()]

env_list = compose['services']['api']['environment']
update_env_var(env_list, 'MLFLOW_TRACKING_URI', config['mlflow']['tracking_uri'])
update_env_var(env_list, 'API_EXPERIMENT', config['api']['experiment_name'])
update_env_var(env_list, 'API_PORT', str(config['api']['port']))
update_env_var(env_list, 'MLFLOW_HOST', config['mlflow'].get('hostname', 'mlflow'))
update_env_var(env_list, 'MLFLOW_PORT', str(config['mlflow']['port']))

compose['services']['api']['ports'] = [f"{config['api']['external_port']}:{config['api']['port']}"]
compose['services']['api']['container_name'] = config['containers']['api']

# Update container names
compose['services']['postgres']['container_name'] = config['containers']['postgres']
compose['services']['mlflow']['container_name'] = config['containers']['mlflow']
compose['services']['airflow-webserver']['container_name'] = config['containers']['webserver']
compose['services']['airflow-scheduler']['container_name'] = config['containers']['scheduler']

# Update network name
if 'networks' in compose and 'ml-pipeline-network' in compose['networks']:
    compose['networks']['ml-pipeline-network']['name'] = config['network']['name']

# Write updated docker-compose.yml with better formatting
with open('docker-compose.yml', 'w') as f:
    # Use default_flow_style=False to get block style (more readable)
    yaml.dump(compose, f, default_flow_style=False, sort_keys=False, allow_unicode=True, width=1000, indent=2)

print("docker-compose.yml updated successfully from dev.yml")
PYTHON_SCRIPT

echo "Configuration sync complete!"
