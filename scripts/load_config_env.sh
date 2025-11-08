#!/bin/bash
# Load configuration from dev.yml and export as environment variables
# Usage: source scripts/load_config_env.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Check if Python and yaml are available
if command -v python3 &> /dev/null; then
    python3 -c "import yaml" 2>/dev/null || pip install pyyaml
    
    # Export environment variables from dev.yml
    eval $(python3 << 'PYTHON_SCRIPT'
import yaml
import os
from pathlib import Path

config_file = Path('dev.yml')
if config_file.exists():
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Export PostgreSQL config
    print(f"export POSTGRES_USER={config['postgres']['user']}")
    print(f"export POSTGRES_PASSWORD={config['postgres']['password']}")
    print(f"export POSTGRES_DB={config['postgres']['database']}")
    print(f"export POSTGRES_PORT={config['postgres']['port']}")
    print(f"export POSTGRES_HOST={config['postgres']['host']}")
    
    # Export Airflow config
    print(f"export AIRFLOW_HOME={config['airflow']['home']}")
    print(f"export AIRFLOW__CORE__EXECUTOR={config['airflow']['executor']}")
    print(f"export AIRFLOW__CORE__LOAD_EXAMPLES={str(config['airflow']['load_examples']).lower()}")
    print(f"export AIRFLOW_ADMIN_USERNAME={config['airflow']['admin_username']}")
    print(f"export AIRFLOW_ADMIN_PASSWORD={config['airflow']['admin_password']}")
    print(f"export AIRFLOW_WEBSERVER_PORT={config['airflow']['webserver_port']}")
    
    # Export MLflow config
    print(f"export MLFLOW_TRACKING_URI={config['mlflow']['tracking_uri']}")
    print(f"export MLFLOW_PORT={config['mlflow']['port']}")
    print(f"export MLFLOW_EXTERNAL_PORT={config['mlflow']['external_port']}")
PYTHON_SCRIPT
)
else
    echo "Warning: python3 not found. Cannot load config from dev.yml"
fi

