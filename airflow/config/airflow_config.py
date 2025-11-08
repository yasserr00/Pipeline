"""
Airflow Configuration

This module contains Airflow configuration settings for both local and production environments.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# ============================================================================
# AIRFLOW PATHS
# ============================================================================

AIRFLOW_HOME = os.path.join(PROJECT_ROOT, "airflow")
DAGS_FOLDER = os.path.join(AIRFLOW_HOME, "dags")
LOGS_FOLDER = os.path.join(AIRFLOW_HOME, "logs")
PLUGINS_FOLDER = os.path.join(AIRFLOW_HOME, "plugins")

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

# For local development (SQLite)
LOCAL_DB_PATH = os.path.join(AIRFLOW_HOME, "airflow.db")
SQLITE_DATABASE_URL = f"sqlite:///{LOCAL_DB_PATH}"

# For production (PostgreSQL) - uncomment and configure
# POSTGRES_DATABASE_URL = "postgresql+psycopg2://user:password@localhost:5432/airflow"

# Current database URL (defaults to SQLite for local)
DATABASE_URL = SQLITE_DATABASE_URL

# ============================================================================
# EXECUTOR CONFIGURATION
# ============================================================================

# For local: SequentialExecutor (runs one task at a time)
# For production: LocalExecutor or CeleryExecutor
EXECUTOR = "SequentialExecutor"  # Options: SequentialExecutor, LocalExecutor, CeleryExecutor

# ============================================================================
# WEBSERVER CONFIGURATION
# ============================================================================

WEBSERVER_HOST = "localhost"
WEBSERVER_PORT = 8080
WEBSERVER_WORKERS = 4

# ============================================================================
# SCHEDULER CONFIGURATION
# ============================================================================

SCHEDULER_HEARTBEAT_SEC = 5
SCHEDULER_DAG_DIR_LIST_INTERVAL = 300  # Check for new DAGs every 5 minutes

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ============================================================================
# SECURITY CONFIGURATION
# ============================================================================

# For local development - use simple authentication
# For production - configure proper authentication
AUTHENTICATE = False
AUTH_ROLE_PUBLIC = "Public"

# ============================================================================
# ENVIRONMENT DETECTION
# ============================================================================

def is_production():
    """Check if running in production environment."""
    return os.getenv("AIRFLOW_ENV", "local").lower() == "production"


def get_database_url():
    """Get database URL based on environment."""
    if is_production():
        return os.getenv("AIRFLOW_DATABASE_URL", DATABASE_URL)
    return SQLITE_DATABASE_URL


def get_executor():
    """Get executor based on environment."""
    if is_production():
        return os.getenv("AIRFLOW_EXECUTOR", "LocalExecutor")
    return EXECUTOR


# ============================================================================
# CONFIGURATION FOR AIRFLOW.CFG
# ============================================================================

AIRFLOW_CONFIG = {
    'core': {
        'dags_folder': DAGS_FOLDER,
        'executor': get_executor(),
        'sql_alchemy_conn': get_database_url(),
        'parallelism': 32,
        'dag_concurrency': 16,
        'max_active_runs_per_dag': 16,
        'load_examples': False,
    },
    'scheduler': {
        'dag_dir_list_interval': SCHEDULER_DAG_DIR_LIST_INTERVAL,
        'job_heartbeat_sec': SCHEDULER_HEARTBEAT_SEC,
    },
    'webserver': {
        'web_server_host': WEBSERVER_HOST,
        'web_server_port': WEBSERVER_PORT,
        'workers': WEBSERVER_WORKERS,
    },
    'logging': {
        'logging_level': LOG_LEVEL,
        'log_format': LOG_FORMAT,
        'simple_log_format': LOG_FORMAT,
    },
}


def print_config():
    """Print current Airflow configuration."""
    print("=" * 70)
    print("AIRFLOW CONFIGURATION")
    print("=" * 70)
    print(f"Environment: {'PRODUCTION' if is_production() else 'LOCAL'}")
    print(f"Airflow Home: {AIRFLOW_HOME}")
    print(f"DAGs Folder: {DAGS_FOLDER}")
    print(f"Logs Folder: {LOGS_FOLDER}")
    print(f"Database: {get_database_url()}")
    print(f"Executor: {get_executor()}")
    print("=" * 70)


if __name__ == "__main__":
    print_config()

