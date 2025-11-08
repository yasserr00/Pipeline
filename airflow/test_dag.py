#!/usr/bin/env python3
"""
Test script to verify DAG can be imported
"""

import sys
import os
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "airflow" / "dags"))

print("Testing DAG import...")
print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}")

try:
    # Try importing the DAG
    from dags.ml_pipeline_dag import dag
    print("DAG imported successfully!")
    print(f"   DAG ID: {dag.dag_id}")
    print(f"   Tasks: {[task.task_id for task in dag.tasks]}")
    print(f"   Schedule: {dag.schedule_interval}")
except Exception as e:
    print(f"Error importing DAG: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

