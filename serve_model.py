"""
Model Serving Script

This script loads the best model from MLflow and serves it via a simple Flask API
for making predictions.
"""

import os
import sys
import mlflow
import mlflow.sklearn
from flask import Flask
from flask_cors import CORS

# Add project root to path
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)

from mlflow_config import setup_mlflow
from controller import create_routes


def load_model_from_mlflow(experiment_name: str, run_name: str = None):
    """
    Load the best model from MLflow.
    
    Args:
        experiment_name (str): Name of the MLflow experiment
        run_name (str): Name of the specific run. If None, loads the best run.
    
    Returns:
        tuple: (model, model_name, problem_type)
    """
    print(f"Loading model from MLflow experiment: {experiment_name}")
    
    setup_mlflow(experiment_name)
    
    # Search for runs
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found!")
    
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if len(runs) == 0:
        raise ValueError("No runs found in experiment!")
    
    # If run_name specified, use it; otherwise find best run
    if run_name:
        run = runs[runs['tags.mlflow.runName'] == run_name]
        if len(run) == 0:
            raise ValueError(f"Run '{run_name}' not found!")
        best_run = run.iloc[0]
    else:
        # Find best run based on primary metric
        # For regression: highest R2, for classification: highest Accuracy
        if 'metrics.test_R2' in runs.columns:
            # Regression
            best_run = runs.loc[runs['metrics.test_R2'].idxmax()]
            problem_type = "regression"
        elif 'metrics.test_Accuracy' in runs.columns:
            # Classification
            best_run = runs.loc[runs['metrics.test_Accuracy'].idxmax()]
            problem_type = "classification"
        else:
            # Fallback: use first run
            best_run = runs.iloc[0]
            problem_type = "regression"  # default
    
    run_id = best_run['run_id']
    model_name = best_run['tags.mlflow.runName']
    
    print(f"Loading model: {model_name} (Run ID: {run_id})")
    
    # Load model
    model_uri = f"runs:/{run_id}/model"
    loaded_model = mlflow.sklearn.load_model(model_uri)
    
    # Get problem type from tags
    if 'tags.problem_type' in best_run:
        problem_type = best_run['tags.problem_type']
    else:
        problem_type = "regression"  # default
    
    print(f"Model loaded successfully!")
    print(f"   Model: {model_name}")
    print(f"   Problem Type: {problem_type}")
    
    return loaded_model, model_name, problem_type


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Serve ML model via Flask API")
    parser.add_argument("--experiment", type=str, default="House_Price_Prediction",
                       help="MLflow experiment name")
    parser.add_argument("--run-name", type=str, default=None,
                       help="Specific run name to load (optional)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000,
                       help="Port to bind to")
    
    args = parser.parse_args()
    
    # Initialize Flask app with frontend folder structure
    front_path = os.path.join(project_root, 'front')
    app = Flask(__name__, 
                template_folder=os.path.join(front_path, 'templates'),
                static_folder=os.path.join(front_path, 'static'))
    CORS(app)  # Enable CORS for frontend requests
    
    # Load model
    try:
        model, model_name, problem_type = load_model_from_mlflow(
            args.experiment, args.run_name
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Register all routes from controller
    create_routes(app, model, model_name, problem_type)
    
    print(f"\nStarting Flask server...")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"\nWeb Interface:")
    print(f"   Open your browser to: http://localhost:{args.port}")
    print(f"\nAPI Endpoints:")
    print(f"   GET  /                 - Web interface (frontend)")
    print(f"   GET  /health           - Health check")
    print(f"   POST /predict          - Single prediction")
    print(f"   POST /predict_batch    - Batch predictions")
    print(f"\nExample API request:")
    print(f'   curl -X POST http://localhost:{args.port}/predict \\')
    print(f'        -H "Content-Type: application/json" \\')
    print(f'        -d \'{{"features": {{"feature1": value1, "feature2": value2}}}}\'')
    print(f"\nServer is ready! Press Ctrl+C to stop.\n")
    
    app.run(host=args.host, port=args.port, debug=False)
