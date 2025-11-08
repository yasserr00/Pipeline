"""
Main Application Entry Point

This is the main entry point for the ML Model Serving application.
Uses Object-Oriented Programming principles for better organization and maintainability.
"""

import os
import sys
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from flask import Flask
from flask_cors import CORS

# Add project root to path
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)

from mlflow_config import setup_mlflow
from controller import create_routes


class ModelLoader:
    """Handles loading ML models from MLflow."""
    
    def __init__(self, experiment_name: str, run_name: str = None):
        """
        Initialize ModelLoader.
        
        Args:
            experiment_name (str): Name of the MLflow experiment
            run_name (str): Specific run name to load (optional)
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.model = None
        self.model_name = None
        self.problem_type = None
    
    def load(self):
        """
        Load the best model from MLflow.
        
        Returns:
            tuple: (model, model_name, problem_type)
        
        Raises:
            ValueError: If experiment or run not found
        """
        print(f"Loading model from MLflow experiment: {self.experiment_name}")
        
        setup_mlflow(self.experiment_name)
        
        # Search for runs
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{self.experiment_name}' not found!")
        
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        if len(runs) == 0:
            raise ValueError("No runs found in experiment!")
        
        # Find the best run
        best_run = self._find_best_run(runs)
        
        run_id = best_run['run_id']
        self.model_name = best_run['tags.mlflow.runName']
        
        print(f"Loading model: {self.model_name} (Run ID: {run_id})")
        
        # Load model - try different URI formats
        model_uri = f"runs:/{run_id}/model"
        try:
            self.model = mlflow.sklearn.load_model(model_uri)
        except Exception as e:
            # Try alternative path
            print(f"Warning: Could not load from {model_uri}, trying alternative...")
            try:
                # Try with experiment ID
                model_uri = f"{experiment.experiment_id}/{run_id}/artifacts/model"
                self.model = mlflow.sklearn.load_model(model_uri)
            except Exception as e2:
                # Try direct file path
                import os
                mlruns_path = os.path.join(os.getcwd(), "mlruns")
                model_path = os.path.join(mlruns_path, str(experiment.experiment_id), run_id, "artifacts", "model")
                if os.path.exists(model_path):
                    self.model = mlflow.sklearn.load_model(model_path)
                else:
                    raise ValueError(f"Model not found for run {run_id}. Path checked: {model_path}. Original error: {e}")
        
        # Get problem type from tags
        if 'tags.problem_type' in best_run:
            self.problem_type = best_run['tags.problem_type']
        else:
            self.problem_type = "regression"  # default
        
        print(f"Model loaded successfully!")
        print(f"   Model: {self.model_name}")
        print(f"   Problem Type: {self.problem_type}")
        
        return self.model, self.model_name, self.problem_type
    
    def _find_best_run(self, runs):
        """
        Find the best run based on metrics, filtering out runs without models.
        
        Args:
            runs: DataFrame of MLflow runs
        
        Returns:
            Series: Best run with a valid model
        """
        import os
        
        if self.run_name:
            run = runs[runs['tags.mlflow.runName'] == self.run_name]
            if len(run) == 0:
                raise ValueError(f"Run '{self.run_name}' not found!")
            selected_run = run.iloc[0]
        else:
            # Find best run based on primary metric
            if 'metrics.test_R2' in runs.columns:
                # Regression - highest R2
                # Filter runs that have valid R2 values
                runs_with_metric = runs[runs['metrics.test_R2'].notna()]
                if len(runs_with_metric) > 0:
                    best_run = runs_with_metric.loc[runs_with_metric['metrics.test_R2'].idxmax()]
                    self.problem_type = "regression"
                else:
                    best_run = runs.iloc[0]
                    self.problem_type = "regression"
            elif 'metrics.test_Accuracy' in runs.columns:
                # Classification - highest Accuracy
                runs_with_metric = runs[runs['metrics.test_Accuracy'].notna()]
                if len(runs_with_metric) > 0:
                    best_run = runs_with_metric.loc[runs_with_metric['metrics.test_Accuracy'].idxmax()]
                    self.problem_type = "classification"
                else:
                    best_run = runs.iloc[0]
                    self.problem_type = "classification"
            else:
                # Fallback: use first run
                best_run = runs.iloc[0]
                self.problem_type = "regression"
            
            selected_run = best_run
        
        # Verify the run has a model artifact
        run_id = selected_run['run_id']
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        mlruns_path = os.path.join(os.getcwd(), "mlruns")
        model_path = os.path.join(mlruns_path, str(experiment.experiment_id), run_id, "artifacts", "model")
        
        # Check if model directory exists and has files
        if not os.path.exists(model_path) or not os.listdir(model_path):
            # Try to find a run with a valid model
            print(f"Warning: Run {run_id} does not have a model artifact. Searching for runs with models...")
            
            # Filter runs that have models
            runs_with_models = []
            for idx, run in runs.iterrows():
                check_run_id = run['run_id']
                check_model_path = os.path.join(mlruns_path, str(experiment.experiment_id), check_run_id, "artifacts", "model")
                if os.path.exists(check_model_path) and os.listdir(check_model_path):
                    runs_with_models.append((idx, run))
            
            if len(runs_with_models) == 0:
                raise ValueError("No runs with valid model artifacts found!")
            
            # Use the best run from runs with models
            if 'metrics.test_R2' in runs.columns:
                best_with_model = max(runs_with_models, key=lambda x: x[1].get('metrics.test_R2', 0) if pd.notna(x[1].get('metrics.test_R2')) else 0)
            elif 'metrics.test_Accuracy' in runs.columns:
                best_with_model = max(runs_with_models, key=lambda x: x[1].get('metrics.test_Accuracy', 0) if pd.notna(x[1].get('metrics.test_Accuracy')) else 0)
            else:
                best_with_model = runs_with_models[0]
            
            selected_run = best_with_model[1]
            print(f"Selected run with valid model: {selected_run['tags.mlflow.runName']} (Run ID: {selected_run['run_id']})")
        
        return selected_run


class FlaskApp:
    """Manages Flask application setup and configuration."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 5000):
        """
        Initialize FlaskApp.
        
        Args:
            host (str): Host to bind to
            port (int): Port to bind to
        """
        self.host = host
        self.port = port
        self.app = None
    
    def create_app(self, model, model_name: str, problem_type: str):
        """
        Create and configure Flask application.
        
        Args:
            model: Loaded ML model
            model_name (str): Name of the model
            problem_type (str): Type of problem (regression/classification)
        
        Returns:
            Flask: Configured Flask application
        """
        # Initialize Flask app with frontend folder structure
        front_path = os.path.join(os.path.dirname(__file__), 'front')
        self.app = Flask(__name__, 
                        template_folder=os.path.join(front_path, 'templates'),
                        static_folder=os.path.join(front_path, 'static'))
        
        # Enable CORS
        CORS(self.app)
        
        # Register routes
        create_routes(self.app, model, model_name, problem_type)
        
        return self.app
    
    def run(self):
        """Start the Flask server."""
        if self.app is None:
            raise RuntimeError("Flask app not created. Call create_app() first.")
        
        self._print_startup_info()
        self.app.run(host=self.host, port=self.port, debug=False)
    
    def _print_startup_info(self):
        """Print server startup information."""
        print(f"\nStarting Flask server...")
        print(f"   Host: {self.host}")
        print(f"   Port: {self.port}")
        print(f"\nWeb Interface:")
        print(f"   Open your browser to: http://localhost:{self.port}")
        print(f"\nAPI Endpoints:")
        print(f"   GET  /                 - Web interface (frontend)")
        print(f"   GET  /health           - Health check")
        print(f"   POST /predict          - Single prediction")
        print(f"   POST /predict_batch    - Batch predictions")
        print(f"\nExample API request:")
        print(f'   curl -X POST http://localhost:{self.port}/predict \\')
        print(f'        -H "Content-Type: application/json" \\')
        print(f'        -d \'{{"features": {{"feature1": value1, "feature2": value2}}}}\'')
        print(f"\nServer is ready! Press Ctrl+C to stop.\n")


class MLModelServer:
    """Main application class that orchestrates the ML model server."""
    
    def __init__(self, experiment_name: str, run_name: str = None, 
                 host: str = "0.0.0.0", port: int = 5000):
        """
        Initialize MLModelServer.
        
        Args:
            experiment_name (str): MLflow experiment name
            run_name (str): Specific run name (optional)
            host (str): Server host
            port (int): Server port
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.host = host
        self.port = port
        
        self.model_loader = ModelLoader(experiment_name, run_name)
        self.flask_app = FlaskApp(host, port)
    
    def start(self):
        """Start the ML model server."""
        try:
            # Load model
            model, model_name, problem_type = self.model_loader.load()
            
            # Create Flask app
            self.flask_app.create_app(model, model_name, problem_type)
            
            # Start server
            self.flask_app.run()
            
        except Exception as e:
            print(f"Error starting server: {e}")
            sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ML Model Serving Application - OOP Implementation"
    )
    parser.add_argument(
        "--experiment", 
        type=str, 
        default="House_Price_Prediction",
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--run-name", 
        type=str, 
        default=None,
        help="Specific run name to load (optional)"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=5000,
        help="Port to bind to"
    )
    
    args = parser.parse_args()
    
    # Create and start server
    server = MLModelServer(
        experiment_name=args.experiment,
        run_name=args.run_name,
        host=args.host,
        port=args.port
    )
    
    server.start()


if __name__ == "__main__":
    main()

