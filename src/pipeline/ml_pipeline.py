"""
ML Pipeline Class - OOP wrapper for training pipeline functions

This class encapsulates all the training pipeline logic from train_model.py
and makes it reusable in Airflow DAGs.
"""

import os
import sys
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import from train_model.py
import train_model
from mlflow_config import setup_mlflow
from src.utils import (
    load_data, validate_data, split_features_target,
    calculate_regression_metrics, calculate_classification_metrics,
    compare_models, get_best_model
)


class MLPipeline:
    """Main pipeline class that orchestrates the ML training process."""
    
    def __init__(self, config: dict = None):
        """
        Initialize ML Pipeline.
        
        Args:
            config (dict): Configuration dictionary. If None, uses default from train_model.CONFIG
        """
        if config is None:
            self.config = train_model.CONFIG.copy()
        else:
            self.config = config
        
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_processed = None
        self.X_test_processed = None
        self.models = None
        self.all_results = {}
        self.best_model_name = None
        self.best_metrics = None
    
    def extract_data(self, data_path: str = None) -> dict:
        """
        Extract and load data from CSV file.
        
        Args:
            data_path (str): Path to data file. If None, uses config value.
        
        Returns:
            dict: Extraction metadata
        """
        if data_path is None:
            data_path = self.config.get("data_path", "data/housing.csv")
        
        # Load data
        self.df = load_data(data_path)
        
        # Validate data
        validate_data(self.df, self.config["target_variable"])
        
        return {
            'status': 'success',
            'rows': self.df.shape[0],
            'columns': self.df.shape[1],
            'file_path': data_path
        }
    
    def validate_data_quality(self, missing_threshold: float = 0.05) -> dict:
        """
        Validate data quality.
        
        Args:
            missing_threshold (float): Maximum acceptable percentage of missing values
        
        Returns:
            dict: Validation results
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call extract_data() first.")
        
        issues = []
        
        # Check missing values
        missing_pct = self.df.isnull().sum() / len(self.df)
        high_missing = missing_pct[missing_pct > missing_threshold]
        
        if len(high_missing) > 0:
            for col, pct in high_missing.items():
                issues.append(f"Column '{col}' has {pct:.2%} missing values (threshold: {missing_threshold:.2%})")
        
        # Check data types
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            issues.append("No numeric columns found!")
        
        # Check target variable
        target = self.config["target_variable"]
        if target in self.df.columns:
            if self.df[target].min() < 0 and self.config["problem_type"] == "regression":
                issues.append(f"Target variable '{target}' has negative values")
        
        return {
            'status': 'success' if len(issues) <= 5 else 'failed',
            'issues_count': len(issues),
            'issues': issues
        }
    
    def preprocess_data(self) -> dict:
        """
        Preprocess data using functions from train_model.py.
        
        Returns:
            dict: Preprocessing results
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call extract_data() first.")
        
        # Split features and target
        X, y = split_features_target(self.df, self.config["target_variable"])
        
        # Train/Test Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.config["test_size"],
            random_state=self.config["random_state"]
        )
        
        # Preprocess using train_model function
        self.X_train_processed, self.X_test_processed, self.y_train, self.y_test = train_model.preprocess_data(
            self.X_train, self.X_test, self.y_train, self.y_test, self.config
        )
        
        return {
            'status': 'success',
            'features_count': self.X_train_processed.shape[1],
            'samples_count': self.X_train_processed.shape[0]
        }
    
    def train_models(self) -> dict:
        """
        Train all models using functions from train_model.py.
        
        Returns:
            dict: Training results with best model info
        """
        if self.X_train_processed is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")
        
        # Setup MLflow
        setup_mlflow(self.config["experiment_name"])
        
        # Get models
        self.models = train_model.get_models(self.config["problem_type"])
        
        # Train all models
        self.all_results = {}
        
        for model_name, model in self.models.items():
            metrics = train_model.train_and_log_model(
                model_name, model, self.X_train_processed, self.X_test_processed,
                self.y_train, self.y_test, self.config["problem_type"], 
                self.config["cv_folds"]
            )
            self.all_results[model_name] = metrics
        
        # Get best model
        self.best_model_name, self.best_metrics = get_best_model(
            self.all_results, self.config["problem_type"]
        )
        
        # Get best run ID and metric from MLflow
        experiment = mlflow.get_experiment_by_name(self.config["experiment_name"])
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        best_run_id = None
        best_metric_value = 0.0
        
        # Determine metric column based on problem type
        if self.config["problem_type"] == "regression":
            metric_col = 'metrics.test_R2'
        else:
            metric_col = 'metrics.test_Accuracy'
        
        if len(runs) > 0 and metric_col in runs.columns:
            runs_with_metric = runs[runs[metric_col].notna()]
            if len(runs_with_metric) > 0:
                # Find run matching best model name
                best_run = runs_with_metric[
                    runs_with_metric['tags.mlflow.runName'] == self.best_model_name
                ]
                if len(best_run) > 0:
                    best_run = best_run.iloc[0]
                    best_run_id = str(best_run.get('run_id', ''))
                    best_metric_value = float(best_run[metric_col])
                else:
                    # If exact match not found, get the best run by metric
                    best_run = runs_with_metric.sort_values(metric_col, ascending=False).iloc[0]
                    best_run_id = str(best_run.get('run_id', ''))
                    best_metric_value = float(best_run[metric_col])
        
        return {
            'status': 'success',
            'best_model': self.best_model_name,
            'best_metric': best_metric_value,
            'best_run_id': best_run_id
        }
    
    def evaluate_model(self, new_model_metric: float, production_metric: float = None, improvement_threshold: float = 0.01) -> dict:
        """
        Evaluate model and compare with production.
        
        Args:
            new_model_metric (float): New model metric value
            production_metric (float): Current production model metric
            improvement_threshold (float): Minimum improvement required to promote
        
        Returns:
            dict: Evaluation results
        """
        promote_to_production = False
        improvement = 0.0
        
        if production_metric is not None and new_model_metric > 0:
            improvement = new_model_metric - production_metric
            if improvement >= improvement_threshold:
                promote_to_production = True
        else:
            # No production model, promote this one
            promote_to_production = True
        
        return {
            'status': 'success',
            'promote_to_production': promote_to_production,
            'new_model_metric': new_model_metric,
            'production_model_metric': production_metric,
            'improvement': improvement
        }
    
    def run_full_pipeline(self) -> dict:
        """
        Run the complete pipeline from start to finish.
        
        Returns:
            dict: Complete pipeline results
        """
        results = {}
        
        # Step 1: Extract data
        results['extract'] = self.extract_data()
        
        # Step 2: Validate data quality
        results['validate'] = self.validate_data_quality()
        
        if results['validate']['status'] == 'failed':
            return results
        
        # Step 3: Preprocess data
        results['preprocess'] = self.preprocess_data()
        
        # Step 4: Train models
        results['train'] = self.train_models()
        
        # Step 5: Evaluate model
        results['evaluate'] = self.evaluate_model()
        
        return results

