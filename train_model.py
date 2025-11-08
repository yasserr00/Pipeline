"""
Main Training Script for ML Model with MLflow Tracking

This script:
1. Loads and validates data
2. Performs EDA and data cleaning
3. Creates feature engineering
4. Preprocesses data (scaling, encoding)
5. Trains multiple models with cross-validation
6. Logs everything to MLflow
7. Selects and saves the best model
"""

import os
import sys
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)

from mlflow_config import setup_mlflow, log_model_info
from src.utils import (
    load_data, validate_data, split_features_target,
    handle_missing_values, encode_categorical_features, scale_features,
    create_interaction_features,
    calculate_regression_metrics, calculate_classification_metrics,
    compare_models, get_best_model, plot_feature_importance
)


# ============================================================================
# CONFIGURATION - MODIFY THESE VALUES FOR YOUR PROJECT
# ============================================================================

CONFIG = {
    # Data Configuration
    "data_path": "data/housing.csv",  # Path to your CSV file
    "target_variable": "price",  # Name of target column
    "problem_type": "regression",  # "regression" or "classification"
    
    # Train/Test Split
    "test_size": 0.2,
    "random_state": 42,
    
    # Cross-Validation
    "cv_folds": 5,
    
    # MLflow Configuration
    "experiment_name": "House_Price_Prediction",
    
    # Feature Engineering
    "create_interactions": True,  # Create interaction features
    "create_polynomials": False,  # Create polynomial features
    "handle_missing": True,  # Handle missing values
    "scale_features": True,  # Scale numeric features
    "encode_categorical": True,  # Encode categorical features
}


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

def get_models(problem_type: str):
    """
    Get list of models to train based on problem type.
    
    Args:
        problem_type (str): "regression" or "classification"
    
    Returns:
        dict: Dictionary of model names and model instances
    """
    if problem_type == "regression":
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            "XGBoost": xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        }
    else:  # classification
        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            "XGBoost": xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        }
    
    return models


# ============================================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

def perform_eda(df: pd.DataFrame, target_column: str):
    """
    Perform basic exploratory data analysis.
    
    Args:
        df (pd.DataFrame): The dataset
        target_column (str): Name of target column
    """
    print("\n" + "=" * 70)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)
    
    # Basic Info
    print(f"\nDataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nColumn Names:")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        missing = df[col].isnull().sum()
        print(f"   {i:2d}. {col:20s} ({dtype}) - Missing: {missing}")
    
    # Missing Values Summary
    missing_summary = df.isnull().sum()
    if missing_summary.sum() > 0:
        print(f"\nMissing Values Summary:")
        for col, count in missing_summary[missing_summary > 0].items():
            pct = (count / len(df)) * 100
            print(f"   {col:20s}: {count:6,} ({pct:.2f}%)")
    else:
        print(f"\nNo missing values found!")
    
    # Target Variable Statistics
    if target_column in df.columns:
        print(f"\nTarget Variable Statistics ({target_column}):")
        print(df[target_column].describe().to_string())
        
        # Check if classification or regression
        if df[target_column].dtype == 'object' or df[target_column].nunique() < 20:
            print(f"\nTarget Distribution:")
            print(df[target_column].value_counts().to_string())
    
    # Numeric Features Summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 0:
        print(f"\nNumeric Features Summary:")
        print(df[numeric_cols].describe().to_string())
    
    print("\n" + "=" * 70)


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def preprocess_data(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                   y_train: pd.Series, y_test: pd.Series,
                   config: dict) -> tuple:
    """
    Preprocess training and test data.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        config: Configuration dictionary
    
    Returns:
        tuple: Preprocessed X_train, X_test, y_train, y_test
    """
    print("\n" + "=" * 70)
    print("DATA PREPROCESSING")
    print("=" * 70)
    
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    
    # Handle Missing Values
    if config.get("handle_missing", True):
        print("\n1. Handling Missing Values...")
        X_train_processed = handle_missing_values(X_train_processed, strategy="mean")
        X_test_processed = handle_missing_values(X_test_processed, strategy="mean")
    
    # Identify Categorical Columns
    categorical_cols = X_train_processed.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X_train_processed.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"\nFeature Types:")
    print(f"   Numeric: {len(numeric_cols)} features")
    print(f"   Categorical: {len(categorical_cols)} features")
    
    # Encode Categorical Features
    if config.get("encode_categorical", True) and len(categorical_cols) > 0:
        print(f"\n2. Encoding Categorical Features...")
        X_train_processed, X_test_processed = encode_categorical_features(
            X_train_processed, X_test_processed, categorical_cols
        )
    
    # Create Interaction Features
    if config.get("create_interactions", False) and len(numeric_cols) >= 2:
        print(f"\n3. Creating Interaction Features...")
        # Create interactions between top numeric features
        top_features = numeric_cols[:min(3, len(numeric_cols))]
        if len(top_features) >= 2:
            feature_pairs = [(top_features[0], top_features[1])]
            X_train_processed = create_interaction_features(X_train_processed, feature_pairs)
            X_test_processed = create_interaction_features(X_test_processed, feature_pairs)
    
    # Scale Features
    if config.get("scale_features", True):
        print(f"\n4. Scaling Numeric Features...")
        X_train_processed, X_test_processed, scaler = scale_features(
            X_train_processed, X_test_processed, numeric_cols
        )
    
    print(f"\nPreprocessing Complete!")
    print(f"   Final Training Shape: {X_train_processed.shape}")
    print(f"   Final Test Shape: {X_test_processed.shape}")
    
    return X_train_processed, X_test_processed, y_train, y_test


# ============================================================================
# MODEL TRAINING WITH MLFLOW
# ============================================================================

def train_and_log_model(model_name: str, model, X_train: pd.DataFrame, 
                       X_test: pd.DataFrame, y_train: pd.Series, 
                       y_test: pd.Series, problem_type: str, 
                       cv_folds: int = 5) -> dict:
    """
    Train a model and log everything to MLflow.
    
    Args:
        model_name (str): Name of the model
        model: Model instance
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        problem_type (str): "regression" or "classification"
        cv_folds (int): Number of cross-validation folds
    
    Returns:
        dict: Dictionary of metrics
    """
    print(f"\n{'='*70}")
    print(f"Training: {model_name}")
    print(f"{'='*70}")
    
    with mlflow.start_run(run_name=model_name):
        # Log model info
        log_model_info(model_name, problem_type, CONFIG["target_variable"])
        
        # Extract hyperparameters
        hyperparams = model.get_params()
        print(f"\nHyperparameters:")
        for key, value in hyperparams.items():
            print(f"   {key}: {value}")
            mlflow.log_param(key, value)
        
        # Train the model
        print(f"\nTraining model...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Cross-Validation
        print(f"\nPerforming {cv_folds}-fold Cross-Validation...")
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        if problem_type == "regression":
            cv_scores = cross_val_score(model, X_train, y_train, 
                                       cv=kfold, scoring='neg_mean_squared_error')
            cv_metric_name = "CV_RMSE"
            cv_metric_value = np.sqrt(-cv_scores.mean())
        else:
            cv_scores = cross_val_score(model, X_train, y_train, 
                                       cv=kfold, scoring='accuracy')
            cv_metric_name = "CV_Accuracy"
            cv_metric_value = cv_scores.mean()
        
        print(f"   {cv_metric_name}: {cv_metric_value:.4f} (+/- {cv_scores.std() * 2:.4f})")
        mlflow.log_metric(cv_metric_name, cv_metric_value)
        
        # Calculate Metrics
        if problem_type == "regression":
            train_metrics = calculate_regression_metrics(y_train, y_train_pred)
            test_metrics = calculate_regression_metrics(y_test, y_test_pred)
            
            # Log metrics with train/test prefix
            for metric_name, metric_value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", metric_value)
            for metric_name, metric_value in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", metric_value)
            
            # Use test metrics for comparison
            metrics = test_metrics
            print(metrics, problem_type)
            
        else:  # classification
            train_metrics = calculate_classification_metrics(y_train, y_train_pred)
            test_metrics = calculate_classification_metrics(y_test, y_test_pred)
            
            # Log metrics
            for metric_name, metric_value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", metric_value)
            for metric_name, metric_value in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", metric_value)
            
            metrics = test_metrics
            print(metrics, problem_type)
        
        # Log model artifact
        mlflow.sklearn.log_model(model, "model")
        print(f"   Model logged to MLflow")
        
        # Feature Importance
        try:
            importances = plot_feature_importance(model, X_train.columns.tolist(), model_name)
            if importances is not None:
                # Log feature importance as artifact
                importance_df = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                importance_path = f"models/feature_importance_{model_name.lower().replace(' ', '_')}.csv"
                importance_df.to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path)
                
                print(f"   Feature importance logged")
        except Exception as e:
            print(f"   Warning: Could not log feature importance: {e}")
        
        # Log data info
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_test_samples", len(X_test))
        
        print(f"{model_name} training complete!")
        
        return metrics


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("ML MODEL TRAINING WITH MLFLOW TRACKING")
    print("=" * 70)
    
    # Ensure directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Setup MLflow
    print("\nSetting up MLflow...")
    setup_mlflow(CONFIG["experiment_name"])
    
    # Load Data
    print("\nLoading Data...")
    df = load_data(CONFIG["data_path"])
    
    # Validate Data
    validate_data(df, CONFIG["target_variable"])
    
    # Perform EDA
    perform_eda(df, CONFIG["target_variable"])
    
    # Split Features and Target
    print("\nSplitting Features and Target...")
    X, y = split_features_target(df, CONFIG["target_variable"])
    
    # Train/Test Split
    print(f"\nSplitting Data (Train: {1-CONFIG['test_size']:.0%}, Test: {CONFIG['test_size']:.0%})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=CONFIG["test_size"], 
        random_state=CONFIG["random_state"]
    )
    print(f"   Training set: {X_train.shape[0]:,} samples")
    print(f"   Test set: {X_test.shape[0]:,} samples")
    
    # Preprocess Data
    X_train_processed, X_test_processed, y_train, y_test = preprocess_data(
        X_train, X_test, y_train, y_test, CONFIG
    )
    
    # Get Models
    models = get_models(CONFIG["problem_type"])
    
    # Train All Models
    print("\n" + "=" * 70)
    print("TRAINING ALL MODELS")
    print("=" * 70)
    
    all_results = {}
    
    for model_name, model in models.items():
        metrics = train_and_log_model(
            model_name, model, X_train_processed, X_test_processed,
            y_train, y_test, CONFIG["problem_type"], CONFIG["cv_folds"]
        )
        all_results[model_name] = metrics
    
    # Compare Models
    comparison_df = compare_models(all_results, CONFIG["problem_type"])
    
    # Get Best Model
    best_model_name, best_metrics = get_best_model(
        all_results, CONFIG["problem_type"]
    )
    
    # Mark Best Model as Production
    print(f"\nMarking best model as production...")
    try:
        # Get the run ID of the best model
        experiment = mlflow.get_experiment_by_name(CONFIG["experiment_name"])
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        # Find the run for the best model
        best_run = runs[runs['tags.mlflow.runName'] == best_model_name].iloc[0]
        best_run_id = best_run['run_id']
        
        # Register the model
        model_uri = f"runs:/{best_run_id}/model"
        mlflow.register_model(model_uri, best_model_name)
        
        print(f"Best model '{best_model_name}' registered and ready for production!")
    except Exception as e:
        print(f"Warning: Could not register model: {e}")
    
    # Save comparison table
    comparison_df.to_csv("models/model_comparison.csv", index=True)
    print(f"\nModel comparison saved to: models/model_comparison.csv")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nView MLflow UI: mlflow ui")
    print(f"Best Model: {best_model_name}")
    print(f"Models saved in: models/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

