"""
Model Evaluation Module

This module handles model evaluation, metric calculation, and comparison.
Supports both regression and classification problems.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Args:
        y_true (np.ndarray): True target values
        y_pred (np.ndarray): Predicted target values
    
    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MSE": mse
    }
    
    return metrics


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                     average: str = "weighted") -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true (np.ndarray): True target labels
        y_pred (np.ndarray): Predicted target labels
        average (str): Averaging strategy for multi-class problems
    
    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    }
    
    return metrics


def print_metrics(metrics: Dict[str, float], problem_type: str):
    """
    Print metrics in a formatted way.
    
    Args:
        metrics (Dict[str, float]): Dictionary of metrics
        problem_type (str): Type of problem ('regression' or 'classification')
    """
    print(f"\n{'Regression' if problem_type == 'regression' else 'Classification'} Metrics:")
    print("=" * 50)
    for metric_name, metric_value in metrics.items():
        print(f"   {metric_name:15s}: {metric_value:.4f}")
    print("=" * 50)


def compare_models(results: Dict[str, Dict[str, float]], problem_type: str) -> pd.DataFrame:
    """
    Create a comparison table of all models.
    
    Args:
        results (Dict[str, Dict[str, float]]): Dictionary with model names as keys
                                               and metrics as values
        problem_type (str): Type of problem ('regression' or 'classification')
    
    Returns:
        pd.DataFrame: Comparison table
    """
    comparison_df = pd.DataFrame(results).T
    comparison_df.index.name = "Model"
    
    print("\n" + "=" * 70)
    print("MODEL COMPARISON TABLE")
    print("=" * 70)
    print(comparison_df.to_string())
    print("=" * 70)
    
    return comparison_df


def get_best_model(results: Dict[str, Dict[str, float]], problem_type: str, 
                   primary_metric: str = None) -> Tuple[str, Dict[str, float]]:
    """
    Determine the best model based on primary metric.
    
    Args:
        results (Dict[str, Dict[str, float]]): Dictionary with model results
        problem_type (str): Type of problem ('regression' or 'classification')
        primary_metric (str): Primary metric to optimize. 
                             If None, uses R2 for regression, Accuracy for classification.
    
    Returns:
        Tuple[str, Dict[str, float]]: Best model name and its metrics
    """
    if primary_metric is None:
        primary_metric = "R2" if problem_type == "regression" else "Accuracy"
    
    # For regression, higher R2 is better; for classification, higher Accuracy is better
    # For RMSE/MAE, lower is better
    if problem_type == "regression":
        if primary_metric in ["R2"]:
            best_model = max(results.items(), key=lambda x: x[1].get(primary_metric, -np.inf))
        else:  # MAE, RMSE, MSE
            best_model = min(results.items(), key=lambda x: x[1].get(primary_metric, np.inf))
    else:  # classification
        best_model = max(results.items(), key=lambda x: x[1].get(primary_metric, -np.inf))
    
    model_name, metrics = best_model
    
    print(f"\nBest Model: {model_name}")
    print(f"   Primary Metric ({primary_metric}): {metrics.get(primary_metric, 'N/A'):.4f}")
    
    return model_name, metrics


def plot_feature_importance(model, feature_names: list, model_name: str, top_n: int = 15):
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (list): List of feature names
        model_name (str): Name of the model
        top_n (int): Number of top features to display
    """
    try:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 6))
        plt.title(f"Top {top_n} Feature Importance - {model_name}", fontsize=14, fontweight='bold')
        plt.barh(range(top_n), importances[indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel("Importance", fontsize=12)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f"models/feature_importance_{model_name.lower().replace(' ', '_')}.png", 
                   dpi=300, bbox_inches='tight')
        print(f"   Feature importance plot saved: models/feature_importance_{model_name.lower().replace(' ', '_')}.png")
        plt.close()
        
        return importances
    except AttributeError:
        print(f"   Warning: Model {model_name} doesn't support feature importance")
        return None

