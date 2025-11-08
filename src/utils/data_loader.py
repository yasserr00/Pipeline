"""
Data Loading and Validation Module

This module handles loading CSV data, performing basic validation,
and providing data summary information.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        pd.DataFrame: Loaded dataset
    
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is empty or invalid
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    
    if df.empty:
        raise ValueError("The dataset is empty!")
    
    print(f"Data loaded successfully!")
    print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    return df


def validate_data(df: pd.DataFrame, target_column: str) -> bool:
    """
    Validate that the dataset has the required target column.
    
    Args:
        df (pd.DataFrame): The dataset
        target_column (str): Name of the target column
    
    Returns:
        bool: True if validation passes
    
    Raises:
        ValueError: If target column is missing
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset!")
    
    # Check for missing values in target
    missing_target = df[target_column].isna().sum()
    if missing_target > 0:
        print(f"Warning: {missing_target} missing values in target column")
    
    print(f"Data validation passed!")
    return True


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics of the dataset.
    
    Args:
        df (pd.DataFrame): The dataset
    
    Returns:
        dict: Summary statistics
    """
    summary = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2
    }
    
    return summary


def split_features_target(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataset into features and target.
    
    Args:
        df (pd.DataFrame): The dataset
        target_column (str): Name of the target column
    
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features and target
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    print(f"Features: {X.shape[1]} columns")
    print(f"Target: {target_column}")
    
    return X, y

