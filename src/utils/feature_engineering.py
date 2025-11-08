"""
Feature Engineering Module

This module handles feature creation, transformation, and preprocessing.
It includes functions for creating meaningful features and preparing data for modeling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import List, Tuple, Optional


def create_interaction_features(df: pd.DataFrame, feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Create interaction features by multiplying pairs of features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        feature_pairs (List[Tuple[str, str]]): List of (feature1, feature2) tuples
    
    Returns:
        pd.DataFrame: Dataframe with new interaction features
    """
    df_new = df.copy()
    
    for feat1, feat2 in feature_pairs:
        if feat1 in df.columns and feat2 in df.columns:
            new_feat_name = f"{feat1}_x_{feat2}"
            df_new[new_feat_name] = df[feat1] * df[feat2]
            print(f"   Created interaction feature: {new_feat_name}")
    
    return df_new


def create_polynomial_features(df: pd.DataFrame, features: List[str], degree: int = 2) -> pd.DataFrame:
    """
    Create polynomial features for specified columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        features (List[str]): List of feature names to create polynomials for
        degree (int): Degree of polynomial (default: 2)
    
    Returns:
        pd.DataFrame: Dataframe with polynomial features
    """
    df_new = df.copy()
    
    for feat in features:
        if feat in df.columns and df[feat].dtype in [np.number]:
            for d in range(2, degree + 1):
                new_feat_name = f"{feat}_pow{d}"
                df_new[new_feat_name] = df[feat] ** d
                print(f"   Created polynomial feature: {new_feat_name}")
    
    return df_new


def handle_missing_values(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    
    Returns:
        pd.DataFrame: Dataframe with handled missing values
    """
    df_new = df.copy()
    missing_before = df_new.isnull().sum().sum()
    
    if missing_before == 0:
        print("   No missing values found")
        return df_new
    
    print(f"   Handling {missing_before} missing values using '{strategy}' strategy")
    
    for col in df_new.columns:
        if df_new[col].isnull().sum() > 0:
            if strategy == "mean" and df_new[col].dtype in [np.number]:
                df_new[col].fillna(df_new[col].mean(), inplace=True)
            elif strategy == "median" and df_new[col].dtype in [np.number]:
                df_new[col].fillna(df_new[col].median(), inplace=True)
            elif strategy == "mode":
                df_new[col].fillna(df_new[col].mode()[0], inplace=True)
            elif strategy == "drop":
                df_new.dropna(subset=[col], inplace=True)
    
    missing_after = df_new.isnull().sum().sum()
    print(f"   Missing values handled: {missing_before} -> {missing_after}")
    
    return df_new


def encode_categorical_features(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                               categorical_columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Encode categorical features using Label Encoding.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        categorical_columns (List[str]): List of categorical column names
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Encoded training and test features
    """
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    
    encoders = {}
    
    for col in categorical_columns:
        if col in X_train.columns:
            le = LabelEncoder()
            X_train_encoded[col] = le.fit_transform(X_train[col].astype(str))
            X_test_encoded[col] = le.transform(X_test[col].astype(str))
            encoders[col] = le
            print(f"   Encoded categorical feature: {col}")
    
    return X_train_encoded, X_test_encoded


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                   numeric_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale numeric features using StandardScaler.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        numeric_columns (Optional[List[str]]): List of numeric columns to scale. 
                                              If None, scales all numeric columns.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]: Scaled features and scaler
    """
    if numeric_columns is None:
        numeric_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    scaler = StandardScaler()
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
    X_test_scaled[numeric_columns] = scaler.transform(X_test[numeric_columns])
    
    print(f"   Scaled {len(numeric_columns)} numeric features")
    
    return X_train_scaled, X_test_scaled, scaler

