"""
Feature Preprocessor for Predictions

This module handles preprocessing of features for predictions to match
the exact preprocessing done during training.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class FeaturePreprocessor:
    """Preprocesses features for prediction to match training preprocessing."""
    
    def __init__(self, expected_features: Optional[List[str]] = None):
        """
        Initialize FeaturePreprocessor.
        
        Args:
            expected_features (List[str]): List of feature names the model expects
        """
        self.expected_features = expected_features or []
        self.interaction_features = []
    
    def detect_interaction_features(self, feature_names: List[str]) -> List[tuple]:
        """
        Detect interaction features from feature names.
        
        Args:
            feature_names (List[str]): List of feature names
        
        Returns:
            List[tuple]: List of (feat1, feat2) tuples for interaction features
        """
        interactions = []
        for feat_name in feature_names:
            if '_x_' in feat_name:
                parts = feat_name.split('_x_')
                if len(parts) == 2:
                    interactions.append((parts[0], parts[1]))
        return interactions
    
    def preprocess(self, features: Dict) -> pd.DataFrame:
        """
        Preprocess features to match training format.
        
        Args:
            features (Dict): Dictionary of feature names and values
        
        Returns:
            pd.DataFrame: Preprocessed features DataFrame
        """
        df = pd.DataFrame([features])
        
        # Create interaction features if needed
        if self.expected_features:
            # Detect which interaction features are expected
            for feat_name in self.expected_features:
                if '_x_' in feat_name:
                    parts = feat_name.split('_x_')
                    if len(parts) == 2:
                        feat1, feat2 = parts[0], parts[1]
                        # Check if both base features exist
                        if feat1 in df.columns and feat2 in df.columns:
                            df[feat_name] = df[feat1] * df[feat2]
                        elif feat1 in df.columns or feat2 in df.columns:
                            # One feature missing, set interaction to 0
                            df[feat_name] = 0
                            print(f"Warning: Cannot create '{feat_name}' - missing base features")
        
        # Ensure all expected features are present
        if self.expected_features:
            # Add missing features with default value 0
            for feat in self.expected_features:
                if feat not in df.columns:
                    df[feat] = 0
                    print(f"Warning: Missing feature '{feat}', using default value 0")
            
            # Select only expected features in the correct order
            df = df[self.expected_features]
        
        return df
    
    def get_expected_features_from_model(self, model) -> List[str]:
        """
        Get expected feature names from the model.
        
        Args:
            model: Trained model (should have feature_names_in_ attribute)
        
        Returns:
            List[str]: List of expected feature names
        """
        if hasattr(model, 'feature_names_in_'):
            return list(model.feature_names_in_)
        elif hasattr(model, 'feature_importances_'):
            # For models without feature_names_in_, we need to get from MLflow
            return []
        return []


def preprocess_prediction_features(features: Dict, model, 
                                   expected_features: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Preprocess features for prediction.
    
    Args:
        features (Dict): Input features dictionary
        model: Trained model
        expected_features (List[str]): Expected feature names (optional)
    
    Returns:
        pd.DataFrame: Preprocessed features DataFrame
    """
    # Get expected features from model if not provided
    if not expected_features:
        if hasattr(model, 'feature_names_in_'):
            expected_features = list(model.feature_names_in_)
        else:
            # Fallback: use features provided, but this might cause issues
            expected_features = list(features.keys())
            print("Warning: Model doesn't have feature_names_in_ attribute. Using provided features.")
    
    preprocessor = FeaturePreprocessor(expected_features)
    
    # Preprocess features
    df = preprocessor.preprocess(features)
    
    return df

