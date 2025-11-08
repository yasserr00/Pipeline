"""
Flask API Routes Controller

This module contains all Flask route handlers for the ML model serving API.
"""

import os
import sys
from flask import Blueprint, request, jsonify, render_template

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.utils.feature_preprocessor import preprocess_prediction_features


def create_routes(app, model, model_name, problem_type):
    """
    Create and register all Flask routes.
    
    Args:
        app: Flask application instance
        model: Loaded ML model
        model_name: Name of the model
        problem_type: Type of problem (regression/classification)
    """
    import pandas as pd
    
    @app.route('/')
    def index():
        """Serve the frontend interface."""
        return render_template('index.html')

    @app.route('/health', methods=['GET', 'OPTIONS'])
    def health_check():
        """Health check endpoint."""
        # Handle CORS preflight
        if request.method == 'OPTIONS':
            response = jsonify({})
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
            response.headers.add('Access-Control-Allow-Methods', 'GET')
            return response
        
        if model is None:
            response = jsonify({
                "status": "unhealthy",
                "error": "Model not loaded"
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 503
        
        response = jsonify({
            "status": "healthy",
            "model": model_name,
            "problem_type": problem_type
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    @app.route('/predict', methods=['POST', 'OPTIONS'])
    def predict():
        """
        Prediction endpoint.
        
        Expects JSON with features:
        {
            "features": {
                "feature1": value1,
                "feature2": value2,
                ...
            }
        }
        """
        # Handle CORS preflight
        if request.method == 'OPTIONS':
            response = jsonify({})
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
            response.headers.add('Access-Control-Allow-Methods', 'POST')
            return response
        
        if model is None:
            response = jsonify({"error": "Model not loaded. Please restart the server."})
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 500
        
        try:
            if not request.is_json:
                response = jsonify({"error": "Request must be JSON"})
                response.headers.add('Access-Control-Allow-Origin', '*')
                return response, 400
            
            data = request.json
            
            if 'features' not in data:
                response = jsonify({"error": "Missing 'features' key in request"})
                response.headers.add('Access-Control-Allow-Origin', '*')
                return response, 400
            
            features = data['features']
            
            # Preprocess features to match training format
            # Get expected features from model
            expected_features = None
            if hasattr(model, 'feature_names_in_'):
                expected_features = list(model.feature_names_in_)
            
            df = preprocess_prediction_features(features, model, expected_features)
            
            # Make prediction
            prediction = model.predict(df)[0]
            
            # For classification, get probabilities if available
            result = {"prediction": float(prediction) if problem_type == "regression" else str(prediction)}
            
            if problem_type == "classification" and hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(df)[0]
                classes = model.classes_
                result["probabilities"] = {
                    str(cls): float(prob) for cls, prob in zip(classes, probabilities)
                }
            
            response = jsonify(result)
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        
        except Exception as e:
            import traceback
            error_details = str(e)
            print(f"Prediction error: {error_details}")
            print(traceback.format_exc())
            response = jsonify({"error": error_details})
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 500

    @app.route('/predict_batch', methods=['POST', 'OPTIONS'])
    def predict_batch():
        """
        Batch prediction endpoint.
        
        Expects JSON with list of feature dictionaries:
        {
            "features_list": [
                {"feature1": value1, "feature2": value2, ...},
                {"feature1": value1, "feature2": value2, ...},
                ...
            ]
        }
        """
        # Handle CORS preflight
        if request.method == 'OPTIONS':
            response = jsonify({})
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
            response.headers.add('Access-Control-Allow-Methods', 'POST')
            return response
        
        if model is None:
            response = jsonify({"error": "Model not loaded. Please restart the server."})
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 500
        
        try:
            if not request.is_json:
                response = jsonify({"error": "Request must be JSON"})
                response.headers.add('Access-Control-Allow-Origin', '*')
                return response, 400
            
            data = request.json
            
            if 'features_list' not in data:
                response = jsonify({"error": "Missing 'features_list' key in request"})
                response.headers.add('Access-Control-Allow-Origin', '*')
                return response, 400
            
            features_list = data['features_list']
            
            # Get expected features from model
            expected_features = None
            if hasattr(model, 'feature_names_in_'):
                expected_features = list(model.feature_names_in_)
            
            # Preprocess each feature set
            processed_features = []
            for features in features_list:
                df_single = preprocess_prediction_features(features, model, expected_features)
                processed_features.append(df_single.iloc[0].to_dict())
            
            # Convert to DataFrame
            df = pd.DataFrame(processed_features)
            
            # Ensure correct column order
            if expected_features:
                df = df[expected_features]
            
            # Make predictions
            predictions = model.predict(df)
            
            results = []
            for i, pred in enumerate(predictions):
                result = {
                    "index": i,
                    "prediction": float(pred) if problem_type == "regression" else str(pred)
                }
                
                if problem_type == "classification" and hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(df.iloc[[i]])[0]
                    classes = model.classes_
                    result["probabilities"] = {
                        str(cls): float(prob) for cls, prob in zip(classes, probabilities)
                    }
                
                results.append(result)
            
            response = jsonify({"predictions": results})
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        
        except Exception as e:
            import traceback
            error_details = str(e)
            print(f"Batch prediction error: {error_details}")
            print(traceback.format_exc())
            response = jsonify({"error": error_details})
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 500

