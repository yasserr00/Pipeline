"""
Sample Data Generator

This script generates sample data for testing the ML pipeline.
You can modify this to generate data for your specific use case.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_housing_data(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate sample housing data for regression problem.
    
    Args:
        n_samples (int): Number of samples to generate
        random_state (int): Random seed
    
    Returns:
        pd.DataFrame: Generated housing dataset
    """
    np.random.seed(random_state)
    
    # Generate features
    data = {
        'square_feet': np.random.normal(2000, 500, n_samples).astype(int),
        'bedrooms': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.3, 0.3, 0.1]),
        'bathrooms': np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], n_samples, 
                                     p=[0.1, 0.15, 0.25, 0.2, 0.15, 0.1, 0.05]),
        'age': np.random.exponential(10, n_samples).astype(int),
        'location_score': np.random.uniform(1, 10, n_samples),
        'garage': np.random.choice([0, 1, 2], n_samples, p=[0.2, 0.6, 0.2]),
        'pool': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'school_rating': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.15, 0.2, 0.3, 0.25]),
        'crime_rate': np.random.uniform(0, 100, n_samples),
        'distance_to_city': np.random.uniform(1, 50, n_samples),
        'lot_size': np.random.normal(8000, 2000, n_samples).astype(int),
        'year_built': np.random.randint(1950, 2024, n_samples),
        'energy_efficiency': np.random.choice(['A', 'B', 'C', 'D'], n_samples, p=[0.2, 0.3, 0.3, 0.2]),
        'neighborhood': np.random.choice(['Downtown', 'Suburbs', 'Rural', 'Urban'], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate target variable (price) with some relationship to features
    price = (
        100 * df['square_feet'] +
        20000 * df['bedrooms'] +
        15000 * df['bathrooms'] -
        1000 * df['age'] +
        5000 * df['location_score'] +
        10000 * df['garage'] +
        25000 * df['pool'] +
        8000 * df['school_rating'] -
        200 * df['crime_rate'] -
        500 * df['distance_to_city'] +
        2 * df['lot_size'] +
        np.random.normal(0, 20000, n_samples)  # Add noise
    )
    
    # Ensure positive prices
    df['price'] = np.maximum(price, 50000)
    
    return df


def generate_classification_data(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate sample data for classification problem (loan approval).
    
    Args:
        n_samples (int): Number of samples to generate
        random_state (int): Random seed
    
    Returns:
        pd.DataFrame: Generated classification dataset
    """
    np.random.seed(random_state)
    
    # Generate features
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 20000, n_samples).astype(int),
        'credit_score': np.random.normal(650, 100, n_samples).astype(int),
        'employment_years': np.random.exponential(5, n_samples).astype(int),
        'debt_to_income': np.random.uniform(0.1, 0.8, n_samples),
        'loan_amount': np.random.normal(200000, 100000, n_samples).astype(int),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                     n_samples, p=[0.3, 0.4, 0.2, 0.1]),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], 
                                          n_samples, p=[0.4, 0.5, 0.1]),
    }
    
    df = pd.DataFrame(data)
    
    # Generate target (loan approval) based on features
    approval_score = (
        (df['income'] > 40000).astype(int) * 2 +
        (df['credit_score'] > 650).astype(int) * 3 +
        (df['debt_to_income'] < 0.4).astype(int) * 2 +
        (df['employment_years'] > 2).astype(int) * 1 +
        np.random.choice([-1, 0, 1], n_samples, p=[0.2, 0.6, 0.2])
    )
    
    df['approved'] = (approval_score >= 4).map({True: 'yes', False: 'no'})
    
    return df


if __name__ == "__main__":
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Generate regression data (housing)
    print("Generating housing data (regression)...")
    housing_df = generate_housing_data(n_samples=10000)
    housing_path = data_dir / "housing.csv"
    housing_df.to_csv(housing_path, index=False)
    print(f"Saved {len(housing_df):,} samples to {housing_path}")
    print(f"   Shape: {housing_df.shape}")
    print(f"   Columns: {', '.join(housing_df.columns)}")
    
    # Generate classification data (loan approval)
    print("\nGenerating loan approval data (classification)...")
    loan_df = generate_classification_data(n_samples=10000)
    loan_path = data_dir / "loan_approval.csv"
    loan_df.to_csv(loan_path, index=False)
    print(f"Saved {len(loan_df):,} samples to {loan_path}")
    print(f"   Shape: {loan_df.shape}")
    print(f"   Columns: {', '.join(loan_df.columns)}")
    
    print("\nSample data generation complete!")
    print("\nTo use the data:")
    print("  - For regression: Update CONFIG in train_model.py to use 'data/housing.csv'")
    print("  - For classification: Update CONFIG in train_model.py to use 'data/loan_approval.csv'")

