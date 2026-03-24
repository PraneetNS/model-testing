#!/usr/bin/env python3
"""
Create Synthetic Customer Churn Dataset
Generates realistic customer data for ML Guard demonstration
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os
import random

def create_customer_churn_dataset(n_samples=10000):
    """
    Create a synthetic customer churn dataset similar to telecom data.

    Features:
    - Demographics: age, gender, location
    - Service: tenure, contract type, monthly charges, total charges
    - Usage: data usage, call minutes, support tickets
    - Target: churn (1 = churned, 0 = retained)
    """

    np.random.seed(42)
    random.seed(42)

    # Generate customer IDs
    customer_ids = [f"CUST_{i:06d}" for i in range(n_samples)]

    # Demographics
    ages = np.random.normal(40, 12, n_samples).clip(18, 80).astype(int)
    genders = np.random.choice(['Male', 'Female'], n_samples, p=[0.52, 0.48])
    locations = np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples, p=[0.6, 0.3, 0.1])

    # Service information
    tenure_months = np.random.exponential(24, n_samples).clip(1, 72).astype(int)
    contract_types = np.random.choice(['Month-to-month', 'One year', 'Two year'],
                                    n_samples, p=[0.55, 0.25, 0.20])

    # Usage patterns
    monthly_charges = np.random.normal(65, 25, n_samples).clip(20, 150)
    total_charges = monthly_charges * tenure_months * np.random.uniform(0.8, 1.2, n_samples)

    # Service usage
    data_usage_gb = np.random.exponential(50, n_samples).clip(0, 500)
    call_minutes = np.random.exponential(300, n_samples).clip(0, 2000)
    support_tickets = np.random.poisson(1.5, n_samples).clip(0, 10)

    # Additional features
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.4, 0.2])
    streaming_tv = np.random.choice(['Yes', 'No'], n_samples, p=[0.4, 0.6])
    streaming_movies = np.random.choice(['Yes', 'No'], n_samples, p=[0.35, 0.65])
    payment_method = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
                                    n_samples, p=[0.4, 0.2, 0.2, 0.2])

    # Create churn probability based on features
    churn_prob = np.zeros(n_samples)

    # Higher churn for month-to-month contracts
    churn_prob += (contract_types == 'Month-to-month') * 0.3

    # Higher churn for higher monthly charges
    churn_prob += (monthly_charges > 80) * 0.2

    # Higher churn for younger customers
    churn_prob += (ages < 30) * 0.15

    # Higher churn for many support tickets
    churn_prob += (support_tickets > 3) * 0.25

    # Lower churn for longer tenure
    churn_prob -= (tenure_months > 36) * 0.2

    # Lower churn for two-year contracts
    churn_prob -= (contract_types == 'Two year') * 0.25

    # Add some randomness
    churn_prob += np.random.normal(0, 0.1, n_samples)

    # Convert to binary churn
    churn_prob = 1 / (1 + np.exp(-churn_prob))  # sigmoid
    churn = (churn_prob > np.random.uniform(0, 1, n_samples)).astype(int)

    # Ensure some balance
    target_churn_rate = 0.25
    current_rate = churn.mean()
    if current_rate < target_churn_rate:
        # Increase churn rate
        additional_churn = np.random.choice(n_samples,
                                          int((target_churn_rate - current_rate) * n_samples),
                                          replace=False)
        churn[additional_churn] = 1
    elif current_rate > target_churn_rate:
        # Decrease churn rate
        retain_indices = np.where(churn == 0)[0]
        churn_to_retain = np.random.choice(retain_indices,
                                         int((current_rate - target_churn_rate) * n_samples),
                                         replace=False)
        churn[churn_to_retain] = 1

    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'age': ages,
        'gender': genders,
        'location': locations,
        'tenure_months': tenure_months,
        'contract_type': contract_types,
        'monthly_charges': monthly_charges.round(2),
        'total_charges': total_charges.round(2),
        'data_usage_gb': data_usage_gb.round(1),
        'call_minutes': call_minutes.round(0),
        'support_tickets': support_tickets,
        'internet_service': internet_service,
        'streaming_tv': streaming_tv,
        'streaming_movies': streaming_movies,
        'payment_method': payment_method,
        'churn': churn
    })

    return df

def train_churn_model(df):
    """
    Train a Random Forest model on the churn dataset.
    Returns the trained model and feature information.
    """

    # Prepare features
    df_model = df.copy()

    # Encode categorical features
    categorical_features = ['gender', 'location', 'contract_type', 'internet_service',
                          'streaming_tv', 'streaming_movies', 'payment_method']

    encoders = {}
    for feature in categorical_features:
        encoder = LabelEncoder()
        df_model[feature] = encoder.fit_transform(df_model[feature])
        encoders[feature] = encoder

    # Features and target
    feature_cols = [col for col in df_model.columns if col not in ['customer_id', 'churn']]
    X = df_model[feature_cols]
    y = df_model['churn']

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )

    model.fit(X, y)

    return model, feature_cols, encoders

def save_datasets_and_model():
    """
    Create datasets, train model, and save everything for ML Guard demo.
    """

    print("Creating synthetic customer churn dataset...")

    # Create dataset
    df = create_customer_churn_dataset(n_samples=100000)

    print(f"Dataset created: {len(df)} samples")
    print(f"Churn rate: {df['churn'].mean():.1%}")
    print(f"Features: {len(df.columns) - 2}")  # Excluding customer_id and churn

    # Split into train/validation/test
    train_val, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['churn'])
    train_df, val_df = train_test_split(train_val, test_size=0.25, random_state=42, stratify=train_val['churn'])

    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")

    # Train model
    print("\nTraining Random Forest model...")
    model, feature_cols, encoders = train_churn_model(train_df)

    # Evaluate on validation set
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    val_encoded = val_df.copy()
    for feature, encoder in encoders.items():
        val_encoded[feature] = encoder.transform(val_encoded[feature])

    X_val = val_encoded[feature_cols]
    y_val_pred = model.predict(X_val)
    y_val_true = val_encoded['churn']

    print("Model Performance on Validation Set:")
    print(f"Accuracy: {accuracy_score(y_val_true, y_val_pred):.3f}")
    print(f"Precision: {precision_score(y_val_true, y_val_pred):.3f}")
    print(f"Recall: {recall_score(y_val_true, y_val_pred):.3f}")
    print(f"F1 Score: {f1_score(y_val_true, y_val_pred):.3f}")

    # Save datasets
    os.makedirs('data', exist_ok=True)

    train_df.to_csv('data/train_data.csv', index=False)
    val_df.to_csv('data/validation_data.csv', index=False)
    test_df.to_csv('data/test_data.csv', index=False)

    print("\nDatasets saved to 'data/' directory")

    # Save model
    import joblib
    os.makedirs('models', exist_ok=True)

    model_data = {
        'model': model,
        'feature_columns': feature_cols,
        'encoders': encoders,
        'training_info': {
            'n_samples': len(train_df),
            'churn_rate': train_df['churn'].mean(),
            'features': feature_cols
        }
    }

    joblib.dump(model_data, 'models/churn_model.pkl')
    print("Model saved to 'models/churn_model.pkl'")

    # Create model metadata
    metadata = {
        'model_version': 'v2.1.3',
        'framework': 'scikit-learn',
        'algorithm': 'Random Forest',
        'training_date': pd.Timestamp.now().isoformat(),
        'performance': {
            'accuracy': float(accuracy_score(y_val_true, y_val_pred)),
            'precision': float(precision_score(y_val_true, y_val_pred)),
            'recall': float(recall_score(y_val_true, y_val_pred)),
            'f1_score': float(f1_score(y_val_true, y_val_pred))
        },
        'features': feature_cols,
        'target': 'churn',
        'description': 'Customer churn prediction model for telecom company'
    }

    import json
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("Model metadata saved to 'models/model_metadata.json'")

    return df, model, feature_cols, encoders

if __name__ == "__main__":
    # Create and save everything
    df, model, features, encoders = save_datasets_and_model()

    print("\n" + "="*50)
    print("DATASET CREATION COMPLETE")
    print("="*50)
    print(f"Total customers: {len(df):,}")
    print(f"Churn rate: {df['churn'].mean():.1%}")
    print(f"Features: {len(df.columns) - 2}")
    print("\nReady for ML Guard demonstration!")
    print("Run: python demo-with-data.py")
    print("="*50)