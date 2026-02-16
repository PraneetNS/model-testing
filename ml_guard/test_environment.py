#!/usr/bin/env python3
"""Test script to verify the environment setup for ML Guard."""

import sys
import os
import subprocess

def test_environment():
    """Test the current environment setup."""
    print("Testing ML Guard Environment Setup")
    print("=" * 50)

    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")
    print()

    # Test core dependencies
    dependencies = [
        'structlog',
        'fastapi',
        'pydantic',
        'streamlit',
        'pandas',
        'numpy',
        'sklearn',
        'joblib'
    ]

    print("Testing Dependencies:")
    failed_deps = []
    for dep in dependencies:
        try:
            module = __import__(dep)
            version = getattr(module, '__version__', 'unknown')
            print(f"  [OK] {dep}: {version}")
        except ImportError as e:
            print(f"  [ERROR] {dep}: NOT FOUND - {e}")
            failed_deps.append(dep)

    if failed_deps:
        print(f"\n[ERROR] Missing dependencies: {', '.join(failed_deps)}")
        print("Run: pip install -r requirements-streamlit.txt")
        return False

    print("\n[OK] All dependencies found!")

    # Test backend imports
    print("\nTesting Backend Imports:")
    try:
        backend_path = os.path.join(os.getcwd(), 'backend')
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)

        from app.services.test_orchestrator import TestOrchestrator
        from app.services.model_registry import ModelRegistry
        print("  [OK] TestOrchestrator imported")
        print("  [OK] ModelRegistry imported")

        # Test instantiation
        test_orch = TestOrchestrator()
        model_reg = ModelRegistry()
        print("  [OK] Services instantiated successfully")

    except Exception as e:
        print(f"  [ERROR] Backend import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n[OK] Backend services working!")

    # Test model and data loading
    print("\nTesting Model & Data Loading:")
    try:
        import joblib
        import pandas as pd

        # Test model loading
        model_path = 'examples/customer_churn/models/churn_model.pkl'
        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
            print("  [OK] Model loaded successfully")
            print(f"    Model type: {type(model_data.get('model', model_data)).__name__}")
        else:
            print(f"  [ERROR] Model file not found: {model_path}")

        # Test data loading
        data_files = [
            'examples/customer_churn/data/train_data.csv',
            'examples/customer_churn/data/validation_data.csv',
            'examples/customer_churn/data/test_data.csv'
        ]

        for data_file in data_files:
            if os.path.exists(data_file):
                df = pd.read_csv(data_file)
                print(f"  [OK] {os.path.basename(data_file)}: {len(df)} rows")
            else:
                print(f"  [ERROR] Data file not found: {data_file}")

    except Exception as e:
        print(f"  [ERROR] Model/Data loading failed: {e}")
        return False

    print("\n[OK] All tests passed!")
    print("\nEnvironment is ready for ML Guard!")
    return True

if __name__ == "__main__":
    success = test_environment()
    sys.exit(0 if success else 1)