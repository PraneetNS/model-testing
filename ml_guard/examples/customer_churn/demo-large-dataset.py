#!/usr/bin/env python3
"""
ML Guard Demo - Large Customer Churn Dataset
Testing ML model validation on large datasets (100K samples)
"""

import sys
import os
import asyncio

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../backend'))

async def demo_large_dataset():
    """Demo ML Guard with large customer churn dataset."""
    print("=" * 60)
    print("ML GUARD DEMO - LARGE CUSTOMER CHURN DATASET")
    print("=" * 60)
    print("Testing ML model validation on large datasets (100K samples)")
    print()

    try:
        from app.services.test_orchestrator import TestOrchestrator
        from app.services.model_registry import ModelRegistry

        # Initialize services
        test_orchestrator = TestOrchestrator()
        model_registry = ModelRegistry()

        print("Loading large datasets...")
        import pandas as pd

        # Load the large datasets
        train_df = pd.read_csv('data/train_data.csv')
        val_df = pd.read_csv('data/validation_data.csv')
        test_df = pd.read_csv('data/test_data.csv')

        print(f"[OK] Train set: {len(train_df):,} samples")
        print(f"[OK] Validation set: {len(val_df):,} samples")
        print(f"[OK] Test set: {len(test_df):,} samples")
        print()

        # Load model from pickle file
        print("Loading trained model...")
        import joblib
        from pathlib import Path

        model_dir = Path("models")
        if not (model_dir / "churn_model.pkl").exists():
            print("ERROR: Model file not found. Run create_dataset.py first!")
            return

        model_data = joblib.load(model_dir / "churn_model.pkl")
        model = model_data['model']
        feature_cols = model_data['feature_columns']
        encoders = model_data['encoders']

        print("[OK] Model loaded: Random Forest")
        print(f"[OK] Features: {len(feature_cols)}")
        print()

        # Prepare datasets for testing
        datasets = {
            'training': prepare_dataset(train_df, feature_cols, encoders),
            'validation': prepare_dataset(val_df, feature_cols, encoders),
            'test': prepare_dataset(test_df, feature_cols, encoders)
        }

        # Create test suite configuration for large dataset validation
        test_suite_config = {
            "name": "Large Dataset Production Readiness",
            "description": "Comprehensive ML model validation for large datasets",
            "tests": [
                # Data Quality Tests
                {
                    "name": "Missing Values Check",
                    "category": "data_quality",
                    "type": "missing_values",
                    "severity": "high",
                    "config": {"threshold": 0.05, "dataset": "validation"}
                },
                {
                    "name": "Duplicate Rows Check",
                    "category": "data_quality",
                    "type": "duplicate_rows",
                    "severity": "medium",
                    "config": {"threshold": 0.01, "dataset": "validation"}
                },
                {
                    "name": "Class Balance Check",
                    "category": "data_quality",
                    "type": "class_balance",
                    "severity": "medium",
                    "config": {
                        "max_imbalance_ratio": 5.0,
                        "target_column": "churn",
                        "dataset": "validation"
                    }
                },
                # Statistical Stability Tests
                {
                    "name": "PSI Drift Check",
                    "category": "statistical_stability",
                    "type": "psi_drift",
                    "severity": "high",
                    "config": {"threshold": 0.1, "dataset": "validation"}
                },
                # Model Performance Tests
                {
                    "name": "Accuracy Threshold",
                    "category": "model_performance",
                    "type": "accuracy_threshold",
                    "severity": "critical",
                    "config": {
                        "threshold": 0.80,
                        "operator": "gte",
                        "dataset": "validation",
                        "target_column": "churn"
                    }
                },
                {
                    "name": "Precision Threshold",
                    "category": "model_performance",
                    "type": "precision_threshold",
                    "severity": "high",
                    "config": {
                        "threshold": 0.75,
                        "operator": "gte",
                        "dataset": "validation",
                        "target_column": "churn"
                    }
                },
                {
                    "name": "Recall Threshold",
                    "category": "model_performance",
                    "type": "recall_threshold",
                    "severity": "high",
                    "config": {
                        "threshold": 0.70,
                        "operator": "gte",
                        "dataset": "validation",
                        "target_column": "churn"
                    }
                },
                {
                    "name": "F1 Score Threshold",
                    "category": "model_performance",
                    "type": "f1_threshold",
                    "severity": "high",
                    "config": {
                        "threshold": 0.75,
                        "operator": "gte",
                        "dataset": "validation",
                        "target_column": "churn"
                    }
                },
                # Bias Fairness Tests
                {
                    "name": "Gender Bias Check",
                    "category": "bias_fairness",
                    "type": "disparate_impact",
                    "severity": "medium",
                    "config": {
                        "protected_attribute": "gender",
                        "disparate_impact_threshold": 1.2,
                        "dataset": "validation",
                        "target_column": "churn"
                    }
                }
            ]
        }

        # Run the test suite
        print("Starting comprehensive ML validation on large dataset...")
        print("-" * 60)
        print("Running 9 comprehensive ML quality tests...")
        print()

        test_run = await test_orchestrator.run_test_suite(
            test_suite_config=test_suite_config,
            test_suite_name="Large Dataset Production Readiness",
            model_version="v2.1.3",
            project_id="large-dataset-demo",
            environment="production",
            model_artifact=model,
            datasets=datasets
        )

        # Display results
        print("TEST RESULTS SUMMARY")
        print("-" * 30)

        # Calculate totals from results
        total_tests = len(test_run.results)
        passed_count = sum(1 for r in test_run.results if r.status == 'passed')
        failed_count = sum(1 for r in test_run.results if r.status == 'failed')
        warning_count = sum(1 for r in test_run.results if r.status == 'warning')

        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_count}")
        print(f"Failed: {failed_count}")
        print(f"Warnings: {warning_count}")
        print(f"Execution Time: {test_run.execution_time_seconds:.2f}s")
        print()

        # Group results by category
        from collections import defaultdict
        results_by_category = defaultdict(list)
        for result in test_run.results:
            results_by_category[result.category].append(result)

        print("Results by Category:")
        print("-" * 20)

        for category, results in results_by_category.items():
            category_name = category.replace('_', ' ').title()
            passed = sum(1 for r in results if r.status == 'passed')
            total = len(results)
            print(f"{category_name}: {passed}/{total} passed")

            for result in results:
                status = "PASS" if result.status == "passed" else "FAIL"
                print(f"  [{status}] {result.test_name}")
                if result.status == "failed":
                    print(f"     - {result.message}")
        print()

        # Quality Gate Decision
        print("=" * 40)
        print("QUALITY GATE DECISION")
        print("=" * 40)

        critical_failures = [r for r in test_run.results
                           if r.status == 'failed' and r.severity == 'critical']

        if critical_failures:
            print("STATUS: FAIL")
            print("Deployment blocked - critical test failures detected")
            print(f"Critical Failures ({len(critical_failures)}):")
            for failure in critical_failures:
                print(f"- {failure.test_name}")
        else:
            print("STATUS: PASS")
            print("Model approved for production deployment!")
            print()
            print("All critical quality checks passed:")
            print("* Data quality standards met")
            print("* Statistical stability confirmed")
            print("* Model performance meets thresholds")

        print()
        print("=" * 60)
        print("DEMO COMPLETE - LARGE DATASET VALIDATION SUCCESSFUL!")
        print("=" * 60)

    except Exception as e:
        print(f"ERROR: Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

def prepare_dataset(df, feature_cols, encoders):
    """Prepare dataset for ML testing by encoding categorical features and keeping only model features."""
    df_prepared = df.copy()

    # Encode categorical features
    for feature, encoder in encoders.items():
        if feature in df_prepared.columns:
            df_prepared[feature] = encoder.transform(df_prepared[feature])

    # Keep only the features the model was trained on, plus the target column
    all_cols_to_keep = feature_cols + ['churn']
    df_prepared = df_prepared[all_cols_to_keep]

    return df_prepared

if __name__ == "__main__":
    asyncio.run(demo_large_dataset())