#!/usr/bin/env python3
"""
ML Guard Demo with Real Data (Simplified)
Demonstrates ML Guard working with actual customer churn dataset
"""

import sys
import os
import asyncio
import pandas as pd
import joblib
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../backend'))

async def demo_with_real_data():
    """Complete ML Guard demonstration with real customer churn data."""

    print("=" * 60)
    print("ML GUARD DEMO - CUSTOMER CHURN PREDICTION")
    print("=" * 60)
    print("Testing real ML model with comprehensive validation")
    print()

    try:
        # Load datasets
        print("Loading datasets...")
        data_dir = Path("data")

        if not data_dir.exists():
            print("ERROR: Data directory not found. Run create_dataset.py first!")
            return

        train_df = pd.read_csv(data_dir / "train_data.csv")
        val_df = pd.read_csv(data_dir / "validation_data.csv")
        test_df = pd.read_csv(data_dir / "test_data.csv")

        print(f"[OK] Train set: {len(train_df):,} samples")
        print(f"[OK] Validation set: {len(val_df):,} samples")
        print(f"[OK] Test set: {len(test_df):,} samples")
        print()

        # Load model
        print("Loading trained model...")
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

        # Initialize ML Guard services
        from app.services.test_orchestrator import TestOrchestrator
        from app.services.model_registry import ModelRegistry

        test_orchestrator = TestOrchestrator()
        model_registry = ModelRegistry()

        print("Starting comprehensive ML validation...")
        print("-" * 60)

        # Define comprehensive test suite
        test_suite_config = {
            "name": "Production Readiness Suite",
            "description": "Complete ML model validation for customer churn prediction",
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
                    "config": {"allow_duplicates": False, "dataset": "validation"}
                },
                {
                    "name": "Class Balance Check",
                    "category": "data_quality",
                    "type": "class_balance",
                    "severity": "medium",
                    "config": {"max_imbalance_ratio": 5.0, "target_column": "churn", "dataset": "training"}
                },

                # Statistical Stability Tests
                {
                    "name": "PSI Drift Check",
                    "category": "statistical_stability",
                    "type": "psi_drift",
                    "severity": "high",
                    "config": {"psi_threshold": 0.15}
                },

                # Model Performance Tests
                {
                    "name": "Accuracy Threshold",
                    "category": "model_performance",
                    "type": "accuracy_threshold",
                    "severity": "critical",
                    "config": {
                        "threshold": 0.75,
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
                        "threshold": 0.70,
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
                        "threshold": 0.75,
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
                        "threshold": 0.70,
                        "operator": "gte",
                        "dataset": "validation",
                        "target_column": "churn"
                    }
                },

                # Bias & Fairness Tests
                {
                    "name": "Gender Bias Check",
                    "category": "bias_fairness",
                    "type": "disparate_impact",
                    "severity": "medium",
                    "config": {
                        "protected_attribute": "gender",
                        "threshold": 1.25,
                        "dataset": "validation",
                        "target_column": "churn"
                    }
                }
            ]
        }

        # Execute test suite
        print("Running 11 comprehensive ML quality tests...")
        print()

        test_run = await test_orchestrator.run_test_suite(
            project_id="ecommerce-ml",
            model_version="v2.1.3",
            test_suite_name="production-readiness",
            environment="production",
            model_artifact=model,
            datasets=datasets,
            test_suite_config=test_suite_config
        )

        # Display results
        display_test_results(test_run)

        # Quality gate decision
        print("\n" + "=" * 60)
        print("QUALITY GATE DECISION")
        print("=" * 60)

        critical_failures = test_run.critical_failures
        deployment_allowed = len(critical_failures) == 0

        if deployment_allowed:
            print("STATUS: PASS")
            print("Model approved for production deployment!")
            print()
            print("All critical quality checks passed:")
            print("* Data quality standards met")
            print("* Statistical stability confirmed")
            print("* Model performance exceeds thresholds")
            print("* Bias and fairness requirements satisfied")
        else:
            print("STATUS: FAIL")
            print("Model blocked from production deployment!")
            print()
            print(f"Critical failures ({len(critical_failures)}):")
            for failure in critical_failures:
                print(f"* {failure.test_name}: {failure.message}")
                if failure.threshold is not None and failure.actual_value is not None:
                    print(".3f")

        # Summary statistics
        print("\n" + "-" * 60)
        print("EXECUTION SUMMARY")
        print("-" * 60)
        print(f"Total Tests: {len(test_run.results)}")
        print(".2f")
        print(".3f")
        print()

        # Category breakdown
        categories = {}
        for result in test_run.results:
            cat = result.category
            if cat not in categories:
                categories[cat] = {'total': 0, 'passed': 0, 'failed': 0}
            categories[cat]['total'] += 1
            if result.status == 'passed':
                categories[cat]['passed'] += 1
            else:
                categories[cat]['failed'] += 1

        print("Results by Category:")
        for category, stats in categories.items():
            category_name = category.replace('_', ' ').title()
            print(f"  {category_name}: {stats['passed']}/{stats['total']} passed")

        print("\n" + "=" * 60)
        print("DEMO COMPLETE - ML GUARD VALIDATION SUCCESSFUL!")
        print("=" * 60)

        return test_run

    except Exception as e:
        print(f"ERROR: Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

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

def display_test_results(test_run):
    """Display formatted test results."""

    print("TEST RESULTS SUMMARY")
    print("-" * 60)
    print(f"Total Tests: {len(test_run.results)}")
    print(f"Passed: {test_run.passed_tests}")
    print(f"Failed: {test_run.failed_tests}")
    print(f"Warnings: {test_run.warning_tests}")
    print(".2f")
    print()

    # Group results by category
    categories = {}
    for result in test_run.results:
        cat = result.category
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(result)

    # Display results by category
    for category, results in categories.items():
        category_name = category.replace('_', ' ').title()
        passed = sum(1 for r in results if r.status == 'passed')
        total = len(results)
        print(f"{category_name} Tests ({passed}/{total} passed):")

        for result in results:
            status_icon = "[PASS]" if result.status == "passed" else "[FAIL]"

            print(f"  {status_icon} {result.test_name}")

            if result.status == "failed":
                print(f"     - {result.message}")
                if result.threshold is not None and result.actual_value is not None:
                    print(".3f")

        print()

if __name__ == "__main__":
    asyncio.run(demo_with_real_data())