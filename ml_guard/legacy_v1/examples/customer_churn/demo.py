#!/usr/bin/env python3
"""
ML Guard Demo - Customer Churn Model Validation
Demonstrates FireFlink-style ML model testing
"""

import sys
import os
import json
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../backend'))

from app.services.test_orchestrator import TestOrchestrator
from app.services.model_registry import ModelRegistry
from app.api.endpoints.quality_gate import quality_gate_check
from pydantic import BaseModel

class QualityGateRequest(BaseModel):
    project_id: str = "ecommerce-ml"
    model_version: str = "v2.1.3"
    test_suite: str = "production-readiness"
    environment: str = "staging"

def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")

def print_test_results(results):
    """Print formatted test results."""
    print(f"\n📊 SUMMARY")
    print(f"• Total Tests: {len(results)}")
    print(f"• Passed: {sum(1 for r in results if r.status == 'passed')}")
    print(f"• Failed: {sum(1 for r in results if r.status == 'failed')}")
    print(f"• Warnings: {sum(1 for r in results if r.status == 'warning')}")
    print(f"• Execution Time: {results[0].execution_time_seconds:.1f}s" if results else "N/A")

    failed_tests = [r for r in results if r.status == "failed"]
    if failed_tests:
        print(f"\n❌ FAILED TESTS")
        print(f"{'─'*30}")
        for i, test in enumerate(failed_tests, 1):
            print(f"\n{i}. {test.test_name}")
            print(f"   Status: {test.status.upper()}")
            if test.threshold is not None and test.actual_value is not None:
                print(f"   Expected: {test.threshold}")
                print(f"   Actual: {test.actual_value:.3f}")
            print(f"   Impact: {test.severity.capitalize()}")

async def demonstrate_quality_gate():
    """Demonstrate the quality gate API."""
    print_header("ML GUARD QUALITY GATE DEMO")
    print("FireFlink-style ML Model Testing Platform")
    print("Customer Churn Prediction Model Validation")

    # Initialize services
    test_orchestrator = TestOrchestrator()
    model_registry = ModelRegistry()

    try:
        # Simulate model registration
        print("\n🔧 Setting up demo environment...")

        # For demo purposes, we'll use mock data since we don't have actual model files
        print("✓ Model registry initialized")
        print("✓ Test orchestrator ready")

        # Create test suite configuration
        test_suite_config = {
            "name": "Production Readiness",
            "description": "Comprehensive ML model validation for production deployment",
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
                    "config": {"max_imbalance_ratio": 10.0, "target_column": "churn", "dataset": "training"}
                },

                # Statistical Stability Tests
                {
                    "name": "PSI Drift Check",
                    "category": "statistical_stability",
                    "type": "psi_drift",
                    "severity": "high",
                    "config": {"psi_threshold": 0.1}
                },

                # Model Performance Tests
                {
                    "name": "Accuracy Threshold",
                    "category": "model_performance",
                    "type": "accuracy_threshold",
                    "severity": "critical",
                    "config": {
                        "threshold": 0.85,
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
                        "threshold": 0.80,
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

                # Bias & Fairness Tests
                {
                    "name": "Gender Bias Check",
                    "category": "bias_fairness",
                    "type": "disparate_impact",
                    "severity": "high",
                    "config": {
                        "protected_attribute": "gender",
                        "threshold": 1.2,
                        "dataset": "validation",
                        "target_column": "churn"
                    }
                }
            ]
        }

        print("\n🧪 Executing Production Readiness Test Suite...")
        print("Running 18 ML quality tests...")

        # Execute test suite
        test_run = await test_orchestrator.run_test_suite(
            project_id="ecommerce-ml",
            model_version="v2.1.3",
            test_suite_name="production-readiness",
            environment="staging"
        )

        # Display results
        print_test_results(test_run.results)

        # Demonstrate quality gate decision
        print_header("QUALITY GATE DECISION")

        critical_failures = test_run.critical_failures
        deployment_allowed = len(critical_failures) == 0

        if deployment_allowed:
            print("✅ QUALITY GATE: PASS")
            print("🚀 Deployment allowed - all critical tests passed")
        else:
            print("❌ QUALITY GATE: FAIL")
            print("🛑 Deployment blocked - critical test failures detected")

            print(f"\n🔍 Critical Failures ({len(critical_failures)}):")
            for failure in critical_failures:
                print(f"• {failure.test_name}: {failure.message}")

        # Show recommendations
        if not deployment_allowed:
            print(f"\n💡 Recommendations:")
            recommendations = [
                "Model retraining required - performance below acceptable thresholds",
                "Review training data quality and distribution",
                "Address bias concerns in protected attributes",
                "Consider hyperparameter tuning or architecture changes"
            ]
            for rec in recommendations:
                print(f"• {rec}")

        print_header("DEMO COMPLETE")
        print("ML Guard successfully validated the customer churn model")
        print("using the same patterns as FireFlink software testing.")

        return test_run

    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    """Main entry point for the demo."""
    await demonstrate_quality_gate()

if __name__ == "__main__":
    # Run async demo
    import asyncio
    asyncio.run(main())