#!/usr/bin/env python3
"""
ML Guard Demo - Customer Churn Model Validation (Simplified)
Demonstrates FireFlink-style ML model testing
"""

import sys
import os
import asyncio

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../backend'))

async def demo():
    """Simple ML Guard demonstration."""
    print("=" * 50)
    print("ML GUARD QUALITY GATE DEMO")
    print("=" * 50)
    print("FireFlink-style ML Model Testing Platform")
    print("Customer Churn Prediction Model Validation")
    print()

    try:
        from app.services.test_orchestrator import TestOrchestrator
        from app.services.model_registry import ModelRegistry

        # Initialize services
        test_orchestrator = TestOrchestrator()
        model_registry = ModelRegistry()

        print("Setting up demo environment...")

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
                }
            ]
        }

        print("Executing Production Readiness Test Suite...")
        print("Running ML quality tests...")
        print()

        # Execute test suite
        test_run = await test_orchestrator.run_test_suite(
            project_id="ecommerce-ml",
            model_version="v2.1.3",
            test_suite_name="production-readiness",
            environment="staging"
        )

        # Display results
        print("TEST RESULTS SUMMARY")
        print("-" * 30)
        print(f"Total Tests: {len(test_run.results)}")
        print(f"Passed: {test_run.passed_tests}")
        print(f"Failed: {test_run.failed_tests}")
        print(f"Warnings: {test_run.warning_tests}")
        print(".1f")
        print()

        failed_tests = [r for r in test_run.results if r.status == "failed"]
        if failed_tests:
            print("FAILED TESTS")
            print("-" * 30)
            for i, test in enumerate(failed_tests, 1):
                print(f"{i}. {test.test_name}")
                print(f"   Status: {test.status.upper()}")
                if test.threshold is not None and test.actual_value is not None:
                    print(".3f")
                print()

        # Quality gate decision
        print("QUALITY GATE DECISION")
        print("-" * 30)

        critical_failures = test_run.critical_failures
        deployment_allowed = len(critical_failures) == 0

        if deployment_allowed:
            print("STATUS: PASS")
            print("Deployment allowed - all critical tests passed")
        else:
            print("STATUS: FAIL")
            print("Deployment blocked - critical test failures detected")

            print(f"Critical Failures ({len(critical_failures)}):")
            for failure in critical_failures:
                print(f"- {failure.test_name}: {failure.message}")

        print()
        print("=" * 50)
        print("DEMO COMPLETE")
        print("ML Guard successfully validated the model")
        print("using FireFlink-style testing patterns.")

        return test_run

    except Exception as e:
        print(f"Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(demo())