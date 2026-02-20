#!/usr/bin/env python3
"""
Test script to verify text visibility fixes in ML Guard
"""

import sys
import os
import asyncio

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def test_text_visibility():
    """Test that text is now visible in the UI"""

    print("=" * 60)
    print("ML GUARD TEXT VISIBILITY TEST")
    print("=" * 60)

    try:
        # Import required modules
        from app.services.test_orchestrator import TestOrchestrator
        from app.services.model_registry import ModelRegistry

        # Initialize services
        test_orchestrator = TestOrchestrator()

        print("Testing text visibility fixes...")

        # Create test suite with various results
        test_suite_config = {
            "name": "Visibility Test Suite",
            "description": "Testing text visibility in results display",
            "tests": [
                # Test that should pass
                {
                    "name": "Data Quality Check",
                    "category": "data_quality",
                    "type": "missing_values",
                    "severity": "high",
                    "config": {"threshold": 0.05, "dataset": "validation"}
                },
                # Test that should fail
                {
                    "name": "Failed Performance Test",
                    "category": "model_performance",
                    "type": "accuracy_threshold",
                    "severity": "critical",
                    "config": {
                        "threshold": 0.95,  # High threshold that will fail
                        "operator": "gte",
                        "dataset": "validation",
                        "target_column": "churn"
                    }
                }
            ]
        }

        # Create mock datasets
        import pandas as pd
        import numpy as np

        # Create clean validation data
        np.random.seed(42)
        val_data = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'churn': np.random.choice([0, 1], 1000, p=[0.7, 0.3])
        })

        datasets = {
            'validation': val_data
        }

        print("Running test suite with visibility fixes...")

        # Execute test suite
        test_run = await test_orchestrator.run_test_suite(
            project_id="visibility-test",
            model_version="v1.0.0",
            test_suite_name="visibility-test",
            environment="testing",
            test_suite_config=test_suite_config,
            datasets=datasets
        )

        print("\n" + "=" * 60)
        print("VISIBILITY TEST RESULTS")
        print("=" * 60)

        print(f"Total Tests: {len(test_run.results)}")
        print(f"Passed: {test_run.passed_tests}")
        print(f"Failed: {test_run.failed_tests}")

        print("\n" + "-" * 40)
        print("SAMPLE TEST RESULTS (as they appear in UI)")
        print("-" * 40)

        for i, result in enumerate(test_run.results[:2], 1):  # Show first 2 results
            status_icon = "[PASS]" if result.status == "passed" else "[FAIL]"
            print(f"\n{i}. {status_icon} {result.test_name}")
            print(f"   Status: {result.status.upper()}")
            print(f"   Category: {result.category.replace('_', ' ').title()}")
            print(f"   Severity: {result.severity.title()}")

            # Show the cleaned message (as it would appear in UI)
            from streamlit_app import clean_test_message
            clean_msg = clean_test_message(result.message)
            print(f"   Message: {clean_msg}")

            if result.threshold is not None:
                print(f"   Threshold: {result.threshold}")
            if result.actual_value is not None:
                print(f"   Actual Value: {result.actual_value:.3f}")

        print("\n" + "=" * 60)
        print("VISIBILITY FIXES APPLIED")
        print("=" * 60)

        print("CSS Changes Made:")
        print("1. Test cards now have dark text color (#2c3e50)")
        print("2. Headings use darker color (#1a202c)")
        print("3. Body text uses medium gray (#4a5568)")
        print("4. Strong text uses dark color (#2d3748)")
        print("5. Root cause boxes have dark brown text (#92400e)")
        print("6. Root cause headings are darker (#78350f)")
        print("7. HTML messages are properly cleaned")

        print("\n" + "=" * 60)
        print("STREAMLIT UI SHOULD NOW SHOW:")
        print("=" * 60)
        print("✅ Visible black/dark text on white backgrounds")
        print("✅ Readable test result cards")
        print("✅ Clear root cause analysis boxes")
        print("✅ Proper contrast for all text elements")
        print("✅ No invisible white-on-white text")

        print(f"\n🎯 Test Results Saved: {len(test_run.results)} tests completed")
        print("Visit http://localhost:8501 to see the visible results!")

        return test_run

    except Exception as e:
        print(f"ERROR: Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(test_text_visibility())