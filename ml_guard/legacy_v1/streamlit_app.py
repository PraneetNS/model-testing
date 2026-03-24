#!/usr/bin/env python3
"""
Fireflink-Style ML Guard Streamlit Application
Advanced ML Model Testing Platform with NLP Capabilities
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime
import joblib
import plotly.express as px
import plotly.graph_objects as go
import re

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Page configuration
st.set_page_config(
    page_title="ML Guard - Fireflink Style",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Fireflink-like design
st.markdown("""
<style>
    /* Fireflink-inspired color scheme */
    :root {
        --fireflink-orange: #FF6B35;
        --fireflink-blue: #2E3440;
        --fireflink-light: #ECEFF4;
        --fireflink-dark: #1E2026;
        --success: #56B949;
        --warning: #F4A261;
        --error: #E63946;
    }

    /* Main container */
    .main-header {
        background: linear-gradient(135deg, var(--fireflink-orange), var(--fireflink-blue));
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }

    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
    }

    /* Test cards */
    .test-card {
        background: white;
        color: #2c3e50; /* Dark text for readability */
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid var(--fireflink-orange);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .test-card h4 {
        color: #1a202c; /* Darker heading color */
        margin-top: 0;
        margin-bottom: 0.5rem;
    }

    .test-card p {
        color: #4a5568; /* Medium gray for body text */
        margin: 0.25rem 0;
        line-height: 1.4;
    }

    .test-card strong {
        color: #2d3748; /* Darker for strong text */
    }

    .test-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }

    .test-card.pass {
        border-left-color: var(--success);
        background: linear-gradient(135deg, #f8fff8, white);
    }

    .test-card.fail {
        border-left-color: var(--error);
        background: linear-gradient(135deg, #fff8f8, white);
    }

    .test-card.warning {
        border-left-color: var(--warning);
        background: linear-gradient(135deg, #fffef8, white);
    }

    /* Root cause analysis boxes */
    .root-cause-box {
        background: linear-gradient(135deg, #fef3cd, #fef9f0);
        color: #92400e; /* Dark brown text for contrast on yellow background */
        border: 2px solid #f59e0b;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
        border-left: 4px solid #f59e0b;
        box-shadow: 0 2px 4px rgba(245, 158, 11, 0.1);
    }
    .root-cause-box h5 {
        color: #78350f; /* Darker brown for headings */
        margin-top: 0;
        margin-bottom: 12px;
        font-size: 1.1em;
        font-weight: bold;
    }
    .root-cause-box p {
        color: #92400e; /* Consistent dark brown for body text */
        margin: 8px 0;
        line-height: 1.4;
    }
    .root-cause-box strong {
        color: #451a03; /* Very dark brown for strong text */
        font-weight: bold;
    }

    /* Status badges */
    .status-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
    }

    .status-pass {
        background: var(--success);
        color: white;
    }

    .status-fail {
        background: var(--error);
        color: white;
    }

    .status-warning {
        background: var(--warning);
        color: white;
    }

    /* Sidebar styling */
    .sidebar-content {
        background: var(--fireflink-light);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }

    /* NLP input area */
    .nlp-input {
        background: linear-gradient(135deg, #f8f9fa, white);
        border: 2px solid var(--fireflink-orange);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }

    /* Results section */
    .results-section {
        background: white;
        border-radius: 10px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Progress bars */
    .progress-bar {
        height: 8px;
        border-radius: 4px;
        background: #e9ecef;
        margin: 0.5rem 0;
    }

    .progress-fill {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, var(--fireflink-orange), var(--success));
        transition: width 0.3s ease;
    }

    /* Metrics cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 0.5rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: var(--fireflink-orange);
    }

    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        text-transform: uppercase;
        font-weight: 600;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
        animation: fadeIn 0.5s ease-in-out;
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: var(--fireflink-orange);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--fireflink-blue);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'test_results' not in st.session_state:
    st.session_state.test_results = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'datasets_loaded' not in st.session_state:
    st.session_state.datasets_loaded = False
if 'nlp_query' not in st.session_state:
    st.session_state.nlp_query = ""
if 'selected_tests' not in st.session_state:
    st.session_state.selected_tests = []

# Load NLP model (simplified version without spaCy for compatibility)
@st.cache_resource
def load_nlp_model():
    """Load simplified NLP processing for understanding test requests."""
    # Simple keyword-based NLP without spaCy dependency
    return {
        'accuracy': ['accuracy', 'precision', 'recall', 'f1', 'performance', 'metrics', 'score'],
        'data_quality': ['missing', 'duplicate', 'null', 'quality', 'data', 'clean'],
        'bias': ['bias', 'fairness', 'discrimination', 'gender', 'race', 'fair'],
        'drift': ['drift', 'stability', 'psi', 'change', 'distribution', 'stable'],
        'robustness': ['robust', 'stress', 'edge', 'boundary', 'adversarial', 'reliable'],
        'all': ['all', 'comprehensive', 'complete', 'everything', 'full', 'test']
    }

def parse_nlp_query(query, nlp_keywords):
    """Parse natural language query using keyword matching."""
    query_lower = query.lower()

    selected_tests = []

    # Check for each test type
    for test_type, keywords in nlp_keywords.items():
        for keyword in keywords:
            if keyword in query_lower:
                if test_type == 'all':
                    return ['accuracy', 'data_quality', 'bias', 'drift', 'robustness']
                if test_type not in selected_tests:
                    selected_tests.append(test_type)
                break

    # Default to comprehensive test suite if nothing specific found
    if not selected_tests:
        selected_tests = ['accuracy', 'data_quality', 'bias', 'drift']

    return selected_tests

def load_ml_guard_services():
    """Load ML Guard backend services."""
    try:
        import sys
        import os

        # Ensure backend is in the Python path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        backend_path = os.path.join(current_dir, 'backend')

        if not os.path.exists(backend_path):
            st.error(f"Backend directory not found: {backend_path}")
            st.error("Make sure you're running from the ml_guard directory")
            return None, None

        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)

        # Import and setup logging first (optional)
        try:
            from app.core.logging import setup_logging
            setup_logging()
        except Exception:
            pass  # Logging is optional

        # Import the services
        from app.services.test_orchestrator import TestOrchestrator
        from app.services.model_registry import ModelRegistry

        # Create instances
        test_orch = TestOrchestrator()
        model_reg = ModelRegistry()

        return test_orch, model_reg

    except ImportError as e:
        st.error(f"Import Error: {str(e)}")
        st.error("Required packages may be missing.")
        st.error("Try: pip install -r requirements-streamlit.txt")
        return None, None
    except Exception as e:
        st.error(f"Failed to load ML Guard services: {str(e)}")
        return None, None

def create_test_suite_from_selection(selected_tests):
    """Create comprehensive test suite configuration based on selected tests."""
    test_configs = {
        'accuracy': [
            {
                "name": "Accuracy Threshold (Critical)",
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
                "name": "Precision Threshold (High Priority)",
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
                "name": "Recall Threshold (High Priority)",
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
                "name": "F1 Score Threshold (Balanced Metric)",
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
            {
                "name": "ROC-AUC Threshold (Discrimination Ability)",
                "category": "model_performance",
                "type": "roc_auc_threshold",
                "severity": "medium",
                "config": {
                    "threshold": 0.80,
                    "operator": "gte",
                    "dataset": "validation",
                    "target_column": "churn"
                }
            }
        ],
        'data_quality': [
            {
                "name": "Missing Values Check (Data Integrity)",
                "category": "data_quality",
                "type": "missing_values",
                "severity": "high",
                "config": {"threshold": 0.05, "dataset": "validation"}
            },
            {
                "name": "Missing Values Check - Training Data",
                "category": "data_quality",
                "type": "missing_values",
                "severity": "high",
                "config": {"threshold": 0.05, "dataset": "training"}
            },
            {
                "name": "Duplicate Rows Check (Data Uniqueness)",
                "category": "data_quality",
                "type": "duplicate_rows",
                "severity": "medium",
                "config": {"allow_duplicates": False, "dataset": "validation"}
            },
            {
                "name": "Duplicate Rows Check - Training Data",
                "category": "data_quality",
                "type": "duplicate_rows",
                "severity": "medium",
                "config": {"allow_duplicates": False, "dataset": "training"}
            },
            {
                "name": "Class Balance Check (Target Distribution)",
                "category": "data_quality",
                "type": "class_balance",
                "severity": "medium",
                "config": {
                    "max_imbalance_ratio": 5.0,
                    "target_column": "churn",
                    "dataset": "training"
                }
            },
            {
                "name": "Class Balance Check - Validation Data",
                "category": "data_quality",
                "type": "class_balance",
                "severity": "medium",
                "config": {
                    "max_imbalance_ratio": 5.0,
                    "target_column": "churn",
                    "dataset": "validation"
                }
            }
        ],
        'bias': [
            {
                "name": "Gender Bias Check (Fairness Audit)",
                "category": "bias_fairness",
                "type": "disparate_impact",
                "severity": "medium",
                "config": {
                    "protected_attribute": "gender",
                    "threshold": 1.25,
                    "dataset": "validation",
                    "target_column": "churn"
                }
            },
            {
                "name": "Age Group Bias Check (Demographic Fairness)",
                "category": "bias_fairness",
                "type": "disparate_impact",
                "severity": "low",
                "config": {
                    "protected_attribute": "age_group",
                    "threshold": 1.25,
                    "dataset": "validation",
                    "target_column": "churn"
                }
            },
            {
                "name": "Location Bias Check (Geographic Fairness)",
                "category": "bias_fairness",
                "type": "disparate_impact",
                "severity": "low",
                "config": {
                    "protected_attribute": "location",
                    "threshold": 1.25,
                    "dataset": "validation",
                    "target_column": "churn"
                }
            }
        ],
        'drift': [
            {
                "name": "PSI Drift Check (Population Stability)",
                "category": "statistical_stability",
                "type": "psi_drift",
                "severity": "high",
                "config": {"psi_threshold": 0.15}
            },
            {
                "name": "KS Test Drift Check (Distribution Comparison)",
                "category": "statistical_stability",
                "type": "ks_test",
                "severity": "medium",
                "config": {"significance_level": 0.05}
            },
            {
                "name": "Feature Correlation Stability Check",
                "category": "statistical_stability",
                "type": "correlation_stability",
                "severity": "medium",
                "config": {"correlation_threshold": 0.1}
            }
        ],
        'robustness': [
            {
                "name": "Prediction Stability Check (Noise Resistance)",
                "category": "robustness",
                "type": "prediction_stability",
                "severity": "medium",
                "config": {
                    "stability_threshold": 0.95,
                    "noise_level": 0.01,
                    "dataset": "validation"
                }
            },
            {
                "name": "Input Perturbation Sensitivity",
                "category": "robustness",
                "type": "input_perturbation",
                "severity": "low",
                "config": {
                    "perturbation_factor": 0.1,
                    "dataset": "validation"
                }
            }
        ]
    }

    selected_configs = []
    for test_type in selected_tests:
        if test_type in test_configs:
            selected_configs.extend(test_configs[test_type])

    return {
        "name": f"Custom Test Suite - {', '.join(selected_tests).title()}",
        "description": f"Custom ML model validation suite for {', '.join(selected_tests)}",
        "tests": selected_configs
    }

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

async def run_ml_tests(test_orchestrator, test_suite_config, model, datasets):
    """Run ML tests asynchronously."""
    try:
        test_run = await test_orchestrator.run_test_suite(
            test_suite_config=test_suite_config,
            test_suite_name=test_suite_config["name"],
            model_version="v1.0.0",
            project_id="streamlit-demo",
            environment="testing",
            model_artifact=model,
            datasets=datasets
        )
        return test_run
    except Exception as e:
        st.error(f"Test execution failed: {str(e)}")
        return None

def clean_test_message(message):
    """Clean and format test messages for display, removing HTML tags and making human-readable."""
    if not message:
        return "Test completed successfully"

    # Remove HTML tags completely
    import re
    # Remove HTML tags
    message = re.sub(r'<[^>]+>', '', message)
    # Decode HTML entities
    import html
    message = html.unescape(message)

    # Replace common error patterns with cleaner messages
    replacements = {
        "Dataset 'validation' not found": "Validation dataset not provided or not found",
        "Target column 'target' not found": "Target column not found in dataset. Expected column with prediction targets.",
        "Target column 'churn' not found": "Target column 'churn' not found in dataset. Please ensure your dataset contains a column named 'churn' with the target values.",
        "Model prediction failed:": "Model prediction error:",
        "The feature names should match those that were passed during fit": "Feature mismatch between training and prediction data",
        "Feature names unseen at fit time": "Model was not trained on these features",
        "Feature names seen at fit time, yet now missing": "Required features are missing from prediction data",
        "Result:": "",  # Remove "Result:" prefix
        "Execution Time:": "Test took:"  # Make execution time more readable
    }

    for old, new in replacements.items():
        message = message.replace(old, new)

    # Clean up extra whitespace and formatting
    message = message.strip()
    # Remove multiple spaces
    message = re.sub(r'\s+', ' ', message)

    return message

def extract_root_cause(result):
    """Extract root cause analysis from test results."""
    if result.status == "passed":
        return None

    root_causes = {
        "Dataset 'validation' not found": {
            "description": "The validation dataset required for this test was not found or not uploaded.",
            "causes": "Dataset not uploaded, incorrect dataset name, or file loading error.",
            "actions": "Upload the validation dataset or check that the dataset name matches the test configuration."
        },
        "Target column": {
            "description": "The target column specified in the test configuration does not exist in the dataset.",
            "causes": "Incorrect column name, missing target variable, or dataset schema mismatch.",
            "actions": "Check the target column name in your dataset and update the test configuration accordingly."
        },
        "Feature names should match": {
            "description": "The features used for prediction don't match the features the model was trained on.",
            "causes": "Dataset preprocessing mismatch, categorical encoding differences, or feature engineering inconsistencies.",
            "actions": "Ensure consistent preprocessing between training and prediction datasets. Check categorical feature encoding."
        },
        "feature names unseen at fit time": {
            "description": "The model encountered features during prediction that it was never trained on.",
            "causes": "New features in prediction data, feature name changes, or incomplete feature set.",
            "actions": "Use only features that were present during model training. Check feature consistency."
        },
        "feature names seen at fit time, yet now missing": {
            "description": "Required features that the model was trained on are missing from the prediction data.",
            "causes": "Incomplete prediction dataset, feature removal, or data pipeline issues.",
            "actions": "Ensure all features used during training are present in prediction data."
        },
        "Model prediction failed": {
            "description": "The model failed to generate predictions on the provided data.",
            "causes": "Data type mismatches, missing values, or incompatible data format.",
            "actions": "Check data types, handle missing values, and ensure data format matches training data."
        }
    }

    for error_pattern, analysis in root_causes.items():
        if error_pattern.lower() in result.message.lower():
            return analysis

    # Generic root cause for unknown errors
    return {
        "description": "An unexpected error occurred during test execution.",
        "causes": "Code error, data incompatibility, or system issue.",
        "actions": "Check the detailed error message and ensure all prerequisites are met for this test type."
    }

def display_test_results(test_run):
    """Display test results in Fireflink-style format with improved error handling and root cause analysis."""
    if not test_run or not hasattr(test_run, 'results'):
        st.error("No test results available")
        return

    # Overall metrics
    total_tests = len(test_run.results)
    passed_count = sum(1 for r in test_run.results if r.status == 'passed')
    failed_count = sum(1 for r in test_run.results if r.status == 'failed')
    warning_count = sum(1 for r in test_run.results if r.status == 'warning')

    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Tests", total_tests)
    with col2:
        st.metric("Passed", passed_count, delta=f"{passed_count/total_tests*100:.1f}%" if total_tests > 0 else "0%")
    with col3:
        st.metric("Failed", failed_count, delta=f"-{failed_count/total_tests*100:.1f}%" if failed_count > 0 and total_tests > 0 else "0%")
    with col4:
        st.metric("Warnings", warning_count)
    with col5:
        st.metric("Execution Time", ".2f")

    # Progress bar
    progress_percentage = (passed_count / total_tests) * 100 if total_tests > 0 else 0
    st.markdown(f"""
    <div class="progress-bar">
        <div class="progress-fill" style="width: {progress_percentage}%"></div>
    </div>
    """, unsafe_allow_html=True)

    # Group results by category
    from collections import defaultdict
    results_by_category = defaultdict(list)
    for result in test_run.results:
        results_by_category[result.category].append(result)

    # Display results by category
    for category, results in results_by_category.items():
        category_name = category.replace('_', ' ').title()
        passed = sum(1 for r in results if r.status == 'passed')
        total = len(results)

        with st.expander(f"📊 {category_name} Tests ({passed}/{total} passed)", expanded=True):
            for result in results:
                status_class = "pass" if result.status == "passed" else "fail"
                status_icon = "✅" if result.status == "passed" else "❌"

                # Clean and format the message (escape HTML and improve readability)
                clean_message = clean_test_message(result.message)
                root_cause = extract_root_cause(result)

                st.markdown(f"""
                <div class="test-card {status_class} fade-in">
                    <h4>{status_icon} {result.test_name}</h4>
                    <p><strong>Status:</strong> <span class="status-badge status-{result.status}">{result.status.upper()}</span></p>
                    <p><strong>Category:</strong> {result.category.replace('_', ' ').title()}</p>
                    <p><strong>Severity:</strong> {result.severity.title()}</p>
                    {"<p><strong>Threshold:</strong> " + str(result.threshold) + "</p>" if result.threshold else ""}
                    {"<p><strong>Actual Value:</strong> " + str(result.actual_value) + "</p>" if result.actual_value else ""}
                    <p><strong>Result:</strong> {clean_message}</p>
                    <p><small>Execution Time: {result.execution_time_seconds:.3f}s</small></p>
                </div>
                """, unsafe_allow_html=True)

                # Show root cause analysis for failed tests
                if result.status == "failed" and root_cause:
                    with st.expander(f"🔍 Root Cause Analysis: {result.test_name}", expanded=False):
                        st.markdown(f"""
                        <div class="root-cause-box">
                            <h5>⚠️ Issue Details</h5>
                            <p><strong>What happened:</strong> {root_cause['description']}</p>
                            <p><strong>Possible causes:</strong> {root_cause['causes']}</p>
                            <p><strong>Recommended actions:</strong> {root_cause['actions']}</p>
                        </div>
                        """, unsafe_allow_html=True)

    # Quality Gate Decision
    critical_failures = [r for r in test_run.results
                        if r.status == 'failed' and r.severity == 'critical']

    if critical_failures:
        st.error("🚫 **DEPLOYMENT BLOCKED** - Critical test failures detected")
        with st.expander("Critical Failures Details"):
            for failure in critical_failures:
                st.write(f"- **{failure.test_name}**: {failure.message}")
    else:
        st.success("✅ **QUALITY GATE PASSED** - Model approved for deployment")
        st.balloons()

def main():
    """Main Streamlit application."""
    # Header
    st.markdown("""
    <div class="main-header fade-in">
        <h1>🔥 ML Guard</h1>
        <p>Fireflink-Style ML Model Testing Platform with NLP Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)

        st.markdown("## 🎯 Test Management")

        # Model Upload Section
        st.markdown("### 📤 Model Upload")
        uploaded_model = st.file_uploader("Upload ML Model (.pkl)", type=['pkl'])

        if uploaded_model is not None:
            # Save uploaded model temporarily
            model_path = Path("temp_model.pkl")
            with open(model_path, "wb") as f:
                f.write(uploaded_model.getbuffer())
            st.success("Model uploaded successfully!")

            # Load model info
            try:
                model_data = joblib.load(model_path)
                st.session_state.model_loaded = True
                st.info(f"Model Type: {type(model_data.get('model', model_data)).__name__}")
            except Exception as e:
                st.error(f"Failed to load model: {str(e)}")

        # Dataset Upload Section
        st.markdown("### 📊 Dataset Upload")
        uploaded_train = st.file_uploader("Training Data (.csv)", type=['csv'], key='train')
        uploaded_val = st.file_uploader("Validation Data (.csv)", type=['csv'], key='val')
        uploaded_test = st.file_uploader("Test Data (.csv)", type=['csv'], key='test')

        if uploaded_train and uploaded_val and uploaded_test:
            try:
                train_df = pd.read_csv(uploaded_train)
                val_df = pd.read_csv(uploaded_val)
                test_df = pd.read_csv(uploaded_test)

                st.session_state.datasets = {
                    'training': train_df,
                    'validation': val_df,
                    'test': test_df
                }
                st.session_state.datasets_loaded = True
                st.success("Datasets loaded successfully!")
                st.info(f"Training: {len(train_df)} samples, Validation: {len(val_df)} samples, Test: {len(test_df)} samples")
            except Exception as e:
                st.error(f"Failed to load datasets: {str(e)}")

        st.markdown('</div>', unsafe_allow_html=True)

    # Main content area
    tab1, tab2, tab3 = st.tabs(["🎙️ NLP Testing", "🎯 Manual Testing", "📈 Results Dashboard"])

    # NLP Testing Tab
    with tab1:
        st.markdown("## 🧠 Natural Language Testing")
        st.markdown("Describe what tests you want to run in natural language:")

        # NLP Input
        nlp_query = st.text_area(
            "Test Request",
            placeholder="Example: 'Run accuracy tests and check for bias in my model'",
            height=100,
            key="nlp_input"
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🔍 Analyze Request", type="primary", use_container_width=True):
                if nlp_query.strip():
                    nlp = load_nlp_model()
                    selected_tests = parse_nlp_query(nlp_query, nlp)
                    st.session_state.selected_tests = selected_tests

                    st.success(f"✅ Understood! Running tests for: {', '.join(selected_tests).title()}")

                    # Display selected tests
                    st.markdown("### Selected Test Categories:")
                    for test in selected_tests:
                        st.markdown(f"- **{test.title()}** tests")
                else:
                    st.warning("Please enter a test request")

        with col2:
            if st.button("🚀 Execute NLP Tests", type="primary", use_container_width=True):
                if st.session_state.selected_tests and st.session_state.model_loaded and st.session_state.datasets_loaded:
                    with st.spinner("Running ML tests..."):
                        st.info("Loading ML Guard backend services...")
                        test_orchestrator, _ = load_ml_guard_services()
                        if test_orchestrator:
                            st.success("Backend services loaded successfully!")
                            test_suite_config = create_test_suite_from_selection(st.session_state.selected_tests)

                            # Load model and datasets
                            model_data = joblib.load("temp_model.pkl")
                            model = model_data.get('model', model_data)

                            # Prepare datasets (assuming standard preprocessing)
                            datasets = st.session_state.datasets

                            # Run tests
                            test_run = asyncio.run(run_ml_tests(
                                test_orchestrator,
                                test_suite_config,
                                model,
                                datasets
                            ))

                            if test_run:
                                st.session_state.test_results = test_run
                                st.success("Tests completed successfully!")
                                st.rerun()
                else:
                    missing = []
                    if not st.session_state.selected_tests:
                        missing.append("test selection")
                    if not st.session_state.model_loaded:
                        missing.append("model")
                    if not st.session_state.datasets_loaded:
                        missing.append("datasets")
                    st.error(f"Please provide: {', '.join(missing)}")

    # Manual Testing Tab
    with tab2:
        st.markdown("## 🎯 Manual Test Selection")
        st.markdown("Select specific tests to run:")

        # Test selection checkboxes
        test_options = {
            "accuracy": "📈 Model Performance Tests (Accuracy, Precision, Recall)",
            "data_quality": "🔍 Data Quality Tests (Missing values, Duplicates, Balance)",
            "bias": "⚖️ Bias & Fairness Tests (Discrimination detection)",
            "drift": "📊 Statistical Stability Tests (Drift detection)",
            "robustness": "🛡️ Robustness Tests (Edge cases, Stability)"
        }

        selected_manual_tests = []
        for test_key, test_desc in test_options.items():
            if st.checkbox(test_desc, key=f"manual_{test_key}"):
                selected_manual_tests.append(test_key)

        if st.button("🚀 Run Selected Tests", type="primary", use_container_width=True):
            if selected_manual_tests and st.session_state.model_loaded and st.session_state.datasets_loaded:
                with st.spinner("Running selected tests..."):
                    test_orchestrator, _ = load_ml_guard_services()
                    if test_orchestrator:
                        test_suite_config = create_test_suite_from_selection(selected_manual_tests)

                        # Load model and datasets
                        model_data = joblib.load("temp_model.pkl")
                        model = model_data.get('model', model_data)
                        datasets = st.session_state.datasets

                        # Run tests
                        test_run = asyncio.run(run_ml_tests(
                            test_orchestrator,
                            test_suite_config,
                            model,
                            datasets
                        ))

                        if test_run:
                            st.session_state.test_results = test_run
                            st.success("Tests completed successfully!")
                            st.rerun()
            else:
                missing = []
                if not selected_manual_tests:
                    missing.append("test selection")
                if not st.session_state.model_loaded:
                    missing.append("model")
                if not st.session_state.datasets_loaded:
                    missing.append("datasets")
                st.error(f"Please provide: {', '.join(missing)}")

    # Results Dashboard Tab
    with tab3:
        st.markdown("## 📈 Test Results Dashboard")

        if st.session_state.test_results:
            display_test_results(st.session_state.test_results)

            # Export results
            if st.button("📥 Export Results"):
                results_data = {
                    "run_id": st.session_state.test_results.run_id,
                    "total_tests": len(st.session_state.test_results.results),
                    "passed": sum(1 for r in st.session_state.test_results.results if r.status == 'passed'),
                    "failed": sum(1 for r in st.session_state.test_results.results if r.status == 'failed'),
                    "execution_time": st.session_state.test_results.execution_time_seconds,
                    "results": [
                        {
                            "test_name": r.test_name,
                            "category": r.category,
                            "status": r.status,
                            "severity": r.severity,
                            "message": r.message,
                            "execution_time": r.execution_time_seconds
                        }
                        for r in st.session_state.test_results.results
                    ]
                }

                st.download_button(
                    label="Download JSON Results",
                    data=json.dumps(results_data, indent=2, default=str),
                    file_name=f"ml_guard_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        else:
            st.info("No test results available. Run tests in the NLP or Manual testing tabs first.")

if __name__ == "__main__":
    main()