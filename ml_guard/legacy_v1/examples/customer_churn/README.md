# Customer Churn ML Guard Example

This example demonstrates ML Guard's capabilities using a customer churn prediction model.

## Dataset

The dataset contains customer information for a telecommunications company:
- **Features**: Customer demographics, service usage, billing information
- **Target**: Churn (1 = churned, 0 = retained)
- **Size**: 7,043 samples with 20 features

## Files

- `train_data.csv` - Training dataset
- `validation_data.csv` - Validation dataset
- `test_data.csv` - Test dataset
- `churn_model.pkl` - Trained scikit-learn model
- `demo.py` - Demonstration script

## Running the Example

```bash
# Install dependencies
pip install -r ../../backend/requirements.txt

# Run the demo
cd examples/customer_churn
python demo.py
```

## Expected Output

The demo will:
1. Load the model and datasets
2. Execute the production readiness test suite
3. Show test results with pass/fail status
4. Demonstrate the quality gate API

## Sample Test Results

```
Test Results - Production Readiness Suite
══════════════════════════════════════════

📊 SUMMARY
• Total Tests: 18
• Passed: 16
• Failed: 2
• Warnings: 0
• Execution Time: 28.4s

❌ FAILED TESTS
──────────────

1. Model Accuracy Threshold
   Expected: > 85%
   Actual: 82.3%
   Impact: Critical

2. Gender Bias Detection
   Disparate Impact: 1.35 (threshold: < 1.2)
   Impact: High
```

## Quality Gate Response

```json
{
  "status": "FAIL",
  "deployment_allowed": false,
  "run_id": "ml-run-20240115-001",
  "failures": [
    {
      "test_name": "Model Accuracy Threshold",
      "category": "model_performance",
      "severity": "critical",
      "message": "Accuracy 0.82 below threshold 0.85"
    }
  ],
  "recommendations": [
    "Model retraining required - performance below acceptable thresholds",
    "Review training data quality and distribution"
  ]
}
```

This demonstrates how ML Guard provides comprehensive pre-deployment validation while maintaining the same user experience as FireFlink's software testing platform.