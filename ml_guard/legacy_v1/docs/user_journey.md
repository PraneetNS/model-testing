# ML Guard User Journey

## Visual User Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    FIREFLINK DASHBOARD                          │
├─────────────────────────────────────────────────────────────────┤
│  Project: "E-commerce ML Pipeline"                              │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │ Web Tests   │  │ API Tests   │  │ DB Tests    │  │ ML Tests    │ │
│  │ 15 suites   │  │ 8 suites    │  │ 5 suites    │  │ 4 suites    │ │ ← Click Here
│  │ ✓ All Pass  │  │ ✓ All Pass  │  │ ⚠ 2 Failed  │  │ ? Not Run   │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Step-by-Step User Experience

### Step 1: Access ML Guard from Project Dashboard

**User Action:** Click "ML Tests" in project navigation
**Experience:** Seamless transition to ML testing interface using familiar FireFlink patterns

```
ML Guard Dashboard - E-commerce ML Pipeline
═══════════════════════════════════════════════════════════════

┌─ Models ──────────────────────────────────────┬─ Test Suites ─┐
│                                               │               │
│  🆕 Add Model                                 │  🆕 Create Suite│
│                                               │               │
│  No models added yet.                         │  Suggested:    │
│  Upload your first ML model to get started.   │  • Production  │
│                                               │    Readiness   │
└───────────────────────────────────────────────┴───────────────┘

┌─ Recent Test Runs ──────────────────────────────┬─ Quality Gate ─┐
│                                                 │               │
│  No test runs yet.                              │  Status:      │
│  Run your first ML quality check.               │  Not Configured│
│                                                 │               │
└─────────────────────────────────────────────────┴───────────────┘
```

### Step 2: Add Your First Model

**User Action:** Click "Add Model" button
**Experience:** Guided upload process (no ML expertise required)

```
Add ML Model
═════════════

Step 1: Model Details
─────────────────────
Model Name: Customer Churn Predictor v2.1
Description: Predicts customer churn probability
Model Type: Binary Classification

Step 2: Upload Artifacts
─────────────────────
📁 Model File: [Choose File] churn_model.pkl
📊 Training Data: [Choose File] train_data.csv
📊 Validation Data: [Choose File] val_data.csv
📊 Test Data: [Optional] test_data.csv

Step 3: Feature Schema (Auto-detected)
────────────────────────────────────
✓ 12 features detected
✓ Target: churn (binary)
✓ Protected attributes: gender, age_group

[Analyze & Generate Tests]
```

### Step 3: Auto-Generated Test Suggestions

**System Response:** Intelligent test generation based on data analysis

```
🎯 Auto-Generated Test Suite: "Production Readiness"
══════════════════════════════════════════════════════

Based on your data, we recommend these tests:

┌─ Data Quality Tests ──────────────────────────┐
│ ✅ Schema validation                         │
│ ✅ Missing values < 5%                        │
│ ✅ No duplicate rows                          │
│ ✅ Class balance ratio < 1:10                 │
│ ✅ Feature ranges within expected bounds      │
└───────────────────────────────────────────────┘

┌─ Statistical Stability Tests ─────────────────┐
│ ✅ No significant drift vs training data      │
│ ✅ Feature correlations stable                │
│ ✅ Distribution shifts detected (PSI < 0.1)   │
└─────────────────────────────────────────────────────────────────┘

┌─ Model Performance Tests ─────────────────────┐
│ ✅ Accuracy > 0.85                            │
│ ✅ Precision > 0.80                           │
│ ✅ Recall > 0.75                              │
│ ✅ ROC-AUC > 0.90                             │
└───────────────────────────────────────────────┘

┌─ Bias & Fairness Tests ───────────────────────┐
│ ✅ No bias on gender (disparate impact < 1.2) │
│ ✅ No bias on age_group                       │
│ ✅ Equal opportunity difference < 0.05        │
└───────────────────────────────────────────────┘

[Customize Tests] [Run All Tests]
```

### Step 4: Customize Tests (Optional)

**User Action:** Modify suggested tests or add custom ones
**Experience:** Scriptless rule definition using natural language

```
Customize Test: "Model Accuracy Threshold"
════════════════════════════════════════════

Current Rule: "Ensure model accuracy is above 85%"

Edit Rule:
─────────
Ensure model accuracy is above [85]% on [validation] dataset

Advanced Options:
• Test on: Validation, Test, or Both datasets
• Threshold: 85% (Warning: 80%, Critical: 75%)
• Comparison: Greater than, Greater than or equal

[Save Test] [Add Another Test]
```

### Step 5: Run Tests

**User Action:** Click "Run All Tests"
**Experience:** Familiar FireFlink test execution with real-time progress

```
Test Execution - Production Readiness Suite
═════════════════════════════════════════════

Running 18 ML Tests...
───────────────────────

🟡 Data Quality Tests (4/4)
  ✅ Schema validation                    0.2s
  ✅ Missing values check                 0.8s
  ✅ Duplicate detection                  0.3s
  ✅ Class balance check                  0.5s

🟡 Statistical Stability Tests (3/3)
  ✅ PSI drift check                      2.1s
  🔄 KS test for distributions...         1.8s
  ⏳ Feature correlation analysis...      3.2s

🟡 Model Performance Tests (4/4)
  ⏳ Accuracy calculation...              5.1s
  ⏳ Precision/Recall computation...      4.8s

🟡 Bias & Fairness Tests (3/3)
  ⏳ Gender bias analysis...              6.2s

[View Live Results] [Stop Execution]
```

### Step 6: Review Results

**Experience:** FireFlink-style results with ML-specific insights

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
   Status: FAIL
   Expected: > 85%
   Actual: 82.3%
   Impact: Critical
   Recommendation: Model retraining required

2. Gender Bias Detection
   Status: FAIL
   Metric: Disparate Impact
   Gender Ratio: 1.35 (threshold: < 1.2)
   Impact: High
   Recommendation: Review training data balance

✅ PASSED TESTS (14 more...)
⚠️  WARNINGS (0)

[Generate Report] [Re-run Failed Tests] [Export Results]
```

### Step 7: Detailed Failure Analysis

**User Action:** Click on failed test for details
**Experience:** Root-cause analysis with actionable insights

```
Failure Analysis: Model Accuracy Threshold
════════════════════════════════════════════

📈 Performance Breakdown
────────────────────────
Accuracy by Feature Slice:
• High spending customers: 89.2% ✓
• Medium spending customers: 84.1% ✓
• Low spending customers: 76.5% ❌ ← Major issue

📊 Confusion Matrix
──────────────────
Predicted →   No Churn    Churn
Actual ↓
No Churn      1,245       89
Churn          156        234

🔍 Root Cause Analysis
─────────────────────
• Primary issue: Poor performance on low-spending customers
• Contributing factor: Under-represented in training data (12% vs 28% in validation)
• Suggestion: Collect more low-spending customer data or use class weighting

💡 Recommendations
─────────────────
1. Retrain with balanced dataset
2. Implement class weighting (churn class weight: 2.5x)
3. Add synthetic data generation for minority class
4. Consider ensemble methods for better low-spending prediction
```

### Step 8: CI/CD Integration

**Experience:** Quality gate prevents deployment of failing models

```
GitHub Actions Integration
══════════════════════════

name: ML Model Deployment
on:
  push:
    branches: [main]

jobs:
  ml-quality-gate:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout
      uses: actions/checkout@v3

    - name: ML Quality Gate
      run: |
        curl -X POST ${{ secrets.ML_GUARD_URL }}/api/v1/ml-quality-gate \
          -H "Authorization: Bearer ${{ secrets.ML_GUARD_TOKEN }}" \
          -H "Content-Type: application/json" \
          -d '{
            "project_id": "ecommerce-ml",
            "model_version": "${{ github.sha }}",
            "test_suite": "production-readiness"
          }'
```

## Error States & Recovery

### Model Upload Issues

```
❌ Upload Failed: Invalid Model Format
───────────────────────────────────────
Error: Model file 'churn_model.h5' is not supported.
Supported formats: scikit-learn (.pkl), XGBoost (.json/.pkl), PyTorch (.pt)

💡 Solutions:
• Convert to supported format
• Use joblib.dump() for scikit-learn models
• Save XGBoost as JSON format
```

### Test Execution Errors

```
❌ Test Failed: Data Schema Mismatch
────────────────────────────────────
Test: Schema validation
Error: Feature 'new_feature' not in training schema

🔧 Quick Fix:
• Update feature schema in model configuration
• Remove extra features from validation data
• Add missing features to validation data
```

## Advanced Features

### Custom Test Creation

**User Action:** Click "Create Custom Test"
**Experience:** Scriptless test definition with guided UI

```
Create Custom ML Test
══════════════════════

Test Category: [Performance ▼]
Test Name: Custom Accuracy Check

Rule Definition:
───────────────
Ensure [accuracy ▼] is [above ▼] [90]% on [test ▼] dataset

Advanced Configuration:
• Metric: accuracy, precision, recall, f1, roc_auc
• Operator: above, below, between
• Dataset: training, validation, test, all
• Thresholds: Primary, Warning, Critical

[Add Condition] [Test Rule] [Save Test]
```

This user journey ensures ML Guard feels like a natural extension of FireFlink, requiring no ML expertise while providing comprehensive model validation capabilities.