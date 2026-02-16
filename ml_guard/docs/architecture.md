# FireFlink ML Guard Architecture

## FireFlink UI Pattern Mapping

### Core FireFlink Concepts → ML Guard Equivalents

| FireFlink Concept | ML Guard Implementation | Description |
|------------------|------------------------|-------------|
| **Project** | ML Guard Project | Container for all ML testing assets |
| **Test Suite** | ML Test Suite | Collection of related ML tests (e.g., "Data Quality Suite", "Model Performance Suite") |
| **Test Case** | ML Test Case | Individual test definition (e.g., "Accuracy > 85%", "Missing Values < 5%") |
| **Test Run** | ML Test Run | Execution of ML tests with results |
| **Test Step** | ML Test Assertion | Individual assertion within a test (e.g., check accuracy, check drift) |
| **Test Data** | ML Datasets | Training, validation, test datasets + model artifacts |
| **Environment** | ML Environment | Model execution context (CPU/GPU, dependencies) |
| **CI/CD Integration** | ML Quality Gate | REST API for deployment blocking/allowing |

### UI Navigation Structure

```
FireFlink Project Dashboard
├── Web Tests
├── API Tests
├── Database Tests
└── ML Model Tests ← NEW: ML Guard
    ├── Models
    │   ├── Model Registry
    │   └── Model Versions
    ├── Test Suites
    │   ├── Data Quality Suite
    │   ├── Statistical Stability Suite
    │   ├── Model Performance Suite
    │   ├── Robustness Suite
    │   └── Bias & Fairness Suite
    ├── Test Cases
    │   ├── Individual test definitions
    │   └── Reusable templates
    ├── Test Runs
    │   ├── Manual executions
    │   └── CI/CD triggered runs
    └── Reports
        ├── Test summaries
        ├── Failure analysis
        └── Historical trends
```

## User Journey Mapping

### FireFlink Software Testing Flow → ML Guard Flow

| Step | Software Testing | ML Guard Equivalent |
|------|------------------|-------------------|
| 1 | Create Project | Create ML Guard Project |
| 2 | Add Test Suite | Create ML Test Suite (e.g., "Production Readiness") |
| 3 | Add Test Cases | Define ML Test Cases (scriptless rules) |
| 4 | Configure Test Data | Upload Model + Datasets |
| 5 | Run Tests | Execute ML Test Suite |
| 6 | View Results | See Pass/Fail with explanations |
| 7 | Generate Reports | ML-specific reports with visualizations |
| 8 | CI/CD Integration | Quality Gate API for deployment control |

## Test Categories as First-Class Citizens

### Data Quality Tests
- **UI Representation**: Like "API Response Validation" tests
- **Pass/Fail Logic**: Threshold-based assertions
- **Failure Reports**: Show actual vs expected values with data samples

### Statistical Stability Tests
- **UI Representation**: Like "Database Consistency" tests
- **Visualization**: Drift charts, distribution comparisons
- **Failure Reports**: Statistical significance, affected features

### Model Performance Tests
- **UI Representation**: Like "Web Page Load Time" tests
- **Metrics Dashboard**: Accuracy, precision, recall over time
- **Failure Reports**: Performance degradation analysis

### Robustness Tests
- **UI Representation**: Like "Load Testing" scenarios
- **Pass/Fail Logic**: Stability thresholds under perturbation
- **Failure Reports**: Sensitivity analysis, failure patterns

### Bias & Fairness Tests
- **UI Representation**: Like "Security Compliance" test suites
- **Protected Attributes**: Configurable fairness constraints
- **Failure Reports**: Disparate impact analysis, mitigation suggestions

## Scriptless Test Definition

### English-like Rules → Executable Tests

**User Input (UI Form/YAML):**
```
Ensure model accuracy is above 85%
Detect train vs validation drift with PSI threshold 0.1
Ensure no bias for protected attribute "gender"
Ensure missing values are below 5% for all features
```

**Internal Conversion:**
- Parse rule using NLP patterns
- Map to test category and parameters
- Generate executable test function
- Store as reusable test template

## CI/CD Quality Gate

### FireFlink-style Integration

**API Endpoint:** `POST /api/v1/ml-quality-gate`

**Request:**
```json
{
  "project_id": "ml-guard-project-123",
  "model_version": "v1.2.3",
  "test_suite": "production-readiness",
  "environment": "staging"
}
```

**Response (PASS):**
```json
{
  "status": "PASS",
  "deployment_allowed": true,
  "run_id": "ml-run-456",
  "summary": {
    "total_tests": 25,
    "passed": 25,
    "failed": 0
  }
}
```

**Response (FAIL):**
```json
{
  "status": "FAIL",
  "deployment_allowed": false,
  "run_id": "ml-run-457",
  "summary": {
    "total_tests": 25,
    "passed": 20,
    "failed": 5
  },
  "failures": [
    {
      "test_name": "Model Accuracy Threshold",
      "category": "performance",
      "severity": "critical",
      "message": "Accuracy 0.82 below threshold 0.85",
      "details": "Current accuracy: 0.82, Required: 0.85"
    }
  ],
  "recommendations": [
    "Retrained model required before deployment"
  ]
}
```

## Automated Test Generation

### Smart Suggestions Based on Data

**On Model Upload:**
1. Analyze dataset schema → Generate data quality tests
2. Check class distribution → Generate imbalance tests
3. Identify feature types → Generate appropriate validation tests
4. Detect potential bias attributes → Generate fairness tests

**Risk-Based Suggestions:**
- High cardinality categorical → Suggest distribution tests
- Numerical features → Suggest outlier and range tests
- Time-series data → Suggest drift detection
- Protected attributes present → Suggest bias tests

## Reporting & Visualization

### FireFlink-Style Reports

**Test Summary Dashboard:**
- Overall pass/fail status
- Test category breakdown
- Historical trend charts
- Risk assessment score

**Failure Analysis:**
- Root cause explanations
- Data sample evidence
- Statistical significance
- Suggested remediation steps

**ML-Specific Visualizations:**
- Feature importance changes
- Prediction distribution shifts
- Bias detection charts
- Performance degradation over time

## Backend Architecture

### Service Components

```
ML Guard Backend Service
├── API Layer (FastAPI)
│   ├── Project Management
│   ├── Test Suite Orchestration
│   ├── Model Registry
│   └── Quality Gate API
├── ML Testing Engine
│   ├── Data Quality Tests
│   ├── Statistical Tests
│   ├── Performance Tests
│   ├── Robustness Tests
│   └── Bias Tests
├── Test DSL Parser
│   ├── NLP Rule Processing
│   ├── Template Generation
│   └── Validation Logic
├── Storage Layer
│   ├── Model Artifacts
│   ├── Test Results
│   ├── Datasets
│   └── Configuration
└── Execution Runtime
    ├── Parallel Test Execution
    ├── Resource Management
    └── Result Aggregation
```

### Data Flow

1. **Test Definition**: User creates scriptless rules via UI
2. **DSL Processing**: Rules parsed into executable test configurations
3. **Test Execution**: ML tests run against model + datasets
4. **Result Aggregation**: Individual test results combined into suite results
5. **Quality Gate**: Pass/fail decision with detailed reporting
6. **CI/CD Integration**: REST API response for deployment control

This architecture ensures ML Guard feels like a natural extension of FireFlink's existing testing capabilities, maintaining the same user experience patterns while providing comprehensive ML model validation.