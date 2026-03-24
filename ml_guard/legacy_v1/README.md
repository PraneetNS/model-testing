# FireFlink ML Guard

## Pre-Deployment Machine Learning Model Testing

ML Guard brings FireFlink's scriptless testing philosophy to machine learning models. Just like FireFlink tests Web, API, and Database applications, ML Guard validates ML models before deployment using the same user experience and workflow patterns.

![ML Guard Logo](https://img.shields.io/badge/FireFlink-ML%20Guard-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

## 🔥 What is ML Guard?

ML Guard is a **native FireFlink feature** that adds comprehensive ML model testing to your testing arsenal. It appears seamlessly in your FireFlink project navigation as "ML Model Tests" alongside your existing Web, API, and Database tests.

### Key Features

- **🔬 Scriptless ML Testing** - Define tests using natural language rules
- **🚫 CI/CD Quality Gates** - Block deployments when models fail validation
- **📊 FireFlink-Style Reports** - Familiar dashboards and failure analysis
- **⚡ Parallel Test Execution** - Fast validation with proper orchestration
- **🛡️ 5 Test Categories** - Data Quality, Statistical Stability, Performance, Robustness, Bias & Fairness

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [User Experience](#-user-experience)
- [Test Categories](#-test-categories)
- [API Reference](#-api-reference)
- [CI/CD Integration](#-ci-cd-integration)
- [Architecture](#-architecture)
- [Examples](#-examples)
- [Contributing](#-contributing)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/fireflink/ml-guard.git
cd ml-guard

# Install dependencies
pip install -r backend/requirements.txt

# Start the service
cd backend
uvicorn app.main:app --reload
```

### 2. Access ML Guard

- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health

### 3. Run the Demo

```bash
cd examples/customer_churn
python demo.py
```

## 👤 User Experience

ML Guard feels like a natural extension of FireFlink. Here's how users interact with it:

### Step-by-Step Workflow

1. **Open Project** → Click "ML Model Tests"
2. **Add Model** → Upload your trained model + datasets
3. **Auto-Generate Tests** → ML Guard suggests comprehensive validation tests
4. **Customize Tests** → Enable/disable or modify suggested tests
5. **Run Tests** → Execute validation suite (manual or CI/CD)
6. **Review Results** → See pass/fail status with detailed explanations
7. **Quality Gate** → Deployment allowed/blocked based on results

### Visual User Journey

```
FireFlink Dashboard
├── Web Tests       (15 suites)
├── API Tests       (8 suites)
├── DB Tests        (5 suites)
└── ML Model Tests  ← ML Guard (4 suites)
    ├── Models
    │   ├── Customer Churn Predictor v2.1
    │   └── Recommendation Engine v1.3
    ├── Test Suites
    │   ├── Production Readiness ✓
    │   ├── Data Quality Checks ✓
    │   ├── Bias & Fairness ⚠️
    │   └── Performance Validation ❌
    └── Recent Runs
        ├── Run #123 - 16/18 passed
        └── Run #122 - 18/18 passed
```

### Scriptless Test Definition

Users define ML tests using natural language:

```yaml
# English-like rules
- "Ensure missing values are below 5%"
- "Ensure model accuracy is above 85%"
- "Detect train vs validation drift"
- "Ensure no bias for gender attribute"
- "Ensure class imbalance ratio is below 1:10"
```

These rules are internally converted to executable ML validation tests.

## 🧪 Test Categories

ML Guard implements 5 comprehensive test categories as first-class citizens:

### 1. Data Quality Tests
- **Schema Validation** - Feature consistency across datasets
- **Missing Values** - Configurable thresholds per feature
- **Outliers** - IQR and Z-score based detection
- **Duplicate Rows** - Uniqueness validation
- **Class Balance** - Imbalance ratio checks

### 2. Statistical Stability Tests
- **PSI (Population Stability Index)** - Distribution drift detection
- **KS Test** - Kolmogorov-Smirnov statistical tests
- **Feature Correlation Changes** - Stability of relationships
- **Covariate Shift** - Input distribution monitoring

### 3. Model Performance Tests
- **Accuracy Thresholds** - Classification accuracy validation
- **Precision/Recall/F1** - Binary classification metrics
- **ROC-AUC** - Area under curve validation
- **Overfitting Gap** - Train vs validation performance
- **Confidence Distribution** - Prediction certainty checks

### 4. Robustness Tests
- **Input Perturbation** - Sensitivity to noise
- **Prediction Stability** - Consistency under variation
- **Flip-rate Analysis** - Prediction changes under stress
- **Adversarial Inputs** - Resilience testing

### 5. Bias & Fairness Tests
- **Disparate Impact** - Protected attribute fairness
- **Equal Opportunity** - True positive rate parity
- **Statistical Parity** - Outcome distribution fairness
- **Slice-based Performance** - Performance across subgroups

## 🔌 API Reference

### Quality Gate Endpoint

The core of ML Guard's CI/CD integration:

```http
POST /api/v1/ml-quality-gate
```

**Request:**
```json
{
  "project_id": "ecommerce-ml-pipeline",
  "model_version": "v2.1.3",
  "test_suite": "production-readiness",
  "environment": "staging"
}
```

**Response (PASS):**
```json
{
  "status": "PASS",
  "deployment_allowed": true,
  "run_id": "ml-run-20240115-001",
  "summary": {
    "total_tests": 18,
    "passed": 18,
    "failed": 0
  }
}
```

**Response (FAIL):**
```json
{
  "status": "FAIL",
  "deployment_allowed": false,
  "run_id": "ml-run-20240115-002",
  "summary": {
    "total_tests": 18,
    "passed": 15,
    "failed": 3
  },
  "failures": [
    {
      "test_name": "Model Accuracy Threshold",
      "category": "model_performance",
      "severity": "critical",
      "message": "Accuracy 0.82 below threshold 0.85"
    }
  ],
  "recommendations": [
    "Model retraining required - performance below acceptable thresholds"
  ]
}
```

### Other Endpoints

- `GET /api/v1/projects` - List projects
- `GET /api/v1/models` - List models
- `POST /api/v1/test-runs` - Execute test suite
- `GET /api/v1/test-runs/{run_id}` - Get test results

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
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

    - name: Deploy to Staging
      if: success()
      run: |
        # Deployment steps here
        echo "Model passed quality gate - deploying to staging"
```

### Jenkins Pipeline

```groovy
pipeline {
    agent any

    stages {
        stage('ML Quality Gate') {
            steps {
                script {
                    def response = httpRequest(
                        url: "${env.ML_GUARD_URL}/api/v1/ml-quality-gate",
                        httpMode: 'POST',
                        contentType: 'APPLICATION_JSON',
                        requestBody: """
                            {
                                "project_id": "ecommerce-ml",
                                "model_version": "${env.GIT_COMMIT}",
                                "test_suite": "production-readiness"
                            }
                        """
                    )

                    def result = readJSON text: response.content
                    if (result.status != 'PASS') {
                        error("ML Quality Gate failed: ${result.failures}")
                    }
                }
            }
        }

        stage('Deploy') {
            when {
                expression { currentBuild.result == null || currentBuild.result == 'SUCCESS' }
            }
            steps {
                echo 'Deploying model to production...'
            }
        }
    }
}
```

## 🏗️ Architecture

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

### Tech Stack

- **Backend**: Python 3.8+, FastAPI, Pydantic
- **ML Libraries**: scikit-learn, pandas, numpy, scipy
- **Fairness**: AIF360, Fairlearn
- **Data Quality**: Great Expectations
- **Storage**: File-based (easily extensible to databases)
- **Deployment**: Docker, Kubernetes ready

## 📚 Examples

### Customer Churn Validation

```python
from ml_guard import MLGuard

# Initialize ML Guard
guard = MLGuard(project_id="ecommerce-ml")

# Register model
guard.register_model(
    version="v2.1.3",
    model_path="churn_model.pkl",
    training_data="train.csv",
    validation_data="val.csv"
)

# Run quality checks
results = guard.run_quality_suite("production-readiness")

# Check deployment readiness
if results.deployment_allowed:
    print("🚀 Model ready for deployment!")
else:
    print("🛑 Model failed quality checks:")
    for failure in results.failures:
        print(f"  - {failure.test_name}: {failure.message}")
```

## 🤝 Contributing

We welcome contributions! ML Guard is designed to be extended with new test types and integrations.

### Development Setup

```bash
# Fork and clone
git clone https://github.com/your-username/ml-guard.git
cd ml-guard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt
pip install -r backend/requirements-dev.txt

# Run tests
pytest backend/tests/

# Start development server
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Adding New Test Types

1. **Extend Test Categories**: Add new test methods to existing categories
2. **Create New Categories**: Implement new test category classes
3. **Update DSL Parser**: Add support for new rule patterns
4. **Add API Endpoints**: Expose new functionality via REST API

### Code Standards

- **Type Hints**: Full type annotation coverage
- **Documentation**: Comprehensive docstrings
- **Testing**: 80%+ test coverage
- **Linting**: Black, isort, flake8 compliance

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙋 Support

- **Documentation**: [docs/](docs/) directory
- **Issues**: [GitHub Issues](https://github.com/fireflink/ml-guard/issues)
- **Discussions**: [GitHub Discussions](https://github.com/fireflink/ml-guard/discussions)

## 🎯 Roadmap

- [ ] Frontend UI components (React/Vue)
- [ ] Integration with MLflow/Model Registry
- [ ] Advanced drift detection algorithms
- [ ] Custom test template marketplace
- [ ] Real-time monitoring integration
- [ ] Multi-framework model support (TensorFlow, PyTorch, XGBoost)

---

**ML Guard** - Bringing FireFlink's testing excellence to machine learning. 🚀