# 🔥 ML Guard - Fireflink Style Streamlit UI

A modern, interactive ML model testing platform built with Streamlit, featuring Fireflink-inspired design and advanced NLP capabilities for natural language test execution.

## 🌟 Features

### 🎨 Fireflink-Inspired Design
- Modern, clean UI matching Fireflink's professional aesthetic
- Gradient backgrounds, smooth animations, and responsive design
- Status badges, progress bars, and intuitive navigation
- Dark/light theme compatibility

### 🧠 Natural Language Processing
- **NLP-Powered Test Selection**: Describe tests in plain English
- **Intelligent Query Understanding**: Automatically interprets test requests
- **Smart Test Recommendations**: Suggests relevant tests based on context

### 🧪 Comprehensive Testing Suite
- **Model Performance**: Accuracy, Precision, Recall, F1-Score
- **Data Quality**: Missing values, duplicates, class balance
- **Bias & Fairness**: Discrimination detection, disparate impact
- **Statistical Stability**: Drift detection, PSI analysis
- **Robustness**: Edge cases, feature stability

### 📊 Advanced Visualization
- Real-time test execution progress
- Interactive result dashboards
- Quality gate decision indicators
- Exportable test reports (JSON format)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- ML Guard backend installed

### Installation

1. **Install Dependencies**
```bash
pip install -r requirements-streamlit.txt
```

**Note:** SpaCy is optional. For Python 3.14+, SpaCy is not compatible, but the app includes a keyword-based NLP fallback that works without SpaCy.

2. **Optional: Download SpaCy NLP Model** (for enhanced NLP capabilities)
```bash
# Only needed for Python < 3.14
pip install spacy
python -m spacy download en_core_web_sm
```

3. **Run the Application**
```bash
python run_streamlit.py
```

The application will start at `http://localhost:8501`

## 🎯 Usage Guide

### 1. Upload Your Model
- Use the sidebar to upload your trained ML model (.pkl format)
- Supported formats: scikit-learn models, custom ML models
- Model information is automatically extracted and displayed

### 2. Upload Datasets
- **Training Data**: Historical data used to train your model
- **Validation Data**: Data for model evaluation during development
- **Test Data**: Holdout dataset for final performance assessment

### 3. Choose Testing Method

#### 🗣️ NLP Testing (Recommended)
- **Natural Language Input**: Describe tests in plain English
- **Examples**:
  - "Run accuracy and bias tests on my model"
  - "Check for data quality issues and model performance"
  - "Test everything - comprehensive validation"

#### 🎯 Manual Testing
- **Checkbox Selection**: Choose specific test categories
- **Fine-grained Control**: Select exactly which tests to run
- **Advanced Options**: Configure thresholds and parameters

### 4. Execute Tests
- Click "🚀 Execute NLP Tests" or "🚀 Run Selected Tests"
- Watch real-time progress with animated indicators
- View detailed results in the dashboard

### 5. Analyze Results
- **Quality Gate Decision**: Pass/Fail deployment recommendation
- **Detailed Breakdown**: Results by category and severity
- **Export Options**: Download JSON reports for further analysis

## 🔧 Configuration

### Test Categories

| Category | Tests Included | Description |
|----------|---------------|-------------|
| **Accuracy** | Accuracy, Precision, Recall, F1 | Core model performance metrics |
| **Data Quality** | Missing Values, Duplicates, Balance | Dataset health checks |
| **Bias** | Disparate Impact, Fairness | Discrimination detection |
| **Drift** | PSI Analysis, Stability | Statistical change detection |
| **Robustness** | Feature Stability, Edge Cases | Model reliability tests |

### NLP Keywords

The NLP engine understands these keywords:

- **Performance**: accuracy, precision, recall, f1, performance, metrics
- **Quality**: missing, duplicate, null, quality, data, balance
- **Bias**: bias, fairness, discrimination, gender, race, protected
- **Stability**: drift, stability, psi, change, distribution
- **Robustness**: robust, stress, edge, boundary, adversarial
- **Comprehensive**: all, comprehensive, complete, everything, full

## 📱 Interface Overview

### Main Dashboard
- **Header**: ML Guard branding with Fireflink-style design
- **Tabs**: NLP Testing, Manual Testing, Results Dashboard
- **Metrics**: Real-time test statistics and progress indicators

### Sidebar
- **Model Upload**: Drag-and-drop model file upload
- **Dataset Upload**: Training, validation, and test data upload
- **Status Indicators**: Upload confirmation and data summaries

### Results Visualization
- **Test Cards**: Individual test results with status indicators
- **Progress Bars**: Overall test completion percentage
- **Quality Gate**: Deployment recommendation with reasoning
- **Export Options**: Download results in multiple formats

## 🛠️ Advanced Features

### Custom Test Configurations
- Modify test thresholds in the backend configuration
- Add custom test categories and types
- Integrate with external testing frameworks

### Batch Testing
- Run multiple test suites simultaneously
- Compare results across different models
- Historical performance tracking

### Integration APIs
- REST API endpoints for automated testing
- Webhook notifications for test completion
- Integration with CI/CD pipelines

## 🎨 Customization

### Themes
- Built-in dark/light mode support
- Custom color schemes via CSS variables
- Brand customization options

### Layout
- Responsive design for all screen sizes
- Collapsible sidebar for mobile devices
- Customizable dashboard widgets

## 📊 Sample NLP Queries

```
"Run accuracy tests and check for bias"
"Test data quality and model performance"
"Check everything - comprehensive validation"
"Run drift detection and robustness tests"
"Validate model fairness and statistical stability"
```

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure model is saved in .pkl format
   - Check scikit-learn version compatibility
   - Verify model contains required components

2. **Dataset Upload Issues**
   - Ensure CSV format with headers
   - Check for special characters in column names
   - Verify target column exists in all datasets

3. **NLP Not Understanding Queries**
   - Use simple, direct language
   - Include specific test keywords
   - Check spelling and grammar

4. **Test Execution Failures**
   - Verify all datasets are uploaded
   - Check model compatibility with test types
   - Review error messages in the results dashboard

### Performance Optimization
- Use smaller datasets for initial testing
- Run tests incrementally rather than all at once
- Monitor memory usage with large models

## 🔗 Integration

### With ML Guard Backend
- Automatic test result storage
- Historical test run tracking
- Model registry integration

### With External Tools
- CI/CD pipeline integration
- Monitoring dashboard connections
- Alert system integration

## 📈 Future Enhancements

- [ ] **Advanced NLP**: Context-aware test recommendations
- [ ] **Model Comparison**: Side-by-side test result comparison
- [ ] **Automated Retraining**: Triggered by test failures
- [ ] **Real-time Monitoring**: Continuous model health checks
- [ ] **Multi-language Support**: Internationalization features

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is part of ML Guard - Fireflink Style ML Testing Platform.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the ML Guard documentation
3. Open an issue on the project repository

---

**Built with ❤️ for the ML community**