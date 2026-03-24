# ML Guard Intelligent Profiler: Design Specification

The Model Profiler is a core governance component that automatically determines the "Surface Area of Risk" for any uploaded model.

## 1. Profiling Algorithm

The profiler performs a multi-dimensional analysis upon artifact ingestion:

1.  **Object Introspection**: Uses the `ModelDetector` to identify the framework (sklearn, XGBoost, etc.) and model task (Classification/Regression).
2.  **Structural Analysis**: Scans the dataset for:
    *   **Class Imbalance**: Ratio between majority and minority classes.
    *   **Feature Sparsity**: Missing value percentages per feature and globally.
    *   **Data Scale**: Row and column counts to detect overfitting/stability risks.
3.  **Governance Scanning**: Cross-references feature names against a global list of **Protected Attributes** (e.g., gender, age, race).

## 2. Auto-Test Generation Logic

Based on the profile, the engine dynamically recommends a `TestSuite`:

| Profile Indicator | Recommended Test | Severity |
| :--- | :--- | :--- |
| **Protected Attrs Found** | `disparate_impact` (Fairness) | **Critical** |
| **High Imbalance (>5:1)** | `precision_recall_balance` | **High** |
| **Low Row Count (<1k)** | `input_perturbation` (Robustness) | **Medium** |
| **High Missingness (>10%)** | `data_integrity_scan` | **High** |
| **All Classification** | `auc_roc_threshold` | **High** |

## 3. Risk Model & Scoring Formula

The Risk Model classifies artifacts into four buckets (**Low, Medium, High, Critical**) based on combined heuristics:

### Scoring Formula
The final **Quality Index (QI)** is calculated as:
$$QI = 100 \times \left(1 - \frac{\sum (FailureWeight_i)}{\sum (MaxPossibleWeight_i)}\right)$$

### Profile-Based Dynamic Weights
The profiler generates a "Weighted Scoring Profile" that adjust the penalties:
- **Low Risk Profile**: Critical Fail = -10 points.
- **High Risk Profile**: Critical Fail = -20 points (Strict Mode).

## 4. Extensible Rule Engine Design

The engine uses a **Chain of Responsibility** pattern:
- **Analyzers**: Pure functions that extract stats (e.g., `ImbalanceAnalyzer`).
- **Policy Mappers**: Map stats to risk levels (e.g., `If ProtectedAttrs -> Critical`).
- **Test Generators**: Map risk levels and stats to specific test configurations.

This design allows ML Guard to be easily extended with domain-specific rules (e.g., GDPR compliance rules for EU-based projects).
