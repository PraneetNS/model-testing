import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from ml_guard.core.explainability import generate_shap_explanation

def test_generate_shap_explanation_basic():
    # Mocking shap.Explainer
    with patch("shap.Explainer") as mock_explainer_cls:
        
        # We need realistic mock returns
        mock_explainer = MagicMock()
        mock_explainer_cls.return_value = mock_explainer
        
        # When called on X_reference
        mock_shap_values_ref = MagicMock()
        # Shape: (samples, features) = (10, 5)
        # We simulate feature 0 having high SHAP value on reference
        mock_shap_values_ref.values = np.array([
            [1.0, 0.1, 0.2, 0.3, 0.0] for _ in range(10)
        ])
        
        # When called on X_current
        mock_shap_values_curr = MagicMock()
        # Simulated drift on feature 1 (increases from 0.1 to 2.5)
        mock_shap_values_curr.values = np.array([
            [1.0, 2.5, 0.2, 0.3, 0.0] for _ in range(10)
        ])
        
        mock_explainer.side_effect = [mock_shap_values_ref, mock_shap_values_curr]
        
        X_ref = pd.DataFrame(np.zeros((10, 5)), columns=[f"f_{i}" for i in range(5)])
        X_curr = pd.DataFrame(np.zeros((10, 5)), columns=[f"f_{i}" for i in range(5)])
        
        mock_model = MagicMock()
        
        res = generate_shap_explanation(mock_model, X_ref, X_curr)
        
        # Validate feature importances (should be ranked on reference data)
        # f_0 should be top rank with mean_abs_shap = 1.0
        importances = res["feature_importances"]
        assert len(importances) == 5
        assert importances[0]["feature"] == "f_0"
        assert importances[0]["mean_abs_shap"] == 1.0
        assert importances[0]["rank"] == 1
        
        # Validate top drift contributors 
        drift = res["top_drift_contributors"]
        # f_1 should be top drift contributor because it jumps from 0.1 to 2.5 (delta = 2.4)
        assert len(drift) == 5
        assert drift[0]["feature"] == "f_1"
        assert drift[0]["shap_delta_vs_baseline"] == pytest.approx(2.4)

def test_generate_shap_explanation_no_current():
    with patch("shap.Explainer") as mock_explainer_cls:
        mock_explainer = MagicMock()
        mock_explainer_cls.return_value = mock_explainer
        
        mock_shap_values_ref = MagicMock()
        mock_shap_values_ref.values = np.array([
            [1.0, 0.1, 0.2, 0.3, 0.0] for _ in range(10)
        ])
        mock_explainer.return_value = mock_shap_values_ref
        
        X_ref = pd.DataFrame(np.zeros((10, 5)), columns=[f"f_{i}" for i in range(5)])
        mock_model = MagicMock()
        
        res = generate_shap_explanation(mock_model, X_ref, None)
        
        assert len(res["feature_importances"]) == 5
        # No current data -> no drift
        assert len(res["top_drift_contributors"]) == 0
