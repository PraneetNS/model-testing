import pytest
import pandas as pd
from datetime import datetime, timedelta
from app.services.forecasting.forecaster import GovernanceForecaster
from unittest.mock import MagicMock

def test_compute_breach_above_threshold():
    """Test breach detection when metric exceeds threshold (e.g., PSI)."""
    # Mock DB session
    db = MagicMock()
    forecaster = GovernanceForecaster(db, "test-model")
    
    # Create forecast data
    today = datetime.now()
    dates = [today + timedelta(days=i) for i in range(10)]
    # Values increasing from 0.15 to 0.25 (threshold 0.20)
    values = [0.15 + (0.012 * i) for i in range(10)]
    
    df = pd.DataFrame({
        'ds': dates,
        'yhat': values,
        'yhat_lower': [v - 0.05 for v in values],
        'yhat_upper': [v + 0.05 for v in values]
    })
    
    breach_date, confidence = forecaster._compute_breach(df, 0.20, "psi")
    
    assert breach_date is not None
    # Expect breach around day 5 (0.15 + 0.012 * 5 = 0.21)
    assert breach_date == dates[5].strftime('%Y-%m-%d')
    assert confidence > 0.5

def test_compute_breach_below_threshold():
    """Test breach detection when metric falls below threshold (e.g., Accuracy)."""
    db = MagicMock()
    forecaster = GovernanceForecaster(db, "test-model")
    
    today = datetime.now()
    dates = [today + timedelta(days=i) for i in range(10)]
    # Accuracy decreasing from 0.90 to 0.81 (threshold 0.85)
    values = [0.90 - (0.01 * i) for i in range(10)]
    
    df = pd.DataFrame({
        'ds': dates,
        'yhat': values,
        'yhat_lower': [v - 0.05 for v in values],
        'yhat_upper': [v + 0.05 for v in values]
    })
    
    breach_date, confidence = forecaster._compute_breach(df, 0.85, "accuracy")
    
    assert breach_date is not None
    # Expect breach on day 6 (0.90 - 0.06 = 0.84)
    assert breach_date == dates[6].strftime('%Y-%m-%d')

def test_no_breach_detected():
    """Test case where no breach is forecast within the horizon."""
    db = MagicMock()
    forecaster = GovernanceForecaster(db, "test-model")
    
    today = datetime.now()
    dates = [today + timedelta(days=i) for i in range(10)]
    # Stable values far from threshold
    values = [0.1] * 10
    
    df = pd.DataFrame({
        'ds': dates,
        'yhat': values,
        'yhat_lower': [0.05] * 10,
        'yhat_upper': [0.15] * 10
    })
    
    breach_date, confidence = forecaster._compute_breach(df, 0.20, "psi")
    
    assert breach_date is None
    assert confidence == 0.0

def test_trend_detection():
    """Test improving vs degrading trend logic."""
    db = MagicMock()
    forecaster = GovernanceForecaster(db, "test-model")
    
    # Degrading (increasing PSI)
    df_degrade = pd.DataFrame({
        'ds': [datetime.now() + timedelta(days=i) for i in range(5)],
        'yhat': [0.1, 0.12, 0.14, 0.16, 0.18]
    })
    trend_d, _ = forecaster._get_trend_info(df_degrade)
    assert trend_d == "DEGRADING"

    # Improving (decreasing PSI)
    df_improve = pd.DataFrame({
        'ds': [datetime.now() + timedelta(days=i) for i in range(5)],
        'yhat': [0.2, 0.18, 0.16, 0.14, 0.12]
    })
    trend_i, _ = forecaster._get_trend_info(df_improve)
    assert trend_i == "IMPROVING"
