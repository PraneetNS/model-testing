import pytest
import numpy as np
from ml_guard.core.drift import compute_psi

def test_psi_identical_distributions():
    """Test that identical distributions return 0.0 PSI."""
    expected = [1.0, 2.0, 3.0, 4.0, 5.0] * 10
    actual = [1.0, 2.0, 3.0, 4.0, 5.0] * 10
    psi = compute_psi(expected, actual)
    assert psi < 0.001

def test_psi_shifted_distribution():
    """Test that a shifted distribution returns a high PSI."""
    # Normal distribution center at 0
    expected = np.random.normal(0, 1, 1000)
    # Normal distribution center at 1 (drifted)
    actual = np.random.normal(1.5, 1, 1000)
    
    psi = compute_psi(expected, actual)
    assert psi > 0.25 # Threshold for high drift

def test_psi_missing_bins():
    """Test PSI with actual data missing points in some bins."""
    expected = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    # Actual missing 3's and 4's
    actual = [1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2]
    
    psi = compute_psi(expected, actual)
    assert psi > 0.2

def test_psi_with_nans():
    """Ensure NaNs are filtered out and don't cause crashes."""
    expected = [1, 2, 3, np.nan, 4, 5]
    actual = [1, 2, 3, 4, 10, np.nan]
    psi = compute_psi(expected, actual)
    assert isinstance(psi, float)

def test_psi_extreme_drift():
    """Test extreme case where distributions are entirely disjoint."""
    expected = np.random.uniform(0, 10, 100)
    actual = np.random.uniform(100, 110, 100)
    psi = compute_psi(expected, actual)
    assert psi > 5.0 # Very high drift
