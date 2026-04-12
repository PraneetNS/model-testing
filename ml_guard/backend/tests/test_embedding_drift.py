import numpy as np
import pytest
import sys
import os

# Ensure core is importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from ml_guard.core.drift import compute_embedding_drift

def test_embedding_drift_no_drift():
    np.random.seed(42)
    ref = np.random.normal(0, 1, size=(100, 16))
    cur = np.random.normal(0, 1, size=(100, 16))
    
    result = compute_embedding_drift(ref, cur)
    assert not result['drift_detected']
    assert result['cosine_drift'] < 0.05
    assert result['mmd_score'] < 0.1
    assert 'umap_snapshot' in result

def test_embedding_drift_with_shift():
    np.random.seed(42)
    ref = np.random.normal(0, 1, size=(100, 16))
    cur = np.random.normal(5, 1, size=(100, 16))
    
    result = compute_embedding_drift(ref, cur)
    assert result['drift_detected']
    assert result['cosine_drift'] > 0.05 or result['mmd_score'] > 0.1
