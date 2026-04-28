import pandas as pd
from niyantrana.local.drift import detect_drift

def test_detect_drift_psi():
    ref_df = pd.DataFrame({"feature1": [1.0, 2.0, 3.0, 4.0, 5.0]})
    # Identical distribution
    cur_df = pd.DataFrame({"feature1": [1.1, 2.1, 3.1, 4.1, 5.1]})
    
    report = detect_drift(ref_df, cur_df, method="psi")
    assert not report.overall_drift_detected
    assert report.method == "psi"
    
    # Drifted distribution
    drift_df = pd.DataFrame({"feature1": [10.0, 11.0, 12.0, 13.0, 14.0]})
    report_drifted = detect_drift(ref_df, drift_df, method="psi")
    assert report_drifted.overall_drift_detected
