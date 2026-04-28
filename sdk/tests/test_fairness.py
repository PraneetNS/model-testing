import pandas as pd
from niyantrana.local.fairness import check_fairness

def test_check_fairness_fair():
    # Perfectly fair predictions
    df = pd.DataFrame({
        "label": [1, 1, 0, 0, 1, 1, 0, 0],
        "pred":  [1, 1, 0, 0, 1, 1, 0, 0],
        "group": ["A", "A", "A", "A", "B", "B", "B", "B"]
    })
    
    report = check_fairness(df, "label", "pred", ["group"])
    assert report.overall_fair
    assert report.per_feature[0].demographic_parity_diff == 0.0

def test_check_fairness_biased():
    # Biased against group B
    df = pd.DataFrame({
        "label": [1, 1, 0, 0, 1, 1, 0, 0],
        "pred":  [1, 1, 0, 0, 0, 0, 0, 0], # Model never predicts 1 for B
        "group": ["A", "A", "A", "A", "B", "B", "B", "B"]
    })
    
    report = check_fairness(df, "label", "pred", ["group"])
    assert not report.overall_fair
    assert report.per_feature[0].demographic_parity_diff > 0.1
    assert "Demographic Parity > 0.1" in report.per_feature[0].flags
