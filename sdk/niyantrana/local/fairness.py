import pandas as pd
from typing import List
from ..models import FairnessReport, FeatureFairness

def check_fairness(df: pd.DataFrame, label_col: str, prediction_col: str, sensitive_features: List[str]) -> FairnessReport:
    """
    Check fairness metrics for sensitive features.
    """
    per_feature = []
    overall_fair = True

    for feature in sensitive_features:
        if feature not in df.columns:
            continue
            
        groups = df[feature].unique()
        if len(groups) != 2:
            # Simplified for binary sensitive attributes
            continue
            
        group_0, group_1 = groups[0], groups[1]
        
        df_g0 = df[df[feature] == group_0]
        df_g1 = df[df[feature] == group_1]

        # Demographic Parity: |P(y_pred=1|A=0) - P(y_pred=1|A=1)|
        p_y1_g0 = df_g0[prediction_col].mean() if len(df_g0) > 0 else 0
        p_y1_g1 = df_g1[prediction_col].mean() if len(df_g1) > 0 else 0
        dp_diff = abs(p_y1_g0 - p_y1_g1)

        # Equalized Odds: diff in TPR and FPR
        # TPR = TP / (TP + FN) = P(y_pred=1 | y_true=1)
        # FPR = FP / (FP + TN) = P(y_pred=1 | y_true=0)
        def get_rates(df_g):
            if len(df_g) == 0: return 0, 0
            df_pos = df_g[df_g[label_col] == 1]
            df_neg = df_g[df_g[label_col] == 0]
            tpr = df_pos[prediction_col].mean() if len(df_pos) > 0 else 0
            fpr = df_neg[prediction_col].mean() if len(df_neg) > 0 else 0
            return tpr, fpr

        tpr_g0, fpr_g0 = get_rates(df_g0)
        tpr_g1, fpr_g1 = get_rates(df_g1)
        
        eo_diff = max(abs(tpr_g0 - tpr_g1), abs(fpr_g0 - fpr_g1))

        # Disparate Impact Ratio: min_group_rate / max_group_rate
        min_rate = min(p_y1_g0, p_y1_g1)
        max_rate = max(p_y1_g0, p_y1_g1)
        di_ratio = min_rate / max_rate if max_rate > 0 else 1.0

        flags = []
        if dp_diff > 0.1:
            flags.append("Demographic Parity > 0.1")
        if eo_diff > 0.1:
            flags.append("Equalized Odds > 0.1")
        if di_ratio < 0.8:
            flags.append("Disparate Impact Ratio < 0.8 (4/5ths Rule Violation)")
            
        if flags:
            overall_fair = False

        per_feature.append(FeatureFairness(
            feature=feature,
            demographic_parity_diff=float(dp_diff),
            equalized_odds_diff=float(eo_diff),
            disparate_impact_ratio=float(di_ratio),
            flags=flags
        ))

    return FairnessReport(
        overall_fair=overall_fair,
        per_feature=per_feature
    )
