from app.services.stats_engine import StatsEngine
import pandas as pd

class DriftEngine:
    def __init__(self):
        self.stats = StatsEngine()

    async def analyze_drift(self, baseline_data: pd.DataFrame, current_data: pd.DataFrame):
        results = {}
        global_drift_score = 0
        features = current_data.columns
        
        for feature in features:
            if feature in baseline_data.columns:
                psi = self.stats.calculate_psi(baseline_data[feature], current_data[feature])
                d_stat, p_val = self.stats.calculate_ks(baseline_data[feature], current_data[feature])
                
                results[feature] = {
                    "psi": psi,
                    "ks_p_value": p_val,
                    "is_drifted": psi > 0.25 or p_val < 0.05
                }
                global_drift_score += psi
        
        avg_drift = global_drift_score / len(features) if features.any() else 0
        
        return {
            "feature_drift": results,
            "global_drift_score": avg_drift,
            "deployment_blocked": avg_drift > 0.2
        }
