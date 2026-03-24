class ScoringEngine:
    @staticmethod
    def compute_weighted_score(preflight_metrics, drift_metrics, llm_metrics=None):
        score = 100.0
        
        # Preflight penalties
        if preflight_metrics.get("missing_percent", 0) > 5:
            score -= 10
            
        # Drift penalties
        drift_score = drift_metrics.get("global_drift_score", 0)
        if drift_score > 0.2:
            score -= 20
        elif drift_score > 0.1:
            score -= 10
            
        # LLM penalties
        if llm_metrics:
            if llm_metrics.get("hallucination_score", 1.0) < 0.9:
                score -= 15
        
        return max(0, score)
