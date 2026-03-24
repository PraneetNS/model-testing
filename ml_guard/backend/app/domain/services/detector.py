from typing import Any

class ModelDetector:
    """
    Tier 1 utility to automatically detect model characteristics in the backend.
    """
    @staticmethod
    def detect_type(model: Any) -> str:
        model_name = str(type(model)).lower()
        
        if "classifier" in model_name:
            return "classification"
        elif "regressor" in model_name:
            return "regression"
        elif "pipeline" in model_name:
            if hasattr(model, 'steps'):
                last_step = model.steps[-1][1]
                return ModelDetector.detect_type(last_step)
            return "unknown_pipeline"
        
        return "generic_ml"

    @staticmethod
    def extract_metadata(model: Any) -> dict:
        metadata = {
            "class": str(type(model)),
        }
        if hasattr(model, 'get_params'):
            try:
                metadata["parameters"] = model.get_params()
            except:
                pass
        return metadata
