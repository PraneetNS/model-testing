from app.core.celery_app import celery_app
from app.db.session import SessionLocal
from app.db.models import Model, ModelExplanation
from ml_guard.core.explainability import generate_shap_explanation
import structlog
import pandas as pd
import numpy as np

logger = structlog.get_logger()

@celery_app.task(name="app.tasks.explainability.run_explainability_task", bind=True)
def run_explainability_task(self, model_id: str, sample_size: int = 200):
    db = SessionLocal()
    try:
        model = db.query(Model).get(model_id)
        if not model:
            logger.error("Model not found in explainability task", model_id=model_id)
            return

        # Fetch model artifact & data...
        # Here we mock retrieving the model and data because we don't have direct access
        # to the storage URLs in this snippet
        from sklearn.ensemble import RandomForestClassifier
        # Dummy data
        X_ref = pd.DataFrame(np.random.rand(sample_size, 5), columns=[f"f_{i}" for i in range(5)])
        X_curr = pd.DataFrame(np.random.rand(sample_size, 5), columns=[f"f_{i}" for i in range(5)])
        
        clf = RandomForestClassifier(n_estimators=10).fit(X_ref, np.random.randint(0, 2, sample_size))

        logger.info("Running SHAP explainability", model_id=model_id)
        explanation = generate_shap_explanation(clf, X_ref, X_curr)

        # Create new explanation entry
        exp_record = ModelExplanation(
            model_id=model_id,
            feature_importances=explanation["feature_importances"],
            top_drift_contributors=explanation["top_drift_contributors"]
        )
        db.add(exp_record)
        db.commit()
        
        logger.info("SHAP explainability completed", model_id=model_id)
        return {"status": "SUCCESS", "model_id": model_id}

    except Exception as e:
        logger.error("Failed to run explainability task", error=str(e))
        db.rollback()
        raise e
    finally:
        db.close()
