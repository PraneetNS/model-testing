
import numpy as np
import logging

logger = logging.getLogger(__name__)


def run_security_checks(model, X_train, X_test, y_train=None, y_test=None):
    """Run all security checks on a model."""
    results = {
        "data_poisoning": check_data_poisoning(model, X_train, y_train),
        "extraction_vulnerability": check_extraction_vulnerability(model, X_test),
        "membership_inference": check_membership_inference(model, X_train, X_test),
        "overall_risk": "LOW",
    }

    # Determine overall risk
    risk_levels = [r.get("risk", "LOW") for r in results.values() if isinstance(r, dict)]
    if "CRITICAL" in risk_levels:
        results["overall_risk"] = "CRITICAL"
    elif "HIGH" in risk_levels:
        results["overall_risk"] = "HIGH"
    elif "MEDIUM" in risk_levels:
        results["overall_risk"] = "MEDIUM"

    return results


def check_data_poisoning(model, X_train, y_train=None):
    """
    Detect potential data poisoning by analyzing:
    - Influential training samples (via leverage scores)
    - Label consistency
    - Training data anomalies
    """
    result = {"risk": "LOW", "indicators": [], "score": 0}

    try:
        if hasattr(X_train, 'values'):
            X = X_train.values
        else:
            X = np.array(X_train)

        # Check for high-leverage points using hat matrix approximation
        n_samples = min(len(X), 1000)
        X_sample = X[:n_samples]
        
        if X_sample.shape[1] > 0:
            try:
                XtX_inv = np.linalg.pinv(X_sample.T @ X_sample)
                hat_diag = np.diag(X_sample @ XtX_inv @ X_sample.T)
                threshold = 2 * X_sample.shape[1] / n_samples
                high_leverage = int(np.sum(hat_diag > threshold))
                leverage_ratio = high_leverage / n_samples
                
                if leverage_ratio > 0.1:
                    result["indicators"].append(f"High leverage points: {high_leverage}/{n_samples} ({leverage_ratio:.1%})")
                    result["score"] += 30
            except Exception:
                pass

        # Check predictions on training data for inconsistency
        if y_train is not None and hasattr(model, "predict"):
            y_pred = model.predict(X[:n_samples])
            y_actual = np.array(y_train[:n_samples])
            mismatch = int(np.sum(y_pred != y_actual))
            mismatch_ratio = mismatch / n_samples
            
            if mismatch_ratio > 0.2:
                result["indicators"].append(f"Training prediction mismatch: {mismatch_ratio:.1%}")
                result["score"] += 25

        # Determine risk level
        if result["score"] >= 50:
            result["risk"] = "HIGH"
        elif result["score"] >= 25:
            result["risk"] = "MEDIUM"

    except Exception as e:
        logger.warning("Data poisoning check failed: %s", e)
        result["error"] = str(e)

    return result


def check_extraction_vulnerability(model, X_test):
    """
    Assess model extraction vulnerability:
    - High confidence predictions make extraction easier
    - Model complexity affects extractability
    """
    result = {"risk": "LOW", "indicators": [], "score": 0}

    try:
        if hasattr(X_test, 'values'):
            X = X_test.values
        else:
            X = np.array(X_test)

        n_samples = min(len(X), 500)
        X_sample = X[:n_samples]

        # Check prediction confidence distribution
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_sample)
            max_confidence = np.max(proba, axis=1)
            avg_confidence = float(np.mean(max_confidence))
            high_conf_ratio = float(np.mean(max_confidence > 0.95))

            if avg_confidence > 0.9:
                result["indicators"].append(f"Very high average confidence: {avg_confidence:.3f}")
                result["score"] += 25
            if high_conf_ratio > 0.7:
                result["indicators"].append(f"High-confidence predictions: {high_conf_ratio:.1%}")
                result["score"] += 20

            result["avg_confidence"] = avg_confidence
            result["high_confidence_ratio"] = high_conf_ratio

        # Check model complexity (number of parameters)
        param_count = _estimate_param_count(model)
        if param_count and param_count < 1000:
            result["indicators"].append(f"Low complexity model ({param_count} params) — easier to extract")
            result["score"] += 15
        result["estimated_parameters"] = param_count

        if result["score"] >= 40:
            result["risk"] = "HIGH"
        elif result["score"] >= 20:
            result["risk"] = "MEDIUM"

    except Exception as e:
        logger.warning("Extraction vulnerability check failed: %s", e)
        result["error"] = str(e)

    return result


def check_membership_inference(model, X_train, X_test):
    """
    Assess membership inference risk:
    - Compare prediction confidence on train vs test data
    - Large gap suggests the model memorized training data
    """
    result = {"risk": "LOW", "indicators": [], "score": 0}

    try:
        if not hasattr(model, "predict_proba"):
            result["indicators"].append("Model has no predict_proba — lower inference risk")
            return result

        if hasattr(X_train, 'values'):
            X_tr = X_train.values
        else:
            X_tr = np.array(X_train)
        if hasattr(X_test, 'values'):
            X_te = X_test.values
        else:
            X_te = np.array(X_test)

        n = min(200, len(X_tr), len(X_te))
        
        train_proba = model.predict_proba(X_tr[:n])
        test_proba = model.predict_proba(X_te[:n])

        train_conf = float(np.mean(np.max(train_proba, axis=1)))
        test_conf = float(np.mean(np.max(test_proba, axis=1)))
        confidence_gap = train_conf - test_conf

        result["train_confidence"] = round(train_conf, 4)
        result["test_confidence"] = round(test_conf, 4)
        result["confidence_gap"] = round(confidence_gap, 4)

        if confidence_gap > 0.15:
            result["indicators"].append(f"Large train-test confidence gap: {confidence_gap:.3f}")
            result["risk"] = "HIGH"
            result["score"] = 60
        elif confidence_gap > 0.08:
            result["indicators"].append(f"Moderate confidence gap: {confidence_gap:.3f}")
            result["risk"] = "MEDIUM"
            result["score"] = 35
        else:
            result["indicators"].append(f"Low confidence gap: {confidence_gap:.3f}")

    except Exception as e:
        logger.warning("Membership inference check failed: %s", e)
        result["error"] = str(e)

    return result


def _estimate_param_count(model):
    """Estimate the number of parameters in a model."""
    try:
        if hasattr(model, "n_features_in_") and hasattr(model, "n_estimators"):
            return model.n_features_in_ * model.n_estimators * 10
        if hasattr(model, "coef_"):
            return int(np.prod(model.coef_.shape))
        return None
    except Exception:
        return None
