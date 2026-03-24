"""
Behavioral robustness & sensitivity analysis engine.
All math-based, no simulations, no hallucinated metrics.
"""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from .exceptions import MetricComputationError


def sensitivity_analysis(model, X_ref: pd.DataFrame, epsilon_factor: float = 0.01) -> dict:
    """
    Partial derivative approximation via finite differences.
    Δy/Δx_i ≈ [f(x + ε·e_i) − f(x)] / ε
    Returns sensitivity score per feature, ranked.
    """
    try:
        X_numeric = X_ref.select_dtypes(include=[np.number])
        base_preds = model.predict(X_numeric.values).astype(float)
        sensitivities = {}

        for i, col in enumerate(X_numeric.columns):
            X_perturbed = X_numeric.values.copy()
            col_std = float(X_numeric[col].std()) if X_numeric[col].std() > 0 else 1.0
            eps = epsilon_factor * col_std
            X_perturbed[:, i] += eps
            new_preds = model.predict(X_perturbed).astype(float)
            delta_y = np.abs(new_preds - base_preds)
            sensitivity = float(delta_y.mean() / eps) if eps > 0 else 0.0
            sensitivities[col] = round(sensitivity, 6)

        max_sens = max(sensitivities.values()) if sensitivities else 1.0
        ranked = dict(sorted(sensitivities.items(), key=lambda x: -x[1]))
        normalized = {k: round(v / max_sens, 4) for k, v in ranked.items()}

        high_sensitivity = [k for k, v in normalized.items() if v >= 0.8]

        return {
            "sensitivity_scores": normalized,
            "high_sensitivity_features": high_sensitivity,
            "top_feature": list(ranked.keys())[0] if ranked else None,
        }
    except Exception as e:
        raise MetricComputationError(f"Sensitivity analysis failed: {e}")


def monte_carlo_stability(model, X_ref: pd.DataFrame, n_runs: int = 100, noise_std: float = 0.05) -> dict:
    """
    x' = x + N(0, σ·feature_std)
    Run n_runs times. Measure output variance and prediction flip rate.
    Stability score = 1 − flip_rate
    """
    try:
        X_numeric = X_ref.select_dtypes(include=[np.number]).values.copy()
        base_preds = model.predict(X_numeric)
        all_preds = []

        feature_stds = X_numeric.std(axis=0)
        feature_stds = np.where(feature_stds == 0, 1.0, feature_stds)

        for _ in range(n_runs):
            noise = np.random.normal(0, noise_std * feature_stds, size=X_numeric.shape)
            X_noisy = X_numeric + noise
            all_preds.append(model.predict(X_noisy))

        all_preds_arr = np.array(all_preds)  # (n_runs, n_samples)

        # Flip rate: fraction of samples that changed prediction at least once
        pred_changed = np.any(all_preds_arr != base_preds[np.newaxis, :], axis=0)
        flip_rate = float(pred_changed.mean())
        stability_score = round(1.0 - flip_rate, 4)

        return {
            "n_runs": n_runs,
            "noise_std_factor": noise_std,
            "flip_rate": round(flip_rate, 4),
            "stability_score": stability_score,
            "output_variance": round(float(all_preds_arr.astype(float).var()), 6),
            "status": "STABLE" if stability_score >= 0.90 else "FRAGILE",
        }
    except Exception as e:
        raise MetricComputationError(f"Monte Carlo stability test failed: {e}")


def ood_boundary_test(model, X_ref: pd.DataFrame, sigma_factor: float = 3.0) -> dict:
    """
    Generate synthetic out-of-distribution inputs:
    min(feature) - σ·3  and  max(feature) + σ·3
    Check if model predictions are valid (not NaN, not exploding).
    """
    try:
        X_numeric = X_ref.select_dtypes(include=[np.number])
        results = {}

        for side in ["extreme_low", "extreme_high"]:
            X_ood = X_numeric.copy()
            for col in X_numeric.columns:
                std = float(X_numeric[col].std()) if X_numeric[col].std() > 0 else 1.0
                if side == "extreme_low":
                    X_ood[col] = X_numeric[col].min() - sigma_factor * std
                else:
                    X_ood[col] = X_numeric[col].max() + sigma_factor * std

            try:
                preds = model.predict(X_ood.values)
                has_nan = bool(np.any(np.isnan(preds.astype(float))))
                pred_arr = preds.astype(float)
                pred_range = float(pred_arr.max() - pred_arr.min()) if not has_nan else None
                results[side] = {
                    "has_nan": has_nan,
                    "prediction_range": round(pred_range, 4) if pred_range is not None else None,
                    "unique_outputs": int(len(np.unique(preds))),
                    "status": "NaN detected — UNSTABLE" if has_nan else "OK",
                }
            except Exception as inner_e:
                results[side] = {"error": str(inner_e), "status": "EXCEPTION"}

        return results
    except Exception as e:
        raise MetricComputationError(f"OOD boundary test failed: {e}")


def permutation_importance_analysis(model, X_val: pd.DataFrame, y_val: np.ndarray, n_repeats: int = 10) -> dict:
    """
    Randomly shuffle one feature at a time, measure performance drop.
    If single feature accounts for >60% drop → model fragile to corruption.
    """
    try:
        X_numeric = X_val.select_dtypes(include=[np.number])
        y_arr = np.array(y_val)

        scoring = "accuracy" if hasattr(model, "predict_proba") else "r2"
        result = permutation_importance(
            model, X_numeric.values, y_arr,
            n_repeats=n_repeats, random_state=42, scoring=scoring
        )
        importances = result.importances_mean
        total = importances.sum()

        fi_dict = {}
        fragile_features = []
        for col, imp in zip(X_numeric.columns, importances):
            frac = float(imp / total) if total > 0 else 0.0
            fi_dict[col] = {
                "importance": round(float(imp), 6),
                "fraction_of_total": round(frac, 4),
            }
            if frac > 0.60:
                fragile_features.append(col)

        ranked = dict(sorted(fi_dict.items(), key=lambda x: -x[1]["importance"]))

        return {
            "permutation_importances": ranked,
            "fragile_features": fragile_features,
            "warning": f"Model heavily relies on {', '.join(fragile_features)}. Single-feature corruption = total failure." if fragile_features else None,
        }
    except Exception as e:
        raise MetricComputationError(f"Permutation importance failed: {e}")
