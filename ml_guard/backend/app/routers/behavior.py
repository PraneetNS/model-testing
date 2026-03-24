import io
import os
import sys
import tempfile
import numpy as np
import pandas as pd
import joblib
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))
from ml_guard.core.sensitivity import (
    sensitivity_analysis,
    monte_carlo_stability,
    ood_boundary_test,
    permutation_importance_analysis,
)

from app.db.session import get_db
from sqlalchemy.orm import Session
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from app.core.auth import AuthContext, require_engineer, log_action

# ML Guard core path injection
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if _repo_root not in sys.path:
    sys.path.append(_repo_root)

from ml_guard.core import ONNXModelWrapper

router = APIRouter()

SCENARIO_REGISTRY = {
    "sensitivity_analysis":      "Finite-difference partial derivative approximation per feature (Δy/Δx)",
    "monte_carlo_stability":     "Inject Gaussian noise N(0, σ) 100 times. Measure flip rate and stability score.",
    "ood_boundary_test":         "Synthetic extremes: min-3σ and max+3σ. Check for NaN/explosion.",
    "adversarial_permutation":   "Permutation importance: shuffle each feature, measure performance drop.",
    "noise_perturbation":        "Classic Gaussian noise scenario for a single run.",
    "extreme_values":            "Feed feature min and max as uniform row values.",
    "missing_data_injection":    "30% NaN injection, imputed with column mean.",
    "boundary_inputs":           "Predict at 5th and 95th percentiles of each feature.",
    "adversarial_shifts":        "Shift all features by +2 standard deviations.",
}


def _load_model(model_bytes: bytes, filename: str):
    suffix = ".onnx" if filename.lower().endswith(".onnx") else ".pkl"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(model_bytes)
        tmp_path = tmp.name
    try:
        if suffix == ".onnx":
            return ONNXModelWrapper(tmp_path)
        return joblib.load(tmp_path)
    except Exception as e:
        raise HTTPException(422, f"ModelLoadError: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _predict_safe(model, X_vals):
    try:
        return model.predict(X_vals)
    except Exception as e:
        raise HTTPException(422, f"model.predict() failed: {e}")


@router.get("/behavior/scenarios")
async def list_scenarios():
    return {"scenarios": list(SCENARIO_REGISTRY.keys()), "descriptions": SCENARIO_REGISTRY}


@router.post("/behavior/test")
def run_behavior_test(
    model_file:     UploadFile = File(...),
    reference_file: UploadFile = File(None),
    ref_dataset_url: str = Form(None),
    val_file:       UploadFile = File(None),   # Optional — needed for permutation importance
    scenarios:      str = Form(...),
    label_col:      str = Form("target"),
    auth:           AuthContext = Depends(require_engineer),
    db:             Session = Depends(get_db),
):
    from app.services.storage_service import download_from_url
    
    model_bytes = model_file.file.read()
    model = _load_model(model_bytes, model_file.filename)

    if reference_file:
        ref_bytes = reference_file.file.read()
        filename = reference_file.filename
    elif ref_dataset_url:
        ref_bytes = download_from_url(ref_dataset_url)
        filename = ref_dataset_url
    else:
        raise HTTPException(400, "Provide either a reference file or a reference dataset URL.")

    try:
        if filename.lower().endswith(".parquet"):
            df_ref = pd.read_parquet(io.BytesIO(ref_bytes))
        else:
            try:
                # Try standard UTF-8/Default
                df_ref = pd.read_csv(io.BytesIO(ref_bytes))
            except (UnicodeDecodeError, pd.errors.ParserError):
                # Fallback to Latin-1
                df_ref = pd.read_csv(io.BytesIO(ref_bytes), encoding='latin-1')
    except Exception as e:
        raise HTTPException(422, f"Reference dataset parse failed: {e}")

    if isinstance(model, dict):
        model = model.get("model", model.get("classifier", list(model.values())[0]))

    y_ref = None
    if label_col in df_ref.columns:
        y_ref = df_ref[label_col].values
        df_ref = df_ref.drop(columns=[label_col])

    # Smart Feature Alignment & Encoding
    X_ref_df = pd.get_dummies(df_ref)
    if getattr(model, "feature_names_in_", None) is not None:
        expected_feats = list(model.feature_names_in_)
        for f in expected_feats:
            if f not in X_ref_df.columns:
                X_ref_df[f] = 0
        X_ref = X_ref_df[expected_feats]
    else:
        X_ref = X_ref_df.select_dtypes(include=[np.number])
        if X_ref.shape[1] == 0:
            raise HTTPException(422, "No numeric features mapped.")

    selected = [s.strip() for s in scenarios.split(",") if s.strip()]
    if not selected:
        raise HTTPException(400, "Select at least one scenario.")

    baseline_preds = _predict_safe(model, X_ref.values).astype(float)
    baseline_variance = float(np.var(baseline_preds))
    stress_results = {}

    # ─── Scientific scenarios ───
    if "sensitivity_analysis" in selected:
        try:
            stress_results["sensitivity_analysis"] = sensitivity_analysis(model, X_ref)
        except Exception as e:
            stress_results["sensitivity_analysis"] = {"error": str(e)}

    if "monte_carlo_stability" in selected:
        try:
            stress_results["monte_carlo_stability"] = monte_carlo_stability(model, X_ref)
        except Exception as e:
            stress_results["monte_carlo_stability"] = {"error": str(e)}

    if "ood_boundary_test" in selected:
        try:
            stress_results["ood_boundary_test"] = ood_boundary_test(model, X_ref)
        except Exception as e:
            stress_results["ood_boundary_test"] = {"error": str(e)}

    if "adversarial_permutation" in selected:
        if y_ref is None:
            stress_results["adversarial_permutation"] = {"error": "Label column required for permutation importance."}
        else:
            try:
                stress_results["adversarial_permutation"] = permutation_importance_analysis(model, X_ref, y_ref)
            except Exception as e:
                stress_results["adversarial_permutation"] = {"error": str(e)}

    # ─── Classic structural scenarios ───
    for scenario in selected:
        if scenario in ("sensitivity_analysis", "monte_carlo_stability", "ood_boundary_test", "adversarial_permutation"):
            continue

        try:
            df_mod = X_ref.copy()

            if scenario == "extreme_values":
                for col in df_mod.columns:
                    df_mod[col] = X_ref[col].min()
                low = _predict_safe(model, df_mod.values).astype(float)
                for col in df_mod.columns:
                    df_mod[col] = X_ref[col].max()
                high = _predict_safe(model, df_mod.values).astype(float)
                preds = np.concatenate([low, high])

            elif scenario == "missing_data_injection":
                mask = np.random.rand(*df_mod.shape) < 0.3
                df_mod[mask] = np.nan
                df_mod = df_mod.fillna(df_mod.mean())
                preds = _predict_safe(model, df_mod.values).astype(float)

            elif scenario == "noise_perturbation":
                for col in df_mod.columns:
                    std = float(X_ref[col].std()) if X_ref[col].std() > 0 else 1.0
                    df_mod[col] += np.random.normal(0, 0.1 * std, size=len(df_mod))
                preds = _predict_safe(model, df_mod.values).astype(float)

            elif scenario == "boundary_inputs":
                for col in df_mod.columns:
                    df_mod[col] = X_ref[col].quantile(0.05)
                low = _predict_safe(model, df_mod.values).astype(float)
                for col in df_mod.columns:
                    df_mod[col] = X_ref[col].quantile(0.95)
                high = _predict_safe(model, df_mod.values).astype(float)
                preds = np.concatenate([low, high])

            elif scenario == "adversarial_shifts":
                for col in df_mod.columns:
                    std = float(X_ref[col].std()) if X_ref[col].std() > 0 else 1.0
                    df_mod[col] += 2.0 * std
                preds = _predict_safe(model, df_mod.values).astype(float)

            else:
                stress_results[scenario] = {"error": "Unknown scenario"}
                continue

            variant_var = float(np.var(preds))
            unique, counts = np.unique(preds, return_counts=True)
            stress_results[scenario] = {
                "n_predictions":      len(preds),
                "output_variance":    round(variant_var, 6),
                "variance_change":    round(abs(variant_var - baseline_variance), 6),
                "prediction_distribution": {str(k): int(v) for k, v in zip(unique, counts)},
                "stability_flag":     "STABLE" if abs(variant_var - baseline_variance) < 0.1 else "UNSTABLE",
            }

        except HTTPException:
            raise
        except Exception as e:
            stress_results[scenario] = {"error": str(e)}

    # ─── Robustness score ───
    mc = stress_results.get("monte_carlo_stability", {})
    if isinstance(mc, dict) and "stability_score" in mc:
        robustness_score = round(mc["stability_score"] * 100, 2)
    else:
        unstable = sum(
            1 for k, v in stress_results.items()
            if isinstance(v, dict) and v.get("stability_flag") == "UNSTABLE"
        )
        robustness_score = max(0, 100 - unstable * 20)

    # ─── Log action ───
    log_action(db, auth, "behavior.test", resource_type="model", details={"scenarios": selected, "score": robustness_score})

    return {
        "baseline_variance": baseline_variance,
        "stress_results":    stress_results,
        "robustness_score":  robustness_score,
        "status": "PASSED" if robustness_score >= 60 else "FAILED",
    }
