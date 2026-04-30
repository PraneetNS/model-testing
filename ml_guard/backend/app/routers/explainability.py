"""
Explainability Router.
Endpoints for computing and retrieving model explanations.
Runs SHAP synchronously in a thread so no Celery worker is required.
"""
import io
import joblib
import numpy as np
import pandas as pd
import uuid
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.session import get_db
from app.db.models import Model, ExplainabilityResult, ScanRecord
from app.core.auth import AuthContext, require_role

router = APIRouter()


@router.post("/explainability/compute")
async def compute_explainability(
    model_file: UploadFile = File(...),
    dataset_file: UploadFile = File(...),
    model_id: str = Form(""),
    max_samples: int = Form(50),   # Hard cap: faster results
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer")),
):
    """Compute SHAP explainability synchronously — no Celery required."""
    import asyncio

    # 1. Validate / create model record
    try:
        valid_model_id = str(uuid.UUID(model_id))
    except (ValueError, TypeError, AttributeError):
        valid_model_id = str(uuid.uuid4())

    model_record = (await db.execute(select(Model).filter(Model.id == valid_model_id))).scalars().first()
    if not model_record:
        dummy_model = Model(id=valid_model_id, name=f"Adhoc Explainer {valid_model_id[:6]}")
        db.add(dummy_model)
        await db.commit()

    # 2. Read uploaded files into memory
    model_bytes = await model_file.read()
    data_bytes = await dataset_file.read()

    # 3. Run SHAP in background thread (non-blocking)
    def _run_shap():
        import shap
        import tempfile, os

        # Load model
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp.write(model_bytes)
            tmp_path = tmp.name
        try:
            clf = joblib.load(tmp_path)
            if isinstance(clf, dict):
                clf = clf.get("model", clf.get("pipeline", list(clf.values())[0]))
        finally:
            os.unlink(tmp_path)

        # Load dataset
        try:
            df = pd.read_csv(io.BytesIO(data_bytes))
        except Exception:
            df = pd.read_csv(io.BytesIO(data_bytes), encoding="latin-1")

        if df.empty:
            raise ValueError("Dataset is empty.")

        feature_cols = [c for c in df.columns if c not in ("target", "label", "y")]
        X = pd.get_dummies(df[feature_cols])

        # --- Feature alignment to match model's expectations ---
        if getattr(clf, "feature_names_in_", None) is not None:
            # Exact named alignment (sklearn ≥ 1.0)
            expected = list(clf.feature_names_in_)
            for f in expected:
                if f not in X.columns:
                    X[f] = 0  # pad missing columns with zeros
            X = X[expected]
        elif getattr(clf, "n_features_in_", None) is not None:
            # Unnamed alignment via feature count
            n_expected = clf.n_features_in_
            if X.shape[1] < n_expected:
                # Pad missing columns with zeros
                for i in range(X.shape[1], n_expected):
                    X[f"__pad_{i}__"] = 0
            elif X.shape[1] > n_expected:
                # Truncate extra columns
                X = X.iloc[:, :n_expected]

        # Cap samples hard
        n_samples = min(max_samples, 50, len(X))
        X_sample = X.sample(n_samples, random_state=42) if len(X) > n_samples else X

        importance = None
        method = "unknown"

        # Strategy 1: TreeExplainer (RF, XGB, LightGBM — sub-second)
        try:
            explainer = shap.TreeExplainer(clf)
            sv = explainer.shap_values(X_sample)
            if isinstance(sv, list):
                importance = np.abs(sv[1]).mean(0) if len(sv) > 1 else np.abs(sv[0]).mean(0)
            else:
                importance = np.abs(sv).mean(0)
            method = "shap_tree"
        except Exception:
            pass

        # Strategy 2: LinearExplainer (linear models — fast)
        if importance is None:
            try:
                bg = shap.maskers.Independent(X_sample, max_samples=25)
                explainer = shap.LinearExplainer(clf, bg)
                sv = explainer.shap_values(X_sample)
                importance = np.abs(sv).mean(0)
                method = "shap_linear"
            except Exception:
                pass

        # Strategy 3: Permutation importance (always fast, any model)
        if importance is None:
            baseline = clf.predict(X_sample.values)
            imps = []
            for i in range(X_sample.shape[1]):
                Xp = X_sample.values.copy()
                np.random.shuffle(Xp[:, i])
                pp = clf.predict(Xp)
                imps.append(float(np.mean(np.abs(pp.astype(float) - baseline.astype(float)))))
            importance = np.array(imps)
            method = "permutation"

        feat_imp = sorted(
            [{"feature": name, "importance": float(imp)} for name, imp in zip(X.columns, importance)],
            key=lambda x: x["importance"],
            reverse=True
        )[:10]

        # Interpretability score (0–100) based on concentration of top feature
        top_sum = sum(f["importance"] for f in feat_imp[:3])
        total_sum = sum(f["importance"] for f in feat_imp) or 1
        interp_score = round(min(100, (top_sum / total_sum) * 100 + 30), 1)

        return {
            "method": method,
            "feature_importance": feat_imp,
            "top_features": [f["feature"] for f in feat_imp[:5]],
            "interpretability_score": interp_score,
        }

    try:
        results = await asyncio.to_thread(_run_shap)
    except Exception as e:
        raise HTTPException(500, f"SHAP computation failed: {str(e)}")

    # 4. Persist result
    result_record = ExplainabilityResult(
        model_id=valid_model_id,
        method=results["method"],
        global_importance=results["feature_importance"],
        summary_metrics={
            "interpretability_score": results["interpretability_score"],
            "top_features": results["top_features"],
            "status": "success",
            "n_samples": min(max_samples, 50),
        },
    )
    db.add(result_record)

    scan_rec = ScanRecord(
        model_id=valid_model_id,
        scan_type="explainability",
        checks_run=["shap_attribution"],
        results_json={
            "metrics": {"interpretability_score": results["interpretability_score"]},
            "method": results["method"],
            "feature_importance": results["feature_importance"],
        },
        governance_score=results["interpretability_score"],
        gate_status="PASSED" if results["interpretability_score"] >= 40 else "WARNING",
        trigger_source="explainability_sync",
    )
    db.add(scan_rec)
    await db.commit()

    return {
        "status": "completed",
        "model_id": valid_model_id,
        "scan_id": str(scan_rec.id),
        "method": results["method"],
        "feature_importance": results["feature_importance"],
        "interpretability_score": results["interpretability_score"],
        "top_features": results["top_features"],
    }


@router.get("/explainability/{model_id}")
async def get_explainability(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("viewer")),
):
    try:
        valid_model_id = str(uuid.UUID(model_id))
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(400, "Invalid explainability model_id format.")

    """Get stored explainability results for a model."""
    stmt = select(ExplainabilityResult).filter(
        ExplainabilityResult.model_id == valid_model_id
    ).order_by(ExplainabilityResult.created_at.desc())
    res = await db.execute(stmt)
    results = res.scalars().all()

    if not results:
        raise HTTPException(404, "No explainability results found for this model.")

    return {
        "model_id": model_id,
        "results": [
            {
                "id": str(r.id),
                "method": r.method,
                "global_importance": r.global_importance,
                "summary_metrics": r.summary_metrics,
                "created_at": str(r.created_at),
            }
            for r in results
        ],
    }
