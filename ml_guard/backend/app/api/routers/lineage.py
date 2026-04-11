from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Dict, Any
import networkx as nx

from app.db.session import get_db
from app.db.models import Model, ScanRecord
from ml_guard.core.lineage import build_lineage_graph

router = APIRouter(prefix="/api/lineage", tags=["lineage"])

@router.post("/{model_id}/set-parent")
async def set_parent(model_id: str, parent_id: str, db: AsyncSession = Depends(get_db)):
    model = (await db.execute(select(Model).filter(Model.id == model_id))).scalars().first()
    if not model: raise HTTPException(404, "Model not found")
    
    parent = (await db.execute(select(Model).filter(Model.id == parent_id))).scalars().first()
    if not parent: raise HTTPException(404, "Parent not found")

    # Check depth / circular
    current_parent_id = parent.id
    depth = 0
    while current_parent_id:
        if str(current_parent_id) == str(model.id):
            raise HTTPException(400, "Circular reference detected")
        depth += 1
        if depth > 10:
            raise HTTPException(400, "Max depth exceeded")
        p_model = (await db.execute(select(Model).filter(Model.id == current_parent_id))).scalars().first()
        if not p_model: break
        current_parent_id = p_model.parent_model_id

    model.parent_model_id = parent.id
    await db.commit()
    return {"status": "ok", "parent_model_id": str(parent.id)}


@router.get("/{model_id}/graph")
async def get_graph(model_id: str, db: AsyncSession = Depends(get_db)):
    G = await build_lineage_graph(model_id, db)
    nodes = []
    edges = []
    for node, data in G.nodes(data=True):
        nodes.append(data)
    for u, v in G.edges():
        edges.append({"parent_id": str(u), "child_id": str(v)})
    
    return {"nodes": nodes, "edges": edges}


def extract_dim_score(scan_rec, dim: str) -> float:
    if not scan_rec: return 0.0
    res = scan_rec.results_json or {}
    
    # attempt to find in top level or inside Component Scores
    if dim == "fairness":
        if "fairness_metrics" in res: return float(res["fairness_metrics"].get("overall_fairness", 0.0) * 100)
        return float(res.get("component_scores", {}).get("fairness_score", res.get("fairness_score", 0.0)))
    elif dim == "performance":
        return float(res.get("component_scores", {}).get("performance_score", res.get("performance_score", 0.0)))
    elif dim == "security":
        return float(res.get("component_scores", {}).get("security_score", res.get("security_score", 0.0)))
    elif dim == "behavioral":
        return float(res.get("component_scores", {}).get("behavioral_score", res.get("behavioral_score", 0.0)))
    return 0.0


@router.get("/{model_id}/diff/{other_model_id}")
async def diff_models(model_id: str, other_model_id: str, db: AsyncSession = Depends(get_db)):
    scan_a = (await db.execute(select(ScanRecord).filter(ScanRecord.model_id == model_id).order_by(ScanRecord.created_at.desc()))).scalars().first()
    scan_b = (await db.execute(select(ScanRecord).filter(ScanRecord.model_id == other_model_id).order_by(ScanRecord.created_at.desc()))).scalars().first()

    dimensions = ["fairness", "performance", "security", "behavioral"]
    diffs = []
    for dim in dimensions:
        score_a = extract_dim_score(scan_a, dim)
        score_b = extract_dim_score(scan_b, dim)
        delta = score_a - score_b
        
        direction = "unchanged"
        if delta > 0: direction = "improved"
        elif delta < 0: direction = "degraded"
        
        diffs.append({
            "dimension": dim,
            "score_a": score_a,
            "score_b": score_b,
            "delta": delta,
            "direction": direction
        })
    return diffs
