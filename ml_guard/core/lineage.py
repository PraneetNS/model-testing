import networkx as nx
from typing import Any

async def build_lineage_graph(model_id: str, db: Any) -> nx.DiGraph:
    """
    Builds the full lineage DAG (ancestors and descendants) for `model_id`.
    Returns a networkx DiGraph.
    """
    from sqlalchemy import select
    from app.db.models import Model, ScanRecord

    # Fetch all models (this could be optimized with a recursive CTE for large DBs)
    res = await db.execute(
        select(Model.id, Model.name, Model.version, Model.parent_model_id, Model.created_at)
    )
    all_models = res.all()

    # Fetch latest scans for scores/verdict
    scans_res = await db.execute(
        select(ScanRecord.model_id, ScanRecord.governance_score, ScanRecord.gate_status)
        .order_by(ScanRecord.created_at.desc())
    )
    latest_scans = {}
    for scan in scans_res.all():
        m_id = str(scan.model_id)
        if m_id not in latest_scans:
            latest_scans[m_id] = {
                "score": scan.governance_score,
                "verdict": scan.gate_status
            }

    G = nx.DiGraph()

    for m in all_models:
        m_id = str(m.id)
        scan = latest_scans.get(m_id, {})
        G.add_node(
            m_id,
            id=m_id,
            name=m.name,
            version=m.version,
            score=scan.get("score"),
            verdict=scan.get("verdict"),
            created_at=m.created_at.isoformat() if m.created_at else None
        )
        if m.parent_model_id:
            G.add_edge(str(m.parent_model_id), m_id)

    # Now we only want the component connected to model_id
    if str(model_id) not in G:
        return nx.DiGraph()

    # NetworkX to get weakly connected components
    for comp in nx.weakly_connected_components(G):
        if str(model_id) in comp:
            return G.subgraph(comp).copy()

    return nx.DiGraph()
