import os
import sys
import pytest

# Ensure ml_guard is importable
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if _repo_root not in sys.path:
    sys.path.append(_repo_root)

from ml_guard.core.rag import (
    context_relevance,
    grounding_fidelity,
    retrieval_hit_rate,
    hallucination_risk
)

def test_context_relevance():
    query = "What is the capital of France?"
    chunks_high = [
        "The capital of France is Paris.",
        "Paris is the largest city in France."
    ]
    chunks_low = [
        "Bananas are yellow.",
        "The moon orbits the earth."
    ]
    
    score_high = context_relevance(query, chunks_high)
    score_low = context_relevance(query, chunks_low)
    
    assert score_high > score_low
    assert 0.0 <= score_high <= 1.0

def test_grounding_fidelity_high():
    answer = "The Eiffel Tower is in Paris. Paris is in France."
    chunks = [
        "The famous Eiffel Tower is located in Paris.",
        "Paris is the capital city of France."
    ]
    fid = grounding_fidelity(answer, chunks)
    assert fid >= 0.5  # High overlap

def test_grounding_fidelity_low():
    answer = "Mars is a red planet."
    chunks = [
        "Germany is located in Central Europe.",
        "Berlin is the capital of Germany."
    ]
    fid = grounding_fidelity(answer, chunks)
    assert fid < 0.3  # Very low overlap

def test_hallucination_risk():
    chunks = ["The stock market went up today due to tech stocks."]
    
    # Low risk (fully grounded)
    low_risk_answer = "The stock market went up today."
    assert hallucination_risk(low_risk_answer, chunks) == "low"
    
    # Medium risk (partially grounded)
    med_risk_answer = "The stock market went up today. Also gold went down."
    assert hallucination_risk(med_risk_answer, chunks) in ["low", "medium"]
    
    # High risk (not grounded)
    high_risk_answer = "The moon is made of cheese."
    assert hallucination_risk(high_risk_answer, chunks) == "high"

def test_retrieval_hit_rate():
    queries = ["q1", "q2", "q3"]
    retrieved = [["doc1", "doc2"], ["doc3"], ["doc4"]]
    relevant = [["doc1"], ["doc5"], []]
    
    # q1 has doc1 (hit)
    # q2 has doc3 but needs doc5 (miss)
    # q3 has doc4 but needs nothing? Wait empty relevant means miss?
    
    rate = retrieval_hit_rate(queries, retrieved, relevant)
    assert abs(rate - (1/3)) < 0.001
