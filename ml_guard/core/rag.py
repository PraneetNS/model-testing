import re
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def context_relevance(query: str, retrieved_chunks: List[str]) -> float:
    """Uses TF-IDF cosine similarity between query and each chunk, returns mean score."""
    if not retrieved_chunks or not query.strip():
        return 0.0
    try:
        vectorizer = TfidfVectorizer()
        tfidf_m = vectorizer.fit_transform([query] + retrieved_chunks)
        sim = cosine_similarity(tfidf_m[0:1], tfidf_m[1:]).flatten()
        return float(sim.mean())
    except ValueError:
        # Happens if vocab is empty
        return 0.0

def grounding_fidelity(answer: str, retrieved_chunks: List[str]) -> float:
    """What fraction of answer sentences have token overlap > 0.3 with at least one chunk."""
    if not answer.strip() or not retrieved_chunks:
        return 0.0
    
    # Split roughly by sentence endings
    sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', answer) if s.strip()]
    if not sentences: 
        return 0.0
    
    grounded_count = 0
    for sent in sentences:
        s_tokens = sent.lower().split()
        if not s_tokens: 
            continue
        is_grounded = False
        for chunk in retrieved_chunks:
            c_tokens = set(chunk.lower().split())
            overlap = set(s_tokens).intersection(c_tokens)
            if len(overlap) / len(s_tokens) > 0.3:
                is_grounded = True
                break
        if is_grounded:
            grounded_count += 1
            
    return float(grounded_count / len(sentences))

def retrieval_hit_rate(queries: List[str], retrieved_ids: List[List[str]], relevant_ids: List[List[str]]) -> float:
    """Standard hit@k metric. Fraction of queries with at least one relevant retrieved doc."""
    if not queries or not retrieved_ids or not relevant_ids:
        return 0.0
    hits = 0
    valid_queries = min(len(queries), len(retrieved_ids), len(relevant_ids))
    if valid_queries == 0: 
        return 0.0
    for i in range(valid_queries):
        if set(retrieved_ids[i]).intersection(set(relevant_ids[i])):
            hits += 1
    return float(hits / valid_queries)

def hallucination_risk(answer: str, retrieved_chunks: List[str]) -> str:
    """Returns 'high' if grounding_fidelity < 0.3, 'medium' if < 0.6, else 'low'."""
    fid = grounding_fidelity(answer, retrieved_chunks)
    if fid < 0.3:
        return "high"
    elif fid < 0.6:
        return "medium"
    return "low"
