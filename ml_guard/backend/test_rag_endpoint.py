import requests

res = requests.post("http://127.0.0.1:8000/api/v1/rag-eval/default/log", json={
    "query": "What is the capital of France?",
    "answer": "The capital of France is Paris.",
    "retrieved_chunks": ["Paris is the capital of France."],
    "retrieved_doc_ids": ["doc1"]
})
print("Log Response:", res.json())

res2 = requests.get("http://127.0.0.1:8000/api/v1/rag-eval/default/report")
print("Report Response:", res2.json())
