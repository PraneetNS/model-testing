from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import random
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(request: Request):
    """
    Mock inference endpoint for ML Guard testing.
    """
    body = await request.json()
    # Support both {"features": {...}} and flat {...} payloads
    features = body.get("features", body) if isinstance(body, dict) else {}
    
    # Simulate processing time (latency)
    process_time = random.uniform(5, 50) / 1000 # 5ms - 50ms
    time.sleep(process_time)
    
    # Simple ML Logic simulation:
    income = float(features.get("annual_income", 0))
    credit = float(features.get("credit_score", 0))
    
    prediction = 1 if (income > 50000 and credit > 650) else 0
    confidence = random.uniform(0.7, 0.99)
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "latency_ms": int(process_time * 1000),
        "status": "success",
        "model_id": "MOCK-AUDIT-001"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
