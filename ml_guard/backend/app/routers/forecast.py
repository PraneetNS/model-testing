from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.session import SessionLocal
from app.db.models import Model
from app.services.forecasting.forecaster import GovernanceForecaster
from app.services.forecasting.models import ForecastResult, ModelForecastSummary
import redis
import json
import os

router = APIRouter()

# Simple Redis setup (pull from settings if possible, otherwise assume standard localhost)
r = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, db=0)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/{model_id}", response_model=ModelForecastSummary)
async def get_model_forecast(model_id: str, db: AsyncSession = Depends(get_db)):
    """
    Get governance forecasts for a specific model.
    Cached in Redis for 1 hour.
    """
    cache_key = f"forecast:{model_id}"
    cached = r.get(cache_key)
    if cached:
        return ModelForecastSummary(**json.loads(cached))

    # Check if model exists
    model = (await db.execute(select(Model).filter(Model.id == model_id))).scalars().first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    forecaster = GovernanceForecaster(db, model_id)
    metrics = ["psi", "bias_score", "hallucination_rate", "accuracy"]
    
    forecasts = {}
    for m in metrics:
        forecasts[m] = forecaster.forecast_metric(m)

    # Generate summary
    summary_text = _generate_summary(forecasts)
    
    result = ModelForecastSummary(
        model_id=model_id,
        summary=summary_text,
        forecasts=forecasts
    )
    
    # Save to Redis with 1h TTL
    r.setex(cache_key, 3600, result.json())
    
    return result

@router.get("/{model_id}/summary")
async def get_model_forecast_summary(model_id: str, db: AsyncSession = Depends(get_db)):
    """Plain-English summary for business stakeholders."""
    res = await get_model_forecast(model_id, db)
    return {"model_id": model_id, "summary": res.summary}

def _generate_summary(forecasts: dict) -> str:
    """Consolidate metric findings into a single readable paragraph."""
    parts = []
    
    # Check for immediate breaches
    breaches = [f"{m} (expected {f.breach_date})" for m, f in forecasts.items() if f.breach_date]
    if breaches:
        parts.append(f"CRITICAL: {', '.join(breaches)} metrics are forecast to breach thresholds. Immediate intervention required.")
    
    # Check for negative trends
    degrading = [m for m, f in forecasts.items() if f.trend == "DEGRADING" and not f.breach_date]
    if degrading:
        parts.append(f"WARNING: Potential degradation detected in {', '.join(degrading)} metrics. Monitor risk trajectory.")

    # All good case
    if not parts:
        parts.append("SYSTEM STATUS: All governing metrics stable or improving. No breaches predicted in the 30-day horizon.")
        
    return " ".join(parts)
