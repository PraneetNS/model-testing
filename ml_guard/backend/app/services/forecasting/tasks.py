from app.core.celery_app import celery_app
from app.db.session import SessionLocal
from app.db.models import Model, AlertEvent, AlertRule
from app.services.forecasting.forecaster import GovernanceForecaster
from datetime import datetime, timedelta
import structlog

logger = structlog.get_logger()

@celery_app.task(name="app.services.forecasting.recompute_all_forecasts")
def recompute_all_forecasts():
    """
    Background job to update forecasts for all active models.
    Every 6 hours.
    """
    db = SessionLocal()
    try:
        active_models = db.query(Model).all()
        logger.info("Starting background forecasting job", model_count=len(active_models))
        
        for model in active_models:
            forecaster = GovernanceForecaster(db, str(model.id))
            metrics = ["psi", "bias_score", "hallucination_rate", "accuracy"]
            
            for metric in metrics:
                try:
                    res = forecaster.forecast_metric(metric)
                    
                    if res.breach_date:
                        breach_dt = datetime.strptime(res.breach_date, '%Y-%m-%d')
                        days_to_breach = (breach_dt - datetime.now()).days
                        
                        # Trigger alert if breach is within 7 days
                        if 0 <= days_to_breach <= 7:
                            _trigger_breach_alert(db, model, metric, res.breach_date)
                            
                except Exception as e:
                    logger.error("Forecast computation failed for model/metric", 
                                 model_id=str(model.id), metric=metric, error=str(e))
        
        db.commit()
    finally:
        db.close()

def _trigger_breach_alert(db, model, metric, breach_date):
    """Integrate with existing alert system."""
    # Find a reasonable alert rule for this model
    rule = db.query(AlertRule).filter(AlertRule.is_active == True).first()
    if not rule:
        # Create a dummy rule just so we can log the event if none exists
        rule = AlertRule(name="Default Breach Predictor", condition={}, channels=["ui"])
        db.add(rule)
        db.flush()

    alert_msg = f"PREDICTIVE BREACH WARNING: Model '{model.name}' is forecast to breach {metric.upper()} policy ceiling on {breach_date}."
    
    # Check if we already alerted for this recently to avoid spam (within last 24h)
    existing = db.query(AlertEvent).filter(
        AlertEvent.message == alert_msg,
        AlertEvent.created_at > datetime.now() - timedelta(hours=24)
    ).first()
    
    if not existing:
        event = AlertEvent(
            rule_id=rule.id,
            severity="CRITICAL",
            message=alert_msg,
            delivered=False
        )
        db.add(event)
        logger.info("Predictive alert triggered", model_id=str(model.id), metric=metric)
