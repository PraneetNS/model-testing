import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import select, and_
from sklearn.linear_model import LinearRegression
from app.db.models import ScanRecord, Model
from .models import ForecastResult, ForecastPoint
import structlog

# Try to import Prophet, fallback to None if not installed
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

logger = structlog.get_logger()

class GovernanceForecaster:
    """
    Forecasting service for ML governance metrics.
    Predicts future breaches using Prophet or Linear Regression.
    """
    
    METRIC_MAP = {
        "psi": "max_psi",
        "bias_score": "bias_risk_score",
        "hallucination_rate": "hallucination_risk",
        "accuracy": "accuracy"
    }

    def __init__(self, db: AsyncSession, model_id: str):
        self.db = db
        self.model_id = model_id

    def _get_historical_data(self, metric: str) -> pd.DataFrame:
        """
        Pull historical metric values from ScanRecord results_json.
        """
        # Fetch scan records for this model
        scans = self.db.query(ScanRecord).filter(
            ScanRecord.model_id == self.model_id
        ).order_by(ScanRecord.created_at.asc()).all()

        data = []
        for s in scans:
            # Check results_json or other fields depending on metric
            val = None
            res = s.results_json or {}
            
            if metric == "accuracy":
                val = res.get("metrics", {}).get("accuracy")
            elif metric == "psi":
                # Get max PSI from drift dict
                drift = res.get("drift", {})
                psi_vals = [v.get("PSI", 0) for v in drift.values() if isinstance(v, dict)]
                val = max(psi_vals) if psi_vals else None
            elif metric == "bias_score":
                val = s.fairness_risk_score
            elif metric == "hallucination_rate":
                val = res.get("llm_evaluation", {}).get("hallucination_risk")

            if val is not None:
                # Ensure timezone-naive for Prophet
                ds_naive = s.created_at.replace(tzinfo=None) if s.created_at.tzinfo else s.created_at
                data.append({"ds": ds_naive, "y": float(val)})

        df = pd.DataFrame(data)
        if not df.empty:
            df['ds'] = pd.to_datetime(df['ds'])
            df['y'] = df['y'].astype(float)
        return df

    def _get_threshold(self, metric: str) -> float:
        """
        Retrieve policy threshold for the metric.
        In a real app, this would come from the active PolicyVersion.
        """
        # Default fallback thresholds
        thresholds = {
            "psi": 0.20,
            "bias_score": 0.20,
            "hallucination_rate": 0.10,
            "accuracy": 0.85
        }
        return thresholds.get(metric, 0.5)

    def forecast_metric(self, metric: str, horizon_days: int = 30) -> ForecastResult:
        """
        Generate forecast for a specific metric.
        """
        df = self._get_historical_data(metric)
        threshold = self._get_threshold(metric)
        
        if len(df) < 3:
            return ForecastResult(
                model_id=self.model_id,
                metric=metric,
                forecast_points=[],
                trend="UNKNOWN",
                recommendation="Insufficient historical data (min 3 points required).",
                status="INSUFFICIENT_DATA"
            )

        # 1. Choose Model
        if len(df) >= 10 and PROPHET_AVAILABLE:
            forecast_df = self._run_prophet(df, horizon_days)
        else:
            forecast_df = self._run_linear_regression(df, horizon_days)

        # 2. Extract Forecast Points
        points = []
        for _, row in forecast_df.iterrows():
            points.append(ForecastPoint(
                date=row['ds'].strftime('%Y-%m-%d'),
                value=float(row['yhat']),
                lower=float(row.get('yhat_lower', row['yhat'] * 0.95)),
                upper=float(row.get('yhat_upper', row['yhat'] * 1.05))
            ))

        # 3. Compute Breach Date
        breach_date, confidence = self._compute_breach(forecast_df, threshold, metric)
        
        # 4. Determine Trend
        trend, slope = self._get_trend_info(forecast_df)
        
        # 5. Recommendation logic
        rec = self._get_recommendation(metric, trend, breach_date)

        return ForecastResult(
            model_id=self.model_id,
            metric=metric,
            forecast_points=points,
            breach_date=breach_date,
            breach_confidence=confidence,
            trend=trend,
            recommendation=rec,
            status="SUCCESS"
        )

    def _run_prophet(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Fit Prophet model."""
        m = Prophet(
            daily_seasonality=True,
            weekly_seasonality=False,
            yearly_seasonality=False
        )
        m.fit(df)
        future = m.make_future_dataframe(periods=horizon)
        forecast = m.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    def _run_linear_regression(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Fallback to simple Linear Regression."""
        # Convert dates to ordinal for sklearn
        df['ds_ord'] = df['ds'].apply(lambda x: x.toordinal())
        X = df[['ds_ord']].values
        y = df['y'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict future
        last_date = df['ds'].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, horizon + 1)]
        all_dates = list(df['ds']) + future_dates
        
        X_all = np.array([[d.toordinal()] for d in all_dates])
        y_pred = model.predict(X_all)
        
        # Simple CI calculation (1.96 * std of residuals)
        residuals = y - model.predict(X)
        std_resid = np.std(residuals) if len(residuals) > 1 else 0.05
        
        return pd.DataFrame({
            'ds': all_dates,
            'yhat': y_pred,
            'yhat_lower': y_pred - (1.96 * std_resid),
            'yhat_upper': y_pred + (1.96 * std_resid)
        })

    def _compute_breach(self, df: pd.DataFrame, threshold: float, metric: str) -> Tuple[Optional[str], float]:
        """Find when yhat crosses threshold."""
        forecast_only = df[df['ds'] > datetime.now()]
        
        # For accuracy, breach is BELOW threshold. For others, breach is ABOVE.
        is_breached = False
        breach_row = None
        
        if metric == "accuracy":
            breach_df = forecast_only[forecast_only['yhat'] < threshold]
        else:
            breach_df = forecast_only[forecast_only['yhat'] > threshold]
            
        if not breach_df.empty:
            breach_row = breach_df.iloc[0]
            # Simple confidence based on how far we are from today
            days_to_breach = (breach_row['ds'] - datetime.now()).days
            confidence = max(0.1, 1.0 - (days_to_breach / 60)) # Lower confidence for far future
            return breach_row['ds'].strftime('%Y-%m-%d'), float(confidence)
            
        return None, 0.0

    def _get_trend_info(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Detect Trend."""
        if len(df) < 2: return "STABLE", 0.0
        
        # Recent slope (last 7 days of forecast)
        y = df['yhat'].values
        slope = (y[-1] - y[0]) / len(y)
        
        if abs(slope) < 0.001: return "STABLE", float(slope)
        return ("IMPROVING" if slope < 0 else "DEGRADING"), float(slope)

    def _get_recommendation(self, metric: str, trend: str, breach_date: Optional[str]) -> str:
        if breach_date:
            return f"CRITICAL: {metric.upper()} expected to breach threshold on {breach_date}. Schedule retraining immediately."
        if trend == "DEGRADING":
            return f"WARNING: {metric.upper()} shows a degrading trend. Monitor closely."
        return f"SYSTEM HEALTHY: {metric.upper()} is stable or improving."
