"""Data models for WebScraper App."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class MarketDataResponse(BaseModel):
    """Market data response model."""
    symbol: str
    current_price: float
    price_change_24h: float
    price_change_percentage_24h: float
    market_cap: Optional[float] = None
    volume_24h: Optional[float] = None
    timestamp: datetime

class NewsArticle(BaseModel):
    """News article model."""
    title: str
    summary: Optional[str] = None
    url: str
    published_at: datetime
    source: str
    sentiment_score: Optional[float] = None

class NewsResponse(BaseModel):
    """News response model."""
    articles: List[NewsArticle]
    total_count: int
    source_type: str

class WeatherData(BaseModel):
    """Weather data model."""
    temperature: float
    humidity: Optional[float] = None
    wind_speed: Optional[float] = None
    weather_description: str
    timestamp: datetime
    location: str

class ForecastPoint(BaseModel):
    """Individual forecast point."""
    date: datetime
    predicted_value: float
    confidence_interval_lower: Optional[float] = None
    confidence_interval_upper: Optional[float] = None

class ForecastResponse(BaseModel):
    """Forecast response model."""
    symbol: str
    model_type: str
    forecast_period_days: int
    forecast_points: List[ForecastPoint]
    model_accuracy: Optional[float] = None
    generated_at: datetime

class TechnicalIndicators(BaseModel):
    """Technical indicators model."""
    rsi: Optional[float] = None
    macd: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    moving_average_20: Optional[float] = None
    moving_average_50: Optional[float] = None

class PiCycleAnalysis(BaseModel):
    """Pi cycle analysis model."""
    symbol: str
    current_ratio: float
    signal_strength: str = Field(..., description="Strong Buy, Buy, Hold, Sell, Strong Sell")
    days_to_next_cycle: Optional[int] = None
    confidence: float = Field(..., ge=0, le=1)
    analysis_date: datetime

class CapitalFlowAnalysis(BaseModel):
    """Capital flow analysis model."""
    symbol: str
    net_flow_24h: float
    inflow_strength: str
    outflow_strength: str
    flow_trend: str = Field(..., description="Bullish, Bearish, Neutral")
    technical_indicators: TechnicalIndicators
    analysis_date: datetime