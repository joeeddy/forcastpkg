"""Data models and schemas for the forecasting package."""

from pydantic import BaseModel, Field, ConfigDict, field_serializer
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum


class ModelType(str, Enum):
    """Supported forecasting model types."""
    ARIMA = "arima"
    LINEAR = "linear"
    MOVING_AVERAGE = "moving_average"
    ENSEMBLE = "ensemble"


class DataType(str, Enum):
    """Supported data types for forecasting."""
    STOCK = "stock"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"


class ForecastPoint(BaseModel):
    """Individual forecast point with confidence intervals."""
    date: datetime
    predicted_value: float
    confidence_interval_lower: Optional[float] = None
    confidence_interval_upper: Optional[float] = None
    
    @field_serializer('date')
    def serialize_date(self, value: datetime) -> str:
        return value.isoformat()


class ForecastResult(BaseModel):
    """Complete forecast result with metadata."""
    symbol: str
    model_type: ModelType
    forecast_period_days: int
    forecast_points: List[ForecastPoint]
    model_accuracy: Optional[float] = Field(None, ge=0, le=1)
    model_parameters: Optional[Dict[str, Any]] = None
    generated_at: datetime
    data_source: Optional[str] = None
    
    @field_serializer('generated_at')
    def serialize_generated_at(self, value: datetime) -> str:
        return value.isoformat()


class TechnicalIndicators(BaseModel):
    """Technical analysis indicators."""
    rsi: Optional[float] = Field(None, ge=0, le=100, description="Relative Strength Index")
    macd: Optional[float] = Field(None, description="MACD line value")
    macd_signal: Optional[float] = Field(None, description="MACD signal line")
    macd_histogram: Optional[float] = Field(None, description="MACD histogram")
    bollinger_upper: Optional[float] = Field(None, description="Bollinger Band upper bound")
    bollinger_lower: Optional[float] = Field(None, description="Bollinger Band lower bound")
    bollinger_middle: Optional[float] = Field(None, description="Bollinger Band middle line (SMA)")
    moving_average_10: Optional[float] = Field(None, description="10-day moving average")
    moving_average_20: Optional[float] = Field(None, description="20-day moving average")
    moving_average_50: Optional[float] = Field(None, description="50-day moving average")
    moving_average_200: Optional[float] = Field(None, description="200-day moving average")
    stochastic_k: Optional[float] = Field(None, ge=0, le=100, description="Stochastic %K")
    stochastic_d: Optional[float] = Field(None, ge=0, le=100, description="Stochastic %D")
    williams_r: Optional[float] = Field(None, ge=-100, le=0, description="Williams %R")
    cci: Optional[float] = Field(None, description="Commodity Channel Index")


class MarketData(BaseModel):
    """Market data point."""
    date: datetime
    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    low: float = Field(..., gt=0)
    close: float = Field(..., gt=0)
    volume: Optional[float] = Field(None, ge=0)
    adjusted_close: Optional[float] = Field(None, gt=0)
    
    @field_serializer('date')
    def serialize_date(self, value: datetime) -> str:
        return value.isoformat()
    
    def __post_init__(self):
        """Validate OHLC relationships."""
        if self.high < max(self.open, self.close):
            raise ValueError("High must be >= max(open, close)")
        if self.low > min(self.open, self.close):
            raise ValueError("Low must be <= min(open, close)")


class DataSourceConfig(BaseModel):
    """Configuration for data sources."""
    source_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    rate_limit: Optional[int] = Field(None, gt=0, description="Requests per minute")
    timeout: Optional[int] = Field(30, gt=0, description="Request timeout in seconds")
    retry_attempts: Optional[int] = Field(3, ge=0, description="Number of retry attempts")


class ForecastingConfig(BaseModel):
    """Configuration for forecasting models."""
    default_forecast_days: int = Field(30, gt=0)
    confidence_level: float = Field(0.95, gt=0, lt=1)
    min_data_points: int = Field(30, gt=0)
    max_data_points: int = Field(1000, gt=0)
    
    # ARIMA specific
    arima_auto_select: bool = Field(True, description="Auto-select ARIMA parameters")
    arima_max_p: int = Field(3, ge=0)
    arima_max_d: int = Field(2, ge=0)
    arima_max_q: int = Field(3, ge=0)
    
    # Linear regression specific
    linear_polynomial_degree: int = Field(1, ge=1, le=5)
    
    # Moving average specific
    moving_average_window: int = Field(20, gt=0)


class AnalysisResult(BaseModel):
    """Result of technical analysis."""
    symbol: str
    analysis_date: datetime
    technical_indicators: TechnicalIndicators
    signal_strength: Optional[str] = Field(None, description="Overall signal strength")
    confidence: Optional[float] = Field(None, ge=0, le=1)
    recommendations: Optional[List[str]] = None
    
    @field_serializer('analysis_date')
    def serialize_analysis_date(self, value: datetime) -> str:
        return value.isoformat()


class BacktestResult(BaseModel):
    """Result of model backtesting."""
    model_type: ModelType
    symbol: str
    test_period_start: datetime
    test_period_end: datetime
    accuracy_metrics: Dict[str, float]
    total_predictions: int
    successful_predictions: int
    average_error: float
    max_error: float
    min_error: float
    
    @field_serializer('test_period_start', 'test_period_end')
    def serialize_datetime_fields(self, value: datetime) -> str:
        return value.isoformat()