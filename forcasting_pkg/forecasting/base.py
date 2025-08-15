"""Base classes for forecasting models."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ..models import ForecastResult, ForecastPoint, MarketData, ModelType, ForecastingConfig


class BaseForecaster(ABC):
    """Abstract base class for all forecasting models."""
    
    def __init__(self, config: Optional[ForecastingConfig] = None):
        """Initialize the forecaster with configuration."""
        self.config = config or ForecastingConfig()
        self.model = None
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, data: Union[List[MarketData], pd.DataFrame]) -> 'BaseForecaster':
        """
        Fit the forecasting model to historical data.
        
        Args:
            data: Historical market data
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, steps: int = None) -> ForecastResult:
        """
        Generate forecast predictions.
        
        Args:
            steps: Number of time steps to forecast
            
        Returns:
            Forecast result with predictions and confidence intervals
        """
        pass
    
    @abstractmethod
    def get_model_parameters(self) -> Dict[str, Any]:
        """Get the parameters of the fitted model."""
        pass
    
    def _prepare_data(self, data: Union[List[MarketData], pd.DataFrame]) -> pd.DataFrame:
        """Prepare and validate input data."""
        if isinstance(data, list):
            # Convert list of MarketData to DataFrame
            df = pd.DataFrame([item.dict() for item in data])
        else:
            df = data.copy()
        
        # Ensure required columns exist
        required_columns = ['date', 'close']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Convert date column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date and remove duplicates
        df = df.sort_values('date').drop_duplicates(subset=['date'])
        
        # Validate data length
        if len(df) < self.config.min_data_points:
            raise ValueError(f"Insufficient data points. Need at least {self.config.min_data_points}, got {len(df)}")
        
        # Limit data points if too many
        if len(df) > self.config.max_data_points:
            df = df.tail(self.config.max_data_points)
        
        return df
    
    def _validate_fitted(self):
        """Ensure the model is fitted before prediction."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions. Call fit() first.")
    
    def _calculate_accuracy(self, actual: pd.Series, predicted: pd.Series) -> float:
        """Calculate forecast accuracy using MAPE (Mean Absolute Percentage Error)."""
        try:
            # Remove any NaN values
            mask = ~(np.isnan(actual) | np.isnan(predicted))
            actual_clean = actual[mask]
            predicted_clean = predicted[mask]
            
            if len(actual_clean) == 0:
                return 0.0
            
            # Calculate MAPE
            mape = np.mean(np.abs((actual_clean - predicted_clean) / actual_clean)) * 100
            
            # Convert to accuracy (lower MAPE = higher accuracy)
            accuracy = max(0, 100 - mape) / 100
            return float(accuracy)
        except Exception:
            return 0.5  # Default accuracy if calculation fails
    
    def _generate_forecast_dates(self, last_date: datetime, steps: int) -> List[datetime]:
        """Generate future dates for forecasting."""
        dates = []
        for i in range(1, steps + 1):
            dates.append(last_date + timedelta(days=i))
        return dates


class EnsembleForecaster:
    """Ensemble forecaster that combines multiple forecasting models."""
    
    def __init__(self, forecasters: List[BaseForecaster], weights: Optional[List[float]] = None):
        """
        Initialize ensemble forecaster.
        
        Args:
            forecasters: List of fitted forecasting models
            weights: Optional weights for each forecaster (must sum to 1)
        """
        self.forecasters = forecasters
        self.weights = weights or [1.0 / len(forecasters)] * len(forecasters)
        
        if len(self.weights) != len(self.forecasters):
            raise ValueError("Number of weights must match number of forecasters")
        
        if abs(sum(self.weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
    
    def predict(self, steps: int = None) -> ForecastResult:
        """
        Generate ensemble forecast by combining individual forecasts.
        
        Args:
            steps: Number of time steps to forecast
            
        Returns:
            Combined forecast result
        """
        if not self.forecasters:
            raise ValueError("No forecasters available for ensemble prediction")
        
        # Get forecasts from all models
        forecasts = []
        for forecaster in self.forecasters:
            try:
                forecast = forecaster.predict(steps)
                forecasts.append(forecast)
            except Exception as e:
                print(f"Warning: Forecaster {type(forecaster).__name__} failed: {e}")
                continue
        
        if not forecasts:
            raise ValueError("All forecasters failed to generate predictions")
        
        # Combine forecasts using weighted average
        combined_points = []
        dates = forecasts[0].forecast_points[0].date if forecasts[0].forecast_points else []
        
        for i in range(steps or self.forecasters[0].config.default_forecast_days):
            try:
                weighted_prediction = 0.0
                weighted_lower = 0.0
                weighted_upper = 0.0
                total_weight = 0.0
                
                for j, forecast in enumerate(forecasts):
                    if i < len(forecast.forecast_points):
                        point = forecast.forecast_points[i]
                        weight = self.weights[j] if j < len(self.weights) else 1.0
                        
                        weighted_prediction += point.predicted_value * weight
                        weighted_lower += (point.confidence_interval_lower or point.predicted_value) * weight
                        weighted_upper += (point.confidence_interval_upper or point.predicted_value) * weight
                        total_weight += weight
                
                if total_weight > 0:
                    combined_point = ForecastPoint(
                        date=forecast.forecast_points[i].date,
                        predicted_value=weighted_prediction / total_weight,
                        confidence_interval_lower=weighted_lower / total_weight,
                        confidence_interval_upper=weighted_upper / total_weight
                    )
                    combined_points.append(combined_point)
            except Exception:
                continue
        
        # Calculate ensemble accuracy as weighted average
        ensemble_accuracy = sum(
            forecast.model_accuracy * weight 
            for forecast, weight in zip(forecasts, self.weights)
            if forecast.model_accuracy is not None
        ) / sum(self.weights)
        
        return ForecastResult(
            symbol=forecasts[0].symbol,
            model_type=ModelType.ENSEMBLE,
            forecast_period_days=len(combined_points),
            forecast_points=combined_points,
            model_accuracy=ensemble_accuracy,
            model_parameters={
                "component_models": [f.model_type for f in forecasts],
                "weights": self.weights
            },
            generated_at=datetime.now(),
            data_source="ensemble"
        )