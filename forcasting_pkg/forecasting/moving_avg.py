"""Moving average forecasting implementations."""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Optional
from datetime import datetime

from .base import BaseForecaster
from ..models import ForecastResult, ForecastPoint, MarketData, ModelType, ForecastingConfig


class MovingAverageForecaster(BaseForecaster):
    """Simple Moving Average forecasting model."""
    
    def __init__(self, window: int = None, config: Optional[ForecastingConfig] = None):
        """
        Initialize moving average forecaster.
        
        Args:
            window: Moving average window size
            config: Forecasting configuration
        """
        super().__init__(config)
        self.window = window or (config.moving_average_window if config else 20)
        self.data = None
        self.symbol = None
        self.moving_average = None
        self.trend_slope = None
    
    def fit(self, data: Union[List[MarketData], pd.DataFrame], symbol: str = "UNKNOWN") -> 'MovingAverageForecaster':
        """
        Fit moving average model to historical data.
        
        Args:
            data: Historical market data
            symbol: Symbol identifier
            
        Returns:
            Self for method chaining
        """
        self.symbol = symbol
        df = self._prepare_data(data)
        self.data = df
        
        # Ensure we have enough data for the moving average
        if len(df) < self.window:
            raise ValueError(f"Insufficient data for moving average. Need at least {self.window} points, got {len(df)}")
        
        # Calculate moving average
        df['ma'] = df['close'].rolling(window=self.window).mean()
        
        # Get the last moving average value
        self.moving_average = df['ma'].dropna().iloc[-1]
        
        # Calculate trend from recent data for better forecasting
        recent_prices = df['close'].tail(self.window).values
        recent_days = np.arange(len(recent_prices))
        self.trend_slope = np.polyfit(recent_days, recent_prices, 1)[0]
        
        self.is_fitted = True
        return self
    
    def predict(self, steps: int = None) -> ForecastResult:
        """
        Generate moving average forecast predictions.
        
        Args:
            steps: Number of time steps to forecast
            
        Returns:
            Forecast result with predictions and confidence intervals
        """
        self._validate_fitted()
        
        steps = steps or self.config.default_forecast_days
        
        # Generate forecast points using moving average + trend
        forecast_points = []
        last_date = self.data['date'].max()
        forecast_dates = self._generate_forecast_dates(last_date, steps)
        
        # Calculate standard deviation for confidence intervals
        recent_prices = self.data['close'].tail(self.window)
        std_dev = recent_prices.std()
        
        for i in range(steps):
            # Apply trend to moving average
            predicted_value = self.moving_average + (self.trend_slope * i)
            
            # Confidence intervals based on historical volatility
            confidence_multiplier = 1.96  # 95% confidence interval
            margin_error = confidence_multiplier * std_dev
            
            point = ForecastPoint(
                date=forecast_dates[i],
                predicted_value=float(predicted_value),
                confidence_interval_lower=float(predicted_value - margin_error),
                confidence_interval_upper=float(predicted_value + margin_error)
            )
            forecast_points.append(point)
        
        # Calculate model accuracy
        accuracy = self._calculate_model_accuracy()
        
        return ForecastResult(
            symbol=self.symbol,
            model_type=ModelType.MOVING_AVERAGE,  # Use enum value
            forecast_period_days=steps,
            forecast_points=forecast_points,
            model_accuracy=accuracy,
            model_parameters=self.get_model_parameters(),
            generated_at=datetime.now(),
            data_source="moving_average"
        )
    
    def _calculate_model_accuracy(self) -> float:
        """Calculate accuracy by comparing MA predictions to actual values."""
        try:
            # Calculate moving average for all available data
            ma_series = self.data['close'].rolling(window=self.window).mean()
            
            # Compare with actual prices (shift by 1 to simulate prediction)
            actual = self.data['close'].iloc[self.window:]
            predicted = ma_series.iloc[self.window-1:-1]  # Shifted MA as prediction
            
            if len(actual) == 0 or len(predicted) == 0:
                return 0.6  # Default accuracy
            
            return self._calculate_accuracy(actual, predicted)
        except Exception:
            return 0.6  # Default accuracy for MA
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """Get moving average model parameters."""
        return {
            "window": self.window,
            "moving_average": float(self.moving_average),
            "trend_slope": float(self.trend_slope),
            "method": "simple_moving_average"
        }


class ExponentialMovingAverageForecaster(BaseForecaster):
    """Exponential Moving Average (EMA) forecasting model."""
    
    def __init__(self, alpha: float = None, span: int = None, config: Optional[ForecastingConfig] = None):
        """
        Initialize EMA forecaster.
        
        Args:
            alpha: Smoothing parameter (0 < alpha <= 1)
            span: Span for EMA calculation (alternative to alpha)
            config: Forecasting configuration
        """
        super().__init__(config)
        
        if alpha is not None and span is not None:
            raise ValueError("Specify either alpha or span, not both")
        
        if alpha is not None:
            if not (0 < alpha <= 1):
                raise ValueError("Alpha must be between 0 and 1")
            self.alpha = alpha
            self.span = None
        elif span is not None:
            if span <= 0:
                raise ValueError("Span must be positive")
            self.span = span
            self.alpha = 2.0 / (span + 1)
        else:
            # Default span based on config or 20 days
            self.span = config.moving_average_window if config else 20
            self.alpha = 2.0 / (self.span + 1)
        
        self.data = None
        self.symbol = None
        self.ema_value = None
        self.trend_slope = None
    
    def fit(self, data: Union[List[MarketData], pd.DataFrame], symbol: str = "UNKNOWN") -> 'ExponentialMovingAverageForecaster':
        """
        Fit EMA model to historical data.
        
        Args:
            data: Historical market data
            symbol: Symbol identifier
            
        Returns:
            Self for method chaining
        """
        self.symbol = symbol
        df = self._prepare_data(data)
        self.data = df
        
        # Calculate EMA
        if self.span:
            ema_series = df['close'].ewm(span=self.span).mean()
        else:
            ema_series = df['close'].ewm(alpha=self.alpha).mean()
        
        # Get the last EMA value
        self.ema_value = ema_series.iloc[-1]
        
        # Calculate trend using EMA values
        recent_ema = ema_series.tail(min(20, len(ema_series))).values
        recent_days = np.arange(len(recent_ema))
        self.trend_slope = np.polyfit(recent_days, recent_ema, 1)[0]
        
        self.is_fitted = True
        return self
    
    def predict(self, steps: int = None) -> ForecastResult:
        """
        Generate EMA forecast predictions.
        
        Args:
            steps: Number of time steps to forecast
            
        Returns:
            Forecast result with predictions and confidence intervals
        """
        self._validate_fitted()
        
        steps = steps or self.config.default_forecast_days
        
        # Generate forecast points
        forecast_points = []
        last_date = self.data['date'].max()
        forecast_dates = self._generate_forecast_dates(last_date, steps)
        
        # Calculate volatility for confidence intervals
        returns = self.data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        
        for i in range(steps):
            # EMA with trend adjustment
            predicted_value = self.ema_value + (self.trend_slope * i)
            
            # Confidence intervals based on volatility
            daily_vol = volatility / np.sqrt(252)
            margin_error = 1.96 * daily_vol * predicted_value * np.sqrt(i + 1)
            
            point = ForecastPoint(
                date=forecast_dates[i],
                predicted_value=float(predicted_value),
                confidence_interval_lower=float(predicted_value - margin_error),
                confidence_interval_upper=float(predicted_value + margin_error)
            )
            forecast_points.append(point)
        
        # Calculate model accuracy
        accuracy = self._calculate_ema_accuracy()
        
        return ForecastResult(
            symbol=self.symbol,
            model_type=f"EMA_alpha_{self.alpha:.3f}" if not self.span else f"EMA_span_{self.span}",
            forecast_period_days=steps,
            forecast_points=forecast_points,
            model_accuracy=accuracy,
            model_parameters=self.get_model_parameters(),
            generated_at=datetime.now(),
            data_source="exponential_moving_average"
        )
    
    def _calculate_ema_accuracy(self) -> float:
        """Calculate EMA model accuracy."""
        try:
            # Calculate EMA for comparison
            if self.span:
                ema_series = self.data['close'].ewm(span=self.span).mean()
            else:
                ema_series = self.data['close'].ewm(alpha=self.alpha).mean()
            
            # Compare EMA predictions to actual prices (1-step ahead)
            actual = self.data['close'].iloc[1:]
            predicted = ema_series.iloc[:-1]
            
            return self._calculate_accuracy(actual, predicted)
        except Exception:
            return 0.65  # Default accuracy for EMA
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """Get EMA model parameters."""
        return {
            "alpha": float(self.alpha),
            "span": self.span,
            "ema_value": float(self.ema_value),
            "trend_slope": float(self.trend_slope),
            "method": "exponential_moving_average"
        }


class AdaptiveMovingAverageForecaster(BaseForecaster):
    """Adaptive Moving Average that adjusts to market conditions."""
    
    def __init__(self, min_window: int = 5, max_window: int = 50, config: Optional[ForecastingConfig] = None):
        """
        Initialize adaptive MA forecaster.
        
        Args:
            min_window: Minimum window size
            max_window: Maximum window size
            config: Forecasting configuration
        """
        super().__init__(config)
        self.min_window = min_window
        self.max_window = max_window
        self.optimal_window = None
        self.data = None
        self.symbol = None
        self.moving_average = None
        self.trend_slope = None
    
    def fit(self, data: Union[List[MarketData], pd.DataFrame], symbol: str = "UNKNOWN") -> 'AdaptiveMovingAverageForecaster':
        """
        Fit adaptive MA model by finding optimal window size.
        
        Args:
            data: Historical market data
            symbol: Symbol identifier
            
        Returns:
            Self for method chaining
        """
        self.symbol = symbol
        df = self._prepare_data(data)
        self.data = df
        
        # Find optimal window size
        self.optimal_window = self._find_optimal_window()
        
        # Calculate moving average with optimal window
        df['ma'] = df['close'].rolling(window=self.optimal_window).mean()
        self.moving_average = df['ma'].dropna().iloc[-1]
        
        # Calculate trend
        recent_prices = df['close'].tail(self.optimal_window).values
        recent_days = np.arange(len(recent_prices))
        self.trend_slope = np.polyfit(recent_days, recent_prices, 1)[0]
        
        self.is_fitted = True
        return self
    
    def _find_optimal_window(self) -> int:
        """Find optimal window size by minimizing prediction error."""
        best_window = self.min_window
        best_error = float('inf')
        
        for window in range(self.min_window, min(self.max_window + 1, len(self.data) // 2)):
            try:
                # Calculate moving average
                ma = self.data['close'].rolling(window=window).mean()
                
                # Calculate prediction error (1-step ahead)
                actual = self.data['close'].iloc[window:]
                predicted = ma.iloc[window-1:-1]
                
                if len(actual) > 0 and len(predicted) > 0:
                    # Use MAE as error metric
                    error = np.mean(np.abs(actual - predicted))
                    
                    if error < best_error:
                        best_error = error
                        best_window = window
            except Exception:
                continue
        
        return best_window
    
    def predict(self, steps: int = None) -> ForecastResult:
        """Generate adaptive MA forecast predictions."""
        self._validate_fitted()
        
        steps = steps or self.config.default_forecast_days
        
        # Use the fitted moving average forecaster logic
        forecast_points = []
        last_date = self.data['date'].max()
        forecast_dates = self._generate_forecast_dates(last_date, steps)
        
        # Calculate confidence intervals
        recent_prices = self.data['close'].tail(self.optimal_window)
        std_dev = recent_prices.std()
        
        for i in range(steps):
            predicted_value = self.moving_average + (self.trend_slope * i)
            
            # Adaptive confidence intervals (wider for longer horizons)
            confidence_multiplier = 1.96 * (1 + i * 0.1)  # Increasing uncertainty
            margin_error = confidence_multiplier * std_dev
            
            point = ForecastPoint(
                date=forecast_dates[i],
                predicted_value=float(predicted_value),
                confidence_interval_lower=float(predicted_value - margin_error),
                confidence_interval_upper=float(predicted_value + margin_error)
            )
            forecast_points.append(point)
        
        # Calculate model accuracy
        accuracy = self._calculate_adaptive_accuracy()
        
        return ForecastResult(
            symbol=self.symbol,
            model_type=f"Adaptive_MA_{self.optimal_window}",
            forecast_period_days=steps,
            forecast_points=forecast_points,
            model_accuracy=accuracy,
            model_parameters=self.get_model_parameters(),
            generated_at=datetime.now(),
            data_source="adaptive_moving_average"
        )
    
    def _calculate_adaptive_accuracy(self) -> float:
        """Calculate adaptive MA accuracy."""
        try:
            ma_series = self.data['close'].rolling(window=self.optimal_window).mean()
            actual = self.data['close'].iloc[self.optimal_window:]
            predicted = ma_series.iloc[self.optimal_window-1:-1]
            
            return self._calculate_accuracy(actual, predicted)
        except Exception:
            return 0.7  # Default accuracy
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """Get adaptive MA model parameters."""
        return {
            "optimal_window": self.optimal_window,
            "min_window": self.min_window,
            "max_window": self.max_window,
            "moving_average": float(self.moving_average),
            "trend_slope": float(self.trend_slope),
            "method": "adaptive_moving_average"
        }