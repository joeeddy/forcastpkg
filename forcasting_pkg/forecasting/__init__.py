"""Forecasting module - Various forecasting models and engines."""

from .base import BaseForecaster, EnsembleForecaster
from .arima import ARIMAForecaster, AutoARIMA
from .linear import LinearForecaster, TrendForecaster
from .moving_avg import MovingAverageForecaster, ExponentialMovingAverageForecaster, AdaptiveMovingAverageForecaster

from typing import Union, List, Dict, Any, Optional
import pandas as pd
from datetime import datetime

from ..models import ForecastResult, MarketData, ModelType, ForecastingConfig


class ForecastingEngine:
    """Main forecasting engine that provides a unified interface to all forecasting models."""
    
    def __init__(self, config: Optional[ForecastingConfig] = None):
        """
        Initialize the forecasting engine.
        
        Args:
            config: Forecasting configuration
        """
        self.config = config or ForecastingConfig()
        self._models = {
            ModelType.ARIMA: ARIMAForecaster,
            ModelType.LINEAR: LinearForecaster,
            ModelType.MOVING_AVERAGE: MovingAverageForecaster,
        }
        self._fitted_models = {}
    
    def forecast(
        self, 
        data: Union[List[MarketData], pd.DataFrame], 
        model: Union[str, ModelType] = ModelType.ARIMA,
        symbol: str = "UNKNOWN",
        days: int = None,
        **model_kwargs
    ) -> ForecastResult:
        """
        Generate forecast using specified model.
        
        Args:
            data: Historical market data
            model: Model type to use for forecasting
            symbol: Symbol identifier
            days: Number of days to forecast
            **model_kwargs: Additional arguments for the model
            
        Returns:
            Forecast result
        """
        # Convert string to ModelType if needed
        if isinstance(model, str):
            model = ModelType(model.lower())
        
        # Get the appropriate forecaster class
        if model not in self._models:
            raise ValueError(f"Unknown model type: {model}. Available: {list(self._models.keys())}")
        
        forecaster_class = self._models[model]
        
        # Create and fit the forecaster
        forecaster = forecaster_class(config=self.config, **model_kwargs)
        forecaster.fit(data, symbol)
        
        # Store fitted model for potential reuse
        self._fitted_models[f"{symbol}_{model}"] = forecaster
        
        # Generate forecast
        return forecaster.predict(days)
    
    def compare_models(
        self, 
        data: Union[List[MarketData], pd.DataFrame], 
        symbol: str = "UNKNOWN",
        models: Optional[List[Union[str, ModelType]]] = None,
        days: int = None
    ) -> Dict[str, ForecastResult]:
        """
        Compare multiple forecasting models on the same data.
        
        Args:
            data: Historical market data
            symbol: Symbol identifier
            models: List of models to compare (default: all available)
            days: Number of days to forecast
            
        Returns:
            Dictionary mapping model names to forecast results
        """
        if models is None:
            models = list(self._models.keys())
        
        results = {}
        for model in models:
            try:
                result = self.forecast(data, model, symbol, days)
                model_name = result.model_type
                results[model_name] = result
            except Exception as e:
                print(f"Warning: Model {model} failed for {symbol}: {e}")
                continue
        
        return results
    
    def create_ensemble(
        self,
        data: Union[List[MarketData], pd.DataFrame],
        symbol: str = "UNKNOWN",
        models: Optional[List[Union[str, ModelType]]] = None,
        weights: Optional[List[float]] = None,
        days: int = None
    ) -> ForecastResult:
        """
        Create an ensemble forecast combining multiple models.
        
        Args:
            data: Historical market data
            symbol: Symbol identifier
            models: List of models to include in ensemble
            weights: Optional weights for each model
            days: Number of days to forecast
            
        Returns:
            Ensemble forecast result
        """
        if models is None:
            models = [ModelType.ARIMA, ModelType.LINEAR, ModelType.MOVING_AVERAGE]
        
        # Fit all models
        forecasters = []
        for model in models:
            try:
                if isinstance(model, str):
                    model = ModelType(model.lower())
                
                forecaster_class = self._models[model]
                forecaster = forecaster_class(config=self.config)
                forecaster.fit(data, symbol)
                forecasters.append(forecaster)
            except Exception as e:
                print(f"Warning: Failed to fit {model} for ensemble: {e}")
                continue
        
        if not forecasters:
            raise ValueError("No models successfully fitted for ensemble")
        
        # Create ensemble
        ensemble = EnsembleForecaster(forecasters, weights)
        return ensemble.predict(days)
    
    def add_custom_model(self, name: str, forecaster_class: type):
        """
        Add a custom forecasting model to the engine.
        
        Args:
            name: Name for the custom model
            forecaster_class: Class that inherits from BaseForecaster
        """
        if not issubclass(forecaster_class, BaseForecaster):
            raise ValueError("Custom model must inherit from BaseForecaster")
        
        self._models[name] = forecaster_class
    
    def get_fitted_model(self, symbol: str, model: Union[str, ModelType]) -> Optional[BaseForecaster]:
        """
        Get a previously fitted model.
        
        Args:
            symbol: Symbol identifier
            model: Model type
            
        Returns:
            Fitted forecaster or None if not found
        """
        if isinstance(model, str):
            model = ModelType(model.lower())
        
        key = f"{symbol}_{model}"
        return self._fitted_models.get(key)
    
    def available_models(self) -> List[str]:
        """Get list of available model types."""
        return list(self._models.keys())
    
    def model_info(self, model: Union[str, ModelType]) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model: Model type
            
        Returns:
            Dictionary with model information
        """
        if isinstance(model, str):
            model = ModelType(model.lower()) if model in [m.value for m in ModelType] else model
        
        if model not in self._models:
            raise ValueError(f"Unknown model: {model}")
        
        forecaster_class = self._models[model]
        
        return {
            "name": model,
            "class": forecaster_class.__name__,
            "description": forecaster_class.__doc__ or "No description available",
            "module": forecaster_class.__module__
        }


# Convenience functions for direct model access
def arima_forecast(
    data: Union[List[MarketData], pd.DataFrame], 
    symbol: str = "UNKNOWN",
    days: int = 30,
    order: Optional[tuple] = None,
    config: Optional[ForecastingConfig] = None
) -> ForecastResult:
    """Convenience function for ARIMA forecasting."""
    forecaster = ARIMAForecaster(order=order, config=config)
    forecaster.fit(data, symbol)
    return forecaster.predict(days)


def linear_forecast(
    data: Union[List[MarketData], pd.DataFrame], 
    symbol: str = "UNKNOWN",
    days: int = 30,
    degree: int = 1,
    config: Optional[ForecastingConfig] = None
) -> ForecastResult:
    """Convenience function for linear regression forecasting."""
    forecaster = LinearForecaster(degree=degree, config=config)
    forecaster.fit(data, symbol)
    return forecaster.predict(days)


def moving_average_forecast(
    data: Union[List[MarketData], pd.DataFrame], 
    symbol: str = "UNKNOWN",
    days: int = 30,
    window: int = 20,
    config: Optional[ForecastingConfig] = None
) -> ForecastResult:
    """Convenience function for moving average forecasting."""
    forecaster = MovingAverageForecaster(window=window, config=config)
    forecaster.fit(data, symbol)
    return forecaster.predict(days)


__all__ = [
    "ForecastingEngine",
    "BaseForecaster",
    "EnsembleForecaster",
    "ARIMAForecaster",
    "AutoARIMA",
    "LinearForecaster", 
    "TrendForecaster",
    "MovingAverageForecaster",
    "ExponentialMovingAverageForecaster",
    "AdaptiveMovingAverageForecaster",
    "arima_forecast",
    "linear_forecast", 
    "moving_average_forecast"
]