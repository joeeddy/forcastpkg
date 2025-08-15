"""Linear regression forecasting implementation."""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Optional
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from .base import BaseForecaster
from ..models import ForecastResult, ForecastPoint, MarketData, ModelType, ForecastingConfig


class LinearForecaster(BaseForecaster):
    """Linear regression forecasting model with polynomial features support."""
    
    def __init__(self, degree: int = 1, config: Optional[ForecastingConfig] = None):
        """
        Initialize linear forecaster.
        
        Args:
            degree: Polynomial degree for feature transformation
            config: Forecasting configuration
        """
        super().__init__(config)
        self.degree = degree if degree is not None else (config.linear_polynomial_degree if config else 1)
        self.model = LinearRegression()
        self.poly_features = PolynomialFeatures(degree=self.degree, include_bias=False)
        self.data = None
        self.symbol = None
        self.X_train = None
        self.y_train = None
        self.last_day = None
    
    def fit(self, data: Union[List[MarketData], pd.DataFrame], symbol: str = "UNKNOWN") -> 'LinearForecaster':
        """
        Fit linear regression model to historical data.
        
        Args:
            data: Historical market data
            symbol: Symbol identifier
            
        Returns:
            Self for method chaining
        """
        self.symbol = symbol
        df = self._prepare_data(data)
        self.data = df
        
        # Create time series features
        df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
        self.last_day = df['days_since_start'].max()
        
        # Prepare features and target
        X = df[['days_since_start']].values
        y = df['close'].values
        
        # Apply polynomial transformation if degree > 1
        if self.degree > 1:
            X = self.poly_features.fit_transform(X)
        
        # Fit the model
        self.model.fit(X, y)
        self.X_train = X
        self.y_train = y
        self.is_fitted = True
        
        return self
    
    def predict(self, steps: int = None) -> ForecastResult:
        """
        Generate linear regression forecast predictions.
        
        Args:
            steps: Number of time steps to forecast
            
        Returns:
            Forecast result with predictions and confidence intervals
        """
        self._validate_fitted()
        
        steps = steps or self.config.default_forecast_days
        
        # Generate future time points
        future_days = np.array([[self.last_day + i + 1] for i in range(steps)])
        
        # Apply polynomial transformation if needed
        if self.degree > 1:
            future_X = self.poly_features.transform(future_days)
        else:
            future_X = future_days
        
        # Make predictions
        predictions = self.model.predict(future_X)
        
        # Calculate confidence intervals using residual standard error
        residuals = self.y_train - self.model.predict(self.X_train)
        std_error = np.std(residuals)
        
        # Use t-distribution for confidence intervals (approximation for normal)
        confidence_multiplier = 1.96  # 95% confidence interval
        margin_error = confidence_multiplier * std_error
        
        # Create forecast points
        forecast_points = []
        last_date = self.data['date'].max()
        forecast_dates = self._generate_forecast_dates(last_date, steps)
        
        for i in range(steps):
            predicted_value = float(predictions[i])
            
            point = ForecastPoint(
                date=forecast_dates[i],
                predicted_value=predicted_value,
                confidence_interval_lower=predicted_value - margin_error,
                confidence_interval_upper=predicted_value + margin_error
            )
            forecast_points.append(point)
        
        # Calculate model accuracy
        accuracy = self._calculate_model_accuracy()
        
        return ForecastResult(
            symbol=self.symbol,
            model_type=ModelType.LINEAR,  # Use enum value instead of custom string
            forecast_period_days=steps,
            forecast_points=forecast_points,
            model_accuracy=accuracy,
            model_parameters=self.get_model_parameters(),
            generated_at=datetime.now(),
            data_source="linear_regression"
        )
    
    def _calculate_model_accuracy(self) -> float:
        """Calculate model accuracy using R-squared and MAPE."""
        try:
            y_pred = self.model.predict(self.X_train)
            
            # Calculate R-squared
            r2 = r2_score(self.y_train, y_pred)
            
            # Calculate MAPE-based accuracy
            mape_accuracy = self._calculate_accuracy(pd.Series(self.y_train), pd.Series(y_pred))
            
            # Combine R-squared and MAPE accuracy (weighted average)
            # R-squared weight: 0.3, MAPE accuracy weight: 0.7
            combined_accuracy = 0.3 * max(0, r2) + 0.7 * mape_accuracy
            
            return float(max(0, min(1, combined_accuracy)))
        except Exception:
            return 0.5
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """Get linear regression model parameters."""
        try:
            # Calculate additional metrics
            y_pred = self.model.predict(self.X_train)
            mae = mean_absolute_error(self.y_train, y_pred)
            mse = mean_squared_error(self.y_train, y_pred)
            r2 = r2_score(self.y_train, y_pred)
            
            return {
                "degree": self.degree,
                "coefficients": self.model.coef_.tolist() if hasattr(self.model.coef_, 'tolist') else [float(self.model.coef_)],
                "intercept": float(self.model.intercept_),
                "r_squared": float(r2),
                "mean_absolute_error": float(mae),
                "mean_squared_error": float(mse),
                "rmse": float(np.sqrt(mse)),
                "n_features": len(self.model.coef_) if hasattr(self.model.coef_, '__len__') else 1,
                "training_samples": len(self.y_train)
            }
        except Exception:
            return {
                "degree": self.degree,
                "method": "linear_regression"
            }


class TrendForecaster(LinearForecaster):
    """Trend-based forecasting using linear regression with trend decomposition."""
    
    def __init__(self, detrend: bool = True, config: Optional[ForecastingConfig] = None):
        """
        Initialize trend forecaster.
        
        Args:
            detrend: Whether to remove trend before modeling
            config: Forecasting configuration
        """
        super().__init__(degree=1, config=config)
        self.detrend = detrend
        self.trend_model = None
        self.residual_model = None
        self.trend_data = None
    
    def fit(self, data: Union[List[MarketData], pd.DataFrame], symbol: str = "UNKNOWN") -> 'TrendForecaster':
        """
        Fit trend-based model with optional detrending.
        
        Args:
            data: Historical market data
            symbol: Symbol identifier
            
        Returns:
            Self for method chaining
        """
        self.symbol = symbol
        df = self._prepare_data(data)
        self.data = df
        
        # Create time series features
        df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
        self.last_day = df['days_since_start'].max()
        
        if self.detrend:
            # First, fit trend
            X = df[['days_since_start']].values
            y = df['close'].values
            
            self.trend_model = LinearRegression()
            self.trend_model.fit(X, y)
            
            # Calculate trend and residuals
            trend = self.trend_model.predict(X)
            residuals = y - trend
            
            # Fit residual model (could be more sophisticated)
            self.residual_model = LinearRegression()
            X_residual = X  # Could add more features here
            self.residual_model.fit(X_residual, residuals)
            
            self.X_train = X
            self.y_train = y
            self.trend_data = trend
        else:
            # Standard linear regression
            super().fit(data, symbol)
        
        self.is_fitted = True
        return self
    
    def predict(self, steps: int = None) -> ForecastResult:
        """
        Generate trend-based forecast predictions.
        
        Args:
            steps: Number of time steps to forecast
            
        Returns:
            Forecast result with predictions and confidence intervals
        """
        self._validate_fitted()
        
        if not self.detrend:
            return super().predict(steps)
        
        steps = steps or self.config.default_forecast_days
        
        # Generate future time points
        future_days = np.array([[self.last_day + i + 1] for i in range(steps)])
        
        # Predict trend and residuals
        trend_predictions = self.trend_model.predict(future_days)
        residual_predictions = self.residual_model.predict(future_days)
        
        # Combine trend and residual predictions
        predictions = trend_predictions + residual_predictions
        
        # Calculate confidence intervals
        # Use both trend and residual uncertainties
        trend_residuals = self.y_train - (self.trend_model.predict(self.X_train) + self.residual_model.predict(self.X_train))
        std_error = np.std(trend_residuals)
        
        confidence_multiplier = 1.96
        margin_error = confidence_multiplier * std_error
        
        # Create forecast points
        forecast_points = []
        last_date = self.data['date'].max()
        forecast_dates = self._generate_forecast_dates(last_date, steps)
        
        for i in range(steps):
            predicted_value = float(predictions[i])
            
            point = ForecastPoint(
                date=forecast_dates[i],
                predicted_value=predicted_value,
                confidence_interval_lower=predicted_value - margin_error,
                confidence_interval_upper=predicted_value + margin_error
            )
            forecast_points.append(point)
        
        # Calculate model accuracy
        accuracy = self._calculate_trend_accuracy()
        
        return ForecastResult(
            symbol=self.symbol,
            model_type="Trend_Decomposition_Linear",
            forecast_period_days=steps,
            forecast_points=forecast_points,
            model_accuracy=accuracy,
            model_parameters=self.get_trend_parameters(),
            generated_at=datetime.now(),
            data_source="trend_regression"
        )
    
    def _calculate_trend_accuracy(self) -> float:
        """Calculate accuracy for trend-based model."""
        if not self.detrend:
            return self._calculate_model_accuracy()
        
        try:
            # Predict using both trend and residual models
            trend_pred = self.trend_model.predict(self.X_train)
            residual_pred = self.residual_model.predict(self.X_train)
            y_pred = trend_pred + residual_pred
            
            return self._calculate_accuracy(pd.Series(self.y_train), pd.Series(y_pred))
        except Exception:
            return 0.5
    
    def get_trend_parameters(self) -> Dict[str, Any]:
        """Get trend model parameters."""
        if not self.detrend:
            return self.get_model_parameters()
        
        try:
            base_params = self.get_model_parameters()
            
            trend_params = {
                "trend_coefficient": float(self.trend_model.coef_[0]),
                "trend_intercept": float(self.trend_model.intercept_),
                "residual_coefficient": float(self.residual_model.coef_[0]) if hasattr(self.residual_model.coef_, '__getitem__') else float(self.residual_model.coef_),
                "residual_intercept": float(self.residual_model.intercept_),
                "detrended": self.detrend
            }
            
            return {**base_params, **trend_params}
        except Exception:
            return {
                "method": "trend_decomposition",
                "detrended": self.detrend
            }