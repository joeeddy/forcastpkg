"""Forecasting service using ARIMA and statistical models."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from app.models import ForecastResponse, ForecastPoint
from app.services.market_data import MarketDataService

# Try to import statsmodels for ARIMA, fall back to simple models if unavailable
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

class ForecastingService:
    """Service for generating forecasts using various statistical models."""
    
    def __init__(self):
        self.market_service = MarketDataService()
    
    async def generate_arima_forecast(
        self, 
        symbol: str, 
        forecast_days: int = 30, 
        data_type: str = "stock"
    ) -> ForecastResponse:
        """Generate ARIMA forecast for a financial instrument."""
        try:
            if not STATSMODELS_AVAILABLE:
                # Fall back to linear regression if statsmodels not available
                return await self.generate_linear_forecast(symbol, forecast_days, data_type)
            
            # Fetch historical data
            historical_data = await self.market_service.get_historical_data(
                symbol, days=90, data_type=data_type
            )
            
            if len(historical_data) < 30:
                raise ValueError(f"Insufficient historical data for {symbol}")
            
            # Prepare data
            df = pd.DataFrame(historical_data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            
            prices = df['close']
            
            # Fit ARIMA model
            # Use auto ARIMA-like approach with simple parameter selection
            best_aic = float('inf')
            best_model = None
            best_order = None
            
            # Try different ARIMA parameters
            for p in [1, 2]:
                for d in [0, 1]:
                    for q in [1, 2]:
                        try:
                            model = ARIMA(prices, order=(p, d, q))
                            fitted_model = model.fit()
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_model = fitted_model
                                best_order = (p, d, q)
                        except Exception:
                            continue
            
            if best_model is None:
                # Fall back to simple linear regression
                return await self.generate_linear_forecast(symbol, forecast_days, data_type)
            
            # Generate forecast
            forecast = best_model.forecast(steps=forecast_days)
            conf_int = best_model.get_forecast(steps=forecast_days).conf_int()
            
            # Create forecast points
            forecast_points = []
            start_date = df.index[-1] + timedelta(days=1)
            
            for i in range(forecast_days):
                forecast_date = start_date + timedelta(days=i)
                
                point = ForecastPoint(
                    date=forecast_date,
                    predicted_value=float(forecast.iloc[i]),
                    confidence_interval_lower=float(conf_int.iloc[i, 0]),
                    confidence_interval_upper=float(conf_int.iloc[i, 1])
                )
                forecast_points.append(point)
            
            # Calculate model accuracy using in-sample fit
            accuracy = self._calculate_accuracy(prices, best_model.fittedvalues)
            
            return ForecastResponse(
                symbol=symbol,
                model_type=f"ARIMA{best_order}",
                forecast_period_days=forecast_days,
                forecast_points=forecast_points,
                model_accuracy=accuracy,
                generated_at=datetime.now()
            )
        except Exception as e:
            raise ValueError(f"Error generating ARIMA forecast for {symbol}: {str(e)}")
    
    async def generate_linear_forecast(
        self, 
        symbol: str, 
        forecast_days: int = 30, 
        data_type: str = "stock"
    ) -> ForecastResponse:
        """Generate linear regression forecast as fallback."""
        try:
            # Fetch historical data
            historical_data = await self.market_service.get_historical_data(
                symbol, days=60, data_type=data_type
            )
            
            if len(historical_data) < 10:
                raise ValueError(f"Insufficient historical data for {symbol}")
            
            # Prepare data
            df = pd.DataFrame(historical_data)
            df['date'] = pd.to_datetime(df['date'])
            df.sort_values('date', inplace=True)
            
            # Create time series features
            df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
            
            # Fit linear regression
            X = df[['days_since_start']].values
            y = df['close'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Generate predictions
            last_day = df['days_since_start'].max()
            future_days = np.array([[last_day + i + 1] for i in range(forecast_days)])
            predictions = model.predict(future_days)
            
            # Calculate confidence intervals (simplified approach)
            residuals = y - model.predict(X)
            std_error = np.std(residuals)
            
            forecast_points = []
            start_date = df['date'].max() + timedelta(days=1)
            
            for i in range(forecast_days):
                forecast_date = start_date + timedelta(days=i)
                predicted_value = float(predictions[i])
                
                # Simple confidence interval using standard error
                margin_error = 1.96 * std_error  # 95% confidence interval
                
                point = ForecastPoint(
                    date=forecast_date,
                    predicted_value=predicted_value,
                    confidence_interval_lower=predicted_value - margin_error,
                    confidence_interval_upper=predicted_value + margin_error
                )
                forecast_points.append(point)
            
            # Calculate R-squared as accuracy metric
            from sklearn.metrics import r2_score
            accuracy = r2_score(y, model.predict(X))
            
            return ForecastResponse(
                symbol=symbol,
                model_type="Linear Regression",
                forecast_period_days=forecast_days,
                forecast_points=forecast_points,
                model_accuracy=max(0, accuracy),  # Ensure non-negative
                generated_at=datetime.now()
            )
        except Exception as e:
            raise ValueError(f"Error generating linear forecast for {symbol}: {str(e)}")
    
    async def generate_moving_average_forecast(
        self, 
        symbol: str, 
        forecast_days: int = 30, 
        data_type: str = "stock",
        window: int = 20
    ) -> ForecastResponse:
        """Generate simple moving average forecast."""
        try:
            # Fetch historical data
            historical_data = await self.market_service.get_historical_data(
                symbol, days=max(window * 2, 60), data_type=data_type
            )
            
            if len(historical_data) < window:
                raise ValueError(f"Insufficient historical data for {symbol}")
            
            # Prepare data
            df = pd.DataFrame(historical_data)
            df['date'] = pd.to_datetime(df['date'])
            df.sort_values('date', inplace=True)
            
            # Calculate moving average
            df['ma'] = df['close'].rolling(window=window).mean()
            
            # Use the last moving average value as the forecast
            last_ma = df['ma'].dropna().iloc[-1]
            
            # Calculate trend from recent data
            recent_prices = df['close'].tail(window).values
            recent_days = np.arange(len(recent_prices))
            trend_slope = np.polyfit(recent_days, recent_prices, 1)[0]
            
            # Generate forecast points
            forecast_points = []
            start_date = df['date'].max() + timedelta(days=1)
            
            # Calculate standard deviation for confidence intervals
            std_dev = df['close'].tail(window).std()
            
            for i in range(forecast_days):
                forecast_date = start_date + timedelta(days=i)
                
                # Apply trend to moving average
                predicted_value = last_ma + (trend_slope * i)
                
                # Confidence intervals based on historical volatility
                margin_error = 1.96 * std_dev
                
                point = ForecastPoint(
                    date=forecast_date,
                    predicted_value=float(predicted_value),
                    confidence_interval_lower=float(predicted_value - margin_error),
                    confidence_interval_upper=float(predicted_value + margin_error)
                )
                forecast_points.append(point)
            
            return ForecastResponse(
                symbol=symbol,
                model_type=f"Moving Average ({window} days)",
                forecast_period_days=forecast_days,
                forecast_points=forecast_points,
                model_accuracy=0.7,  # Placeholder accuracy
                generated_at=datetime.now()
            )
        except Exception as e:
            raise ValueError(f"Error generating moving average forecast for {symbol}: {str(e)}")
    
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
    
    async def compare_forecast_models(
        self, 
        symbol: str, 
        forecast_days: int = 30, 
        data_type: str = "stock"
    ) -> Dict[str, ForecastResponse]:
        """Compare different forecasting models for a symbol."""
        try:
            models = {}
            
            # Try ARIMA
            try:
                arima_forecast = await self.generate_arima_forecast(symbol, forecast_days, data_type)
                models['ARIMA'] = arima_forecast
            except Exception:
                pass
            
            # Linear regression
            try:
                linear_forecast = await self.generate_linear_forecast(symbol, forecast_days, data_type)
                models['Linear'] = linear_forecast
            except Exception:
                pass
            
            # Moving average
            try:
                ma_forecast = await self.generate_moving_average_forecast(symbol, forecast_days, data_type)
                models['Moving Average'] = ma_forecast
            except Exception:
                pass
            
            return models
        except Exception as e:
            raise ValueError(f"Error comparing forecast models for {symbol}: {str(e)}")