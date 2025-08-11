"""ARIMA forecasting implementation."""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from .base import BaseForecaster
from ..models import ForecastResult, ForecastPoint, MarketData, ModelType, ForecastingConfig

# Try to import statsmodels for ARIMA, fall back to simple models if unavailable
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available. ARIMA forecaster will use fallback methods.")


class ARIMAForecaster(BaseForecaster):
    """ARIMA (AutoRegressive Integrated Moving Average) forecasting model."""
    
    def __init__(self, order: Optional[Tuple[int, int, int]] = None, config: Optional[ForecastingConfig] = None):
        """
        Initialize ARIMA forecaster.
        
        Args:
            order: ARIMA order (p, d, q). If None, will auto-select
            config: Forecasting configuration
        """
        super().__init__(config)
        self.order = order
        self.auto_select = order is None or config.arima_auto_select if config else True
        self.fitted_model = None
        self.data = None
        self.symbol = None
    
    def fit(self, data: Union[List[MarketData], pd.DataFrame], symbol: str = "UNKNOWN") -> 'ARIMAForecaster':
        """
        Fit ARIMA model to historical data.
        
        Args:
            data: Historical market data
            symbol: Symbol identifier
            
        Returns:
            Self for method chaining
        """
        self.symbol = symbol
        df = self._prepare_data(data)
        self.data = df
        
        # Extract price series
        prices = df['close'].dropna()
        
        if not STATSMODELS_AVAILABLE:
            # Fallback: store data for simple prediction
            self.is_fitted = True
            return self
        
        if self.auto_select:
            self.order = self._auto_select_order(prices)
        
        try:
            # Fit ARIMA model
            model = ARIMA(prices, order=self.order)
            self.fitted_model = model.fit()
            self.is_fitted = True
        except Exception as e:
            # If ARIMA fails, try simpler order
            try:
                self.order = (1, 1, 1)
                model = ARIMA(prices, order=self.order)
                self.fitted_model = model.fit()
                self.is_fitted = True
            except Exception:
                # Last resort: just store data for fallback prediction
                self.is_fitted = True
                print(f"Warning: ARIMA fitting failed for {symbol}. Using fallback method.")
        
        return self
    
    def predict(self, steps: int = None) -> ForecastResult:
        """
        Generate ARIMA forecast predictions.
        
        Args:
            steps: Number of time steps to forecast
            
        Returns:
            Forecast result with predictions and confidence intervals
        """
        self._validate_fitted()
        
        steps = steps or self.config.default_forecast_days
        
        if not STATSMODELS_AVAILABLE or self.fitted_model is None:
            return self._fallback_predict(steps)
        
        try:
            # Generate forecast using ARIMA
            forecast = self.fitted_model.forecast(steps=steps)
            forecast_ci = self.fitted_model.get_forecast(steps=steps).conf_int(alpha=1-self.config.confidence_level)
            
            # Create forecast points
            forecast_points = []
            last_date = self.data['date'].iloc[-1]
            forecast_dates = self._generate_forecast_dates(last_date, steps)
            
            for i in range(steps):
                point = ForecastPoint(
                    date=forecast_dates[i],
                    predicted_value=float(forecast.iloc[i]) if hasattr(forecast, 'iloc') else float(forecast[i]),
                    confidence_interval_lower=float(forecast_ci.iloc[i, 0]) if hasattr(forecast_ci, 'iloc') else float(forecast_ci[i, 0]),
                    confidence_interval_upper=float(forecast_ci.iloc[i, 1]) if hasattr(forecast_ci, 'iloc') else float(forecast_ci[i, 1])
                )
                forecast_points.append(point)
            
            # Calculate model accuracy using in-sample fit
            accuracy = self._calculate_model_accuracy()
            
            return ForecastResult(
                symbol=self.symbol,
                model_type=f"ARIMA{self.order}",
                forecast_period_days=steps,
                forecast_points=forecast_points,
                model_accuracy=accuracy,
                model_parameters=self.get_model_parameters(),
                generated_at=datetime.now(),
                data_source="arima_forecast"
            )
        
        except Exception as e:
            print(f"Warning: ARIMA prediction failed: {e}. Using fallback method.")
            return self._fallback_predict(steps)
    
    def _fallback_predict(self, steps: int) -> ForecastResult:
        """Fallback prediction method when ARIMA is not available."""
        from .linear import LinearForecaster
        
        # Use linear regression as fallback
        linear_forecaster = LinearForecaster(config=self.config)
        linear_forecaster.fit(self.data, self.symbol)
        result = linear_forecaster.predict(steps)
        
        # Update model type to indicate fallback
        result.model_type = "ARIMA_fallback_linear"
        return result
    
    def _auto_select_order(self, series: pd.Series) -> Tuple[int, int, int]:
        """
        Automatically select optimal ARIMA order using grid search.
        
        Args:
            series: Time series data
            
        Returns:
            Optimal (p, d, q) order
        """
        if not STATSMODELS_AVAILABLE:
            return (1, 1, 1)  # Default order
        
        # Determine d (differencing order) using ADF test
        d = self._determine_differencing_order(series)
        
        best_aic = float('inf')
        best_order = (1, d, 1)
        
        # Grid search for p and q
        max_p = min(self.config.arima_max_p, len(series) // 10)
        max_q = min(self.config.arima_max_q, len(series) // 10)
        
        for p in range(0, max_p + 1):
            for q in range(0, max_q + 1):
                try:
                    model = ARIMA(series, order=(p, d, q))
                    fitted_model = model.fit()
                    
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_order = (p, d, q)
                except Exception:
                    continue
        
        return best_order
    
    def _determine_differencing_order(self, series: pd.Series, max_d: int = 2) -> int:
        """
        Determine the appropriate differencing order using ADF test.
        
        Args:
            series: Time series data
            max_d: Maximum differencing order to test
            
        Returns:
            Optimal differencing order
        """
        if not STATSMODELS_AVAILABLE:
            return 1
        
        for d in range(max_d + 1):
            if d == 0:
                test_series = series
            else:
                test_series = series.diff(d).dropna()
            
            try:
                # Perform Augmented Dickey-Fuller test
                adf_result = adfuller(test_series)
                p_value = adf_result[1]
                
                # If p-value < 0.05, series is stationary
                if p_value < 0.05:
                    return d
            except Exception:
                continue
        
        return 1  # Default if no stationarity found
    
    def _calculate_model_accuracy(self) -> float:
        """Calculate model accuracy using fitted values."""
        if not STATSMODELS_AVAILABLE or self.fitted_model is None:
            return 0.7  # Default accuracy for fallback
        
        try:
            actual = self.data['close'].iloc[self.fitted_model.nobs_diffed:]
            fitted = self.fitted_model.fittedvalues
            
            return self._calculate_accuracy(actual, fitted)
        except Exception:
            return 0.7
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """Get ARIMA model parameters."""
        if not STATSMODELS_AVAILABLE or self.fitted_model is None:
            return {
                "order": self.order,
                "method": "fallback",
                "auto_selected": self.auto_select
            }
        
        try:
            return {
                "order": self.order,
                "aic": float(self.fitted_model.aic),
                "bic": float(self.fitted_model.bic),
                "hqic": float(self.fitted_model.hqic),
                "llf": float(self.fitted_model.llf),
                "auto_selected": self.auto_select,
                "method": "arima"
            }
        except Exception:
            return {
                "order": self.order,
                "method": "arima_basic",
                "auto_selected": self.auto_select
            }
    
    def diagnostic_tests(self) -> Dict[str, Any]:
        """
        Perform diagnostic tests on the fitted model.
        
        Returns:
            Dictionary with diagnostic test results
        """
        if not STATSMODELS_AVAILABLE or self.fitted_model is None:
            return {"status": "not_available"}
        
        try:
            # Ljung-Box test for residual autocorrelation
            residuals = self.fitted_model.resid
            ljung_box = acorr_ljungbox(residuals, lags=10, return_df=True)
            
            return {
                "ljung_box_statistic": float(ljung_box['lb_stat'].iloc[-1]),
                "ljung_box_pvalue": float(ljung_box['lb_pvalue'].iloc[-1]),
                "residual_mean": float(residuals.mean()),
                "residual_std": float(residuals.std()),
                "model_summary": str(self.fitted_model.summary()) if hasattr(self.fitted_model, 'summary') else "Not available"
            }
        except Exception as e:
            return {"error": str(e)}


class AutoARIMA(ARIMAForecaster):
    """Auto-ARIMA implementation with enhanced automatic model selection."""
    
    def __init__(self, config: Optional[ForecastingConfig] = None):
        """Initialize Auto-ARIMA with automatic order selection."""
        super().__init__(order=None, config=config)
        self.auto_select = True
    
    def _auto_select_order(self, series: pd.Series) -> Tuple[int, int, int]:
        """
        Enhanced automatic order selection with more sophisticated criteria.
        
        Args:
            series: Time series data
            
        Returns:
            Optimal (p, d, q) order
        """
        if not STATSMODELS_AVAILABLE:
            return (1, 1, 1)
        
        # Determine differencing order
        d = self._determine_differencing_order(series)
        
        # Use information criteria for model selection
        best_criteria = float('inf')
        best_order = (1, d, 1)
        
        candidates = []
        
        # Grid search with expanded range
        max_p = min(self.config.arima_max_p + 2, len(series) // 8)
        max_q = min(self.config.arima_max_q + 2, len(series) // 8)
        
        for p in range(0, max_p + 1):
            for q in range(0, max_q + 1):
                # Skip if model is too complex for data size
                if p + q > len(series) // 5:
                    continue
                
                try:
                    model = ARIMA(series, order=(p, d, q))
                    fitted_model = model.fit()
                    
                    # Use BIC (more conservative than AIC)
                    criteria = fitted_model.bic
                    
                    candidates.append({
                        'order': (p, d, q),
                        'aic': fitted_model.aic,
                        'bic': fitted_model.bic,
                        'hqic': fitted_model.hqic,
                        'criteria': criteria
                    })
                    
                    if criteria < best_criteria:
                        best_criteria = criteria
                        best_order = (p, d, q)
                except Exception:
                    continue
        
        # Store candidate models for analysis
        self.candidate_models = sorted(candidates, key=lambda x: x['criteria'])[:5]
        
        return best_order