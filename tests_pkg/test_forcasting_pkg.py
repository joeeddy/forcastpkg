"""Test suite for the forecasting package."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List

from forcasting_pkg.models import MarketData, DataType, ForecastingConfig, ModelType
from forcasting_pkg.forecasting import ForecastingEngine, ARIMAForecaster, LinearForecaster, MovingAverageForecaster
from forcasting_pkg.analysis import TechnicalAnalyzer
from forcasting_pkg.data import MockDataSource, DataSourceManager


# Test fixtures
@pytest.fixture
def sample_market_data() -> List[MarketData]:
    """Create sample market data for testing."""
    data = []
    base_price = 100.0
    start_date = datetime.now() - timedelta(days=60)
    
    np.random.seed(42)  # For reproducible tests
    
    for i in range(60):
        # Generate realistic OHLC data
        daily_return = np.random.normal(0.001, 0.02)
        price = base_price * (1 + daily_return)
        
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = low + (high - low) * np.random.random()
        close_price = low + (high - low) * np.random.random()
        
        # Ensure OHLC relationships
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)
        
        data_point = MarketData(
            date=start_date + timedelta(days=i),
            open=round(open_price, 2),
            high=round(high, 2),
            low=round(low, 2),
            close=round(close_price, 2),
            volume=np.random.randint(1000000, 10000000),
            adjusted_close=round(close_price, 2)
        )
        data.append(data_point)
        base_price = close_price
    
    return data


@pytest.fixture
def forecasting_config() -> ForecastingConfig:
    """Create test forecasting configuration."""
    return ForecastingConfig(
        default_forecast_days=14,
        confidence_level=0.95,
        min_data_points=10,  # Lower for testing
        arima_max_p=2,
        arima_max_d=1,
        arima_max_q=2
    )


class TestMarketDataModel:
    """Test MarketData model validation."""
    
    def test_valid_market_data(self):
        """Test valid market data creation."""
        data = MarketData(
            date=datetime.now(),
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000000
        )
        assert data.open == 100.0
        assert data.high == 105.0
        assert data.low == 95.0
        assert data.close == 102.0
    
    def test_invalid_price_values(self):
        """Test that invalid price values raise errors."""
        with pytest.raises(ValueError):
            MarketData(
                date=datetime.now(),
                open=-100.0,  # Negative price
                high=105.0,
                low=95.0,
                close=102.0
            )


class TestDataSources:
    """Test data source functionality."""
    
    def test_mock_data_source(self):
        """Test mock data source."""
        source = MockDataSource()
        
        # Test historical data
        data = source.get_historical_data("AAPL", days=30)
        assert len(data) == 30
        assert all(isinstance(d, MarketData) for d in data)
        
        # Test current price
        price = source.get_current_price("AAPL")
        assert isinstance(price, float)
        assert price > 0
    
    def test_data_source_manager(self):
        """Test data source manager with fallback."""
        manager = DataSourceManager([MockDataSource()])
        
        # Test with valid symbol
        data = manager.get_historical_data("AAPL", days=30)
        assert len(data) == 30
        
        # Test current price
        price = manager.get_current_price("AAPL")
        assert price is not None


class TestForecasting:
    """Test forecasting functionality."""
    
    def test_linear_forecaster(self, sample_market_data, forecasting_config):
        """Test linear regression forecaster."""
        forecaster = LinearForecaster(degree=1, config=forecasting_config)
        
        # Fit the model
        forecaster.fit(sample_market_data, "TEST")
        assert forecaster.is_fitted
        
        # Generate predictions
        forecast = forecaster.predict(7)
        assert forecast.symbol == "TEST"
        assert len(forecast.forecast_points) == 7
        assert forecast.model_type == ModelType.LINEAR
        
        # Check predictions are reasonable
        for point in forecast.forecast_points:
            assert point.predicted_value > 0
            assert point.confidence_interval_lower is not None
            assert point.confidence_interval_upper is not None
    
    def test_moving_average_forecaster(self, sample_market_data, forecasting_config):
        """Test moving average forecaster."""
        forecaster = MovingAverageForecaster(window=10, config=forecasting_config)
        
        # Fit the model
        forecaster.fit(sample_market_data, "TEST")
        assert forecaster.is_fitted
        
        # Generate predictions
        forecast = forecaster.predict(5)
        assert len(forecast.forecast_points) == 5
        assert forecast.model_type == ModelType.MOVING_AVERAGE
    
    def test_arima_forecaster_fallback(self, sample_market_data, forecasting_config):
        """Test ARIMA forecaster (should fallback to linear if statsmodels not available)."""
        forecaster = ARIMAForecaster(config=forecasting_config)
        
        # Fit the model
        forecaster.fit(sample_market_data, "TEST")
        assert forecaster.is_fitted
        
        # Generate predictions
        forecast = forecaster.predict(7)
        assert len(forecast.forecast_points) == 7
        # Should either be ARIMA or fallback to linear
        assert forecast.model_type in [ModelType.ARIMA, ModelType.LINEAR] or "ARIMA" in str(forecast.model_type)
    
    def test_forecasting_engine(self, sample_market_data):
        """Test forecasting engine."""
        engine = ForecastingEngine()
        
        # Test available models
        models = engine.available_models()
        assert ModelType.LINEAR in models
        
        # Test forecasting
        forecast = engine.forecast(sample_market_data, "linear", "TEST", days=7)
        assert forecast.symbol == "TEST"
        assert len(forecast.forecast_points) == 7
        
        # Test model comparison
        comparison = engine.compare_models(sample_market_data, "TEST", days=5)
        assert len(comparison) > 0
        assert all("TEST" in result.symbol for result in comparison.values())


class TestTechnicalAnalysis:
    """Test technical analysis functionality."""
    
    def test_technical_analyzer(self, sample_market_data):
        """Test technical analyzer."""
        analyzer = TechnicalAnalyzer()
        
        # Test individual indicators
        rsi = analyzer.calculate_rsi(sample_market_data)
        assert rsi is None or (0 <= rsi <= 100)
        
        macd = analyzer.calculate_macd(sample_market_data)
        assert isinstance(macd, dict)
        assert 'macd' in macd
        
        bollinger = analyzer.calculate_bollinger_bands(sample_market_data)
        assert isinstance(bollinger, dict)
        assert 'upper' in bollinger
        assert 'lower' in bollinger
        
        # Test full analysis
        analysis = analyzer.analyze(sample_market_data, "TEST")
        assert analysis.symbol == "TEST"
        assert analysis.signal_strength in ["Buy", "Sell", "Hold"]
        assert 0 <= analysis.confidence <= 1
    
    def test_technical_indicators_calculation(self, sample_market_data):
        """Test technical indicators with sufficient data."""
        analyzer = TechnicalAnalyzer()
        
        # RSI calculation
        rsi = analyzer.calculate_rsi(sample_market_data, period=14)
        if rsi is not None:
            assert 0 <= rsi <= 100
        
        # Moving average calculation
        ma_20 = analyzer.calculate_moving_average(sample_market_data, 20)
        if ma_20 is not None:
            assert ma_20 > 0
        
        # Stochastic calculation
        stoch = analyzer.calculate_stochastic(sample_market_data)
        if stoch['%K'] is not None:
            assert 0 <= stoch['%K'] <= 100


class TestDataValidation:
    """Test data validation and error handling."""
    
    def test_insufficient_data(self, forecasting_config):
        """Test handling of insufficient data."""
        # Create minimal data
        minimal_data = [
            MarketData(
                date=datetime.now() - timedelta(days=i),
                open=100, high=105, low=95, close=102
            ) for i in range(5)
        ]
        
        forecaster = LinearForecaster(config=forecasting_config)
        
        # Should handle minimal data gracefully
        try:
            forecaster.fit(minimal_data, "TEST")
            forecast = forecaster.predict(3)
            assert len(forecast.forecast_points) == 3
        except ValueError as e:
            # Expected if truly insufficient data
            assert "insufficient" in str(e).lower()
    
    def test_invalid_model_type(self, sample_market_data):
        """Test handling of invalid model types."""
        engine = ForecastingEngine()
        
        with pytest.raises(ValueError):
            engine.forecast(sample_market_data, "invalid_model", "TEST")
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        analyzer = TechnicalAnalyzer()
        
        empty_data = []
        
        # Should handle empty data gracefully
        rsi = analyzer.calculate_rsi(empty_data)
        assert rsi is None


class TestConfiguration:
    """Test configuration functionality."""
    
    def test_custom_config(self, sample_market_data):
        """Test custom configuration usage."""
        config = ForecastingConfig(
            default_forecast_days=21,
            confidence_level=0.90,
            min_data_points=20
        )
        
        engine = ForecastingEngine(config=config)
        forecast = engine.forecast(sample_market_data, "linear", "TEST")
        
        # Should use custom forecast days
        assert forecast.forecast_period_days == 21
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid configuration
        config = ForecastingConfig(confidence_level=0.95)
        assert config.confidence_level == 0.95
        
        # Invalid confidence level should be handled by Pydantic
        with pytest.raises(ValueError):
            ForecastingConfig(confidence_level=1.5)  # > 1.0


class TestModelPersistence:
    """Test model parameters and metadata."""
    
    def test_model_parameters(self, sample_market_data, forecasting_config):
        """Test that models store and return parameters."""
        forecaster = LinearForecaster(config=forecasting_config)
        forecaster.fit(sample_market_data, "TEST")
        
        params = forecaster.get_model_parameters()
        assert isinstance(params, dict)
        assert "method" in params or "degree" in params
    
    def test_forecast_metadata(self, sample_market_data):
        """Test forecast result metadata."""
        engine = ForecastingEngine()
        forecast = engine.forecast(sample_market_data, "linear", "TEST", days=7)
        
        assert forecast.symbol == "TEST"
        assert forecast.forecast_period_days == 7
        assert isinstance(forecast.generated_at, datetime)
        assert forecast.model_parameters is not None


# Integration tests
class TestIntegration:
    """Integration tests for the full package."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from data to visualization."""
        # Get data
        source = MockDataSource()
        data = source.get_historical_data("AAPL", days=60)
        
        # Generate forecast
        engine = ForecastingEngine()
        forecast = engine.forecast(data, "linear", "AAPL", days=14)
        
        # Technical analysis
        analyzer = TechnicalAnalyzer()
        analysis = analyzer.analyze(data, "AAPL")
        
        # Validate results
        assert len(forecast.forecast_points) == 14
        assert analysis.symbol == "AAPL"
        assert forecast.symbol == "AAPL"
    
    def test_multiple_symbols(self):
        """Test processing multiple symbols."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        engine = ForecastingEngine()
        source = MockDataSource()
        
        results = {}
        
        for symbol in symbols:
            data = source.get_historical_data(symbol, days=30)
            forecast = engine.forecast(data, "linear", symbol, days=7)
            results[symbol] = forecast
        
        assert len(results) == 3
        assert all(results[symbol].symbol == symbol for symbol in symbols)


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])