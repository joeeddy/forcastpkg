"""Advanced usage examples for the forecasting package."""

import json
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any

from forcasting_pkg import ForecastingEngine, TechnicalAnalyzer
from forcasting_pkg.data import DataSourceManager, YahooFinanceSource, MockDataSource
from forcasting_pkg.forecasting import ARIMAForecaster, LinearForecaster, EnsembleForecaster
from forcasting_pkg.models import ForecastingConfig, DataType, MarketData


class CustomForecaster:
    """Example of a custom forecasting model."""
    
    def __init__(self, config=None):
        self.config = config or ForecastingConfig()
        self.is_fitted = False
        self.data = None
        self.symbol = None
    
    def fit(self, data: List[MarketData], symbol: str = "UNKNOWN"):
        """Simple trend-following model."""
        self.symbol = symbol
        self.data = data
        self.is_fitted = True
        return self
    
    def predict(self, steps: int = None):
        """Generate predictions using simple trend extrapolation."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        steps = steps or self.config.default_forecast_days
        
        # Calculate simple trend from last 10 days
        recent_prices = [d.close for d in self.data[-10:]]
        trend = (recent_prices[-1] - recent_prices[0]) / len(recent_prices)
        
        # Generate forecast points
        forecast_points = []
        last_price = self.data[-1].close
        last_date = self.data[-1].date
        
        for i in range(steps):
            future_date = last_date + timedelta(days=i+1)
            predicted_value = last_price + (trend * (i + 1))
            
            # Simple confidence intervals
            volatility = pd.Series(recent_prices).std()
            margin = 1.96 * volatility
            
            from forcasting_pkg.models import ForecastPoint
            point = ForecastPoint(
                date=future_date,
                predicted_value=predicted_value,
                confidence_interval_lower=predicted_value - margin,
                confidence_interval_upper=predicted_value + margin
            )
            forecast_points.append(point)
        
        from forcasting_pkg.models import ForecastResult
        return ForecastResult(
            symbol=self.symbol,
            model_type="Custom_Trend_Following",
            forecast_period_days=steps,
            forecast_points=forecast_points,
            model_accuracy=0.75,  # Mock accuracy
            generated_at=datetime.now(),
            data_source="custom_model"
        )


def advanced_data_source_example():
    """Demonstrate advanced data source management."""
    print("=== Advanced Data Source Example ===")
    
    # Create custom data source manager
    data_manager = DataSourceManager()
    
    # Add multiple data sources with priorities
    if YahooFinanceSource:
        try:
            yahoo_source = YahooFinanceSource()
            data_manager.add_source(yahoo_source, priority=0)  # Highest priority
            print("Added Yahoo Finance source")
        except Exception:
            print("Yahoo Finance not available")
    
    # Add mock source as backup
    mock_source = MockDataSource()
    data_manager.add_source(mock_source, priority=1)
    print("Added Mock data source as backup")
    
    # List available sources
    sources = data_manager.list_sources()
    print(f"Available sources: {sources}")
    
    # Fetch data with fallback
    symbols = ["AAPL", "INVALID_SYMBOL", "MSFT"]
    
    for symbol in symbols:
        print(f"\nFetching data for {symbol}...")
        try:
            data = data_manager.get_historical_data(symbol, days=30)
            print(f"  Success: {len(data)} data points")
            
            current_price = data_manager.get_current_price(symbol)
            print(f"  Current price: ${current_price:.2f}" if current_price else "  Current price: N/A")
        except Exception as e:
            print(f"  Failed: {e}")
    
    return data_manager


def custom_model_integration_example():
    """Demonstrate integrating custom forecasting models."""
    print("\n=== Custom Model Integration Example ===")
    
    # Get some data
    from forcasting_pkg.data import get_historical_data
    data = get_historical_data("AAPL", days=60)
    
    if not data:
        print("No data available for custom model example")
        return
    
    # Create and use custom forecaster directly
    print("Using custom forecaster directly...")
    custom_forecaster = CustomForecaster()
    custom_forecaster.fit(data, "AAPL")
    custom_forecast = custom_forecaster.predict(14)
    
    print(f"Custom model: {custom_forecast.model_type}")
    print(f"Predictions: {len(custom_forecast.forecast_points)}")
    
    # Integrate with ForecastingEngine
    print("\nIntegrating with ForecastingEngine...")
    engine = ForecastingEngine()
    
    # Add custom model to engine
    engine.add_custom_model("custom_trend", CustomForecaster)
    
    # List available models
    available_models = engine.available_models()
    print(f"Available models: {available_models}")
    
    # Use custom model through engine
    custom_forecast_2 = engine.forecast(data, "custom_trend", "AAPL", days=14)
    print(f"Custom forecast through engine: {custom_forecast_2.model_type}")
    
    return custom_forecast, custom_forecast_2


def portfolio_analysis_example():
    """Demonstrate portfolio-level analysis."""
    print("\n=== Portfolio Analysis Example ===")
    
    # Define portfolio
    portfolio = {
        "AAPL": 0.3,
        "MSFT": 0.25,
        "GOOGL": 0.25,
        "TSLA": 0.2
    }
    
    print(f"Analyzing portfolio: {portfolio}")
    
    # Get data for all symbols
    portfolio_data = {}
    portfolio_forecasts = {}
    
    engine = ForecastingEngine()
    analyzer = TechnicalAnalyzer()
    
    for symbol, weight in portfolio.items():
        print(f"\nAnalyzing {symbol} (weight: {weight:.1%})...")
        
        try:
            # Get historical data
            from forcasting_pkg.data import get_historical_data
            data = get_historical_data(symbol, days=90)
            
            if not data:
                print(f"  No data for {symbol}")
                continue
            
            portfolio_data[symbol] = data
            
            # Generate forecast
            forecast = engine.forecast(data, "arima", symbol, days=30)
            portfolio_forecasts[symbol] = forecast
            
            # Technical analysis
            analysis = analyzer.analyze(data, symbol)
            
            print(f"  Forecast accuracy: {forecast.model_accuracy:.2%}" if forecast.model_accuracy else "  Accuracy: N/A")
            print(f"  Signal: {analysis.signal_strength}")
            print(f"  30-day prediction: ${forecast.forecast_points[0].predicted_value:.2f}")
            
        except Exception as e:
            print(f"  Error analyzing {symbol}: {e}")
    
    # Calculate portfolio-level metrics
    if portfolio_forecasts:
        print(f"\n=== Portfolio Summary ===")
        
        # Weighted average accuracy
        total_weight = 0
        weighted_accuracy = 0
        
        for symbol, forecast in portfolio_forecasts.items():
            weight = portfolio.get(symbol, 0)
            if forecast.model_accuracy:
                weighted_accuracy += forecast.model_accuracy * weight
                total_weight += weight
        
        if total_weight > 0:
            avg_accuracy = weighted_accuracy / total_weight
            print(f"Portfolio weighted forecast accuracy: {avg_accuracy:.2%}")
        
        # Portfolio predictions summary
        print("\nPortfolio 30-day outlook:")
        for symbol, forecast in portfolio_forecasts.items():
            weight = portfolio.get(symbol, 0)
            first_pred = forecast.forecast_points[0].predicted_value
            print(f"  {symbol} ({weight:.1%}): ${first_pred:.2f}")
    
    return portfolio_data, portfolio_forecasts


def backtesting_example():
    """Demonstrate simple backtesting functionality."""
    print("\n=== Backtesting Example ===")
    
    from forcasting_pkg.data import get_historical_data
    
    # Get extended historical data
    symbol = "AAPL"
    print(f"Backtesting forecasts for {symbol}...")
    
    data = get_historical_data(symbol, days=180)  # 6 months
    
    if not data or len(data) < 90:
        print("Insufficient data for backtesting")
        return
    
    print(f"Using {len(data)} data points for backtesting")
    
    # Split data: first 120 days for training, last 60 for testing
    train_data = data[:120]
    test_data = data[120:]
    
    print(f"Training data: {len(train_data)} points")
    print(f"Testing data: {len(test_data)} points")
    
    # Test different forecast horizons
    forecast_horizons = [7, 14, 30]
    models = ["arima", "linear", "moving_average"]
    
    results = {}
    
    engine = ForecastingEngine()
    
    for horizon in forecast_horizons:
        results[horizon] = {}
        print(f"\nTesting {horizon}-day forecasts...")
        
        for model in models:
            print(f"  Testing {model} model...")
            
            errors = []
            predictions = []
            actuals = []
            
            # Rolling window backtesting
            for i in range(0, len(test_data) - horizon, 7):  # Weekly intervals
                # Use data up to this point for training
                current_train_data = train_data + test_data[:i]
                
                if len(current_train_data) < 30:
                    continue
                
                try:
                    # Generate forecast
                    forecast = engine.forecast(current_train_data, model, symbol, horizon)
                    
                    if forecast.forecast_points:
                        # Compare first prediction with actual
                        predicted = forecast.forecast_points[0].predicted_value
                        actual = test_data[i].close if i < len(test_data) else test_data[-1].close
                        
                        error = abs(predicted - actual) / actual * 100  # MAPE
                        errors.append(error)
                        predictions.append(predicted)
                        actuals.append(actual)
                
                except Exception as e:
                    print(f"    Error in iteration {i}: {e}")
                    continue
            
            if errors:
                avg_error = sum(errors) / len(errors)
                results[horizon][model] = {
                    'avg_error': avg_error,
                    'predictions': len(predictions),
                    'min_error': min(errors),
                    'max_error': max(errors)
                }
                print(f"    Average error: {avg_error:.2f}%")
            else:
                print(f"    No valid predictions for {model}")
    
    # Display results summary
    print(f"\n=== Backtesting Results Summary ===")
    for horizon, models_results in results.items():
        print(f"\n{horizon}-day forecasts:")
        for model, metrics in models_results.items():
            print(f"  {model:15}: {metrics['avg_error']:.2f}% avg error ({metrics['predictions']} predictions)")
    
    return results


def configuration_example():
    """Demonstrate advanced configuration options."""
    print("\n=== Configuration Example ===")
    
    # Create custom configuration
    config = ForecastingConfig(
        default_forecast_days=21,
        confidence_level=0.90,  # 90% confidence intervals
        min_data_points=50,
        arima_auto_select=True,
        arima_max_p=5,
        arima_max_d=2,
        arima_max_q=5,
        linear_polynomial_degree=2,
        moving_average_window=15
    )
    
    print("Created custom configuration:")
    print(f"  Default forecast days: {config.default_forecast_days}")
    print(f"  Confidence level: {config.confidence_level}")
    print(f"  Min data points: {config.min_data_points}")
    print(f"  ARIMA max order: ({config.arima_max_p}, {config.arima_max_d}, {config.arima_max_q})")
    
    # Use configuration with forecasting engine
    engine = ForecastingEngine(config=config)
    
    from forcasting_pkg.data import get_historical_data
    data = get_historical_data("MSFT", days=90)
    
    if data:
        print(f"\nGenerating forecast with custom config...")
        forecast = engine.forecast(data, "arima", "MSFT")
        
        print(f"Forecast generated: {forecast.forecast_period_days} days")
        print(f"Model parameters: {forecast.model_parameters}")
        
        # Check confidence intervals
        if forecast.forecast_points:
            first_point = forecast.forecast_points[0]
            if first_point.confidence_interval_lower and first_point.confidence_interval_upper:
                interval_width = first_point.confidence_interval_upper - first_point.confidence_interval_lower
                print(f"Confidence interval width: ${interval_width:.2f}")
    
    return config, forecast if 'forecast' in locals() else None


def export_import_example():
    """Demonstrate exporting and importing forecast results."""
    print("\n=== Export/Import Example ===")
    
    # Generate some forecasts
    from forcasting_pkg.data import get_historical_data
    data = get_historical_data("AAPL", days=60)
    
    if not data:
        print("No data available for export example")
        return
    
    engine = ForecastingEngine()
    
    # Generate multiple forecasts
    forecasts = {}
    models = ["arima", "linear", "moving_average"]
    
    for model in models:
        try:
            forecast = engine.forecast(data, model, "AAPL", days=14)
            forecasts[model] = forecast
            print(f"Generated {model} forecast")
        except Exception as e:
            print(f"Failed to generate {model} forecast: {e}")
    
    # Export to JSON
    export_data = {
        "symbol": "AAPL",
        "generated_at": datetime.now().isoformat(),
        "forecasts": {}
    }
    
    for model_name, forecast in forecasts.items():
        export_data["forecasts"][model_name] = forecast.dict()
    
    # Save to file
    filename = "forecasts_export.json"
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    
    print(f"Exported forecasts to {filename}")
    
    # Read back and display
    with open(filename, 'r') as f:
        imported_data = json.load(f)
    
    print(f"Imported data for {imported_data['symbol']}")
    print(f"Generated at: {imported_data['generated_at']}")
    print(f"Available forecasts: {list(imported_data['forecasts'].keys())}")
    
    return export_data


def main():
    """Run all advanced examples."""
    print("Forecasting Package - Advanced Usage Examples")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run advanced examples
        advanced_data_source_example()
        custom_model_integration_example()
        portfolio_analysis_example()
        backtesting_example()
        configuration_example()
        export_import_example()
        
        print(f"\n" + "=" * 60)
        print("All advanced examples completed successfully!")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()