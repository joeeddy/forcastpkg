"""Basic usage examples for the forecasting package."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import asyncio
from datetime import datetime

from forcasting_pkg import ForecastingEngine, TechnicalAnalyzer
from forcasting_pkg.data import get_historical_data, get_current_price
from forcasting_pkg.models import DataType
from forcasting_pkg.visualization import plot_forecast, plot_technical_indicators


def basic_forecasting_example():
    """Demonstrate basic forecasting functionality."""
    print("=== Basic Forecasting Example ===")
    
    # Get historical data for Apple stock
    print("Fetching historical data for AAPL...")
    data = get_historical_data("AAPL", days=90, data_type=DataType.STOCK)
    
    if not data:
        print("Failed to fetch data. Using mock data...")
        return
    
    print(f"Loaded {len(data)} data points")
    
    # Create forecasting engine
    engine = ForecastingEngine()
    
    # Generate ARIMA forecast
    print("Generating ARIMA forecast...")
    arima_forecast = engine.forecast(data, "arima", "AAPL", days=30)
    
    print(f"ARIMA Model: {arima_forecast.model_type}")
    print(f"Accuracy: {arima_forecast.model_accuracy:.2%}" if arima_forecast.model_accuracy else "Accuracy: N/A")
    print(f"First prediction: ${arima_forecast.forecast_points[0].predicted_value:.2f}")
    
    # Generate linear forecast
    print("Generating Linear forecast...")
    linear_forecast = engine.forecast(data, "linear", "AAPL", days=30)
    
    print(f"Linear Model: {linear_forecast.model_type}")
    print(f"Accuracy: {linear_forecast.model_accuracy:.2%}" if linear_forecast.model_accuracy else "Accuracy: N/A")
    print(f"First prediction: ${linear_forecast.forecast_points[0].predicted_value:.2f}")
    
    return data, arima_forecast, linear_forecast


def technical_analysis_example():
    """Demonstrate technical analysis functionality."""
    print("\n=== Technical Analysis Example ===")
    
    # Get historical data for Tesla
    print("Fetching historical data for TSLA...")
    data = get_historical_data("TSLA", days=60, data_type=DataType.STOCK)
    
    if not data:
        print("Failed to fetch data. Using mock data...")
        return
    
    print(f"Loaded {len(data)} data points")
    
    # Create technical analyzer
    analyzer = TechnicalAnalyzer()
    
    # Perform analysis
    print("Performing technical analysis...")
    analysis = analyzer.analyze(data, "TSLA")
    
    # Display results
    print(f"Analysis for {analysis.symbol}:")
    print(f"Signal Strength: {analysis.signal_strength}")
    print(f"Confidence: {analysis.confidence:.2%}" if analysis.confidence else "Confidence: N/A")
    
    # Display indicators
    indicators = analysis.technical_indicators
    print("\nTechnical Indicators:")
    if indicators.rsi:
        print(f"  RSI: {indicators.rsi:.2f}")
    if indicators.macd:
        print(f"  MACD: {indicators.macd:.4f}")
    if indicators.moving_average_20:
        print(f"  20-day MA: ${indicators.moving_average_20:.2f}")
    if indicators.moving_average_50:
        print(f"  50-day MA: ${indicators.moving_average_50:.2f}")
    
    # Display recommendations
    if analysis.recommendations:
        print("\nRecommendations:")
        for rec in analysis.recommendations:
            print(f"  • {rec}")
    
    return data, analysis


def model_comparison_example():
    """Demonstrate model comparison functionality."""
    print("\n=== Model Comparison Example ===")
    
    # Get historical data for Microsoft
    print("Fetching historical data for MSFT...")
    data = get_historical_data("MSFT", days=90, data_type=DataType.STOCK)
    
    if not data:
        print("Failed to fetch data. Using mock data...")
        return
    
    print(f"Loaded {len(data)} data points")
    
    # Create forecasting engine
    engine = ForecastingEngine()
    
    # Compare multiple models
    print("Comparing forecasting models...")
    comparison = engine.compare_models(data, "MSFT", days=14)
    
    print(f"\nComparison Results for MSFT (14-day forecast):")
    print("-" * 50)
    
    best_model = None
    best_accuracy = -1
    
    for model_name, forecast in comparison.items():
        accuracy = forecast.model_accuracy if forecast.model_accuracy else 0
        print(f"{model_name:25} | Accuracy: {accuracy:.2%}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model_name
    
    if best_model:
        print(f"\nBest performing model: {best_model} ({best_accuracy:.2%})")
    
    return data, comparison


def crypto_forecasting_example():
    """Demonstrate cryptocurrency forecasting."""
    print("\n=== Cryptocurrency Forecasting Example ===")
    
    # Get Bitcoin data
    print("Fetching historical data for Bitcoin...")
    data = get_historical_data("bitcoin", days=60, data_type=DataType.CRYPTO)
    
    if not data:
        print("Failed to fetch data. Using mock data...")
        return
    
    print(f"Loaded {len(data)} data points")
    
    # Get current price
    current_price = get_current_price("bitcoin", DataType.CRYPTO)
    if current_price:
        print(f"Current Bitcoin price: ${current_price:,.2f}")
    
    # Create forecasting engine
    engine = ForecastingEngine()
    
    # Generate forecast
    print("Generating Bitcoin forecast...")
    forecast = engine.forecast(data, "arima", "BTC", days=7)
    
    print(f"Model: {forecast.model_type}")
    print(f"7-day forecast starting from: {forecast.forecast_points[0].date.strftime('%Y-%m-%d')}")
    
    # Show predictions
    print("\nPrice predictions:")
    for i, point in enumerate(forecast.forecast_points):
        if i < 3:  # Show first 3
            print(f"  Day {i+1}: ${point.predicted_value:,.2f}")
        elif i == 3:
            print(f"  ...")
        elif i == len(forecast.forecast_points) - 1:  # Show last
            print(f"  Day {i+1}: ${point.predicted_value:,.2f}")
    
    return data, forecast


def ensemble_forecasting_example():
    """Demonstrate ensemble forecasting."""
    print("\n=== Ensemble Forecasting Example ===")
    
    # Get data for Google
    print("Fetching historical data for GOOGL...")
    data = get_historical_data("GOOGL", days=90, data_type=DataType.STOCK)
    
    if not data:
        print("Failed to fetch data. Using mock data...")
        return
    
    print(f"Loaded {len(data)} data points")
    
    # Create forecasting engine
    engine = ForecastingEngine()
    
    # Create ensemble forecast
    print("Creating ensemble forecast...")
    ensemble_forecast = engine.create_ensemble(
        data, 
        "GOOGL", 
        models=["arima", "linear", "moving_average"],
        weights=[0.5, 0.3, 0.2],  # Higher weight for ARIMA
        days=21
    )
    
    print(f"Ensemble Model: {ensemble_forecast.model_type}")
    print(f"Accuracy: {ensemble_forecast.model_accuracy:.2%}" if ensemble_forecast.model_accuracy else "Accuracy: N/A")
    print(f"21-day forecast generated with {len(ensemble_forecast.forecast_points)} predictions")
    
    # Show some predictions
    print("\nSample predictions:")
    for i in [0, 6, 13, 20]:  # Weekly intervals
        if i < len(ensemble_forecast.forecast_points):
            point = ensemble_forecast.forecast_points[i]
            print(f"  Week {i//7 + 1}: ${point.predicted_value:.2f}")
    
    return data, ensemble_forecast


def visualization_example():
    """Demonstrate visualization capabilities."""
    print("\n=== Visualization Example ===")
    
    try:
        # Get data and generate forecast
        data = get_historical_data("AAPL", days=60)
        if not data:
            print("No data available for visualization example")
            return
        
        engine = ForecastingEngine()
        forecast = engine.forecast(data, "arima", "AAPL", days=14)
        
        # Generate technical analysis
        analyzer = TechnicalAnalyzer()
        analysis = analyzer.analyze(data, "AAPL")
        
        print("Generating forecast plot...")
        # Save forecast plot
        plot_forecast(data, forecast, "AAPL", save_path="forecast_example.png")
        print("Forecast plot saved as 'forecast_example.png'")
        
        print("Generating technical analysis plot...")
        # Save technical analysis plot
        plot_technical_indicators(
            data, 
            analysis.technical_indicators, 
            "AAPL", 
            save_path="technical_example.png"
        )
        print("Technical analysis plot saved as 'technical_example.png'")
        
    except Exception as e:
        print(f"Visualization example failed: {e}")
        print("This is likely due to missing optional dependencies (matplotlib/plotly)")


def main():
    """Run all examples."""
    print("Forecasting Package - Usage Examples")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run examples
        basic_forecasting_example()
        technical_analysis_example()
        model_comparison_example()
        crypto_forecasting_example()
        ensemble_forecasting_example()
        visualization_example()
        
        print(f"\n" + "=" * 50)
        print("All examples completed successfully!")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure the package is installed correctly.")


if __name__ == "__main__":
    main()