"""Command Line Interface for the forecasting package."""

import argparse
import sys
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd

from ..forecasting import ForecastingEngine
from ..analysis import TechnicalAnalyzer
from ..data import default_data_source
from ..models import DataType, ModelType, ForecastingConfig
from ..visualization import plot_forecast, plot_technical_indicators


def create_parser() -> argparse.ArgumentParser:
    """Create the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Forecasting Package CLI - Financial forecasting and technical analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate ARIMA forecast for AAPL
  forcasting-cli forecast AAPL --model arima --days 30

  # Technical analysis for Bitcoin
  forcasting-cli analyze BTC --crypto --plot

  # Compare multiple models
  forcasting-cli compare MSFT --models arima,linear,moving_average

  # Generate forecast with custom parameters
  forcasting-cli forecast GOOGL --model linear --days 60 --output forecast.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Forecast command
    forecast_parser = subparsers.add_parser('forecast', help='Generate forecasts')
    forecast_parser.add_argument('symbol', help='Financial symbol to forecast')
    forecast_parser.add_argument('--model', choices=['arima', 'linear', 'moving_average'], 
                                default='arima', help='Forecasting model to use')
    forecast_parser.add_argument('--days', type=int, default=30, help='Number of days to forecast')
    forecast_parser.add_argument('--crypto', action='store_true', help='Treat symbol as cryptocurrency')
    forecast_parser.add_argument('--historical-days', type=int, default=90, 
                                help='Number of historical days to use')
    forecast_parser.add_argument('--output', help='Output file path (JSON)')
    forecast_parser.add_argument('--plot', help='Save forecast plot to file')
    forecast_parser.add_argument('--interactive', action='store_true', help='Create interactive plot')
    
    # Analysis command
    analysis_parser = subparsers.add_parser('analyze', help='Perform technical analysis')
    analysis_parser.add_argument('symbol', help='Financial symbol to analyze')
    analysis_parser.add_argument('--crypto', action='store_true', help='Treat symbol as cryptocurrency')
    analysis_parser.add_argument('--historical-days', type=int, default=90,
                                help='Number of historical days to use')
    analysis_parser.add_argument('--output', help='Output file path (JSON)')
    analysis_parser.add_argument('--plot', help='Save technical analysis plot to file')
    analysis_parser.add_argument('--interactive', action='store_true', help='Create interactive plot')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple forecasting models')
    compare_parser.add_argument('symbol', help='Financial symbol to forecast')
    compare_parser.add_argument('--models', default='arima,linear,moving_average',
                               help='Comma-separated list of models to compare')
    compare_parser.add_argument('--days', type=int, default=30, help='Number of days to forecast')
    compare_parser.add_argument('--crypto', action='store_true', help='Treat symbol as cryptocurrency')
    compare_parser.add_argument('--historical-days', type=int, default=90,
                                help='Number of historical days to use')
    compare_parser.add_argument('--output', help='Output file path (JSON)')
    compare_parser.add_argument('--plot', help='Save comparison plot to file')
    compare_parser.add_argument('--interactive', action='store_true', help='Create interactive plot')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show package information')
    info_parser.add_argument('--models', action='store_true', help='Show available models')
    info_parser.add_argument('--version', action='store_true', help='Show version information')
    
    return parser


def forecast_command(args) -> int:
    """Handle forecast command."""
    try:
        print(f"Generating {args.model} forecast for {args.symbol}...")
        
        # Determine data type
        data_type = DataType.CRYPTO if args.crypto else DataType.STOCK
        
        # Get historical data
        print(f"Fetching {args.historical_days} days of historical data...")
        historical_data = default_data_source.get_historical_data(
            args.symbol, args.historical_days, data_type
        )
        
        if not historical_data:
            print(f"Error: No historical data found for {args.symbol}")
            return 1
        
        print(f"Found {len(historical_data)} data points")
        
        # Create forecasting engine
        engine = ForecastingEngine()
        
        # Generate forecast
        print(f"Generating forecast using {args.model} model...")
        forecast_result = engine.forecast(
            historical_data, 
            args.model, 
            args.symbol, 
            args.days
        )
        
        # Display results
        print(f"\nForecast Results for {forecast_result.symbol}:")
        print(f"Model: {forecast_result.model_type}")
        print(f"Forecast period: {forecast_result.forecast_period_days} days")
        if forecast_result.model_accuracy:
            print(f"Model accuracy: {forecast_result.model_accuracy:.2%}")
        
        # Show first few predictions
        print(f"\nFirst 5 predictions:")
        for i, point in enumerate(forecast_result.forecast_points[:5]):
            print(f"  {point.date.strftime('%Y-%m-%d')}: ${point.predicted_value:.2f}")
            if point.confidence_interval_lower and point.confidence_interval_upper:
                print(f"    (CI: ${point.confidence_interval_lower:.2f} - ${point.confidence_interval_upper:.2f})")
        
        if len(forecast_result.forecast_points) > 5:
            print(f"  ... and {len(forecast_result.forecast_points) - 5} more predictions")
        
        # Save to file if requested
        if args.output:
            output_data = {
                "forecast_result": forecast_result.dict(),
                "generated_at": datetime.now().isoformat(),
                "command_args": vars(args)
            }
            
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            print(f"\nResults saved to {args.output}")
        
        # Create plot if requested
        if args.plot:
            print(f"\nGenerating plot...")
            plot_forecast(
                historical_data, 
                forecast_result, 
                args.symbol,
                save_path=args.plot,
                interactive=args.interactive
            )
            print(f"Plot saved to {args.plot}")
        
        return 0
        
    except Exception as e:
        print(f"Error generating forecast: {e}")
        return 1


def analyze_command(args) -> int:
    """Handle analysis command."""
    try:
        print(f"Performing technical analysis for {args.symbol}...")
        
        # Determine data type
        data_type = DataType.CRYPTO if args.crypto else DataType.STOCK
        
        # Get historical data
        print(f"Fetching {args.historical_days} days of historical data...")
        historical_data = default_data_source.get_historical_data(
            args.symbol, args.historical_days, data_type
        )
        
        if not historical_data:
            print(f"Error: No historical data found for {args.symbol}")
            return 1
        
        print(f"Found {len(historical_data)} data points")
        
        # Create analyzer
        analyzer = TechnicalAnalyzer()
        
        # Perform analysis
        print("Calculating technical indicators...")
        analysis_result = analyzer.analyze(historical_data, args.symbol)
        
        # Display results
        print(f"\nTechnical Analysis for {analysis_result.symbol}:")
        print(f"Analysis date: {analysis_result.analysis_date.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if analysis_result.signal_strength:
            print(f"Signal strength: {analysis_result.signal_strength}")
        if analysis_result.confidence:
            print(f"Confidence: {analysis_result.confidence:.2%}")
        
        # Display indicators
        indicators = analysis_result.technical_indicators
        print(f"\nTechnical Indicators:")
        
        if indicators.rsi:
            print(f"  RSI: {indicators.rsi:.2f}")
        if indicators.macd:
            print(f"  MACD: {indicators.macd:.4f}")
        if indicators.macd_signal:
            print(f"  MACD Signal: {indicators.macd_signal:.4f}")
        if indicators.bollinger_upper and indicators.bollinger_lower:
            print(f"  Bollinger Bands: {indicators.bollinger_lower:.2f} - {indicators.bollinger_upper:.2f}")
        if indicators.moving_average_20:
            print(f"  20-day MA: {indicators.moving_average_20:.2f}")
        if indicators.moving_average_50:
            print(f"  50-day MA: {indicators.moving_average_50:.2f}")
        
        # Display recommendations
        if analysis_result.recommendations:
            print(f"\nRecommendations:")
            for rec in analysis_result.recommendations:
                print(f"  • {rec}")
        
        # Save to file if requested
        if args.output:
            output_data = {
                "analysis_result": analysis_result.dict(),
                "generated_at": datetime.now().isoformat(),
                "command_args": vars(args)
            }
            
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            print(f"\nResults saved to {args.output}")
        
        # Create plot if requested
        if args.plot:
            print(f"\nGenerating plot...")
            plot_technical_indicators(
                historical_data,
                analysis_result.technical_indicators,
                args.symbol,
                save_path=args.plot,
                interactive=args.interactive
            )
            print(f"Plot saved to {args.plot}")
        
        return 0
        
    except Exception as e:
        print(f"Error performing analysis: {e}")
        return 1


def compare_command(args) -> int:
    """Handle compare command."""
    try:
        print(f"Comparing forecasting models for {args.symbol}...")
        
        # Parse models
        model_names = [model.strip().lower() for model in args.models.split(',')]
        valid_models = [m for m in model_names if m in ['arima', 'linear', 'moving_average']]
        
        if not valid_models:
            print("Error: No valid models specified")
            return 1
        
        print(f"Models to compare: {', '.join(valid_models)}")
        
        # Determine data type
        data_type = DataType.CRYPTO if args.crypto else DataType.STOCK
        
        # Get historical data
        print(f"Fetching {args.historical_days} days of historical data...")
        historical_data = default_data_source.get_historical_data(
            args.symbol, args.historical_days, data_type
        )
        
        if not historical_data:
            print(f"Error: No historical data found for {args.symbol}")
            return 1
        
        print(f"Found {len(historical_data)} data points")
        
        # Create forecasting engine
        engine = ForecastingEngine()
        
        # Compare models
        print("Generating forecasts with different models...")
        comparison_results = engine.compare_models(
            historical_data,
            args.symbol,
            valid_models,
            args.days
        )
        
        if not comparison_results:
            print("Error: No models produced valid forecasts")
            return 1
        
        # Display results
        print(f"\nModel Comparison Results for {args.symbol}:")
        print(f"Forecast period: {args.days} days")
        print("-" * 60)
        
        for model_name, forecast in comparison_results.items():
            accuracy_text = f"{forecast.model_accuracy:.2%}" if forecast.model_accuracy else "N/A"
            print(f"{model_name:30} | Accuracy: {accuracy_text:>8}")
            
            # Show first prediction
            if forecast.forecast_points:
                first_pred = forecast.forecast_points[0]
                print(f"{'':30} | First prediction: ${first_pred.predicted_value:.2f}")
        
        # Find best model
        best_model = None
        best_accuracy = -1
        for model_name, forecast in comparison_results.items():
            if forecast.model_accuracy and forecast.model_accuracy > best_accuracy:
                best_accuracy = forecast.model_accuracy
                best_model = model_name
        
        if best_model:
            print(f"\nBest performing model: {best_model} (Accuracy: {best_accuracy:.2%})")
        
        # Save to file if requested
        if args.output:
            output_data = {
                "comparison_results": {k: v.dict() for k, v in comparison_results.items()},
                "best_model": best_model,
                "best_accuracy": best_accuracy,
                "generated_at": datetime.now().isoformat(),
                "command_args": vars(args)
            }
            
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            print(f"\nResults saved to {args.output}")
        
        # Create plot if requested
        if args.plot:
            print(f"\nGenerating comparison plot...")
            from ..visualization import ForecastVisualizer
            visualizer = ForecastVisualizer()
            visualizer.plot_model_comparison(
                historical_data,
                comparison_results,
                args.symbol,
                save_path=args.plot,
                interactive=args.interactive
            )
            print(f"Plot saved to {args.plot}")
        
        return 0
        
    except Exception as e:
        print(f"Error comparing models: {e}")
        return 1


def info_command(args) -> int:
    """Handle info command."""
    try:
        if args.version:
            from .. import __version__, __author__
            print(f"Forecasting Package version {__version__}")
            print(f"Author: {__author__}")
            return 0
        
        if args.models:
            engine = ForecastingEngine()
            models = engine.available_models()
            
            print("Available forecasting models:")
            for model in models:
                try:
                    info = engine.model_info(model)
                    print(f"  {model}: {info.get('description', 'No description')}")
                except:
                    print(f"  {model}: Available")
            return 0
        
        # Default info
        print("Forecasting Package - Financial forecasting and technical analysis toolkit")
        print("")
        print("Available commands:")
        print("  forecast  - Generate forecasts using various models")
        print("  analyze   - Perform technical analysis")
        print("  compare   - Compare multiple forecasting models")
        print("  info      - Show package information")
        print("")
        print("Use 'forcasting-cli <command> --help' for detailed help on each command")
        
        return 0
        
    except Exception as e:
        print(f"Error displaying info: {e}")
        return 1


def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    
    args = parser.parse_args()
    
    if args.command == 'forecast':
        return forecast_command(args)
    elif args.command == 'analyze':
        return analyze_command(args)
    elif args.command == 'compare':
        return compare_command(args)
    elif args.command == 'info':
        return info_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())