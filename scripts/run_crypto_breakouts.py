#!/usr/bin/env python3
"""
Example script demonstrating the crypto breakout detection and forecasting workflow.

This script fetches cryptocurrency data from MEXC exchange, applies breakout detection,
and runs forecasting on detected candidates.

Usage:
    python scripts/run_crypto_breakouts.py [options]

Options:
    --symbols LIMIT        Number of symbols to analyze (default: 20)
    --days DAYS           Days of historical data (default: 30)
    --forecast DAYS       Days to forecast (default: 7)
    --min-strength FLOAT  Minimum signal strength (default: 0.5)
    --candidates LIMIT    Max candidates to forecast (default: 5)
    --output PATH         Output directory for results
    --mock               Use mock data (for testing)
    --help               Show this help message
"""

import sys
import argparse
from datetime import datetime
from pathlib import Path

# Add the parent directory to the path so we can import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from forcasting_pkg.crypto import CryptoBreakoutPipeline, create_default_pipeline
    from forcasting_pkg.crypto.breakout_detection import BreakoutDetector
    from forcasting_pkg.crypto.data_ingestion import create_mexc_data_source
    from forcasting_pkg.forecasting import ForecastingEngine
except ImportError as e:
    print(f"Error importing forcasting_pkg: {e}")
    print("Please ensure the package is installed: pip install -e .")
    sys.exit(1)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Crypto breakout detection and forecasting pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--symbols', 
        type=int, 
        default=20,
        help='Number of symbols to analyze (default: 20)'
    )
    
    parser.add_argument(
        '--days', 
        type=int, 
        default=30,
        help='Days of historical data to fetch (default: 30)'
    )
    
    parser.add_argument(
        '--forecast', 
        type=int, 
        default=7,
        help='Days to forecast for candidates (default: 7)'
    )
    
    parser.add_argument(
        '--min-strength', 
        type=float, 
        default=0.5,
        help='Minimum breakout signal strength (default: 0.5)'
    )
    
    parser.add_argument(
        '--candidates', 
        type=int, 
        default=5,
        help='Maximum candidates to forecast (default: 5)'
    )
    
    parser.add_argument(
        '--output', 
        type=str,
        help='Output directory for results (default: auto-generated)'
    )
    
    parser.add_argument(
        '--mock', 
        action='store_true',
        help='Use mock data instead of real MEXC API'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def print_breakout_signals(signals, limit=10):
    """Print breakout signals in a formatted table."""
    if not signals:
        print("No breakout signals detected.")
        return
    
    print(f"\n📈 Top {min(limit, len(signals))} Breakout Signals:")
    print("-" * 80)
    print(f"{'Symbol':<12} {'Type':<16} {'Strength':<8} {'Price':<10} {'Change%':<8} {'Vol Ratio':<8}")
    print("-" * 80)
    
    for signal in signals[:limit]:
        print(f"{signal.symbol:<12} {signal.signal_type:<16} {signal.strength:<8.3f} "
              f"${signal.current_price:<9.2f} {signal.price_change_percent:<7.2f}% "
              f"{signal.volume_ratio:<7.2f}x")


def print_forecasts(forecasts, limit=10):
    """Print forecast results in a formatted table."""
    if not forecasts:
        print("No forecasts generated.")
        return
    
    print(f"\n🔮 Forecasts for Top {min(limit, len(forecasts))} Candidates:")
    print("-" * 90)
    print(f"{'Symbol':<12} {'Model':<12} {'Accuracy':<10} {'Current':<10} {'Forecast':<10} {'Change%':<8}")
    print("-" * 90)
    
    for symbol, forecast in list(forecasts.items())[:limit]:
        if forecast.forecast_points:
            current_price = forecast.forecast_points[0].predicted_value
            future_price = forecast.forecast_points[-1].predicted_value
            change_percent = ((future_price - current_price) / current_price) * 100
            
            accuracy = forecast.model_accuracy or 0.0
            
            print(f"{symbol:<12} {forecast.model_type:<12} {accuracy:<10.3f} "
                  f"${current_price:<9.2f} ${future_price:<9.2f} {change_percent:<7.2f}%")


def print_summary(summary):
    """Print pipeline execution summary."""
    exec_info = summary.get('execution_info', {})
    signal_summary = summary.get('signal_summary', {})
    forecast_summary = summary.get('forecast_summary', {})
    
    print("\n" + "="*60)
    print("📊 PIPELINE EXECUTION SUMMARY")
    print("="*60)
    
    # Execution info
    if exec_info:
        print(f"⏰ Execution time: {exec_info.get('execution_time', 'N/A')}")
        print(f"📈 Symbols analyzed: {exec_info.get('symbols_analyzed', 0)}")
        print(f"🎯 Signals detected: {exec_info.get('signals_detected', 0)}")
        print(f"🔮 Forecasts generated: {exec_info.get('forecasts_generated', 0)}")
    
    # Signal breakdown
    if signal_summary:
        print(f"\n🎯 Signal Analysis:")
        print(f"   Total signals: {signal_summary.get('total_signals', 0)}")
        print(f"   Average strength: {signal_summary.get('avg_strength', 0):.3f}")
        
        signal_types = signal_summary.get('signal_types', {})
        if signal_types:
            print(f"   Signal types:")
            for signal_type, count in signal_types.items():
                print(f"     - {signal_type}: {count}")
    
    # Forecast breakdown
    if forecast_summary:
        print(f"\n🔮 Forecast Analysis:")
        print(f"   Total forecasts: {forecast_summary.get('total_forecasts', 0)}")
        print(f"   Average accuracy: {forecast_summary.get('avg_accuracy', 0):.3f}")
        
        model_types = forecast_summary.get('model_types', {})
        if model_types:
            print(f"   Models used:")
            for model_type, count in model_types.items():
                print(f"     - {model_type}: {count}")


def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("🚀 Crypto Breakout Detection & Forecasting Pipeline")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Symbols to analyze: {args.symbols}")
    print(f"  - Historical data: {args.days} days")
    print(f"  - Forecast period: {args.forecast} days")
    print(f"  - Min signal strength: {args.min_strength}")
    print(f"  - Max candidates: {args.candidates}")
    print(f"  - Using mock data: {args.mock}")
    print()
    
    try:
        # Initialize pipeline
        if args.verbose:
            print("Initializing pipeline components...")
        
        pipeline = create_default_pipeline(use_mock_data=args.mock)
        
        # Run the full pipeline
        results = pipeline.run_full_pipeline(
            symbol_limit=args.symbols,
            historical_days=args.days,
            forecast_days=args.forecast,
            min_signal_strength=args.min_strength,
            max_breakout_candidates=args.candidates
        )
        
        # Display results
        print_breakout_signals(results.get('breakout_signals', []))
        print_forecasts(results.get('forecasts', {}))
        
        # Print summary
        summary = pipeline.get_pipeline_summary(results)
        print_summary(summary)
        
        # Export results
        if args.output or len(results.get('breakout_signals', [])) > 0:
            output_path = pipeline.export_results_to_csv(results, args.output)
            print(f"\n💾 Results exported to: {output_path}")
        
        print(f"\n✅ Pipeline completed successfully!")
        
        # Return success code
        return 0
        
    except KeyboardInterrupt:
        print("\n⏸️  Pipeline interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)