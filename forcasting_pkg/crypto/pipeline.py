"""Pipeline module for orchestrating crypto breakout detection and forecasting."""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import pandas as pd

from ..models import MarketData, ForecastResult
from ..forecasting import ForecastingEngine
from .data_ingestion import MEXCDataSource, create_mexc_data_source
from .breakout_detection import BreakoutDetector, BreakoutSignal, MultiSymbolBreakoutScanner


class CryptoBreakoutPipeline:
    """
    Main pipeline class that orchestrates the entire crypto breakout detection
    and forecasting workflow.
    """
    
    def __init__(self,
                 data_source: Optional[MEXCDataSource] = None,
                 breakout_detector: Optional[BreakoutDetector] = None,
                 forecasting_engine: Optional[ForecastingEngine] = None,
                 use_mock_data: bool = False):
        """
        Initialize the crypto breakout pipeline.
        
        Args:
            data_source: MEXC data source (will create default if None)
            breakout_detector: Breakout detector (will create default if None)
            forecasting_engine: Forecasting engine (will create default if None)
            use_mock_data: Whether to use mock data source
        """
        self.data_source = data_source or create_mexc_data_source(use_mock=use_mock_data)
        self.breakout_detector = breakout_detector or BreakoutDetector()
        self.forecasting_engine = forecasting_engine or ForecastingEngine()
        self.scanner = MultiSymbolBreakoutScanner(self.breakout_detector)
        
    def run_full_pipeline(self,
                         symbol_limit: int = 50,
                         historical_days: int = 30,
                         forecast_days: int = 7,
                         min_signal_strength: float = 0.5,
                         max_breakout_candidates: int = 10) -> Dict[str, Any]:
        """
        Run the complete pipeline: fetch data, detect breakouts, and generate forecasts.
        
        Args:
            symbol_limit: Maximum number of symbols to analyze
            historical_days: Days of historical data to fetch
            forecast_days: Days to forecast for breakout candidates
            min_signal_strength: Minimum breakout signal strength
            max_breakout_candidates: Maximum number of candidates to forecast
            
        Returns:
            Dictionary with pipeline results including signals and forecasts
        """
        print(f"Starting crypto breakout pipeline at {datetime.now()}")
        
        # Step 1: Get top trading symbols
        print("Fetching top trading symbols...")
        symbols = self._get_top_symbols(symbol_limit)
        print(f"Found {len(symbols)} symbols to analyze")
        
        # Step 2: Fetch historical data for all symbols
        print("Fetching historical data...")
        symbol_data = self._fetch_historical_data(symbols, historical_days)
        print(f"Successfully fetched data for {len(symbol_data)} symbols")
        
        # Step 3: Detect breakouts
        print("Detecting breakout signals...")
        breakout_signals = self._detect_breakouts(symbol_data, min_signal_strength)
        print(f"Found {len(breakout_signals)} breakout signals")
        
        # Step 4: Generate forecasts for top candidates
        print("Generating forecasts for breakout candidates...")
        forecasts = self._generate_forecasts(breakout_signals, symbol_data, 
                                           forecast_days, max_breakout_candidates)
        print(f"Generated forecasts for {len(forecasts)} symbols")
        
        # Step 5: Compile results
        results = {
            'pipeline_info': {
                'execution_time': datetime.now(),
                'symbols_analyzed': len(symbol_data),
                'signals_detected': len(breakout_signals),
                'forecasts_generated': len(forecasts),
                'parameters': {
                    'symbol_limit': symbol_limit,
                    'historical_days': historical_days,
                    'forecast_days': forecast_days,
                    'min_signal_strength': min_signal_strength,
                    'max_breakout_candidates': max_breakout_candidates
                }
            },
            'breakout_signals': breakout_signals,
            'forecasts': forecasts,
            'symbol_data': symbol_data  # Include for further analysis if needed
        }
        
        print("Pipeline execution completed successfully!")
        return results
    
    def _get_top_symbols(self, limit: int) -> List[str]:
        """Get top trading symbols by volume."""
        try:
            return self.data_source.get_top_volume_symbols(limit)
        except Exception as e:
            print(f"Warning: Could not fetch top symbols: {e}")
            # Fallback to available symbols
            symbols = self.data_source.get_available_symbols()
            return symbols[:limit]
    
    def _fetch_historical_data(self, 
                             symbols: List[str], 
                             days: int) -> Dict[str, List[MarketData]]:
        """Fetch historical data for all symbols."""
        symbol_data = {}
        failed_symbols = []
        
        for i, symbol in enumerate(symbols):
            try:
                print(f"  Fetching data for {symbol} ({i+1}/{len(symbols)})")
                data = self.data_source.get_historical_data(symbol, days)
                
                if len(data) >= self.breakout_detector.lookback_period:
                    symbol_data[symbol] = data
                else:
                    print(f"  Warning: Insufficient data for {symbol} ({len(data)} points)")
                    
            except Exception as e:
                print(f"  Error fetching data for {symbol}: {e}")
                failed_symbols.append(symbol)
                continue
        
        if failed_symbols:
            print(f"Failed to fetch data for {len(failed_symbols)} symbols: {failed_symbols[:5]}...")
        
        return symbol_data
    
    def _detect_breakouts(self, 
                         symbol_data: Dict[str, List[MarketData]], 
                         min_strength: float) -> List[BreakoutSignal]:
        """Detect breakout signals across all symbols."""
        signals = self.scanner.get_top_signals(
            symbol_data, 
            limit=100,  # Get many signals, we'll filter later
            min_strength=min_strength
        )
        
        # Group signals by type for logging
        signal_types = {}
        for signal in signals:
            signal_types[signal.signal_type] = signal_types.get(signal.signal_type, 0) + 1
        
        print(f"  Signal breakdown: {signal_types}")
        return signals
    
    def _generate_forecasts(self, 
                          signals: List[BreakoutSignal],
                          symbol_data: Dict[str, List[MarketData]],
                          forecast_days: int,
                          max_candidates: int) -> Dict[str, ForecastResult]:
        """Generate forecasts for top breakout candidates."""
        forecasts = {}
        
        # Get unique symbols from top signals
        candidate_symbols = []
        seen_symbols = set()
        
        for signal in signals:
            if signal.symbol not in seen_symbols and len(candidate_symbols) < max_candidates:
                candidate_symbols.append(signal.symbol)
                seen_symbols.add(signal.symbol)
        
        # Generate forecasts for each candidate
        for i, symbol in enumerate(candidate_symbols):
            try:
                print(f"  Generating forecast for {symbol} ({i+1}/{len(candidate_symbols)})")
                
                if symbol in symbol_data:
                    # Try multiple models and use the best one
                    forecast = self._generate_best_forecast(symbol, symbol_data[symbol], forecast_days)
                    if forecast:
                        forecasts[symbol] = forecast
                
            except Exception as e:
                print(f"  Error generating forecast for {symbol}: {e}")
                continue
        
        return forecasts
    
    def _generate_best_forecast(self, 
                              symbol: str, 
                              data: List[MarketData], 
                              days: int) -> Optional[ForecastResult]:
        """Generate forecast using the best available model."""
        models_to_try = ['moving_average', 'linear']  # Start with simpler models
        
        best_forecast = None
        best_accuracy = 0.0
        
        for model_type in models_to_try:
            try:
                forecast = self.forecasting_engine.forecast(
                    data, 
                    model=model_type, 
                    symbol=symbol, 
                    days=days
                )
                
                # Use the forecast with highest accuracy
                if forecast.model_accuracy and forecast.model_accuracy > best_accuracy:
                    best_forecast = forecast
                    best_accuracy = forecast.model_accuracy
                    
            except Exception as e:
                print(f"    Warning: {model_type} model failed for {symbol}: {e}")
                continue
        
        return best_forecast
    
    def get_pipeline_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of pipeline results."""
        signals = results.get('breakout_signals', [])
        forecasts = results.get('forecasts', {})
        
        # Analyze signals
        signal_summary = {
            'total_signals': len(signals),
            'avg_strength': sum(s.strength for s in signals) / len(signals) if signals else 0,
            'signal_types': {},
            'top_signals': signals[:5] if len(signals) >= 5 else signals
        }
        
        for signal in signals:
            signal_type = signal.signal_type
            signal_summary['signal_types'][signal_type] = signal_summary['signal_types'].get(signal_type, 0) + 1
        
        # Analyze forecasts
        forecast_summary = {
            'total_forecasts': len(forecasts),
            'avg_accuracy': 0.0,
            'model_types': {},
            'symbols_with_forecasts': list(forecasts.keys())
        }
        
        if forecasts:
            accuracies = [f.model_accuracy for f in forecasts.values() if f.model_accuracy]
            forecast_summary['avg_accuracy'] = sum(accuracies) / len(accuracies) if accuracies else 0
            
            for forecast in forecasts.values():
                model_type = forecast.model_type
                forecast_summary['model_types'][model_type] = forecast_summary['model_types'].get(model_type, 0) + 1
        
        return {
            'execution_info': results.get('pipeline_info', {}),
            'signal_summary': signal_summary,
            'forecast_summary': forecast_summary
        }
    
    def export_results_to_csv(self, results: Dict[str, Any], output_path: str = None) -> str:
        """Export pipeline results to CSV files."""
        import os
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"crypto_breakout_results_{timestamp}"
        
        os.makedirs(output_path, exist_ok=True)
        
        # Export signals
        signals = results.get('breakout_signals', [])
        if signals:
            signals_data = []
            for signal in signals:
                signals_data.append({
                    'symbol': signal.symbol,
                    'signal_type': signal.signal_type,
                    'detection_time': signal.detection_time,
                    'current_price': signal.current_price,
                    'breakout_level': signal.breakout_level,
                    'strength': signal.strength,
                    'volume_ratio': signal.volume_ratio,
                    'price_change_percent': signal.price_change_percent
                })
            
            signals_df = pd.DataFrame(signals_data)
            signals_file = os.path.join(output_path, 'breakout_signals.csv')
            signals_df.to_csv(signals_file, index=False)
            print(f"Exported signals to {signals_file}")
        
        # Export forecast summaries
        forecasts = results.get('forecasts', {})
        if forecasts:
            forecast_data = []
            for symbol, forecast in forecasts.items():
                forecast_data.append({
                    'symbol': symbol,
                    'model_type': forecast.model_type,
                    'forecast_period_days': forecast.forecast_period_days,
                    'model_accuracy': forecast.model_accuracy,
                    'first_prediction': forecast.forecast_points[0].predicted_value if forecast.forecast_points else None,
                    'last_prediction': forecast.forecast_points[-1].predicted_value if forecast.forecast_points else None,
                    'generated_at': forecast.generated_at
                })
            
            forecasts_df = pd.DataFrame(forecast_data)
            forecasts_file = os.path.join(output_path, 'forecasts_summary.csv')
            forecasts_df.to_csv(forecasts_file, index=False)
            print(f"Exported forecast summaries to {forecasts_file}")
        
        return output_path


def create_default_pipeline(use_mock_data: bool = False) -> CryptoBreakoutPipeline:
    """
    Create a default crypto breakout pipeline with standard configuration.
    
    Args:
        use_mock_data: Whether to use mock data for testing
        
    Returns:
        Configured CryptoBreakoutPipeline instance
    """
    return CryptoBreakoutPipeline(use_mock_data=use_mock_data)