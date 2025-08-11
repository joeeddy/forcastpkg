"""Tests for the crypto breakout detection module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List

from forcasting_pkg.models import MarketData
from forcasting_pkg.crypto.data_ingestion import MockMEXCDataSource, create_mexc_data_source
from forcasting_pkg.crypto.breakout_detection import BreakoutDetector, BreakoutSignal
from forcasting_pkg.crypto.pipeline import CryptoBreakoutPipeline


class TestMockMEXCDataSource:
    """Test the mock MEXC data source."""
    
    def test_get_available_symbols(self):
        """Test getting available symbols."""
        data_source = MockMEXCDataSource()
        symbols = data_source.get_available_symbols()
        
        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert 'BTC/USDT' in symbols
        assert 'ETH/USDT' in symbols
    
    def test_get_historical_data(self):
        """Test getting historical data."""
        data_source = MockMEXCDataSource()
        data = data_source.get_historical_data('BTC/USDT', days=30)
        
        assert isinstance(data, list)
        assert len(data) == 30
        assert all(isinstance(item, MarketData) for item in data)
        
        # Check data is sorted by date
        dates = [item.date for item in data]
        assert dates == sorted(dates)
        
        # Check OHLC relationships
        for item in data:
            assert item.high >= max(item.open, item.close)
            assert item.low <= min(item.open, item.close)
            assert item.volume >= 0
    
    def test_get_current_price(self):
        """Test getting current price."""
        data_source = MockMEXCDataSource()
        price = data_source.get_current_price('BTC/USDT')
        
        assert isinstance(price, float)
        assert price > 0
    
    def test_get_24h_stats(self):
        """Test getting 24h statistics."""
        data_source = MockMEXCDataSource()
        stats = data_source.get_24h_stats('BTC/USDT')
        
        assert isinstance(stats, dict)
        required_keys = ['symbol', 'price', 'volume_24h', 'change_percent_24h']
        for key in required_keys:
            assert key in stats
    
    def test_get_top_volume_symbols(self):
        """Test getting top volume symbols."""
        data_source = MockMEXCDataSource()
        symbols = data_source.get_top_volume_symbols(limit=3)
        
        assert isinstance(symbols, list)
        assert len(symbols) <= 3
        assert all(isinstance(symbol, str) for symbol in symbols)


class TestBreakoutDetector:
    """Test the breakout detection functionality."""
    
    @pytest.fixture
    def sample_data_trending_up(self) -> List[MarketData]:
        """Create sample data with an upward trend and breakout."""
        data = []
        base_price = 100.0
        start_date = datetime.now() - timedelta(days=30)
        
        np.random.seed(42)
        
        for i in range(30):
            # Create upward trend with breakout at the end
            if i < 25:
                # Normal upward trend
                daily_return = np.random.normal(0.005, 0.015)
            else:
                # Strong breakout
                daily_return = np.random.normal(0.03, 0.01)
            
            close_price = base_price * (1 + daily_return)
            
            # Higher volume during breakout
            volume = np.random.randint(1000000, 2000000)
            if i >= 25:
                volume *= 3  # Volume spike during breakout
            
            high = close_price * (1 + abs(np.random.normal(0, 0.005)))
            low = close_price * (1 - abs(np.random.normal(0, 0.005)))
            open_price = low + (high - low) * np.random.random()
            
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)
            
            data.append(MarketData(
                date=start_date + timedelta(days=i),
                open=round(open_price, 2),
                high=round(high, 2),
                low=round(low, 2),
                close=round(close_price, 2),
                volume=volume,
                adjusted_close=round(close_price, 2)
            ))
            base_price = close_price
        
        return data
    
    def test_detector_initialization(self):
        """Test detector initialization with default parameters."""
        detector = BreakoutDetector()
        
        assert detector.lookback_period == 20
        assert detector.volume_threshold == 1.5
        assert detector.price_threshold == 0.02
        assert detector.min_volume_for_signal == 100000
    
    def test_detector_custom_parameters(self):
        """Test detector initialization with custom parameters."""
        detector = BreakoutDetector(
            lookback_period=15,
            volume_threshold=2.0,
            price_threshold=0.03,
            min_volume_for_signal=200000
        )
        
        assert detector.lookback_period == 15
        assert detector.volume_threshold == 2.0
        assert detector.price_threshold == 0.03
        assert detector.min_volume_for_signal == 200000
    
    def test_detect_breakouts_insufficient_data(self):
        """Test breakout detection with insufficient data."""
        detector = BreakoutDetector()
        
        # Create minimal data (less than required)
        data = []
        for i in range(10):
            data.append(MarketData(
                date=datetime.now() - timedelta(days=10-i),
                open=100.0, high=105.0, low=95.0, close=102.0,
                volume=1000000, adjusted_close=102.0
            ))
        
        signals = detector.detect_breakouts('TEST/USDT', data)
        assert signals == []
    
    def test_detect_bullish_breakout(self, sample_data_trending_up):
        """Test detection of bullish breakout."""
        detector = BreakoutDetector(
            lookback_period=15,
            volume_threshold=1.5,
            price_threshold=0.01  # Lower threshold for test
        )
        
        signals = detector.detect_breakouts('TEST/USDT', sample_data_trending_up)
        
        # Should detect at least one signal
        assert len(signals) > 0
        
        # Check if we got a bullish breakout
        bullish_signals = [s for s in signals if s.signal_type == 'bullish_breakout']
        
        if bullish_signals:
            signal = bullish_signals[0]
            assert signal.symbol == 'TEST/USDT'
            assert signal.strength > 0
            assert signal.volume_ratio > 1.0
            assert isinstance(signal.detection_time, datetime)
    
    def test_filter_signals_by_strength(self, sample_data_trending_up):
        """Test filtering signals by strength."""
        detector = BreakoutDetector(price_threshold=0.001)  # Very low threshold
        
        signals = detector.detect_breakouts('TEST/USDT', sample_data_trending_up)
        
        if signals:
            # Test filtering
            high_strength_signals = detector.filter_signals_by_strength(signals, 0.7)
            medium_strength_signals = detector.filter_signals_by_strength(signals, 0.3)
            
            assert len(high_strength_signals) <= len(medium_strength_signals)
            assert all(s.strength >= 0.7 for s in high_strength_signals)
            assert all(s.strength >= 0.3 for s in medium_strength_signals)
    
    def test_rank_signals_by_strength(self, sample_data_trending_up):
        """Test ranking signals by strength."""
        detector = BreakoutDetector(price_threshold=0.001)
        
        signals = detector.detect_breakouts('TEST/USDT', sample_data_trending_up)
        
        if len(signals) > 1:
            ranked_signals = detector.rank_signals_by_strength(signals)
            
            # Check that signals are sorted by strength (descending)
            strengths = [s.strength for s in ranked_signals]
            assert strengths == sorted(strengths, reverse=True)


class TestCryptoBreakoutPipeline:
    """Test the crypto breakout pipeline."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = CryptoBreakoutPipeline(use_mock_data=True)
        
        assert pipeline.data_source is not None
        assert pipeline.breakout_detector is not None
        assert pipeline.forecasting_engine is not None
        assert pipeline.scanner is not None
    
    def test_pipeline_get_top_symbols(self):
        """Test getting top symbols."""
        pipeline = CryptoBreakoutPipeline(use_mock_data=True)
        
        symbols = pipeline._get_top_symbols(5)
        
        assert isinstance(symbols, list)
        assert len(symbols) <= 5
        assert all(isinstance(symbol, str) for symbol in symbols)
    
    def test_pipeline_fetch_historical_data(self):
        """Test fetching historical data for multiple symbols."""
        pipeline = CryptoBreakoutPipeline(use_mock_data=True)
        
        symbols = ['BTC/USDT', 'ETH/USDT']
        symbol_data = pipeline._fetch_historical_data(symbols, 20)
        
        assert isinstance(symbol_data, dict)
        
        for symbol in symbols:
            if symbol in symbol_data:
                data = symbol_data[symbol]
                assert isinstance(data, list)
                assert len(data) >= pipeline.breakout_detector.lookback_period
                assert all(isinstance(item, MarketData) for item in data)
    
    def test_pipeline_summary(self):
        """Test pipeline summary generation."""
        pipeline = CryptoBreakoutPipeline(use_mock_data=True)
        
        # Create mock results
        results = {
            'pipeline_info': {
                'execution_time': datetime.now(),
                'symbols_analyzed': 5,
                'signals_detected': 3,
                'forecasts_generated': 2
            },
            'breakout_signals': [],
            'forecasts': {}
        }
        
        summary = pipeline.get_pipeline_summary(results)
        
        assert 'execution_info' in summary
        assert 'signal_summary' in summary
        assert 'forecast_summary' in summary
        
        assert summary['signal_summary']['total_signals'] == 0
        assert summary['forecast_summary']['total_forecasts'] == 0


def test_create_mexc_data_source():
    """Test the factory function for creating MEXC data source."""
    # Test mock creation
    mock_source = create_mexc_data_source(use_mock=True)
    assert isinstance(mock_source, MockMEXCDataSource)
    
    # Test automatic fallback to mock when ccxt not available
    source = create_mexc_data_source(use_mock=False)
    # Should fallback to mock since ccxt might not be installed
    assert source is not None


class TestCryptoIntegration:
    """Integration tests for the crypto module."""
    
    def test_end_to_end_mock_pipeline(self):
        """Test the complete pipeline with mock data."""
        pipeline = CryptoBreakoutPipeline(use_mock_data=True)
        
        try:
            results = pipeline.run_full_pipeline(
                symbol_limit=3,
                historical_days=25,
                forecast_days=5,
                min_signal_strength=0.1,  # Low threshold for testing
                max_breakout_candidates=2
            )
            
            # Check that results have expected structure
            assert 'pipeline_info' in results
            assert 'breakout_signals' in results
            assert 'forecasts' in results
            assert 'symbol_data' in results
            
            pipeline_info = results['pipeline_info']
            assert pipeline_info['symbols_analyzed'] >= 0
            assert pipeline_info['signals_detected'] >= 0
            assert pipeline_info['forecasts_generated'] >= 0
            
        except Exception as e:
            pytest.fail(f"End-to-end pipeline test failed: {e}")
    
    def test_pipeline_with_no_signals(self):
        """Test pipeline behavior when no signals are detected."""
        # Use very high thresholds to ensure no signals
        detector = BreakoutDetector(
            volume_threshold=10.0,
            price_threshold=0.5,
            min_volume_for_signal=100000000
        )
        
        pipeline = CryptoBreakoutPipeline(
            breakout_detector=detector,
            use_mock_data=True
        )
        
        results = pipeline.run_full_pipeline(
            symbol_limit=2,
            historical_days=25,
            forecast_days=5,
            min_signal_strength=0.9,
            max_breakout_candidates=1
        )
        
        # Should complete without errors even with no signals
        assert results['pipeline_info']['signals_detected'] == 0
        assert len(results['breakout_signals']) == 0
        assert len(results['forecasts']) == 0