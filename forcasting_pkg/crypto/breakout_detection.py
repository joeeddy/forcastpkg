"""Breakout detection module for cryptocurrency analysis."""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass

from ..models import MarketData


@dataclass
class BreakoutSignal:
    """Represents a breakout signal with relevant data."""
    symbol: str
    signal_type: str  # 'bullish_breakout', 'bearish_breakout', 'volume_spike'
    detection_time: datetime
    current_price: float
    breakout_level: float
    strength: float  # 0.0 to 1.0, confidence of the breakout
    volume_ratio: float  # Current volume vs average volume
    price_change_percent: float
    additional_data: Dict[str, Any]


class BreakoutDetector:
    """Detects breakout patterns in cryptocurrency price data."""
    
    def __init__(self,
                 lookback_period: int = 20,
                 volume_threshold: float = 1.5,
                 price_threshold: float = 0.02,
                 min_volume_for_signal: float = 100000):
        """
        Initialize breakout detector.
        
        Args:
            lookback_period: Number of periods to look back for highs/lows
            volume_threshold: Volume multiplier vs average to consider significant
            price_threshold: Minimum price change % to consider a breakout
            min_volume_for_signal: Minimum volume required to generate signal
        """
        self.lookback_period = lookback_period
        self.volume_threshold = volume_threshold
        self.price_threshold = price_threshold
        self.min_volume_for_signal = min_volume_for_signal
    
    def detect_breakouts(self, 
                        symbol: str,
                        data: List[MarketData]) -> List[BreakoutSignal]:
        """
        Detect breakout signals in the given market data.
        
        Args:
            symbol: Trading symbol
            data: List of MarketData objects (should be sorted by date)
            
        Returns:
            List of BreakoutSignal objects
        """
        if len(data) < self.lookback_period + 5:
            return []  # Not enough data
        
        df = self._prepare_dataframe(data)
        signals = []
        
        # Check only the most recent data point for breakouts
        current_idx = len(df) - 1
        current_row = df.iloc[current_idx]
        
        # Calculate indicators
        resistance_level, support_level = self._calculate_support_resistance(df, current_idx)
        avg_volume = self._calculate_average_volume(df, current_idx)
        volume_ratio = current_row['volume'] / avg_volume if avg_volume > 0 else 1.0
        
        # Detect bullish breakout
        bullish_signal = self._detect_bullish_breakout(
            df, current_idx, resistance_level, volume_ratio, symbol
        )
        if bullish_signal:
            signals.append(bullish_signal)
        
        # Detect bearish breakout
        bearish_signal = self._detect_bearish_breakout(
            df, current_idx, support_level, volume_ratio, symbol
        )
        if bearish_signal:
            signals.append(bearish_signal)
        
        # Detect volume spike without clear price breakout
        volume_signal = self._detect_volume_spike(
            df, current_idx, volume_ratio, symbol
        )
        if volume_signal:
            signals.append(volume_signal)
        
        return signals
    
    def _prepare_dataframe(self, data: List[MarketData]) -> pd.DataFrame:
        """Convert MarketData list to pandas DataFrame with indicators."""
        df_data = []
        for item in data:
            df_data.append({
                'date': item.date,
                'open': item.open,
                'high': item.high,
                'low': item.low,
                'close': item.close,
                'volume': item.volume or 0
            })
        
        df = pd.DataFrame(df_data)
        df = df.sort_values('date').reset_index(drop=True)
        
        # Add technical indicators
        df['price_change'] = df['close'].pct_change()
        df['volume_sma'] = df['volume'].rolling(window=self.lookback_period).mean()
        df['high_max'] = df['high'].rolling(window=self.lookback_period).max()
        df['low_min'] = df['low'].rolling(window=self.lookback_period).min()
        
        return df
    
    def _calculate_support_resistance(self, df: pd.DataFrame, current_idx: int) -> Tuple[float, float]:
        """Calculate current support and resistance levels."""
        start_idx = max(0, current_idx - self.lookback_period)
        lookback_data = df.iloc[start_idx:current_idx]
        
        if len(lookback_data) == 0:
            current_price = df.iloc[current_idx]['close']
            return current_price * 1.02, current_price * 0.98
        
        resistance = lookback_data['high'].max()
        support = lookback_data['low'].min()
        
        return resistance, support
    
    def _calculate_average_volume(self, df: pd.DataFrame, current_idx: int) -> float:
        """Calculate average volume over lookback period."""
        start_idx = max(0, current_idx - self.lookback_period)
        lookback_data = df.iloc[start_idx:current_idx]
        
        if len(lookback_data) == 0:
            return df.iloc[current_idx]['volume']
        
        return lookback_data['volume'].mean()
    
    def _detect_bullish_breakout(self, 
                                df: pd.DataFrame, 
                                current_idx: int, 
                                resistance_level: float, 
                                volume_ratio: float,
                                symbol: str) -> Optional[BreakoutSignal]:
        """Detect bullish breakout pattern."""
        current_row = df.iloc[current_idx]
        current_price = current_row['close']
        
        # Check for price breakout above resistance
        price_breakout = current_price > resistance_level
        
        # Calculate price change percentage
        price_change_pct = ((current_price - resistance_level) / resistance_level) * 100
        
        # Check conditions
        volume_confirmation = volume_ratio >= self.volume_threshold
        sufficient_volume = current_row['volume'] >= self.min_volume_for_signal
        significant_move = price_change_pct >= self.price_threshold
        
        if price_breakout and sufficient_volume and significant_move:
            # Calculate signal strength
            strength = min(1.0, (
                (price_change_pct / (self.price_threshold * 5)) * 0.4 +
                (volume_ratio / (self.volume_threshold * 2)) * 0.4 +
                0.2  # Base strength for meeting criteria
            ))
            
            return BreakoutSignal(
                symbol=symbol,
                signal_type='bullish_breakout',
                detection_time=current_row['date'],
                current_price=current_price,
                breakout_level=resistance_level,
                strength=strength,
                volume_ratio=volume_ratio,
                price_change_percent=price_change_pct,
                additional_data={
                    'resistance_level': resistance_level,
                    'volume_confirmation': volume_confirmation,
                    'lookback_period': self.lookback_period
                }
            )
        
        return None
    
    def _detect_bearish_breakout(self, 
                                df: pd.DataFrame, 
                                current_idx: int, 
                                support_level: float, 
                                volume_ratio: float,
                                symbol: str) -> Optional[BreakoutSignal]:
        """Detect bearish breakout pattern."""
        current_row = df.iloc[current_idx]
        current_price = current_row['close']
        
        # Check for price breakout below support
        price_breakout = current_price < support_level
        
        # Calculate price change percentage (negative for bearish)
        price_change_pct = ((current_price - support_level) / support_level) * 100
        
        # Check conditions
        volume_confirmation = volume_ratio >= self.volume_threshold
        sufficient_volume = current_row['volume'] >= self.min_volume_for_signal
        significant_move = abs(price_change_pct) >= self.price_threshold
        
        if price_breakout and sufficient_volume and significant_move:
            # Calculate signal strength
            strength = min(1.0, (
                (abs(price_change_pct) / (self.price_threshold * 5)) * 0.4 +
                (volume_ratio / (self.volume_threshold * 2)) * 0.4 +
                0.2  # Base strength for meeting criteria
            ))
            
            return BreakoutSignal(
                symbol=symbol,
                signal_type='bearish_breakout',
                detection_time=current_row['date'],
                current_price=current_price,
                breakout_level=support_level,
                strength=strength,
                volume_ratio=volume_ratio,
                price_change_percent=price_change_pct,
                additional_data={
                    'support_level': support_level,
                    'volume_confirmation': volume_confirmation,
                    'lookback_period': self.lookback_period
                }
            )
        
        return None
    
    def _detect_volume_spike(self, 
                           df: pd.DataFrame, 
                           current_idx: int, 
                           volume_ratio: float,
                           symbol: str) -> Optional[BreakoutSignal]:
        """Detect volume spike that might precede a breakout."""
        current_row = df.iloc[current_idx]
        current_price = current_row['close']
        
        # Look for volume spikes without clear price breakouts
        high_volume = volume_ratio >= (self.volume_threshold * 1.5)  # Higher threshold
        sufficient_volume = current_row['volume'] >= self.min_volume_for_signal
        
        # Check if there's some price movement (but not necessarily a full breakout)
        prev_close = df.iloc[current_idx - 1]['close'] if current_idx > 0 else current_price
        price_change_pct = ((current_price - prev_close) / prev_close) * 100
        some_movement = abs(price_change_pct) >= (self.price_threshold * 0.5)
        
        if high_volume and sufficient_volume and some_movement:
            # Calculate signal strength based primarily on volume
            strength = min(1.0, (
                (volume_ratio / (self.volume_threshold * 3)) * 0.6 +
                (abs(price_change_pct) / self.price_threshold) * 0.3 +
                0.1  # Base strength
            ))
            
            return BreakoutSignal(
                symbol=symbol,
                signal_type='volume_spike',
                detection_time=current_row['date'],
                current_price=current_price,
                breakout_level=current_price,  # Use current price as reference
                strength=strength,
                volume_ratio=volume_ratio,
                price_change_percent=price_change_pct,
                additional_data={
                    'avg_volume_multiple': volume_ratio,
                    'is_precursor_signal': True,
                    'lookback_period': self.lookback_period
                }
            )
        
        return None
    
    def filter_signals_by_strength(self, 
                                 signals: List[BreakoutSignal], 
                                 min_strength: float = 0.5) -> List[BreakoutSignal]:
        """Filter signals by minimum strength threshold."""
        return [signal for signal in signals if signal.strength >= min_strength]
    
    def rank_signals_by_strength(self, signals: List[BreakoutSignal]) -> List[BreakoutSignal]:
        """Sort signals by strength (highest first)."""
        return sorted(signals, key=lambda x: x.strength, reverse=True)


class MultiSymbolBreakoutScanner:
    """Scanner for detecting breakouts across multiple cryptocurrency symbols."""
    
    def __init__(self, detector: BreakoutDetector):
        """
        Initialize multi-symbol scanner.
        
        Args:
            detector: BreakoutDetector instance to use for analysis
        """
        self.detector = detector
    
    def scan_symbols(self, 
                    symbol_data: Dict[str, List[MarketData]],
                    min_strength: float = 0.5) -> Dict[str, List[BreakoutSignal]]:
        """
        Scan multiple symbols for breakout signals.
        
        Args:
            symbol_data: Dictionary mapping symbols to their market data
            min_strength: Minimum signal strength to include in results
            
        Returns:
            Dictionary mapping symbols to their breakout signals
        """
        results = {}
        
        for symbol, data in symbol_data.items():
            try:
                signals = self.detector.detect_breakouts(symbol, data)
                filtered_signals = self.detector.filter_signals_by_strength(signals, min_strength)
                
                if filtered_signals:
                    results[symbol] = self.detector.rank_signals_by_strength(filtered_signals)
                    
            except Exception as e:
                print(f"Warning: Failed to scan {symbol}: {e}")
                continue
        
        return results
    
    def get_top_signals(self, 
                       symbol_data: Dict[str, List[MarketData]],
                       limit: int = 10,
                       min_strength: float = 0.5) -> List[BreakoutSignal]:
        """
        Get top breakout signals across all symbols.
        
        Args:
            symbol_data: Dictionary mapping symbols to their market data
            limit: Maximum number of signals to return
            min_strength: Minimum signal strength threshold
            
        Returns:
            List of top BreakoutSignal objects sorted by strength
        """
        all_signals = []
        
        scan_results = self.scan_symbols(symbol_data, min_strength)
        for signals in scan_results.values():
            all_signals.extend(signals)
        
        # Sort all signals by strength and return top ones
        sorted_signals = sorted(all_signals, key=lambda x: x.strength, reverse=True)
        return sorted_signals[:limit]