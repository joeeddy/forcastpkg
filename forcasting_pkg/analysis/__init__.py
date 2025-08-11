"""Technical analysis indicators and tools."""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Optional, Tuple
from datetime import datetime

from ..models import TechnicalIndicators, MarketData, AnalysisResult


class TechnicalAnalyzer:
    """Comprehensive technical analysis toolkit."""
    
    def __init__(self):
        """Initialize the technical analyzer."""
        pass
    
    def analyze(self, data: Union[List[MarketData], pd.DataFrame], symbol: str = "UNKNOWN") -> AnalysisResult:
        """
        Perform comprehensive technical analysis.
        
        Args:
            data: Historical market data
            symbol: Symbol identifier
            
        Returns:
            Complete analysis result with indicators and signals
        """
        df = self._prepare_data(data)
        
        # Calculate all technical indicators
        indicators = self.calculate_all_indicators(df)
        
        # Generate trading signals
        signal_strength, confidence, recommendations = self._generate_signals(indicators, df)
        
        return AnalysisResult(
            symbol=symbol,
            analysis_date=datetime.now(),
            technical_indicators=indicators,
            signal_strength=signal_strength,
            confidence=confidence,
            recommendations=recommendations
        )
    
    def calculate_all_indicators(self, data: Union[List[MarketData], pd.DataFrame]) -> TechnicalIndicators:
        """
        Calculate all available technical indicators.
        
        Args:
            data: Historical market data
            
        Returns:
            Technical indicators object
        """
        df = self._prepare_data(data)
        
        return TechnicalIndicators(
            rsi=self.calculate_rsi(df),
            macd=self.calculate_macd(df)['macd'],
            macd_signal=self.calculate_macd(df)['signal'],
            macd_histogram=self.calculate_macd(df)['histogram'],
            bollinger_upper=self.calculate_bollinger_bands(df)['upper'],
            bollinger_lower=self.calculate_bollinger_bands(df)['lower'],
            bollinger_middle=self.calculate_bollinger_bands(df)['middle'],
            moving_average_10=self.calculate_moving_average(df, 10),
            moving_average_20=self.calculate_moving_average(df, 20),
            moving_average_50=self.calculate_moving_average(df, 50),
            moving_average_200=self.calculate_moving_average(df, 200),
            stochastic_k=self.calculate_stochastic(df)['%K'],
            stochastic_d=self.calculate_stochastic(df)['%D'],
            williams_r=self.calculate_williams_r(df),
            cci=self.calculate_cci(df)
        )
    
    def calculate_rsi(self, data: Union[List[MarketData], pd.DataFrame], period: int = 14) -> Optional[float]:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            data: Historical market data
            period: RSI calculation period
            
        Returns:
            RSI value (0-100) or None if insufficient data
        """
        df = self._prepare_data(data)
        
        if len(df) < period + 1:
            return None
        
        # Calculate price changes
        delta = df['close'].diff()
        
        # Separate gains and losses
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None
    
    def calculate_macd(
        self, 
        data: Union[List[MarketData], pd.DataFrame], 
        fast: int = 12, 
        slow: int = 26, 
        signal: int = 9
    ) -> Dict[str, Optional[float]]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            data: Historical market data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
            
        Returns:
            Dictionary with MACD, signal, and histogram values
        """
        df = self._prepare_data(data)
        
        if len(df) < slow:
            return {'macd': None, 'signal': None, 'histogram': None}
        
        # Calculate EMAs
        ema_fast = df['close'].ewm(span=fast).mean()
        ema_slow = df['close'].ewm(span=slow).mean()
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=signal).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return {
            'macd': float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else None,
            'signal': float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else None,
            'histogram': float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else None
        }
    
    def calculate_bollinger_bands(
        self, 
        data: Union[List[MarketData], pd.DataFrame], 
        period: int = 20, 
        std_dev: float = 2
    ) -> Dict[str, Optional[float]]:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: Historical market data
            period: Moving average period
            std_dev: Number of standard deviations
            
        Returns:
            Dictionary with upper, lower, and middle band values
        """
        df = self._prepare_data(data)
        
        if len(df) < period:
            return {'upper': None, 'lower': None, 'middle': None}
        
        # Calculate middle band (SMA)
        middle_band = df['close'].rolling(window=period).mean()
        
        # Calculate standard deviation
        std = df['close'].rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return {
            'upper': float(upper_band.iloc[-1]) if not pd.isna(upper_band.iloc[-1]) else None,
            'lower': float(lower_band.iloc[-1]) if not pd.isna(lower_band.iloc[-1]) else None,
            'middle': float(middle_band.iloc[-1]) if not pd.isna(middle_band.iloc[-1]) else None
        }
    
    def calculate_moving_average(
        self, 
        data: Union[List[MarketData], pd.DataFrame], 
        period: int
    ) -> Optional[float]:
        """
        Calculate Simple Moving Average.
        
        Args:
            data: Historical market data
            period: Moving average period
            
        Returns:
            Moving average value or None if insufficient data
        """
        df = self._prepare_data(data)
        
        if len(df) < period:
            return None
        
        ma = df['close'].rolling(window=period).mean()
        return float(ma.iloc[-1]) if not pd.isna(ma.iloc[-1]) else None
    
    def calculate_stochastic(
        self, 
        data: Union[List[MarketData], pd.DataFrame], 
        k_period: int = 14, 
        d_period: int = 3
    ) -> Dict[str, Optional[float]]:
        """
        Calculate Stochastic Oscillator (%K and %D).
        
        Args:
            data: Historical market data
            k_period: %K calculation period
            d_period: %D smoothing period
            
        Returns:
            Dictionary with %K and %D values
        """
        df = self._prepare_data(data)
        
        if len(df) < k_period:
            return {'%K': None, '%D': None}
        
        # Calculate %K
        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()
        
        k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
        
        # Calculate %D (moving average of %K)
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            '%K': float(k_percent.iloc[-1]) if not pd.isna(k_percent.iloc[-1]) else None,
            '%D': float(d_percent.iloc[-1]) if not pd.isna(d_percent.iloc[-1]) else None
        }
    
    def calculate_williams_r(
        self, 
        data: Union[List[MarketData], pd.DataFrame], 
        period: int = 14
    ) -> Optional[float]:
        """
        Calculate Williams %R.
        
        Args:
            data: Historical market data
            period: Calculation period
            
        Returns:
            Williams %R value (-100 to 0) or None if insufficient data
        """
        df = self._prepare_data(data)
        
        if len(df) < period:
            return None
        
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        
        williams_r = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
        
        return float(williams_r.iloc[-1]) if not pd.isna(williams_r.iloc[-1]) else None
    
    def calculate_cci(
        self, 
        data: Union[List[MarketData], pd.DataFrame], 
        period: int = 20
    ) -> Optional[float]:
        """
        Calculate Commodity Channel Index (CCI).
        
        Args:
            data: Historical market data
            period: Calculation period
            
        Returns:
            CCI value or None if insufficient data
        """
        df = self._prepare_data(data)
        
        if len(df) < period:
            return None
        
        # Calculate typical price
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate moving average of typical price
        ma_tp = typical_price.rolling(window=period).mean()
        
        # Calculate mean deviation
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )
        
        # Calculate CCI
        cci = (typical_price - ma_tp) / (0.015 * mean_deviation)
        
        return float(cci.iloc[-1]) if not pd.isna(cci.iloc[-1]) else None
    
    def _prepare_data(self, data: Union[List[MarketData], pd.DataFrame]) -> pd.DataFrame:
        """Prepare and validate input data."""
        if isinstance(data, list):
            # Convert list of MarketData to DataFrame
            df = pd.DataFrame([item.dict() for item in data])
        else:
            df = data.copy()
        
        # Ensure required columns exist
        required_columns = ['date', 'open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in df.columns:
                # Try to infer missing OHLC data
                if col != 'date' and 'close' in df.columns:
                    df[col] = df['close']  # Use close price as fallback
                else:
                    raise ValueError(f"Required column '{col}' not found in data")
        
        # Convert date column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date and remove duplicates
        df = df.sort_values('date').drop_duplicates(subset=['date'])
        
        return df
    
    def _generate_signals(
        self, 
        indicators: TechnicalIndicators, 
        df: pd.DataFrame
    ) -> Tuple[str, float, List[str]]:
        """
        Generate trading signals based on technical indicators.
        
        Args:
            indicators: Calculated technical indicators
            df: Price data
            
        Returns:
            Tuple of (signal_strength, confidence, recommendations)
        """
        signals = []
        recommendations = []
        
        # RSI signals
        if indicators.rsi is not None:
            if indicators.rsi > 70:
                signals.append(-1)  # Overbought
                recommendations.append("RSI indicates overbought conditions (>70)")
            elif indicators.rsi < 30:
                signals.append(1)  # Oversold
                recommendations.append("RSI indicates oversold conditions (<30)")
            else:
                signals.append(0)  # Neutral
        
        # MACD signals
        if indicators.macd is not None and indicators.macd_signal is not None:
            if indicators.macd > indicators.macd_signal:
                signals.append(1)  # Bullish
                recommendations.append("MACD line above signal line (bullish)")
            else:
                signals.append(-1)  # Bearish
                recommendations.append("MACD line below signal line (bearish)")
        
        # Bollinger Bands signals
        if all([indicators.bollinger_upper, indicators.bollinger_lower, indicators.bollinger_middle]):
            current_price = df['close'].iloc[-1]
            if current_price > indicators.bollinger_upper:
                signals.append(-1)  # Overbought
                recommendations.append("Price above upper Bollinger Band (potential reversal)")
            elif current_price < indicators.bollinger_lower:
                signals.append(1)  # Oversold
                recommendations.append("Price below lower Bollinger Band (potential bounce)")
        
        # Moving average signals
        if indicators.moving_average_20 and indicators.moving_average_50:
            if indicators.moving_average_20 > indicators.moving_average_50:
                signals.append(1)  # Bullish
                recommendations.append("20-day MA above 50-day MA (bullish trend)")
            else:
                signals.append(-1)  # Bearish
                recommendations.append("20-day MA below 50-day MA (bearish trend)")
        
        # Stochastic signals
        if indicators.stochastic_k is not None:
            if indicators.stochastic_k > 80:
                signals.append(-1)  # Overbought
                recommendations.append("Stochastic indicates overbought conditions (>80)")
            elif indicators.stochastic_k < 20:
                signals.append(1)  # Oversold
                recommendations.append("Stochastic indicates oversold conditions (<20)")
        
        # Calculate overall signal
        if signals:
            avg_signal = np.mean(signals)
            confidence = min(1.0, len(signals) / 5.0)  # More indicators = higher confidence
            
            if avg_signal > 0.3:
                signal_strength = "Buy"
            elif avg_signal < -0.3:
                signal_strength = "Sell"
            else:
                signal_strength = "Hold"
        else:
            signal_strength = "Hold"
            confidence = 0.5
            recommendations = ["Insufficient data for reliable signals"]
        
        return signal_strength, confidence, recommendations


# Convenience functions for individual indicators
def rsi(data: Union[List[MarketData], pd.DataFrame], period: int = 14) -> Optional[float]:
    """Calculate RSI for given data."""
    analyzer = TechnicalAnalyzer()
    return analyzer.calculate_rsi(data, period)


def macd(data: Union[List[MarketData], pd.DataFrame]) -> Dict[str, Optional[float]]:
    """Calculate MACD for given data."""
    analyzer = TechnicalAnalyzer()
    return analyzer.calculate_macd(data)


def bollinger_bands(data: Union[List[MarketData], pd.DataFrame]) -> Dict[str, Optional[float]]:
    """Calculate Bollinger Bands for given data."""
    analyzer = TechnicalAnalyzer()
    return analyzer.calculate_bollinger_bands(data)


__all__ = [
    "TechnicalAnalyzer",
    "rsi",
    "macd", 
    "bollinger_bands"
]