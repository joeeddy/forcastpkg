"""Data ingestion module for MEXC exchange cryptocurrency data."""

import time
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

from ..models import MarketData, DataType
from ..data import DataSource


class MEXCDataSource(DataSource):
    """Data source for MEXC exchange using ccxt library."""
    
    def __init__(self, 
                 rate_limit: int = 600,  # requests per minute
                 sandbox: bool = False):
        """
        Initialize MEXC data source.
        
        Args:
            rate_limit: Maximum requests per minute to respect API limits
            sandbox: Whether to use sandbox mode (for testing)
        """
        if not CCXT_AVAILABLE:
            raise ImportError("ccxt library is required for MEXC data source. Install with: pip install ccxt")
        
        self.rate_limit = rate_limit
        self.sandbox = sandbox
        self._last_request_time = 0
        
        # Initialize MEXC exchange
        self.exchange = ccxt.mexc({
            'sandbox': sandbox,
            'rateLimit': int(60000 / rate_limit),  # Convert to milliseconds
            'enableRateLimit': True,
        })
        
    def _rate_limit_wait(self):
        """Ensure we respect rate limits."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        min_interval = 60.0 / self.rate_limit
        
        if time_since_last < min_interval:
            time.sleep(min_interval - time_since_last)
        
        self._last_request_time = time.time()
    
    def get_available_symbols(self) -> List[str]:
        """
        Get all available trading symbols on MEXC.
        
        Returns:
            List of trading symbols (e.g., ['BTC/USDT', 'ETH/USDT'])
        """
        try:
            self._rate_limit_wait()
            markets = self.exchange.load_markets()
            return list(markets.keys())
        except Exception as e:
            raise RuntimeError(f"Failed to fetch MEXC symbols: {e}")
    
    def get_historical_data(self, 
                          symbol: str, 
                          days: int = 30, 
                          data_type: DataType = DataType.CRYPTO) -> List[MarketData]:
        """
        Fetch historical OHLCV data for a cryptocurrency symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            days: Number of days of historical data
            data_type: Type of data (should be CRYPTO)
            
        Returns:
            List of MarketData objects with OHLCV data
        """
        try:
            self._rate_limit_wait()
            
            # Calculate timeframe parameters
            timeframe = '1d'  # Daily candles
            since = self.exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since)
            
            # Convert to MarketData objects
            market_data = []
            for candle in ohlcv:
                timestamp, open_price, high_price, low_price, close_price, volume = candle
                
                market_data.append(MarketData(
                    date=datetime.fromtimestamp(timestamp / 1000),
                    open=float(open_price),
                    high=float(high_price),
                    low=float(low_price),
                    close=float(close_price),
                    volume=float(volume) if volume else 0.0,
                    adjusted_close=float(close_price)  # Crypto doesn't have adjusted close
                ))
            
            return sorted(market_data, key=lambda x: x.date)
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch historical data for {symbol}: {e}")
    
    def get_current_price(self, symbol: str, data_type: DataType = DataType.CRYPTO) -> float:
        """
        Get current price for a cryptocurrency symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            data_type: Type of data (should be CRYPTO)
            
        Returns:
            Current price as float
        """
        try:
            self._rate_limit_wait()
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker['last'])
        except Exception as e:
            raise RuntimeError(f"Failed to fetch current price for {symbol}: {e}")
    
    def get_24h_stats(self, symbol: str) -> Dict[str, Any]:
        """
        Get 24-hour statistics for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dictionary with 24h stats including volume, price change, etc.
        """
        try:
            self._rate_limit_wait()
            ticker = self.exchange.fetch_ticker(symbol)
            
            return {
                'symbol': symbol,
                'price': ticker['last'],
                'change_24h': ticker['change'],
                'change_percent_24h': ticker['percentage'],
                'volume_24h': ticker['baseVolume'],
                'volume_24h_usd': ticker['quoteVolume'],
                'high_24h': ticker['high'],
                'low_24h': ticker['low'],
                'timestamp': ticker['timestamp']
            }
        except Exception as e:
            raise RuntimeError(f"Failed to fetch 24h stats for {symbol}: {e}")
    
    def get_top_volume_symbols(self, limit: int = 100) -> List[str]:
        """
        Get symbols sorted by 24h volume (highest first).
        
        Args:
            limit: Maximum number of symbols to return
            
        Returns:
            List of symbols sorted by volume
        """
        try:
            symbols = self.get_available_symbols()
            volume_data = []
            
            # Get volume data for all symbols (be careful with rate limits)
            for symbol in symbols[:limit * 2]:  # Get more than needed in case some fail
                try:
                    stats = self.get_24h_stats(symbol)
                    volume_data.append((symbol, stats['volume_24h_usd'] or 0))
                except:
                    continue  # Skip symbols that fail
            
            # Sort by volume and return top symbols
            volume_data.sort(key=lambda x: x[1], reverse=True)
            return [symbol for symbol, volume in volume_data[:limit]]
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch top volume symbols: {e}")


class MockMEXCDataSource(DataSource):
    """Mock MEXC data source for testing when ccxt is not available."""
    
    def __init__(self):
        """Initialize mock data source."""
        pass
    
    def get_available_symbols(self) -> List[str]:
        """Return mock symbols."""
        return ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT']
    
    def get_historical_data(self, 
                          symbol: str, 
                          days: int = 30, 
                          data_type: DataType = DataType.CRYPTO) -> List[MarketData]:
        """Generate mock historical data."""
        import numpy as np
        
        data = []
        base_price = 50000.0 if 'BTC' in symbol else 3000.0 if 'ETH' in symbol else 100.0
        start_date = datetime.now() - timedelta(days=days)
        
        np.random.seed(42)  # For reproducible mock data
        
        for i in range(days):
            # Generate realistic OHLC data
            daily_return = np.random.normal(0.002, 0.05)  # Higher volatility for crypto
            close_price = base_price * (1 + daily_return)
            
            high = close_price * (1 + abs(np.random.normal(0, 0.02)))
            low = close_price * (1 - abs(np.random.normal(0, 0.02)))
            open_price = low + (high - low) * np.random.random()
            
            # Ensure OHLC relationships
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)
            
            data.append(MarketData(
                date=start_date + timedelta(days=i),
                open=round(open_price, 2),
                high=round(high, 2),
                low=round(low, 2),
                close=round(close_price, 2),
                volume=np.random.randint(1000000, 50000000),
                adjusted_close=round(close_price, 2)
            ))
            base_price = close_price
        
        return data
    
    def get_current_price(self, symbol: str, data_type: DataType = DataType.CRYPTO) -> float:
        """Return mock current price."""
        if 'BTC' in symbol:
            return 45000.0
        elif 'ETH' in symbol:
            return 2800.0
        else:
            return 150.0
    
    def get_24h_stats(self, symbol: str) -> Dict[str, Any]:
        """Return mock 24h stats."""
        return {
            'symbol': symbol,
            'price': self.get_current_price(symbol),
            'change_24h': 500.0,
            'change_percent_24h': 1.2,
            'volume_24h': 1000000,
            'volume_24h_usd': 45000000000,
            'high_24h': self.get_current_price(symbol) * 1.05,
            'low_24h': self.get_current_price(symbol) * 0.95,
            'timestamp': int(datetime.now().timestamp() * 1000)
        }
    
    def get_top_volume_symbols(self, limit: int = 100) -> List[str]:
        """Return mock top volume symbols."""
        symbols = self.get_available_symbols()
        return symbols[:min(limit, len(symbols))]


def create_mexc_data_source(use_mock: bool = False) -> DataSource:
    """
    Factory function to create MEXC data source.
    
    Args:
        use_mock: If True, return mock data source regardless of ccxt availability
        
    Returns:
        MEXCDataSource or MockMEXCDataSource instance
    """
    if use_mock or not CCXT_AVAILABLE:
        return MockMEXCDataSource()
    else:
        return MEXCDataSource()