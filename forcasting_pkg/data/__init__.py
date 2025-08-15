"""Data source interfaces and implementations for market data."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from ..models import MarketData, DataSourceConfig, DataType

# Try to import optional data source dependencies
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class DataSource(ABC):
    """Abstract base class for all data sources."""
    
    def __init__(self, config: Optional[DataSourceConfig] = None):
        """
        Initialize data source.
        
        Args:
            config: Data source configuration
        """
        self.config = config or DataSourceConfig(source_name="unknown")
    
    @abstractmethod
    def get_historical_data(
        self, 
        symbol: str, 
        days: int = 90, 
        data_type: DataType = DataType.STOCK
    ) -> List[MarketData]:
        """
        Fetch historical market data.
        
        Args:
            symbol: Financial instrument symbol
            days: Number of days of historical data
            data_type: Type of data (stock, crypto, etc.)
            
        Returns:
            List of market data points
        """
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str, data_type: DataType = DataType.STOCK) -> Optional[float]:
        """
        Get current price for a symbol.
        
        Args:
            symbol: Financial instrument symbol
            data_type: Type of data
            
        Returns:
            Current price or None if not available
        """
        pass
    
    def validate_symbol(self, symbol: str, data_type: DataType) -> bool:
        """
        Validate if a symbol is supported by this data source.
        
        Args:
            symbol: Financial instrument symbol
            data_type: Type of data
            
        Returns:
            True if symbol is valid, False otherwise
        """
        try:
            price = self.get_current_price(symbol, data_type)
            return price is not None
        except Exception:
            return False


class YahooFinanceSource(DataSource):
    """Yahoo Finance data source implementation."""
    
    def __init__(self, config: Optional[DataSourceConfig] = None):
        """Initialize Yahoo Finance data source."""
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance package is required for YahooFinanceSource")
        
        default_config = DataSourceConfig(
            source_name="yahoo_finance",
            base_url="https://finance.yahoo.com",
            rate_limit=2000,  # requests per hour
            timeout=30
        )
        
        super().__init__(config or default_config)
    
    def get_historical_data(
        self, 
        symbol: str, 
        days: int = 90, 
        data_type: DataType = DataType.STOCK
    ) -> List[MarketData]:
        """Fetch historical data from Yahoo Finance."""
        try:
            # Convert crypto symbols if needed
            if data_type == DataType.CRYPTO:
                symbol = self._convert_crypto_symbol(symbol)
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Fetch data using yfinance
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist.empty:
                return []
            
            # Convert to MarketData objects
            market_data = []
            for date, row in hist.iterrows():
                data_point = MarketData(
                    date=date.to_pydatetime(),
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=float(row['Volume']) if not pd.isna(row['Volume']) else None,
                    adjusted_close=float(row['Close'])  # Yahoo Finance automatically adjusts
                )
                market_data.append(data_point)
            
            return market_data
        
        except Exception as e:
            print(f"Error fetching Yahoo Finance data for {symbol}: {e}")
            return []
    
    def get_current_price(self, symbol: str, data_type: DataType = DataType.STOCK) -> Optional[float]:
        """Get current price from Yahoo Finance."""
        try:
            if data_type == DataType.CRYPTO:
                symbol = self._convert_crypto_symbol(symbol)
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Try different price fields
            price_fields = ['currentPrice', 'regularMarketPrice', 'price', 'ask', 'bid']
            for field in price_fields:
                if field in info and info[field] is not None:
                    return float(info[field])
            
            # Fallback to latest close price
            hist = ticker.history(period="1d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
            
            return None
        
        except Exception:
            return None
    
    def _convert_crypto_symbol(self, symbol: str) -> str:
        """Convert crypto symbol to Yahoo Finance format."""
        crypto_mapping = {
            'bitcoin': 'BTC-USD',
            'btc': 'BTC-USD',
            'ethereum': 'ETH-USD',
            'eth': 'ETH-USD',
            'litecoin': 'LTC-USD',
            'ltc': 'LTC-USD',
            'cardano': 'ADA-USD',
            'ada': 'ADA-USD',
            'polkadot': 'DOT-USD',
            'dot': 'DOT-USD'
        }
        
        symbol_lower = symbol.lower()
        if symbol_lower in crypto_mapping:
            return crypto_mapping[symbol_lower]
        
        # If not in mapping, try adding -USD suffix
        if not symbol.upper().endswith('-USD'):
            return f"{symbol.upper()}-USD"
        
        return symbol.upper()


class MockDataSource(DataSource):
    """Mock data source for testing and fallback scenarios."""
    
    def __init__(self, config: Optional[DataSourceConfig] = None):
        """Initialize mock data source."""
        default_config = DataSourceConfig(
            source_name="mock_data",
            base_url="mock://localhost",
            rate_limit=None,  # No rate limit for mock data
            timeout=1
        )
        
        super().__init__(config or default_config)
        
        # Generate some realistic-looking base prices
        self.base_prices = {
            'AAPL': 150.0,
            'GOOGL': 2500.0,
            'MSFT': 300.0,
            'TSLA': 800.0,
            'NVDA': 400.0,
            'BTC-USD': 45000.0,
            'ETH-USD': 3000.0,
            'bitcoin': 45000.0,
            'ethereum': 3000.0
        }
    
    def get_historical_data(
        self, 
        symbol: str, 
        days: int = 90, 
        data_type: DataType = DataType.STOCK
    ) -> List[MarketData]:
        """Generate mock historical data."""
        # Get base price for symbol
        base_price = self.base_prices.get(symbol.upper(), 100.0)
        
        market_data = []
        current_date = datetime.now() - timedelta(days=days)
        
        # Generate realistic price movement
        price = base_price
        np.random.seed(hash(symbol) % 2**32)  # Consistent random data for same symbol
        
        for i in range(days):
            # Random walk with slight upward bias
            daily_return = np.random.normal(0.001, 0.02)  # 0.1% mean return, 2% volatility
            price *= (1 + daily_return)
            
            # Generate OHLC data
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = low + (high - low) * np.random.random()
            close_price = low + (high - low) * np.random.random()
            
            # Ensure OHLC relationships are valid
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)
            
            volume = np.random.randint(1000000, 10000000) if data_type == DataType.STOCK else np.random.randint(100000, 1000000)
            
            data_point = MarketData(
                date=current_date + timedelta(days=i),
                open=round(open_price, 2),
                high=round(high, 2),
                low=round(low, 2),
                close=round(close_price, 2),
                volume=volume,
                adjusted_close=round(close_price, 2)
            )
            market_data.append(data_point)
            
            price = close_price  # Use close as next day's base
        
        return market_data
    
    def get_current_price(self, symbol: str, data_type: DataType = DataType.STOCK) -> Optional[float]:
        """Get mock current price."""
        base_price = self.base_prices.get(symbol.upper(), 100.0)
        
        # Add some random variation
        np.random.seed(hash(symbol + str(datetime.now().date())) % 2**32)
        variation = np.random.normal(0, 0.01)  # 1% daily variation
        
        return round(base_price * (1 + variation), 2)


class DataSourceManager:
    """Manager class for handling multiple data sources with fallback."""
    
    def __init__(self, primary_sources: Optional[List[DataSource]] = None):
        """
        Initialize data source manager.
        
        Args:
            primary_sources: List of primary data sources to try
        """
        self.sources = primary_sources or []
        
        # Always add mock source as fallback
        self.fallback_source = MockDataSource()
        
        # Add Yahoo Finance if available
        if YFINANCE_AVAILABLE and not any(isinstance(s, YahooFinanceSource) for s in self.sources):
            try:
                self.sources.insert(0, YahooFinanceSource())
            except Exception:
                pass
    
    def get_historical_data(
        self, 
        symbol: str, 
        days: int = 90, 
        data_type: DataType = DataType.STOCK
    ) -> List[MarketData]:
        """
        Get historical data trying sources in order with fallback.
        
        Args:
            symbol: Financial instrument symbol
            days: Number of days of historical data
            data_type: Type of data
            
        Returns:
            List of market data points
        """
        # Try each source in order
        for source in self.sources:
            try:
                data = source.get_historical_data(symbol, days, data_type)
                if data:  # If we got data, return it
                    return data
            except Exception as e:
                print(f"Source {source.config.source_name} failed for {symbol}: {e}")
                continue
        
        # If all primary sources fail, use fallback
        print(f"All primary sources failed for {symbol}, using mock data")
        return self.fallback_source.get_historical_data(symbol, days, data_type)
    
    def get_current_price(self, symbol: str, data_type: DataType = DataType.STOCK) -> Optional[float]:
        """Get current price with fallback."""
        for source in self.sources:
            try:
                price = source.get_current_price(symbol, data_type)
                if price is not None:
                    return price
            except Exception:
                continue
        
        # Fallback to mock data
        return self.fallback_source.get_current_price(symbol, data_type)
    
    def add_source(self, source: DataSource, priority: int = -1):
        """
        Add a new data source.
        
        Args:
            source: Data source to add
            priority: Position in the source list (0 = highest priority, -1 = append)
        """
        if priority == -1:
            self.sources.append(source)
        else:
            self.sources.insert(priority, source)
    
    def remove_source(self, source_name: str) -> bool:
        """
        Remove a data source by name.
        
        Args:
            source_name: Name of the source to remove
            
        Returns:
            True if source was removed, False if not found
        """
        for i, source in enumerate(self.sources):
            if source.config.source_name == source_name:
                self.sources.pop(i)
                return True
        return False
    
    def list_sources(self) -> List[str]:
        """Get list of available source names."""
        return [source.config.source_name for source in self.sources]


# Default data source manager instance
default_data_source = DataSourceManager()


# Convenience functions
def get_historical_data(symbol: str, days: int = 90, data_type: DataType = DataType.STOCK) -> List[MarketData]:
    """Get historical data using default data source manager."""
    return default_data_source.get_historical_data(symbol, days, data_type)


def get_current_price(symbol: str, data_type: DataType = DataType.STOCK) -> Optional[float]:
    """Get current price using default data source manager."""
    return default_data_source.get_current_price(symbol, data_type)


__all__ = [
    "DataSource",
    "YahooFinanceSource", 
    "MockDataSource",
    "DataSourceManager",
    "default_data_source",
    "get_historical_data",
    "get_current_price"
]