"""Market data service using Yahoo Finance and CoinGecko APIs."""

import yfinance as yf
import ccxt
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import asyncio
import httpx

from app.models import MarketDataResponse

class MarketDataService:
    """Service for fetching market data from various sources."""
    
    def __init__(self):
        self.coingecko_base_url = "https://api.coingecko.com/api/v3"
    
    async def get_stock_data(self, symbol: str) -> MarketDataResponse:
        """Fetch stock data using Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="2d")
            
            if hist.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            current_price = hist['Close'].iloc[-1]
            previous_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            price_change = current_price - previous_price
            price_change_percentage = (price_change / previous_price) * 100 if previous_price != 0 else 0
            
            return MarketDataResponse(
                symbol=symbol,
                current_price=float(current_price),
                price_change_24h=float(price_change),
                price_change_percentage_24h=float(price_change_percentage),
                market_cap=info.get('marketCap'),
                volume_24h=float(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else None,
                timestamp=datetime.now()
            )
        except Exception as e:
            raise ValueError(f"Error fetching stock data for {symbol}: {str(e)}")
    
    async def get_crypto_data(self, symbol: str) -> MarketDataResponse:
        """Fetch cryptocurrency data using CoinGecko API."""
        try:
            # Convert symbol to CoinGecko ID format
            symbol_lower = symbol.lower()
            if symbol_lower == 'btc':
                coin_id = 'bitcoin'
            elif symbol_lower == 'eth':
                coin_id = 'ethereum'
            elif symbol_lower == 'ada':
                coin_id = 'cardano'
            else:
                coin_id = symbol_lower
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.coingecko_base_url}/simple/price",
                    params={
                        'ids': coin_id,
                        'vs_currencies': 'usd',
                        'include_24hr_change': 'true',
                        'include_market_cap': 'true',
                        'include_24hr_vol': 'true'
                    }
                )
                response.raise_for_status()
                data = response.json()
            
            if coin_id not in data:
                raise ValueError(f"Cryptocurrency {symbol} not found")
            
            coin_data = data[coin_id]
            
            return MarketDataResponse(
                symbol=symbol.upper(),
                current_price=coin_data['usd'],
                price_change_24h=coin_data['usd'] * (coin_data.get('usd_24h_change', 0) / 100),
                price_change_percentage_24h=coin_data.get('usd_24h_change', 0),
                market_cap=coin_data.get('usd_market_cap'),
                volume_24h=coin_data.get('usd_24h_vol'),
                timestamp=datetime.now()
            )
        except Exception as e:
            raise ValueError(f"Error fetching crypto data for {symbol}: {str(e)}")
    
    async def get_market_overview(self) -> Dict[str, List[MarketDataResponse]]:
        """Get overview of major market indices and cryptocurrencies."""
        try:
            # Major stock indices
            stock_symbols = ['^GSPC', '^IXIC', '^DJI', '^RUT']  # S&P 500, NASDAQ, Dow Jones, Russell 2000
            crypto_symbols = ['bitcoin', 'ethereum', 'cardano', 'solana']
            
            stocks_data = []
            crypto_data = []
            
            # Fetch stock data
            for symbol in stock_symbols:
                try:
                    data = await self.get_stock_data(symbol)
                    stocks_data.append(data)
                except Exception:
                    continue  # Skip if data unavailable
            
            # Fetch crypto data
            for symbol in crypto_symbols:
                try:
                    data = await self.get_crypto_data(symbol)
                    crypto_data.append(data)
                except Exception:
                    continue  # Skip if data unavailable
            
            return {
                'stocks': stocks_data,
                'crypto': crypto_data
            }
        except Exception as e:
            raise ValueError(f"Error fetching market overview: {str(e)}")
    
    async def get_historical_data(self, symbol: str, days: int = 30, data_type: str = "stock") -> List[Dict]:
        """Fetch historical data for analysis."""
        try:
            if data_type == "stock":
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=f"{days}d")
                
                if hist.empty:
                    raise ValueError(f"No historical data found for {symbol}")
                
                return [
                    {
                        'date': date.strftime('%Y-%m-%d'),
                        'open': float(row['Open']),
                        'high': float(row['High']),
                        'low': float(row['Low']),
                        'close': float(row['Close']),
                        'volume': float(row['Volume'])
                    }
                    for date, row in hist.iterrows()
                ]
            else:
                # For crypto, we'll use a simplified approach with daily data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                # This is a simplified implementation - in production you might want to use
                # more sophisticated crypto historical data APIs
                return []
                
        except Exception as e:
            raise ValueError(f"Error fetching historical data for {symbol}: {str(e)}")