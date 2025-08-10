"""Market data API endpoints."""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Any
import asyncio

from app.services.market_data import MarketDataService
from app.models import MarketDataResponse

router = APIRouter()
market_service = MarketDataService()

@router.get("/stock/{symbol}", response_model=MarketDataResponse)
async def get_stock_data(symbol: str):
    """Get current stock data for a symbol."""
    try:
        return await market_service.get_stock_data(symbol)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/crypto/{symbol}", response_model=MarketDataResponse)
async def get_crypto_data(symbol: str):
    """Get current cryptocurrency data for a symbol."""
    try:
        return await market_service.get_crypto_data(symbol)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/overview")
async def get_market_overview():
    """Get overview of major market indices and cryptocurrencies."""
    try:
        return await market_service.get_market_overview()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/historical/{symbol}")
async def get_historical_data(
    symbol: str,
    days: int = Query(30, ge=1, le=365, description="Number of days of historical data"),
    data_type: str = Query("stock", regex="^(stock|crypto)$", description="Type of data: stock or crypto")
):
    """Get historical data for a symbol."""
    try:
        data = await market_service.get_historical_data(symbol, days, data_type)
        return {
            "symbol": symbol,
            "data_type": data_type,
            "days_requested": days,
            "data_points": len(data),
            "historical_data": data
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/compare")
async def compare_symbols(
    symbols: str = Query(..., description="Comma-separated list of symbols to compare"),
    data_type: str = Query("stock", regex="^(stock|crypto)$", description="Type of data: stock or crypto")
):
    """Compare multiple symbols."""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        
        if len(symbol_list) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 symbols allowed")
        
        comparison_data = {}
        
        # Fetch data for all symbols concurrently
        if data_type == "stock":
            tasks = [market_service.get_stock_data(symbol) for symbol in symbol_list]
        else:
            tasks = [market_service.get_crypto_data(symbol) for symbol in symbol_list]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for symbol, result in zip(symbol_list, results):
            if isinstance(result, Exception):
                comparison_data[symbol] = {"error": str(result)}
            else:
                comparison_data[symbol] = result.dict()
        
        return {
            "comparison_type": data_type,
            "symbols_requested": symbol_list,
            "comparison_data": comparison_data
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/search")
async def search_symbols(
    query: str = Query(..., min_length=1, description="Search query for symbols")
):
    """Search for stock/crypto symbols (simplified implementation)."""
    try:
        # This is a simplified implementation
        # In production, you might want to use a proper symbol search API
        
        query_upper = query.upper()
        
        # Common stock symbols
        common_stocks = {
            "AAPL": "Apple Inc.",
            "GOOGL": "Alphabet Inc.",
            "MSFT": "Microsoft Corporation",
            "AMZN": "Amazon.com Inc.",
            "TSLA": "Tesla Inc.",
            "META": "Meta Platforms Inc.",
            "NVDA": "NVIDIA Corporation",
            "SPY": "SPDR S&P 500 ETF",
            "QQQ": "Invesco QQQ Trust"
        }
        
        # Common crypto symbols
        common_crypto = {
            "BITCOIN": "Bitcoin",
            "BTC": "Bitcoin",
            "ETHEREUM": "Ethereum", 
            "ETH": "Ethereum",
            "ADA": "Cardano",
            "SOL": "Solana",
            "DOT": "Polkadot",
            "MATIC": "Polygon"
        }
        
        matches = {}
        
        # Search in stocks
        for symbol, name in common_stocks.items():
            if query_upper in symbol or query_upper in name.upper():
                matches[symbol] = {"name": name, "type": "stock"}
        
        # Search in crypto
        for symbol, name in common_crypto.items():
            if query_upper in symbol or query_upper in name.upper():
                matches[symbol] = {"name": name, "type": "crypto"}
        
        return {
            "query": query,
            "matches": matches,
            "total_matches": len(matches)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))