"""Analysis and forecasting API endpoints."""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Optional

from app.services.forecasting import ForecastingService
from app.services.analysis import AnalysisService
from app.models import ForecastResponse, PiCycleAnalysis, CapitalFlowAnalysis

router = APIRouter()
forecasting_service = ForecastingService()
analysis_service = AnalysisService()

@router.get("/forecast/{symbol}", response_model=ForecastResponse)
async def generate_forecast(
    symbol: str,
    forecast_days: int = Query(30, ge=1, le=90, description="Number of days to forecast"),
    model_type: str = Query("arima", regex="^(arima|linear|moving_average)$", description="Forecasting model to use"),
    data_type: str = Query("stock", regex="^(stock|crypto)$", description="Type of data: stock or crypto")
):
    """Generate forecast for a financial instrument."""
    try:
        if model_type == "arima":
            return await forecasting_service.generate_arima_forecast(symbol, forecast_days, data_type)
        elif model_type == "linear":
            return await forecasting_service.generate_linear_forecast(symbol, forecast_days, data_type)
        elif model_type == "moving_average":
            return await forecasting_service.generate_moving_average_forecast(symbol, forecast_days, data_type)
        else:
            raise HTTPException(status_code=400, detail="Invalid model type")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/forecast/compare/{symbol}")
async def compare_forecast_models(
    symbol: str,
    forecast_days: int = Query(30, ge=1, le=90, description="Number of days to forecast"),
    data_type: str = Query("stock", regex="^(stock|crypto)$", description="Type of data: stock or crypto")
):
    """Compare different forecasting models for a symbol."""
    try:
        models_comparison = await forecasting_service.compare_forecast_models(symbol, forecast_days, data_type)
        
        return {
            "symbol": symbol,
            "forecast_days": forecast_days,
            "data_type": data_type,
            "models": {model_name: forecast.dict() for model_name, forecast in models_comparison.items()},
            "models_count": len(models_comparison)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/pi-cycle/{symbol}", response_model=PiCycleAnalysis)
async def get_pi_cycle_analysis(symbol: str = "bitcoin"):
    """Get Pi Cycle Top analysis for cryptocurrency."""
    try:
        return await analysis_service.calculate_pi_cycle_analysis(symbol)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/capital-flow/{symbol}", response_model=CapitalFlowAnalysis)
async def get_capital_flow_analysis(
    symbol: str,
    data_type: str = Query("stock", regex="^(stock|crypto)$", description="Type of data: stock or crypto")
):
    """Get capital flow analysis for a symbol."""
    try:
        return await analysis_service.calculate_capital_flow_analysis(symbol, data_type)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/technical-indicators/{symbol}")
async def get_technical_indicators(
    symbol: str,
    data_type: str = Query("stock", regex="^(stock|crypto)$", description="Type of data: stock or crypto")
):
    """Get technical indicators for a symbol."""
    try:
        indicators = await analysis_service.calculate_technical_indicators(symbol, data_type)
        return {
            "symbol": symbol,
            "data_type": data_type,
            "technical_indicators": indicators.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/market-sentiment")
async def get_market_sentiment(
    symbols: str = Query("AAPL,GOOGL,bitcoin,ethereum", description="Comma-separated list of symbols")
):
    """Get comprehensive market sentiment analysis for multiple symbols."""
    try:
        symbol_list = [s.strip() for s in symbols.split(",")]
        
        if len(symbol_list) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 symbols allowed")
        
        sentiment_data = await analysis_service.get_market_sentiment_analysis(symbol_list)
        
        return {
            "symbols_analyzed": symbol_list,
            "sentiment_analysis": sentiment_data,
            "analysis_timestamp": sentiment_data[list(sentiment_data.keys())[0]].get("timestamp") if sentiment_data else None
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/portfolio-analysis")
async def analyze_portfolio(
    symbols: str = Query(..., description="Comma-separated list of portfolio symbols"),
    weights: Optional[str] = Query(None, description="Comma-separated weights for each symbol (optional)")
):
    """Analyze a portfolio of symbols."""
    try:
        symbol_list = [s.strip() for s in symbols.split(",")]
        
        if len(symbol_list) > 20:
            raise HTTPException(status_code=400, detail="Maximum 20 symbols allowed in portfolio")
        
        # Parse weights if provided
        weight_list = None
        if weights:
            try:
                weight_list = [float(w.strip()) for w in weights.split(",")]
                if len(weight_list) != len(symbol_list):
                    raise ValueError("Number of weights must match number of symbols")
                
                # Normalize weights to sum to 1
                total_weight = sum(weight_list)
                weight_list = [w / total_weight for w in weight_list]
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid weights format")
        else:
            # Equal weights
            weight_list = [1.0 / len(symbol_list)] * len(symbol_list)
        
        # Get sentiment analysis for all symbols
        sentiment_data = await analysis_service.get_market_sentiment_analysis(symbol_list)
        
        # Calculate portfolio metrics
        portfolio_sentiment = 0
        total_confidence = 0
        symbol_analysis = {}
        
        for i, symbol in enumerate(symbol_list):
            if symbol in sentiment_data:
                analysis = sentiment_data[symbol]
                weight = weight_list[i]
                
                portfolio_sentiment += analysis.get("sentiment_score", 0) * weight
                total_confidence += analysis.get("confidence", 0) * weight
                
                symbol_analysis[symbol] = {
                    "weight": weight,
                    "analysis": analysis
                }
        
        # Determine overall portfolio sentiment
        if portfolio_sentiment > 0.1:
            portfolio_label = "Bullish"
        elif portfolio_sentiment < -0.1:
            portfolio_label = "Bearish"
        else:
            portfolio_label = "Neutral"
        
        return {
            "portfolio_symbols": symbol_list,
            "portfolio_weights": weight_list,
            "portfolio_sentiment_score": portfolio_sentiment,
            "portfolio_sentiment_label": portfolio_label,
            "portfolio_confidence": total_confidence,
            "individual_analysis": symbol_analysis,
            "symbols_analyzed": len(symbol_analysis),
            "analysis_timestamp": "2024-01-01T00:00:00"  # Placeholder
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/risk-assessment/{symbol}")
async def assess_risk(
    symbol: str,
    data_type: str = Query("stock", regex="^(stock|crypto)$", description="Type of data: stock or crypto")
):
    """Assess risk metrics for a symbol."""
    try:
        # Get technical indicators for risk assessment
        indicators = await analysis_service.calculate_technical_indicators(symbol, data_type)
        capital_flow = await analysis_service.calculate_capital_flow_analysis(symbol, data_type)
        
        # Calculate risk score based on indicators
        risk_score = 0.5  # Base risk (medium)
        risk_factors = []
        
        # RSI-based risk
        if indicators.rsi:
            if indicators.rsi > 80:
                risk_score += 0.2
                risk_factors.append("Overbought conditions (RSI > 80)")
            elif indicators.rsi < 20:
                risk_score += 0.3
                risk_factors.append("Oversold conditions (RSI < 20)")
        
        # MACD-based risk
        if indicators.macd:
            if abs(indicators.macd) > 5:
                risk_score += 0.1
                risk_factors.append("High MACD divergence")
        
        # Flow-based risk
        if capital_flow.flow_trend == "Bearish":
            risk_score += 0.1
            risk_factors.append("Bearish capital flow trend")
        
        # Cap risk score at 1.0
        risk_score = min(1.0, risk_score)
        
        # Determine risk level
        if risk_score > 0.8:
            risk_level = "Very High"
        elif risk_score > 0.6:
            risk_level = "High"
        elif risk_score > 0.4:
            risk_level = "Medium"
        elif risk_score > 0.2:
            risk_level = "Low"
        else:
            risk_level = "Very Low"
        
        return {
            "symbol": symbol,
            "data_type": data_type,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "technical_indicators": indicators.dict(),
            "capital_flow_analysis": capital_flow.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))