"""Analysis service for Pi cycles and capital flows analysis."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import asyncio

from app.models import PiCycleAnalysis, CapitalFlowAnalysis, TechnicalIndicators
from app.services.market_data import MarketDataService

class AnalysisService:
    """Service for advanced market analysis including Pi cycles and capital flows."""
    
    def __init__(self):
        self.market_service = MarketDataService()
    
    async def calculate_pi_cycle_analysis(self, symbol: str = "bitcoin") -> PiCycleAnalysis:
        """
        Calculate Pi Cycle Top analysis for cryptocurrency (primarily Bitcoin).
        Pi Cycle Top = 111 Day Moving Average × 2 crosses above 350 Day Moving Average
        """
        try:
            # Fetch extended historical data for moving averages
            historical_data = await self.market_service.get_historical_data(
                symbol, days=400, data_type="crypto"
            )
            
            if not historical_data or len(historical_data) < 350:
                # Fallback to mock data for demonstration
                return self._generate_mock_pi_cycle_analysis(symbol)
            
            df = pd.DataFrame(historical_data)
            df['date'] = pd.to_datetime(df['date'])
            df.sort_values('date', inplace=True)
            
            # Calculate moving averages
            df['ma_111'] = df['close'].rolling(window=111).mean()
            df['ma_350'] = df['close'].rolling(window=350).mean()
            df['pi_cycle_top'] = df['ma_111'] * 2
            
            # Get the latest values
            latest_data = df.dropna().tail(1)
            if latest_data.empty:
                return self._generate_mock_pi_cycle_analysis(symbol)
            
            current_ratio = float(latest_data['pi_cycle_top'].iloc[0] / latest_data['ma_350'].iloc[0])
            
            # Determine signal strength based on ratio
            if current_ratio > 1.2:
                signal_strength = "Strong Sell"
                confidence = min(0.9, (current_ratio - 1.0) * 2)
            elif current_ratio > 1.1:
                signal_strength = "Sell"
                confidence = 0.7
            elif current_ratio > 0.9:
                signal_strength = "Hold"
                confidence = 0.6
            elif current_ratio > 0.8:
                signal_strength = "Buy"
                confidence = 0.7
            else:
                signal_strength = "Strong Buy"
                confidence = min(0.9, (1.0 - current_ratio) * 3)
            
            # Estimate days to next cycle (simplified calculation)
            # Bitcoin cycles are roughly 4 years (1461 days)
            days_since_last_peak = 400  # Placeholder
            cycle_length = 1461
            days_to_next_cycle = cycle_length - (days_since_last_peak % cycle_length)
            
            return PiCycleAnalysis(
                symbol=symbol.upper(),
                current_ratio=current_ratio,
                signal_strength=signal_strength,
                days_to_next_cycle=days_to_next_cycle,
                confidence=confidence,
                analysis_date=datetime.now()
            )
        except Exception as e:
            # Return mock analysis if real calculation fails
            return self._generate_mock_pi_cycle_analysis(symbol)
    
    def _generate_mock_pi_cycle_analysis(self, symbol: str) -> PiCycleAnalysis:
        """Generate mock Pi cycle analysis for demonstration."""
        import random
        
        # Generate realistic-looking mock data
        current_ratio = random.uniform(0.7, 1.3)
        
        if current_ratio > 1.15:
            signal_strength = "Strong Sell"
            confidence = 0.85
        elif current_ratio > 1.05:
            signal_strength = "Sell"
            confidence = 0.75
        elif current_ratio > 0.95:
            signal_strength = "Hold"
            confidence = 0.65
        elif current_ratio > 0.85:
            signal_strength = "Buy"
            confidence = 0.75
        else:
            signal_strength = "Strong Buy"
            confidence = 0.85
        
        return PiCycleAnalysis(
            symbol=symbol.upper(),
            current_ratio=current_ratio,
            signal_strength=signal_strength,
            days_to_next_cycle=random.randint(200, 800),
            confidence=confidence,
            analysis_date=datetime.now()
        )
    
    async def calculate_technical_indicators(self, symbol: str, data_type: str = "stock") -> TechnicalIndicators:
        """Calculate various technical indicators."""
        try:
            # Fetch historical data
            historical_data = await self.market_service.get_historical_data(
                symbol, days=60, data_type=data_type
            )
            
            if not historical_data or len(historical_data) < 20:
                return self._generate_mock_technical_indicators()
            
            df = pd.DataFrame(historical_data)
            df['date'] = pd.to_datetime(df['date'])
            df.sort_values('date', inplace=True)
            
            # Calculate RSI (Relative Strength Index)
            rsi = self._calculate_rsi(df['close'])
            
            # Calculate MACD
            macd_line, macd_signal = self._calculate_macd(df['close'])
            macd = macd_line - macd_signal if macd_line is not None and macd_signal is not None else None
            
            # Calculate Bollinger Bands
            bollinger_upper, bollinger_lower = self._calculate_bollinger_bands(df['close'])
            
            # Calculate Moving Averages
            ma_20 = df['close'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else None
            ma_50 = df['close'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else None
            
            return TechnicalIndicators(
                rsi=float(rsi) if rsi is not None else None,
                macd=float(macd) if macd is not None else None,
                bollinger_upper=float(bollinger_upper) if bollinger_upper is not None else None,
                bollinger_lower=float(bollinger_lower) if bollinger_lower is not None else None,
                moving_average_20=float(ma_20) if ma_20 is not None else None,
                moving_average_50=float(ma_50) if ma_50 is not None else None
            )
        except Exception:
            return self._generate_mock_technical_indicators()
    
    def _generate_mock_technical_indicators(self) -> TechnicalIndicators:
        """Generate mock technical indicators for demonstration."""
        import random
        
        return TechnicalIndicators(
            rsi=random.uniform(30, 70),
            macd=random.uniform(-2, 2),
            bollinger_upper=random.uniform(100, 200),
            bollinger_lower=random.uniform(80, 120),
            moving_average_20=random.uniform(90, 150),
            moving_average_50=random.uniform(85, 155)
        )
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> Optional[float]:
        """Calculate Relative Strength Index."""
        try:
            if len(prices) < period + 1:
                return None
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.iloc[-1]
        except Exception:
            return None
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[Optional[float], Optional[float]]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        try:
            if len(prices) < slow:
                return None, None
            
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            macd_signal = macd_line.ewm(span=signal).mean()
            
            return macd_line.iloc[-1], macd_signal.iloc[-1]
        except Exception:
            return None, None
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[Optional[float], Optional[float]]:
        """Calculate Bollinger Bands."""
        try:
            if len(prices) < period:
                return None, None
            
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            return upper_band.iloc[-1], lower_band.iloc[-1]
        except Exception:
            return None, None
    
    async def calculate_capital_flow_analysis(self, symbol: str, data_type: str = "stock") -> CapitalFlowAnalysis:
        """Calculate capital flow analysis with technical indicators."""
        try:
            # Get technical indicators
            technical_indicators = await self.calculate_technical_indicators(symbol, data_type)
            
            # Get recent market data for flow calculation
            try:
                market_data = await self.market_service.get_stock_data(symbol) if data_type == "stock" else await self.market_service.get_crypto_data(symbol)
                volume_24h = market_data.volume_24h or 0
                price_change_24h = market_data.price_change_24h
            except Exception:
                volume_24h = 1000000  # Mock volume
                price_change_24h = 0
            
            # Calculate net flow (simplified approach)
            # Positive price change with high volume suggests inflow
            net_flow_24h = volume_24h * (price_change_24h / 100) if price_change_24h else 0
            
            # Determine flow strengths
            if abs(net_flow_24h) > volume_24h * 0.05:  # > 5% of volume
                if net_flow_24h > 0:
                    inflow_strength = "Strong"
                    outflow_strength = "Weak"
                else:
                    inflow_strength = "Weak"
                    outflow_strength = "Strong"
            else:
                inflow_strength = "Moderate"
                outflow_strength = "Moderate"
            
            # Determine overall trend
            rsi = technical_indicators.rsi or 50
            macd = technical_indicators.macd or 0
            
            if rsi > 60 and macd > 0:
                flow_trend = "Bullish"
            elif rsi < 40 and macd < 0:
                flow_trend = "Bearish"
            else:
                flow_trend = "Neutral"
            
            return CapitalFlowAnalysis(
                symbol=symbol.upper(),
                net_flow_24h=net_flow_24h,
                inflow_strength=inflow_strength,
                outflow_strength=outflow_strength,
                flow_trend=flow_trend,
                technical_indicators=technical_indicators,
                analysis_date=datetime.now()
            )
        except Exception as e:
            # Return mock analysis if calculation fails
            return self._generate_mock_capital_flow_analysis(symbol)
    
    def _generate_mock_capital_flow_analysis(self, symbol: str) -> CapitalFlowAnalysis:
        """Generate mock capital flow analysis for demonstration."""
        import random
        
        net_flow = random.uniform(-1000000, 1000000)
        flows = ["Weak", "Moderate", "Strong"]
        trends = ["Bullish", "Bearish", "Neutral"]
        
        return CapitalFlowAnalysis(
            symbol=symbol.upper(),
            net_flow_24h=net_flow,
            inflow_strength=random.choice(flows),
            outflow_strength=random.choice(flows),
            flow_trend=random.choice(trends),
            technical_indicators=self._generate_mock_technical_indicators(),
            analysis_date=datetime.now()
        )
    
    async def get_market_sentiment_analysis(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get comprehensive market sentiment analysis for multiple symbols."""
        try:
            sentiment_data = {}
            
            # Process symbols concurrently
            tasks = []
            for symbol in symbols:
                # Determine data type based on symbol
                data_type = "crypto" if symbol.lower() in ['bitcoin', 'ethereum', 'btc', 'eth'] else "stock"
                
                tasks.append(self._get_symbol_sentiment(symbol, data_type))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for symbol, result in zip(symbols, results):
                if isinstance(result, Exception):
                    continue
                sentiment_data[symbol] = result
            
            return sentiment_data
        except Exception as e:
            raise ValueError(f"Error calculating market sentiment: {str(e)}")
    
    async def _get_symbol_sentiment(self, symbol: str, data_type: str) -> Dict:
        """Get sentiment analysis for a single symbol."""
        try:
            # Get technical analysis
            technical_indicators = await self.calculate_technical_indicators(symbol, data_type)
            
            # Get capital flow analysis
            capital_flow = await self.calculate_capital_flow_analysis(symbol, data_type)
            
            # Calculate overall sentiment score
            sentiment_score = 0
            factors = 0
            
            # RSI factor
            if technical_indicators.rsi:
                if technical_indicators.rsi > 70:
                    sentiment_score -= 0.3  # Overbought
                elif technical_indicators.rsi < 30:
                    sentiment_score += 0.3  # Oversold
                else:
                    sentiment_score += (50 - technical_indicators.rsi) / 100
                factors += 1
            
            # MACD factor
            if technical_indicators.macd:
                sentiment_score += min(0.2, max(-0.2, technical_indicators.macd / 10))
                factors += 1
            
            # Flow trend factor
            if capital_flow.flow_trend == "Bullish":
                sentiment_score += 0.2
            elif capital_flow.flow_trend == "Bearish":
                sentiment_score -= 0.2
            factors += 1
            
            # Normalize sentiment score
            if factors > 0:
                sentiment_score = sentiment_score / factors
            
            # Convert to sentiment label
            if sentiment_score > 0.1:
                sentiment_label = "Bullish"
            elif sentiment_score < -0.1:
                sentiment_label = "Bearish"
            else:
                sentiment_label = "Neutral"
            
            return {
                "sentiment_score": sentiment_score,
                "sentiment_label": sentiment_label,
                "technical_indicators": technical_indicators.dict(),
                "capital_flow": capital_flow.dict(),
                "confidence": min(1.0, abs(sentiment_score) + 0.5)
            }
        except Exception:
            return {
                "sentiment_score": 0.0,
                "sentiment_label": "Neutral",
                "confidence": 0.5
            }