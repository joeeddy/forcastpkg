"""Basic tests for the WebScraper Analysis API."""

import pytest
import asyncio
from fastapi.testclient import TestClient
import os
from unittest.mock import patch, AsyncMock

# Set up test environment
os.environ["API_TOKEN"] = "test_token_123"

from app.main import app

client = TestClient(app)

# Authentication headers for tests
TEST_HEADERS = {"Authorization": "Bearer test_token_123"}
INVALID_HEADERS = {"Authorization": "Bearer invalid_token"}

class TestAuthentication:
    """Test authentication functionality."""
    
    def test_root_endpoint_no_auth(self):
        """Test that root endpoint works without authentication."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"] == "WebScraper Analysis API"
    
    def test_health_endpoint_no_auth(self):
        """Test that health endpoint works without authentication."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_protected_endpoint_no_auth(self):
        """Test that protected endpoints require authentication."""
        response = client.get("/api/market/stock/AAPL")
        assert response.status_code == 403  # FastAPI returns 403 for missing auth
    
    def test_protected_endpoint_invalid_auth(self):
        """Test that protected endpoints reject invalid tokens."""
        response = client.get("/api/market/stock/AAPL", headers=INVALID_HEADERS)
        assert response.status_code == 401
    
    def test_protected_endpoint_valid_auth(self):
        """Test that protected endpoints accept valid tokens."""
        with patch('app.services.market_data.MarketDataService.get_stock_data') as mock_get_stock:
            # Mock the service to avoid external API calls in tests
            mock_data = {
                "symbol": "AAPL",
                "current_price": 150.0,
                "price_change_24h": 2.5,
                "price_change_percentage_24h": 1.69,
                "market_cap": 2500000000000,
                "volume_24h": 50000000,
                "timestamp": "2024-01-01T00:00:00"
            }
            mock_get_stock.return_value = AsyncMock(return_value=mock_data)
            
            response = client.get("/api/market/stock/AAPL", headers=TEST_HEADERS)
            # We expect this to work with valid auth, even if the service is mocked
            assert response.status_code in [200, 400, 500]  # Various outcomes depending on mock

class TestMarketAPI:
    """Test market data API endpoints."""
    
    @patch('app.services.market_data.MarketDataService.get_stock_data')
    def test_get_stock_data_endpoint(self, mock_get_stock):
        """Test stock data endpoint."""
        from app.models import MarketDataResponse
        from datetime import datetime
        
        # Mock successful response
        mock_response = MarketDataResponse(
            symbol="AAPL",
            current_price=150.0,
            price_change_24h=2.5,
            price_change_percentage_24h=1.69,
            market_cap=2500000000000,
            volume_24h=50000000,
            timestamp=datetime.now()
        )
        mock_get_stock.return_value = mock_response
        
        response = client.get("/api/market/stock/AAPL", headers=TEST_HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "AAPL"
        assert data["current_price"] == 150.0
    
    def test_get_stock_data_invalid_symbol(self):
        """Test stock data endpoint with invalid symbol."""
        response = client.get("/api/market/stock/INVALID", headers=TEST_HEADERS)
        # Should return an error (400 or 500 depending on implementation)
        assert response.status_code in [400, 500]

class TestNewsAPI:
    """Test news API endpoints."""
    
    @patch('app.services.news.NewsService.get_google_news')
    def test_get_google_news_endpoint(self, mock_get_news):
        """Test Google News endpoint."""
        from app.models import NewsResponse, NewsArticle
        from datetime import datetime
        
        # Mock successful response
        mock_articles = [
            NewsArticle(
                title="Test News Article",
                summary="Test summary",
                url="https://example.com/news",
                published_at=datetime.now(),
                source="Test Source"
            )
        ]
        mock_response = NewsResponse(
            articles=mock_articles,
            total_count=1,
            source_type="Google News RSS"
        )
        mock_get_news.return_value = mock_response
        
        response = client.get("/api/news/google", headers=TEST_HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 1
        assert len(data["articles"]) == 1
    
    def test_sentiment_analysis_endpoint(self):
        """Test sentiment analysis endpoint."""
        test_text = "This is great news for the market"
        response = client.get(f"/api/news/sentiment/{test_text}", headers=TEST_HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert "sentiment_score" in data
        assert "sentiment_label" in data
        assert data["text"] == test_text

class TestWeatherAPI:
    """Test weather API endpoints."""
    
    @patch('app.services.weather.WeatherService.get_current_weather')
    def test_get_current_weather_endpoint(self, mock_get_weather):
        """Test current weather endpoint."""
        from app.models import WeatherData
        from datetime import datetime
        
        # Mock successful response
        mock_response = WeatherData(
            temperature=22.5,
            humidity=65.0,
            wind_speed=10.0,
            weather_description="Partly cloudy",
            timestamp=datetime.now(),
            location="New York"
        )
        mock_get_weather.return_value = mock_response
        
        response = client.get("/api/weather/current/New York", headers=TEST_HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert data["location"] == "New York"
        assert data["temperature"] == 22.5
    
    def test_get_weather_invalid_location(self):
        """Test weather endpoint with invalid location."""
        response = client.get("/api/weather/current/InvalidLocation123", headers=TEST_HEADERS)
        # Should return an error
        assert response.status_code in [400, 500]

class TestAnalysisAPI:
    """Test analysis and forecasting API endpoints."""
    
    @patch('app.services.forecasting.ForecastingService.generate_arima_forecast')
    def test_generate_forecast_endpoint(self, mock_forecast):
        """Test forecast generation endpoint."""
        from app.models import ForecastResponse, ForecastPoint
        from datetime import datetime, timedelta
        
        # Mock successful response
        forecast_points = [
            ForecastPoint(
                date=datetime.now() + timedelta(days=i),
                predicted_value=150.0 + i,
                confidence_interval_lower=145.0 + i,
                confidence_interval_upper=155.0 + i
            )
            for i in range(5)
        ]
        
        mock_response = ForecastResponse(
            symbol="AAPL",
            model_type="ARIMA(1,1,1)",
            forecast_period_days=5,
            forecast_points=forecast_points,
            model_accuracy=0.85,
            generated_at=datetime.now()
        )
        mock_forecast.return_value = mock_response
        
        response = client.get("/api/analysis/forecast/AAPL?forecast_days=5&model_type=arima", headers=TEST_HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "AAPL"
        assert data["forecast_period_days"] == 5
        assert len(data["forecast_points"]) == 5
    
    @patch('app.services.analysis.AnalysisService.calculate_pi_cycle_analysis')
    def test_pi_cycle_analysis_endpoint(self, mock_pi_cycle):
        """Test Pi cycle analysis endpoint."""
        from app.models import PiCycleAnalysis
        from datetime import datetime
        
        # Mock successful response
        mock_response = PiCycleAnalysis(
            symbol="BITCOIN",
            current_ratio=0.95,
            signal_strength="Buy",
            days_to_next_cycle=450,
            confidence=0.75,
            analysis_date=datetime.now()
        )
        mock_pi_cycle.return_value = mock_response
        
        response = client.get("/api/analysis/pi-cycle/bitcoin", headers=TEST_HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "BITCOIN"
        assert data["signal_strength"] == "Buy"

class TestIntegration:
    """Integration tests for the full API."""
    
    def test_api_documentation_accessible(self):
        """Test that API documentation is accessible."""
        response = client.get("/docs")
        assert response.status_code == 200
        
        response = client.get("/redoc")
        assert response.status_code == 200
    
    def test_openapi_schema(self):
        """Test that OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert data["info"]["title"] == "WebScraper Analysis API"

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])