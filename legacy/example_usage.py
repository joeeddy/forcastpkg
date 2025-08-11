"""Example usage of the WebScraper Analysis API."""

import httpx
import asyncio
import json
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"
API_TOKEN = "your_secret_token_here"  # Replace with your actual token

# Headers for authentication
HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

async def example_market_data():
    """Example: Fetch market data."""
    print("=== Market Data Examples ===")
    
    async with httpx.AsyncClient() as client:
        # Get stock data
        print("\n1. Getting Apple stock data...")
        response = await client.get(f"{API_BASE_URL}/api/market/stock/AAPL", headers=HEADERS)
        if response.status_code == 200:
            data = response.json()
            print(f"AAPL: ${data['current_price']:.2f} ({data['price_change_percentage_24h']:+.2f}%)")
        else:
            print(f"Error: {response.status_code} - {response.text}")
        
        # Get crypto data
        print("\n2. Getting Bitcoin data...")
        response = await client.get(f"{API_BASE_URL}/api/market/crypto/bitcoin", headers=HEADERS)
        if response.status_code == 200:
            data = response.json()
            print(f"Bitcoin: ${data['current_price']:,.2f} ({data['price_change_percentage_24h']:+.2f}%)")
        else:
            print(f"Error: {response.status_code} - {response.text}")
        
        # Get market overview
        print("\n3. Getting market overview...")
        response = await client.get(f"{API_BASE_URL}/api/market/overview", headers=HEADERS)
        if response.status_code == 200:
            data = response.json()
            print(f"Market overview: {len(data.get('stocks', []))} stocks, {len(data.get('crypto', []))} cryptocurrencies")
        else:
            print(f"Error: {response.status_code} - {response.text}")

async def example_news():
    """Example: Fetch news data."""
    print("\n\n=== News Examples ===")
    
    async with httpx.AsyncClient() as client:
        # Get Google News
        print("\n1. Getting financial news...")
        response = await client.get(f"{API_BASE_URL}/api/news/financial?max_articles=5", headers=HEADERS)
        if response.status_code == 200:
            data = response.json()
            print(f"Found {data['total_count']} financial news articles")
            for article in data['articles'][:3]:
                print(f"- {article['title'][:80]}...")
        else:
            print(f"Error: {response.status_code} - {response.text}")
        
        # Get Reddit posts
        print("\n2. Getting Reddit news...")
        response = await client.get(f"{API_BASE_URL}/api/news/reddit/worldnews?max_posts=5", headers=HEADERS)
        if response.status_code == 200:
            data = response.json()
            print(f"Found {data['total_count']} Reddit posts from r/worldnews")
            for article in data['articles'][:3]:
                print(f"- {article['title'][:80]}...")
        else:
            print(f"Error: {response.status_code} - {response.text}")

async def example_weather():
    """Example: Fetch weather data."""
    print("\n\n=== Weather Examples ===")
    
    async with httpx.AsyncClient() as client:
        # Get current weather
        print("\n1. Getting weather for New York...")
        response = await client.get(f"{API_BASE_URL}/api/weather/current/New York", headers=HEADERS)
        if response.status_code == 200:
            data = response.json()
            print(f"New York: {data['temperature']:.1f}°C, {data['weather_description']}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
        
        # Get weather for multiple cities
        print("\n2. Getting weather for multiple cities...")
        cities = "London,Tokyo,Sydney"
        response = await client.get(f"{API_BASE_URL}/api/weather/multiple?cities={cities}", headers=HEADERS)
        if response.status_code == 200:
            data = response.json()
            print(f"Weather for {data['successful_fetches']} cities:")
            for city, weather in data['weather_data'].items():
                print(f"- {city}: {weather['temperature']:.1f}°C, {weather['weather_description']}")
        else:
            print(f"Error: {response.status_code} - {response.text}")

async def example_forecasting():
    """Example: Generate forecasts."""
    print("\n\n=== Forecasting Examples ===")
    
    async with httpx.AsyncClient() as client:
        # Generate ARIMA forecast
        print("\n1. Generating ARIMA forecast for AAPL...")
        response = await client.get(
            f"{API_BASE_URL}/api/analysis/forecast/AAPL?forecast_days=7&model_type=arima", 
            headers=HEADERS
        )
        if response.status_code == 200:
            data = response.json()
            print(f"Generated {data['model_type']} forecast for {data['symbol']}")
            print(f"Forecast period: {data['forecast_period_days']} days")
            print(f"Model accuracy: {data['model_accuracy']:.2%}")
            print("First 3 forecast points:")
            for point in data['forecast_points'][:3]:
                print(f"- {point['date'][:10]}: ${point['predicted_value']:.2f}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
        
        # Compare forecast models
        print("\n2. Comparing forecast models for Bitcoin...")
        response = await client.get(
            f"{API_BASE_URL}/api/analysis/forecast/compare/bitcoin?forecast_days=14&data_type=crypto",
            headers=HEADERS
        )
        if response.status_code == 200:
            data = response.json()
            print(f"Compared {data['models_count']} models for {data['symbol']}")
            for model_name, forecast in data['models'].items():
                print(f"- {model_name}: accuracy {forecast['model_accuracy']:.2%}")
        else:
            print(f"Error: {response.status_code} - {response.text}")

async def example_analysis():
    """Example: Advanced analysis."""
    print("\n\n=== Analysis Examples ===")
    
    async with httpx.AsyncClient() as client:
        # Pi Cycle analysis
        print("\n1. Pi Cycle analysis for Bitcoin...")
        response = await client.get(f"{API_BASE_URL}/api/analysis/pi-cycle/bitcoin", headers=HEADERS)
        if response.status_code == 200:
            data = response.json()
            print(f"Pi Cycle signal: {data['signal_strength']} (confidence: {data['confidence']:.2%})")
            print(f"Current ratio: {data['current_ratio']:.3f}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
        
        # Technical indicators
        print("\n2. Technical indicators for TSLA...")
        response = await client.get(f"{API_BASE_URL}/api/analysis/technical-indicators/TSLA", headers=HEADERS)
        if response.status_code == 200:
            data = response.json()
            indicators = data['technical_indicators']
            print(f"RSI: {indicators.get('rsi', 'N/A')}")
            print(f"MACD: {indicators.get('macd', 'N/A')}")
            print(f"20-day MA: ${indicators.get('moving_average_20', 'N/A')}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
        
        # Market sentiment
        print("\n3. Market sentiment analysis...")
        symbols = "AAPL,GOOGL,bitcoin"
        response = await client.get(f"{API_BASE_URL}/api/analysis/market-sentiment?symbols={symbols}", headers=HEADERS)
        if response.status_code == 200:
            data = response.json()
            print(f"Sentiment analysis for {len(data['symbols_analyzed'])} symbols:")
            for symbol, analysis in data['sentiment_analysis'].items():
                sentiment = analysis.get('sentiment_label', 'N/A')
                confidence = analysis.get('confidence', 0)
                print(f"- {symbol}: {sentiment} (confidence: {confidence:.2%})")
        else:
            print(f"Error: {response.status_code} - {response.text}")

async def example_portfolio_analysis():
    """Example: Portfolio analysis."""
    print("\n\n=== Portfolio Analysis Example ===")
    
    async with httpx.AsyncClient() as client:
        # Portfolio analysis
        symbols = "AAPL,GOOGL,MSFT,TSLA"
        weights = "0.3,0.3,0.2,0.2"
        response = await client.get(
            f"{API_BASE_URL}/api/analysis/portfolio-analysis?symbols={symbols}&weights={weights}",
            headers=HEADERS
        )
        if response.status_code == 200:
            data = response.json()
            print(f"Portfolio sentiment: {data['portfolio_sentiment_label']}")
            print(f"Portfolio score: {data['portfolio_sentiment_score']:.3f}")
            print(f"Portfolio confidence: {data['portfolio_confidence']:.2%}")
            print(f"Analyzed {data['symbols_analyzed']} symbols")
        else:
            print(f"Error: {response.status_code} - {response.text}")

async def main():
    """Run all examples."""
    print("WebScraper Analysis API - Example Usage")
    print("=" * 50)
    print(f"API Base URL: {API_BASE_URL}")
    print(f"Using token: {API_TOKEN[:10]}...")
    
    try:
        await example_market_data()
        await example_news()
        await example_weather()
        await example_forecasting()
        await example_analysis()
        await example_portfolio_analysis()
        
        print(f"\n\nExample run completed at {datetime.now()}")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure the API server is running and your token is correct.")

if __name__ == "__main__":
    asyncio.run(main())