# WebScraper Analysis API

A modern, comprehensive web scraping and analysis platform built with FastAPI. This application provides real-time data aggregation, forecasting, and analysis capabilities for financial markets, news, weather, and more.

## 🚀 Features

### Data Sources & APIs
- **Market Data**: Stocks, cryptocurrencies using Yahoo Finance and CoinGecko
- **News Aggregation**: Google News RSS feeds and Reddit scraping
- **Weather Data**: Open-Meteo API (no API key required)
- **Financial Analysis**: Technical indicators, capital flows, market sentiment
- **Forecasting**: ARIMA, linear regression, and moving average models
- **Pi Cycles Analysis**: Cryptocurrency market cycle analysis

### Technical Features
- **FastAPI Framework**: Modern, fast web framework with automatic API documentation
- **Token Authentication**: Simple, secure token-based authentication
- **Modular Architecture**: Easily extensible for new data sources
- **Async Operations**: High-performance asynchronous data processing
- **Comprehensive APIs**: RESTful endpoints with detailed documentation
- **Error Handling**: Robust error handling and validation

## 📋 Requirements

- Python 3.8+
- Internet connection for data fetching
- API token for authentication (configured in `.env`)

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd WebscraperApp
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
cp .env.example .env
```

Edit `.env` and set your API token:
```env
API_TOKEN=your_secret_token_here
```

### 5. Run the Application
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## 📖 API Documentation

### Interactive Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Authentication
All API endpoints (except `/` and `/health`) require authentication using a Bearer token:

```bash
curl -H "Authorization: Bearer your_secret_token_here" \
     http://localhost:8000/api/market/stock/AAPL
```

## 🔗 API Endpoints

### Market Data (`/api/market`)

#### Get Stock Data
```http
GET /api/market/stock/{symbol}
```
Example: `/api/market/stock/AAPL`

#### Get Cryptocurrency Data
```http
GET /api/market/crypto/{symbol}
```
Example: `/api/market/crypto/bitcoin`

#### Market Overview
```http
GET /api/market/overview
```

#### Historical Data
```http
GET /api/market/historical/{symbol}?days=30&data_type=stock
```

#### Compare Symbols
```http
GET /api/market/compare?symbols=AAPL,GOOGL,MSFT&data_type=stock
```

### News (`/api/news`)

#### Google News
```http
GET /api/news/google?query=bitcoin&max_articles=20
```

#### Reddit Posts
```http
GET /api/news/reddit/{subreddit}?max_posts=20
```
Example: `/api/news/reddit/worldnews`

#### Financial News
```http
GET /api/news/financial?symbol=AAPL&max_articles=15
```

#### Trending Topics
```http
GET /api/news/trending
```

#### Sentiment Analysis
```http
GET /api/news/sentiment/{text}
```

### Weather (`/api/weather`)

#### Current Weather
```http
GET /api/weather/current/{location}
```
Example: `/api/weather/current/New York`

#### Weather Forecast
```http
GET /api/weather/forecast/{location}?days=7
```

#### Multiple Cities
```http
GET /api/weather/multiple?cities=London,Tokyo,Sydney
```

#### Weather Alerts
```http
GET /api/weather/alerts/{location}
```

### Analysis & Forecasting (`/api/analysis`)

#### Generate Forecast
```http
GET /api/analysis/forecast/{symbol}?forecast_days=30&model_type=arima&data_type=stock
```

Model types: `arima`, `linear`, `moving_average`

#### Compare Forecast Models
```http
GET /api/analysis/forecast/compare/{symbol}?forecast_days=30&data_type=stock
```

#### Pi Cycle Analysis
```http
GET /api/analysis/pi-cycle/{symbol}
```
Example: `/api/analysis/pi-cycle/bitcoin`

#### Capital Flow Analysis
```http
GET /api/analysis/capital-flow/{symbol}?data_type=stock
```

#### Technical Indicators
```http
GET /api/analysis/technical-indicators/{symbol}?data_type=stock
```

#### Market Sentiment
```http
GET /api/analysis/market-sentiment?symbols=AAPL,GOOGL,bitcoin
```

#### Portfolio Analysis
```http
GET /api/analysis/portfolio-analysis?symbols=AAPL,GOOGL,MSFT&weights=0.4,0.3,0.3
```

#### Risk Assessment
```http
GET /api/analysis/risk-assessment/{symbol}?data_type=stock
```

## 💡 Usage Examples

### Python Client Example
```python
import httpx
import asyncio

async def get_market_data():
    headers = {"Authorization": "Bearer your_secret_token_here"}
    
    async with httpx.AsyncClient() as client:
        # Get Apple stock data
        response = await client.get(
            "http://localhost:8000/api/market/stock/AAPL", 
            headers=headers
        )
        data = response.json()
        print(f"AAPL: ${data['current_price']:.2f}")
        
        # Get Bitcoin forecast
        response = await client.get(
            "http://localhost:8000/api/analysis/forecast/bitcoin?forecast_days=7&data_type=crypto",
            headers=headers
        )
        forecast = response.json()
        print(f"Bitcoin 7-day forecast: {forecast['model_type']}")

asyncio.run(get_market_data())
```

### cURL Examples
```bash
# Get stock data
curl -H "Authorization: Bearer your_token" \
     "http://localhost:8000/api/market/stock/AAPL"

# Get weather forecast
curl -H "Authorization: Bearer your_token" \
     "http://localhost:8000/api/weather/forecast/London?days=5"

# Get financial news
curl -H "Authorization: Bearer your_token" \
     "http://localhost:8000/api/news/financial?symbol=TSLA&max_articles=10"
```

## 🧪 Testing

### Run Tests
```bash
pytest tests/ -v
```

### Run Example Script
```bash
python examples/example_usage.py
```

## 🔧 Configuration

### Environment Variables (`.env`)
```env
# Required: API authentication token
API_TOKEN=your_secret_token_here

# Optional: Rate limiting
RATE_LIMIT_PER_MINUTE=60

# Optional: Cache settings
CACHE_EXPIRY_MINUTES=15

# Optional: External API settings
REDDIT_USER_AGENT=WebScraperApp/1.0
NEWS_MAX_ARTICLES=50
```

## 🚀 Deployment

### Production Deployment
```bash
# Install production server
pip install gunicorn

# Run with Gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🔌 Extensibility

### Adding New Data Sources

1. **Create Service Module**:
```python
# app/services/new_service.py
class NewDataService:
    async def fetch_data(self):
        # Implementation here
        pass
```

2. **Create API Router**:
```python
# app/api/new_api.py
from fastapi import APIRouter
router = APIRouter()

@router.get("/endpoint")
async def get_data():
    # Implementation here
    pass
```

3. **Register Router**:
```python
# app/main.py
from app.api import new_api
app.include_router(new_api.router, prefix="/api/new", tags=["New Data"])
```

### Custom Models
```python
# app/models/__init__.py
class CustomDataModel(BaseModel):
    field1: str
    field2: float
    timestamp: datetime
```

## 🛡️ Security

- **Token Authentication**: All endpoints protected with Bearer token
- **Input Validation**: Pydantic models for request/response validation
- **Rate Limiting**: Configurable rate limiting (optional)
- **CORS**: Configurable CORS settings
- **Error Handling**: Secure error messages without sensitive data exposure

## 📊 Monitoring & Logging

The application includes:
- Health check endpoint (`/health`)
- Structured error responses
- Request/response logging (configurable)
- Performance monitoring capabilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 External APIs Used

- **Yahoo Finance**: Stock market data
- **CoinGecko**: Cryptocurrency data
- **Google News RSS**: News aggregation
- **Reddit**: Social media posts
- **Open-Meteo**: Weather data (no API key required)

## ⚠️ Disclaimers

- Financial data is for informational purposes only
- Not financial advice - do your own research
- Weather forecasts are estimates and may not be accurate
- External API availability may affect service functionality
- Rate limits may apply to external APIs

## 📞 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check the API documentation at `/docs`
- Review example usage in `examples/`

---

**Built with ❤️ using FastAPI, Python, and modern web technologies.**