"""Main FastAPI application for WebScraper App."""

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

from app.auth import get_auth_dependency
from app.api import market, news, weather, analysis

# Load environment variables
load_dotenv()

# Verify API token is configured
if not os.getenv("API_TOKEN"):
    raise ValueError("API_TOKEN environment variable must be set. Copy .env.example to .env and configure your token.")

app = FastAPI(
    title="WebScraper Analysis API",
    description="A modern web scraping and analysis platform providing market data, news, weather, and forecasting capabilities.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers with authentication
auth_dependency = get_auth_dependency()

app.include_router(
    market.router, 
    prefix="/api/market", 
    tags=["Market Data"],
    dependencies=[Depends(auth_dependency)]
)

app.include_router(
    news.router, 
    prefix="/api/news", 
    tags=["News"],
    dependencies=[Depends(auth_dependency)]
)

app.include_router(
    weather.router, 
    prefix="/api/weather", 
    tags=["Weather"],
    dependencies=[Depends(auth_dependency)]
)

app.include_router(
    analysis.router, 
    prefix="/api/analysis", 
    tags=["Analysis & Forecasting"],
    dependencies=[Depends(auth_dependency)]
)

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint providing API information."""
    return {
        "message": "WebScraper Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "market": "/api/market",
            "news": "/api/news", 
            "weather": "/api/weather",
            "analysis": "/api/analysis"
        }
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "WebScraper Analysis API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)