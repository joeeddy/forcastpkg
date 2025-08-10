"""News aggregation API endpoints."""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional

from app.services.news import NewsService
from app.models import NewsResponse

router = APIRouter()
news_service = NewsService()

@router.get("/google", response_model=NewsResponse)
async def get_google_news(
    query: str = Query("", description="Search query for news (empty for general news)"),
    max_articles: int = Query(20, ge=1, le=50, description="Maximum number of articles to fetch")
):
    """Get news from Google News RSS feed."""
    try:
        return await news_service.get_google_news(query, max_articles)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reddit/{subreddit}", response_model=NewsResponse)
async def get_reddit_posts(
    subreddit: str,
    max_posts: int = Query(20, ge=1, le=50, description="Maximum number of posts to fetch")
):
    """Get posts from a specific Reddit subreddit."""
    try:
        return await news_service.get_reddit_posts(subreddit, max_posts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/financial", response_model=NewsResponse)
async def get_financial_news(
    symbol: str = Query("", description="Stock/crypto symbol to filter news (optional)"),
    max_articles: int = Query(15, ge=1, le=50, description="Maximum number of articles to fetch")
):
    """Get financial news, optionally filtered by symbol."""
    try:
        return await news_service.get_financial_news(symbol, max_articles)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trending")
async def get_trending_topics():
    """Get trending topics from various news sources."""
    try:
        topics = await news_service.get_trending_topics()
        return {
            "trending_topics": topics,
            "total_topics": len(topics),
            "source": "Multiple sources aggregation"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sentiment/{text}")
async def analyze_sentiment(text: str):
    """Analyze sentiment of provided text."""
    try:
        sentiment_score = news_service.calculate_sentiment_score(text)
        
        # Convert score to label
        if sentiment_score > 0.1:
            sentiment_label = "Positive"
        elif sentiment_score < -0.1:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
        
        return {
            "text": text,
            "sentiment_score": sentiment_score,
            "sentiment_label": sentiment_label,
            "confidence": min(1.0, abs(sentiment_score) + 0.5)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/aggregate")
async def get_aggregated_news(
    sources: str = Query("google,reddit", description="Comma-separated list of sources (google, reddit)"),
    query: str = Query("", description="Search query for filtering news"),
    max_per_source: int = Query(10, ge=1, le=25, description="Maximum articles per source")
):
    """Get aggregated news from multiple sources."""
    try:
        source_list = [s.strip().lower() for s in sources.split(",")]
        all_articles = []
        source_results = {}
        
        for source in source_list:
            try:
                if source == "google":
                    result = await news_service.get_google_news(query, max_per_source)
                    source_results["google"] = {
                        "articles_count": len(result.articles),
                        "source_type": result.source_type
                    }
                    all_articles.extend(result.articles)
                
                elif source == "reddit":
                    # Use news subreddit for general news
                    subreddit = "news" if not query else "worldnews"
                    result = await news_service.get_reddit_posts(subreddit, max_per_source)
                    source_results["reddit"] = {
                        "articles_count": len(result.articles),
                        "source_type": result.source_type
                    }
                    all_articles.extend(result.articles)
                    
            except Exception as e:
                source_results[source] = {"error": str(e)}
        
        # Sort articles by publication date (newest first)
        all_articles.sort(key=lambda x: x.published_at, reverse=True)
        
        return {
            "aggregated_articles": [article.dict() for article in all_articles],
            "total_articles": len(all_articles),
            "sources_used": list(source_results.keys()),
            "source_results": source_results,
            "query_used": query or "general news"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/headlines")
async def get_top_headlines(
    category: str = Query("general", description="News category"),
    count: int = Query(10, ge=1, le=20, description="Number of headlines to fetch")
):
    """Get top headlines from various sources."""
    try:
        # Map categories to appropriate search queries
        category_queries = {
            "general": "",
            "business": "business finance economy",
            "technology": "technology tech startup",
            "crypto": "cryptocurrency bitcoin ethereum",
            "stocks": "stock market trading",
            "politics": "politics government election"
        }
        
        query = category_queries.get(category, category)
        news_response = await news_service.get_google_news(query, count)
        
        headlines = []
        for article in news_response.articles:
            headlines.append({
                "title": article.title,
                "url": article.url,
                "published_at": article.published_at,
                "source": article.source
            })
        
        return {
            "category": category,
            "headlines": headlines,
            "total_headlines": len(headlines)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))