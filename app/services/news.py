"""News aggregation service using Google News RSS and Reddit scraping."""

import feedparser
import requests
from bs4 import BeautifulSoup
import httpx
from datetime import datetime, timedelta
from typing import List, Optional
import re
from urllib.parse import quote_plus

from app.models import NewsArticle, NewsResponse

class NewsService:
    """Service for aggregating news from various sources."""
    
    def __init__(self):
        self.google_news_base_url = "https://news.google.com/rss"
        self.reddit_base_url = "https://www.reddit.com"
        self.user_agent = "WebScraperApp/1.0"
    
    async def get_google_news(self, query: str = "", max_articles: int = 20) -> NewsResponse:
        """Fetch news from Google News RSS feed."""
        try:
            if query:
                url = f"{self.google_news_base_url}/search?q={quote_plus(query)}&hl=en&gl=US&ceid=US:en"
            else:
                url = f"{self.google_news_base_url}?hl=en&gl=US&ceid=US:en"
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers={'User-Agent': self.user_agent})
                response.raise_for_status()
                
            feed = feedparser.parse(response.content)
            
            articles = []
            for entry in feed.entries[:max_articles]:
                # Parse the publication date
                published_at = datetime.now()
                if hasattr(entry, 'published'):
                    try:
                        from dateutil import parser
                        published_at = parser.parse(entry.published)
                    except:
                        pass
                
                # Extract source from the link or title
                source = "Google News"
                if hasattr(entry, 'source'):
                    source = entry.source.get('title', 'Google News')
                
                article = NewsArticle(
                    title=entry.title,
                    summary=entry.summary if hasattr(entry, 'summary') else None,
                    url=entry.link,
                    published_at=published_at,
                    source=source
                )
                articles.append(article)
            
            return NewsResponse(
                articles=articles,
                total_count=len(articles),
                source_type="Google News RSS"
            )
        except Exception as e:
            raise ValueError(f"Error fetching Google News: {str(e)}")
    
    async def get_reddit_posts(self, subreddit: str = "news", max_posts: int = 20) -> NewsResponse:
        """Fetch posts from Reddit."""
        try:
            url = f"{self.reddit_base_url}/r/{subreddit}.json"
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={'User-Agent': self.user_agent},
                    params={'limit': max_posts}
                )
                response.raise_for_status()
                data = response.json()
            
            articles = []
            if 'data' in data and 'children' in data['data']:
                for post in data['data']['children']:
                    post_data = post['data']
                    
                    # Convert Reddit timestamp to datetime
                    published_at = datetime.fromtimestamp(post_data.get('created_utc', 0))
                    
                    # Get post URL - prefer the external URL if available
                    url = post_data.get('url', f"https://reddit.com{post_data.get('permalink', '')}")
                    
                    article = NewsArticle(
                        title=post_data.get('title', ''),
                        summary=post_data.get('selftext', '')[:200] + '...' if post_data.get('selftext') else None,
                        url=url,
                        published_at=published_at,
                        source=f"Reddit r/{subreddit}"
                    )
                    articles.append(article)
            
            return NewsResponse(
                articles=articles,
                total_count=len(articles),
                source_type=f"Reddit r/{subreddit}"
            )
        except Exception as e:
            raise ValueError(f"Error fetching Reddit posts from r/{subreddit}: {str(e)}")
    
    async def get_financial_news(self, symbol: str = "", max_articles: int = 15) -> NewsResponse:
        """Get financial news, optionally filtered by symbol."""
        try:
            queries = []
            if symbol:
                queries = [f"{symbol} stock", f"{symbol} financial news", f"{symbol} earnings"]
            else:
                queries = ["financial markets", "stock market", "cryptocurrency"]
            
            all_articles = []
            for query in queries:
                try:
                    news_response = await self.get_google_news(query, max_articles // len(queries))
                    all_articles.extend(news_response.articles)
                except Exception:
                    continue  # Skip failed queries
            
            # Remove duplicates based on title similarity
            unique_articles = []
            seen_titles = set()
            
            for article in all_articles:
                # Simple deduplication by title
                title_words = set(article.title.lower().split())
                is_duplicate = False
                
                for seen_title in seen_titles:
                    seen_words = set(seen_title.split())
                    if len(title_words.intersection(seen_words)) / len(title_words.union(seen_words)) > 0.7:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_articles.append(article)
                    seen_titles.add(article.title.lower())
                
                if len(unique_articles) >= max_articles:
                    break
            
            return NewsResponse(
                articles=unique_articles[:max_articles],
                total_count=len(unique_articles),
                source_type="Financial News Aggregation"
            )
        except Exception as e:
            raise ValueError(f"Error fetching financial news: {str(e)}")
    
    async def get_trending_topics(self) -> List[str]:
        """Get trending topics from various sources."""
        try:
            trending_topics = []
            
            # Get trending from Reddit
            try:
                popular_response = await self.get_reddit_posts("popular", 10)
                for article in popular_response.articles:
                    # Extract potential trending topics from titles
                    words = re.findall(r'\b[A-Z][A-Za-z]+\b', article.title)
                    trending_topics.extend(words[:2])  # Take first 2 capitalized words
            except Exception:
                pass
            
            # Add some common financial trending topics
            default_topics = ["Bitcoin", "Tesla", "Apple", "AI", "Federal Reserve", "Inflation"]
            trending_topics.extend(default_topics)
            
            # Remove duplicates and return top topics
            unique_topics = list(dict.fromkeys(trending_topics))
            return unique_topics[:10]
        except Exception as e:
            return ["Bitcoin", "Tesla", "Apple", "AI", "Federal Reserve"]  # Fallback topics
    
    def calculate_sentiment_score(self, text: str) -> float:
        """Calculate a simple sentiment score for text (placeholder implementation)."""
        # This is a very basic sentiment analysis
        # In production, you might want to use libraries like TextBlob or VADER
        positive_words = ['good', 'great', 'excellent', 'positive', 'up', 'gain', 'profit', 'bull', 'rise']
        negative_words = ['bad', 'terrible', 'negative', 'down', 'loss', 'bear', 'fall', 'crash', 'decline']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        return (positive_count - negative_count) / total_words