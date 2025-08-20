"""
Data Collection Module for AI Trading Assistant
Handles market data fetching and news scraping
"""

import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketDataCollector:
    """Collects market data for stocks and cryptocurrencies"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_stock_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch stock data using yfinance
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Fetching stock data for {symbol}")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Ensure column names are consistent
            data.columns = [col.title() for col in data.columns]
            data.index.name = 'Date'
            
            logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_crypto_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch cryptocurrency data using Binance API
        
        Args:
            symbol: Crypto symbol (e.g., 'BTCUSDT')
            period: Time period in days
            interval: Data interval ('1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M')
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Fetching crypto data for {symbol}")
            
            # Binance API endpoint
            base_url = "https://api.binance.com/api/v3/klines"
            
            # Calculate start time
            end_time = int(time.time() * 1000)
            start_time = end_time - (int(period) * 24 * 60 * 60 * 1000)
            
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': start_time,
                'endTime': end_time,
                'limit': 1000
            }
            
            response = self.session.get(base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Close Time', 'Quote Asset Volume', 'Number of Trades',
                'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
            ])
            
            # Convert to proper types
            df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
            df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')
            
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Set index and rename columns
            df.set_index('Open Time', inplace=True)
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            logger.info(f"Successfully fetched {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching crypto data for {symbol}: {str(e)}")
            return pd.DataFrame()

class NewsCollector:
    """Collects financial news from various sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.news_sources = {
            'yahoo_finance': 'https://finance.yahoo.com/news/',
            'google_news': 'https://news.google.com/search?q=finance&hl=en-US&gl=US&ceid=US:en',
            'marketwatch': 'https://www.marketwatch.com/latest-news'
        }
    
    def scrape_yahoo_finance(self, query: str = "stock market") -> List[Dict]:
        """
        Scrape news from Yahoo Finance
        
        Args:
            query: Search query for news
            
        Returns:
            List of news articles with title, summary, and timestamp
        """
        try:
            logger.info(f"Scraping Yahoo Finance news for query: {query}")
            
            # Construct search URL
            search_url = f"https://finance.yahoo.com/news/search?q={query.replace(' ', '+')}"
            
            response = self.session.get(search_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = []
            
            # Find news articles (this is a simplified approach)
            news_items = soup.find_all('h3', class_='Mb(5px)')[:10]  # Limit to 10 articles
            
            for item in news_items:
                link = item.find('a')
                if link:
                    title = link.get_text(strip=True)
                    url = "https://finance.yahoo.com" + link.get('href', '')
                    
                    articles.append({
                        'title': title,
                        'url': url,
                        'source': 'Yahoo Finance',
                        'timestamp': datetime.now().isoformat()
                    })
            
            logger.info(f"Scraped {len(articles)} articles from Yahoo Finance")
            return articles
            
        except Exception as e:
            logger.error(f"Error scraping Yahoo Finance: {str(e)}")
            return []
    
    def scrape_google_news(self, query: str = "finance") -> List[Dict]:
        """
        Scrape news from Google News
        
        Args:
            query: Search query for news
            
        Returns:
            List of news articles with title, summary, and timestamp
        """
        try:
            logger.info(f"Scraping Google News for query: {query}")
            
            search_url = f"https://news.google.com/search?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
            
            response = self.session.get(search_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = []
            
            # Find news articles
            news_items = soup.find_all('article')[:10]  # Limit to 10 articles
            
            for item in news_items:
                title_elem = item.find('h3')
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    link = title_elem.find('a')
                    url = "https://news.google.com" + link.get('href', '') if link else ''
                    
                    articles.append({
                        'title': title,
                        'url': url,
                        'source': 'Google News',
                        'timestamp': datetime.now().isoformat()
                    })
            
            logger.info(f"Scraped {len(articles)} articles from Google News")
            return articles
            
        except Exception as e:
            logger.error(f"Error scraping Google News: {str(e)}")
            return []
    
    def get_news_for_symbol(self, symbol: str, limit: int = 20) -> List[Dict]:
        """
        Get news articles related to a specific stock/crypto symbol
        
        Args:
            symbol: Stock or crypto symbol
            limit: Maximum number of articles to return
            
        Returns:
            List of news articles
        """
        try:
            logger.info(f"Fetching news for symbol: {symbol}")
            
            all_articles = []
            
            # Get news from multiple sources
            yahoo_articles = self.scrape_yahoo_finance(symbol)
            google_articles = self.scrape_google_news(symbol)
            
            all_articles.extend(yahoo_articles)
            all_articles.extend(google_articles)
            
            # Remove duplicates and limit results
            seen_titles = set()
            unique_articles = []
            
            for article in all_articles:
                if article['title'] not in seen_titles and len(unique_articles) < limit:
                    seen_titles.add(article['title'])
                    unique_articles.append(article)
            
            logger.info(f"Retrieved {len(unique_articles)} unique news articles for {symbol}")
            return unique_articles
            
        except Exception as e:
            logger.error(f"Error getting news for symbol {symbol}: {str(e)}")
            return []
    
    def get_market_sentiment_news(self, limit: int = 30) -> List[Dict]:
        """
        Get general market sentiment news
        
        Args:
            limit: Maximum number of articles to return
            
        Returns:
            List of news articles
        """
        try:
            logger.info("Fetching general market sentiment news")
            
            all_articles = []
            
            # Get general finance news
            yahoo_articles = self.scrape_yahoo_finance("stock market")
            google_articles = self.scrape_google_news("finance market")
            
            all_articles.extend(yahoo_articles)
            all_articles.extend(google_articles)
            
            # Remove duplicates and limit results
            seen_titles = set()
            unique_articles = []
            
            for article in all_articles:
                if article['title'] not in seen_titles and len(unique_articles) < limit:
                    seen_titles.add(article['title'])
                    unique_articles.append(article)
            
            logger.info(f"Retrieved {len(unique_articles)} unique market sentiment articles")
            return unique_articles
            
        except Exception as e:
            logger.error(f"Error getting market sentiment news: {str(e)}")
            return []

# Example usage and testing
if __name__ == "__main__":
    # Test market data collection
    market_collector = MarketDataCollector()
    
    # Test stock data
    aapl_data = market_collector.get_stock_data('AAPL', period='6mo')
    print(f"AAPL data shape: {aapl_data.shape}")
    if not aapl_data.empty:
        print(aapl_data.head())
    
    # Test crypto data
    btc_data = market_collector.get_crypto_data('BTCUSDT', period='30')
    print(f"BTC data shape: {btc_data.shape}")
    if not btc_data.empty:
        print(btc_data.head())
    
    # Test news collection
    news_collector = NewsCollector()
    
    # Test symbol-specific news
    aapl_news = news_collector.get_news_for_symbol('AAPL', limit=5)
    print(f"AAPL news count: {len(aapl_news)}")
    for article in aapl_news[:2]:
        print(f"- {article['title']} ({article['source']})")
    
    # Test general market news
    market_news = news_collector.get_market_sentiment_news(limit=5)
    print(f"Market news count: {len(market_news)}")
    for article in market_news[:2]:
        print(f"- {article['title']} ({article['source']})")
