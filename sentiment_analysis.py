"""
Sentiment Analysis Module for AI Trading Assistant
Uses FinBERT and alternative models for financial text sentiment analysis
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from textblob import TextBlob
import nltk
import logging
from typing import Dict, List, Optional, Tuple, Union
import re
import json
from datetime import datetime

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialSentimentAnalyzer:
    """Analyzes sentiment of financial news and text"""
    
    def __init__(self, model_name: str = "ProsusAI/finbert", use_gpu: bool = False):
        """
        Initialize the sentiment analyzer
        
        Args:
            model_name: Name of the pre-trained model to use
            use_gpu: Whether to use GPU if available
        """
        self.model_name = model_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.device = "cuda" if self.use_gpu else "cpu"
        
        # Initialize the model
        self._load_model()
        
        # Sentiment labels mapping
        self.label_mapping = {
            0: "negative",
            1: "neutral", 
            2: "positive"
        }
        
        # Financial-specific keywords for enhanced analysis
        self.bullish_keywords = [
            'bullish', 'rally', 'surge', 'jump', 'climb', 'gain', 'rise', 'up',
            'strong', 'positive', 'growth', 'earnings', 'profit', 'beat', 'exceed',
            'upgrade', 'buy', 'outperform', 'bull market', 'recovery'
        ]
        
        self.bearish_keywords = [
            'bearish', 'decline', 'fall', 'drop', 'plunge', 'crash', 'sell-off',
            'weak', 'negative', 'loss', 'miss', 'disappoint', 'downgrade', 'sell',
            'underperform', 'bear market', 'recession', 'correction'
        ]
        
        # Market-specific terms
        self.market_terms = [
            'earnings', 'revenue', 'profit', 'loss', 'guidance', 'analyst',
            'upgrade', 'downgrade', 'target price', 'price target', 'valuation',
            'fundamentals', 'technical analysis', 'support', 'resistance'
        ]
    
    def _load_model(self):
        """Load the pre-trained sentiment analysis model"""
        try:
            logger.info(f"Loading sentiment model: {self.model_name}")
            
            if self.model_name == "ProsusAI/finbert":
                # Load FinBERT specifically
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                
                if self.use_gpu:
                    self.model = self.model.to(self.device)
                
                # Create pipeline
                self.pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if self.use_gpu else -1
                )
                
            else:
                # Load generic sentiment model
                self.pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.model_name,
                    device=0 if self.use_gpu else -1
                )
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.info("Falling back to TextBlob for sentiment analysis")
            self.pipeline = None
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            if not text or len(text.strip()) == 0:
                return {
                    'sentiment': 'neutral',
                    'confidence': 0.0,
                    'score': 0.0,
                    'keywords_found': [],
                    'market_terms': []
                }
            
            # Clean text
            cleaned_text = self._clean_text(text)
            
            # Analyze sentiment
            if self.pipeline:
                result = self._analyze_with_model(cleaned_text)
            else:
                result = self._analyze_with_textblob(cleaned_text)
            
            # Extract keywords and market terms
            keywords_found = self._extract_keywords(cleaned_text)
            market_terms = self._extract_market_terms(cleaned_text)
            
            # Combine results
            result.update({
                'keywords_found': keywords_found,
                'market_terms': market_terms,
                'text_length': len(text),
                'cleaned_text': cleaned_text
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing text: {str(e)}")
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'score': 0.0,
                'keywords_found': [],
                'market_terms': [],
                'error': str(e)
            }
    
    def _analyze_with_model(self, text: str) -> Dict:
        """Analyze sentiment using the loaded model"""
        try:
            # Truncate text if too long for model
            max_length = 512 if self.model_name == "ProsusAI/finbert" else 1024
            if len(text) > max_length:
                text = text[:max_length]
            
            # Get prediction
            result = self.pipeline(text)[0]
            
            if self.model_name == "ProsusAI/finbert":
                # FinBERT specific processing
                label = result['label']
                score = result['score']
                
                # Map FinBERT labels to standard sentiment
                if label == 'LABEL_0':
                    sentiment = 'negative'
                elif label == 'LABEL_1':
                    sentiment = 'neutral'
                else:
                    sentiment = 'positive'
                
                return {
                    'sentiment': sentiment,
                    'confidence': score,
                    'score': self._sentiment_to_score(sentiment, score)
                }
            else:
                # Generic model processing
                label = result['label'].lower()
                score = result['score']
                
                # Normalize sentiment labels
                if 'neg' in label or 'negative' in label:
                    sentiment = 'negative'
                elif 'pos' in label or 'positive' in label:
                    sentiment = 'positive'
                else:
                    sentiment = 'neutral'
                
                return {
                    'sentiment': sentiment,
                    'confidence': score,
                    'score': self._sentiment_to_score(sentiment, score)
                }
                
        except Exception as e:
            logger.error(f"Error with model analysis: {str(e)}")
            return self._analyze_with_textblob(text)
    
    def _analyze_with_textblob(self, text: str) -> Dict:
        """Fallback sentiment analysis using TextBlob"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Convert polarity to sentiment
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            # Calculate confidence based on subjectivity
            confidence = abs(polarity) + (1 - subjectivity) / 2
            
            return {
                'sentiment': sentiment,
                'confidence': min(confidence, 1.0),
                'score': polarity
            }
            
        except Exception as e:
            logger.error(f"Error with TextBlob analysis: {str(e)}")
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'score': 0.0
            }
    
    def _sentiment_to_score(self, sentiment: str, confidence: float) -> float:
        """Convert sentiment and confidence to a numerical score"""
        base_scores = {
            'positive': 1.0,
            'neutral': 0.0,
            'negative': -1.0
        }
        
        base_score = base_scores.get(sentiment, 0.0)
        return base_score * confidence
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        try:
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', '', text)
            
            # Remove URLs
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            
            # Remove special characters but keep important punctuation
            text = re.sub(r'[^\w\s\.\,\!\?\-\$\%]', '', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Convert to lowercase
            text = text.lower().strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return text
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract bullish/bearish keywords from text"""
        try:
            words = text.lower().split()
            found_keywords = []
            
            for word in words:
                if word in self.bullish_keywords:
                    found_keywords.append(f"bullish:{word}")
                elif word in self.bearish_keywords:
                    found_keywords.append(f"bearish:{word}")
            
            return found_keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []
    
    def _extract_market_terms(self, text: str) -> List[str]:
        """Extract market-specific terms from text"""
        try:
            words = text.lower().split()
            found_terms = []
            
            for word in words:
                if word in self.market_terms:
                    found_terms.append(word)
            
            return found_terms
            
        except Exception as e:
            logger.error(f"Error extracting market terms: {str(e)}")
            return []
    
    def analyze_news_batch(self, news_articles: List[Dict]) -> List[Dict]:
        """
        Analyze sentiment of multiple news articles
        
        Args:
            news_articles: List of news article dictionaries
            
        Returns:
            List of articles with sentiment analysis added
        """
        try:
            logger.info(f"Analyzing sentiment for {len(news_articles)} news articles")
            
            analyzed_articles = []
            
            for article in news_articles:
                # Analyze title sentiment
                title_sentiment = self.analyze_text(article.get('title', ''))
                
                # Create enhanced article with sentiment
                enhanced_article = article.copy()
                enhanced_article['sentiment_analysis'] = {
                    'title_sentiment': title_sentiment['sentiment'],
                    'title_confidence': title_sentiment['confidence'],
                    'title_score': title_sentiment['score'],
                    'keywords_found': title_sentiment['keywords_found'],
                    'market_terms': title_sentiment['market_terms']
                }
                
                analyzed_articles.append(enhanced_article)
            
            logger.info(f"Completed sentiment analysis for {len(analyzed_articles)} articles")
            return analyzed_articles
            
        except Exception as e:
            logger.error(f"Error analyzing news batch: {str(e)}")
            return news_articles
    
    def get_market_sentiment_summary(self, news_articles: List[Dict]) -> Dict:
        """
        Get overall market sentiment summary from news articles
        
        Args:
            news_articles: List of analyzed news articles
            
        Returns:
            Dictionary with market sentiment summary
        """
        try:
            if not news_articles:
                return {
                    'overall_sentiment': 'neutral',
                    'sentiment_score': 0.0,
                    'confidence': 0.0,
                    'article_count': 0,
                    'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0}
                }
            
            # Calculate sentiment distribution
            sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
            total_score = 0.0
            total_confidence = 0.0
            
            for article in news_articles:
                if 'sentiment_analysis' in article:
                    sentiment = article['sentiment_analysis']['title_sentiment']
                    confidence = article['sentiment_analysis']['title_confidence']
                    score = article['sentiment_analysis']['title_score']
                    
                    sentiment_counts[sentiment] += 1
                    total_score += score
                    total_confidence += confidence
            
            # Calculate overall metrics
            article_count = len(news_articles)
            avg_score = total_score / article_count if article_count > 0 else 0.0
            avg_confidence = total_confidence / article_count if article_count > 0 else 0.0
            
            # Determine overall sentiment
            if avg_score > 0.1:
                overall_sentiment = 'positive'
            elif avg_score < -0.1:
                overall_sentiment = 'negative'
            else:
                overall_sentiment = 'neutral'
            
            return {
                'overall_sentiment': overall_sentiment,
                'sentiment_score': avg_score,
                'confidence': avg_confidence,
                'article_count': article_count,
                'sentiment_distribution': sentiment_counts,
                'positive_ratio': sentiment_counts['positive'] / article_count if article_count > 0 else 0.0,
                'negative_ratio': sentiment_counts['negative'] / article_count if article_count > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error calculating market sentiment summary: {str(e)}")
            return {
                'overall_sentiment': 'neutral',
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'article_count': 0,
                'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0}
            }

# Example usage and testing
if __name__ == "__main__":
    # Test sentiment analysis
    analyzer = FinancialSentimentAnalyzer()
    
    # Test individual text analysis
    test_texts = [
        "Apple stock surges 5% on strong earnings report",
        "Market crashes as investors panic over economic data",
        "Analysts upgrade Tesla target price on positive outlook",
        "Company reports disappointing quarterly results"
    ]
    
    print("Testing individual text sentiment analysis:")
    for text in test_texts:
        result = analyzer.analyze_text(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Score: {result['score']:.3f}")
        print(f"Keywords: {result['keywords_found']}")
        print(f"Market Terms: {result['market_terms']}")
    
    # Test news batch analysis
    test_news = [
        {'title': 'Stock market rallies on positive economic data', 'source': 'Test'},
        {'title': 'Tech stocks fall amid earnings concerns', 'source': 'Test'},
        {'title': 'Federal Reserve signals potential rate cuts', 'source': 'Test'}
    ]
    
    print("\n\nTesting news batch analysis:")
    analyzed_news = analyzer.analyze_news_batch(test_news)
    
    for article in analyzed_news:
        print(f"\nTitle: {article['title']}")
        print(f"Sentiment: {article['sentiment_analysis']['title_sentiment']}")
        print(f"Confidence: {article['sentiment_analysis']['title_confidence']:.3f}")
    
    # Test market sentiment summary
    summary = analyzer.get_market_sentiment_summary(analyzed_news)
    print(f"\n\nMarket Sentiment Summary:")
    print(f"Overall Sentiment: {summary['overall_sentiment']}")
    print(f"Sentiment Score: {summary['sentiment_score']:.3f}")
    print(f"Confidence: {summary['confidence']:.3f}")
    print(f"Sentiment Distribution: {summary['sentiment_distribution']}")
