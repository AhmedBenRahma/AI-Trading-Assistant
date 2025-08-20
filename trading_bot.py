"""
Trading Bot Module for AI Trading Assistant
Integrates all modules to generate trading signals and detailed explanations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Union
import json
import warnings

# Import our custom modules
from data_collector import MarketDataCollector, NewsCollector
from feature_engineering import TechnicalIndicators, DataPreprocessor, FeatureSelector
from sentiment_analysis import FinancialSentimentAnalyzer
from prediction_model import LSTMPredictor

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AITradingBot:
    """Main AI trading bot that integrates all components"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the AI trading bot
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.market_collector = MarketDataCollector()
        self.news_collector = NewsCollector()
        self.technical_indicators = TechnicalIndicators()
        self.data_preprocessor = DataPreprocessor()
        self.feature_selector = FeatureSelector()
        self.sentiment_analyzer = FinancialSentimentAnalyzer()
        self.lstm_predictor = LSTMPredictor(
            lookback_days=self.config['lookback_days'],
            prediction_days=self.config['prediction_days']
        )
        
        # State variables
        self.is_trained = False
        self.current_data = {}
        self.current_news = {}
        self.last_signals = {}
        
        logger.info("AI Trading Bot initialized successfully")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'lookback_days': 30,
            'prediction_days': 1,
            'technical_confidence_weight': 0.6,
            'sentiment_confidence_weight': 0.4,
            'min_confidence_threshold': 0.6,
            'update_frequency_hours': 4,
            'max_news_articles': 20,
            'feature_selection_method': 'correlation',
            'n_selected_features': 25
        }
    
    def collect_market_data(self, symbol: str, asset_type: str = 'stock', 
                           period: str = '1y') -> pd.DataFrame:
        """
        Collect market data for a given symbol
        
        Args:
            symbol: Stock or crypto symbol
            asset_type: 'stock' or 'crypto'
            period: Time period for data collection
            
        Returns:
            DataFrame with market data
        """
        try:
            logger.info(f"Collecting market data for {symbol} ({asset_type})")
            
            if asset_type.lower() == 'stock':
                data = self.market_collector.get_stock_data(symbol, period=period)
            elif asset_type.lower() == 'crypto':
                data = self.market_collector.get_crypto_data(symbol, period=period)
            else:
                raise ValueError(f"Unsupported asset type: {asset_type}")
            
            if data.empty:
                logger.warning(f"No market data collected for {symbol}")
                return pd.DataFrame()
            
            # Store current data
            self.current_data[symbol] = data
            
            logger.info(f"Successfully collected {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error collecting market data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def collect_news_data(self, symbol: str, limit: int = 20) -> List[Dict]:
        """
        Collect news data for a given symbol
        
        Args:
            symbol: Stock or crypto symbol
            limit: Maximum number of news articles
            
        Returns:
            List of news articles
        """
        try:
            logger.info(f"Collecting news data for {symbol}")
            
            # Get symbol-specific news
            symbol_news = self.news_collector.get_news_for_symbol(symbol, limit=limit//2)
            
            # Get general market news
            market_news = self.news_collector.get_market_sentiment_news(limit=limit//2)
            
            # Combine and analyze sentiment
            all_news = symbol_news + market_news
            
            # Analyze sentiment for all news
            analyzed_news = self.sentiment_analyzer.analyze_news_batch(all_news)
            
            # Store current news
            self.current_news[symbol] = analyzed_news
            
            logger.info(f"Successfully collected and analyzed {len(analyzed_news)} news articles for {symbol}")
            return analyzed_news
            
        except Exception as e:
            logger.error(f"Error collecting news data for {symbol}: {str(e)}")
            return []
    
    def prepare_features_and_train(self, symbol: str) -> bool:
        """
        Prepare features and train the LSTM model
        
        Args:
            symbol: Symbol to train model for
            
        Returns:
            True if training successful, False otherwise
        """
        try:
            logger.info(f"Preparing features and training model for {symbol}")
            
            if symbol not in self.current_data:
                logger.error(f"No market data available for {symbol}")
                return False
            
            data = self.current_data[symbol]
            
            # Calculate technical indicators
            data_with_indicators = self.technical_indicators.calculate_all_indicators(data)
            
            if data_with_indicators.empty:
                logger.error(f"Failed to calculate technical indicators for {symbol}")
                return False
            
            # Select most important features
            selected_features = self.feature_selector.select_features(
                data_with_indicators,
                target_col='Close',
                n_features=self.config['n_selected_features']
            )
            
            if not selected_features:
                logger.error(f"No features selected for {symbol}")
                return False
            
            # Prepare data for LSTM
            features, targets = self.lstm_predictor.prepare_data(data_with_indicators, target_col='Close')
            
            if len(features) == 0:
                logger.error(f"No features prepared for {symbol}")
                return False
            
            # Scale features
            scaled_features = self.data_preprocessor.scale_features(features, fit=True)
            
            # Split data for training
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                scaled_features, targets, test_size=0.2, random_state=42
            )
            
            # Train LSTM model
            logger.info(f"Training LSTM model for {symbol}")
            history = self.lstm_predictor.train(X_train, y_train)
            
            # Evaluate model
            test_metrics = self.lstm_predictor.evaluate(X_test, y_test)
            
            logger.info(f"Model training completed for {symbol}")
            logger.info(f"Test R² Score: {test_metrics['r2']:.4f}")
            
            self.is_trained = True
            return True
            
        except Exception as e:
            logger.error(f"Error preparing features and training for {symbol}: {str(e)}")
            return False
    
    def generate_trading_signal(self, symbol: str) -> Dict:
        """
        Generate trading signal for a given symbol
        
        Args:
            symbol: Symbol to generate signal for
            
        Returns:
            Dictionary with trading signal and explanation
        """
        try:
            logger.info(f"Generating trading signal for {symbol}")
            
            if not self.is_trained:
                logger.warning(f"Model not trained for {symbol}, attempting to train now")
                if not self.prepare_features_and_train(symbol):
                    return self._generate_fallback_signal(symbol)
            
            # Get current market data
            if symbol not in self.current_data:
                self.collect_market_data(symbol)
            
            if symbol not in self.current_news:
                self.collect_news_data(symbol)
            
            # Generate technical signal
            technical_signal = self._generate_technical_signal(symbol)
            
            # Generate sentiment signal
            sentiment_signal = self._generate_sentiment_signal(symbol)
            
            # Combine signals
            combined_signal = self._combine_signals(technical_signal, sentiment_signal)
            
            # Generate detailed explanation
            explanation = self._generate_signal_explanation(symbol, technical_signal, sentiment_signal, combined_signal)
            
            # Create final signal
            final_signal = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'signal': combined_signal['action'],
                'confidence': combined_signal['confidence'],
                'technical_analysis': technical_signal,
                'sentiment_analysis': sentiment_signal,
                'explanation': explanation,
                'recommended_action': combined_signal['action'],
                'risk_level': self._assess_risk_level(combined_signal['confidence'])
            }
            
            # Store last signal
            self.last_signals[symbol] = final_signal
            
            logger.info(f"Generated {combined_signal['action']} signal for {symbol} with confidence {combined_signal['confidence']:.3f}")
            return final_signal
            
        except Exception as e:
            logger.error(f"Error generating trading signal for {symbol}: {str(e)}")
            return self._generate_fallback_signal(symbol)
    
    def _generate_technical_signal(self, symbol: str) -> Dict:
        """Generate signal based on technical analysis"""
        try:
            data = self.current_data[symbol]
            
            if data.empty:
                return {'action': 'hold', 'confidence': 0.0, 'indicators': {}}
            
            # Get latest data
            latest = data.iloc[-1]
            prev = data.iloc[-2] if len(data) > 1 else latest
            
            indicators = {}
            signals = []
            
            # Moving Average signals
            if 'SMA_50' in data.columns and 'SMA_200' in data.columns:
                if latest['SMA_50'] > latest['SMA_200']:
                    signals.append(('golden_cross', 0.8))
                else:
                    signals.append(('death_cross', -0.8))
            
            # RSI signals
            if 'RSI' in data.columns:
                rsi = latest['RSI']
                if rsi < 30:
                    signals.append(('oversold_rsi', 0.7))
                elif rsi > 70:
                    signals.append(('overbought_rsi', -0.7))
                else:
                    signals.append(('neutral_rsi', 0.0))
            
            # MACD signals
            if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                if latest['MACD'] > latest['MACD_Signal']:
                    signals.append(('macd_bullish', 0.6))
                else:
                    signals.append(('macd_bearish', -0.6))
            
            # Bollinger Bands signals
            if 'BB_Position' in data.columns:
                bb_pos = latest['BB_Position']
                if bb_pos < 0.2:
                    signals.append(('bb_oversold', 0.6))
                elif bb_pos > 0.8:
                    signals.append(('bb_overbought', -0.6))
            
            # Price momentum
            price_change = (latest['Close'] - prev['Close']) / prev['Close']
            if abs(price_change) > 0.02:  # 2% change
                if price_change > 0:
                    signals.append(('price_momentum_up', 0.5))
                else:
                    signals.append(('price_momentum_down', -0.5))
            
            # Calculate overall technical signal
            if signals:
                total_score = sum(score for _, score in signals)
                avg_score = total_score / len(signals)
                
                # Convert to action and confidence
                if avg_score > 0.3:
                    action = 'buy'
                    confidence = min(abs(avg_score), 0.9)
                elif avg_score < -0.3:
                    action = 'sell'
                    confidence = min(abs(avg_score), 0.9)
                else:
                    action = 'hold'
                    confidence = 0.5
            else:
                action = 'hold'
                confidence = 0.5
            
            return {
                'action': action,
                'confidence': confidence,
                'indicators': {name: score for name, score in signals},
                'price_change': price_change,
                'latest_price': latest['Close']
            }
            
        except Exception as e:
            logger.error(f"Error generating technical signal: {str(e)}")
            return {'action': 'hold', 'confidence': 0.0, 'indicators': {}}
    
    def _generate_sentiment_signal(self, symbol: str) -> Dict:
        """Generate signal based on sentiment analysis"""
        try:
            if symbol not in self.current_news:
                return {'action': 'hold', 'confidence': 0.0, 'sentiment': 'neutral'}
            
            news_articles = self.current_news[symbol]
            
            if not news_articles:
                return {'action': 'hold', 'confidence': 0.0, 'sentiment': 'neutral'}
            
            # Get market sentiment summary
            sentiment_summary = self.sentiment_analyzer.get_market_sentiment_summary(news_articles)
            
            # Convert sentiment to trading signal
            sentiment_score = sentiment_summary['sentiment_score']
            confidence = sentiment_summary['confidence']
            
            if sentiment_score > 0.2:
                action = 'buy'
            elif sentiment_score < -0.2:
                action = 'sell'
            else:
                action = 'hold'
            
            return {
                'action': action,
                'confidence': confidence,
                'sentiment': sentiment_summary['overall_sentiment'],
                'sentiment_score': sentiment_score,
                'article_count': sentiment_summary['article_count'],
                'sentiment_distribution': sentiment_summary['sentiment_distribution']
            }
            
        except Exception as e:
            logger.error(f"Error generating sentiment signal: {str(e)}")
            return {'action': 'hold', 'confidence': 0.0, 'sentiment': 'neutral'}
    
    def _combine_signals(self, technical_signal: Dict, sentiment_signal: Dict) -> Dict:
        """Combine technical and sentiment signals"""
        try:
            # Get weights from config
            tech_weight = self.config['technical_confidence_weight']
            sent_weight = self.config['sentiment_confidence_weight']
            
            # Calculate weighted confidence
            tech_conf = technical_signal['confidence'] * tech_weight
            sent_conf = sentiment_signal['confidence'] * sent_weight
            
            # Determine action based on agreement
            tech_action = technical_signal['action']
            sent_action = sentiment_signal['action']
            
            if tech_action == sent_action:
                # Signals agree
                action = tech_action
                confidence = (tech_conf + sent_conf) / 2
            else:
                # Signals disagree - use higher confidence
                if tech_conf > sent_conf:
                    action = tech_action
                    confidence = tech_conf * 0.8  # Reduce confidence due to disagreement
                else:
                    action = sent_action
                    confidence = sent_conf * 0.8
            
            # Ensure confidence is within bounds
            confidence = max(0.1, min(0.95, confidence))
            
            return {
                'action': action,
                'confidence': confidence,
                'technical_weight': tech_weight,
                'sentiment_weight': sent_weight
            }
            
        except Exception as e:
            logger.error(f"Error combining signals: {str(e)}")
            return {'action': 'hold', 'confidence': 0.5}
    
    def _generate_signal_explanation(self, symbol: str, technical_signal: Dict, 
                                   sentiment_signal: Dict, combined_signal: Dict) -> str:
        """Generate detailed explanation for the trading signal"""
        try:
            explanation_parts = []
            
            # Header
            explanation_parts.append(f"Trading Signal Analysis for {symbol}")
            explanation_parts.append("=" * 50)
            
            # Overall recommendation
            action = combined_signal['action'].upper()
            confidence = combined_signal['confidence']
            explanation_parts.append(f"\nRECOMMENDATION: {action}")
            explanation_parts.append(f"Confidence Level: {confidence:.1%}")
            
            # Technical analysis explanation
            explanation_parts.append("\nTECHNICAL ANALYSIS:")
            explanation_parts.append("-" * 20)
            
            if technical_signal['indicators']:
                for indicator, score in technical_signal['indicators'].items():
                    if abs(score) > 0.3:
                        if score > 0:
                            explanation_parts.append(f"✓ {indicator.replace('_', ' ').title()}: Bullish signal")
                        else:
                            explanation_parts.append(f"✗ {indicator.replace('_', ' ').title()}: Bearish signal")
            
            if 'price_change' in technical_signal:
                price_change = technical_signal['price_change']
                if abs(price_change) > 0.01:
                    direction = "increased" if price_change > 0 else "decreased"
                    explanation_parts.append(f"• Price has {direction} by {abs(price_change):.1%} recently")
            
            # Sentiment analysis explanation
            explanation_parts.append("\nMARKET SENTIMENT:")
            explanation_parts.append("-" * 20)
            
            if 'sentiment' in sentiment_signal:
                sentiment = sentiment_signal['sentiment']
                explanation_parts.append(f"• Overall sentiment: {sentiment.title()}")
            
            if 'sentiment_score' in sentiment_signal:
                score = sentiment_signal['sentiment_score']
                if score > 0.1:
                    explanation_parts.append("• News sentiment is generally positive")
                elif score < -0.1:
                    explanation_parts.append("• News sentiment is generally negative")
                else:
                    explanation_parts.append("• News sentiment is neutral")
            
            if 'article_count' in sentiment_signal:
                count = sentiment_signal['article_count']
                explanation_parts.append(f"• Analyzed {count} recent news articles")
            
            # Risk assessment
            explanation_parts.append("\nRISK ASSESSMENT:")
            explanation_parts.append("-" * 20)
            
            if confidence > 0.8:
                explanation_parts.append("• High confidence signal - Strong technical and sentiment alignment")
            elif confidence > 0.6:
                explanation_parts.append("• Medium confidence signal - Good technical and sentiment alignment")
            else:
                explanation_parts.append("• Low confidence signal - Mixed or weak signals")
            
            # Disclaimer
            explanation_parts.append("\nDISCLAIMER:")
            explanation_parts.append("-" * 20)
            explanation_parts.append("This analysis is for educational purposes only. Always conduct your own research and consider consulting with a financial advisor before making investment decisions.")
            
            return "\n".join(explanation_parts)
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return f"Error generating explanation: {str(e)}"
    
    def _assess_risk_level(self, confidence: float) -> str:
        """Assess risk level based on signal confidence"""
        if confidence > 0.8:
            return "Low"
        elif confidence > 0.6:
            return "Medium"
        else:
            return "High"
    
    def _generate_fallback_signal(self, symbol: str) -> Dict:
        """Generate fallback signal when model training fails"""
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'signal': 'hold',
            'confidence': 0.3,
            'technical_analysis': {'action': 'hold', 'confidence': 0.3},
            'sentiment_analysis': {'action': 'hold', 'confidence': 0.3},
            'explanation': f"Unable to generate complete analysis for {symbol}. Model training failed or insufficient data available. Recommend HOLD position until more data is available.",
            'recommended_action': 'hold',
            'risk_level': 'High',
            'error': 'Model training failed'
        }
    
    def get_portfolio_signals(self, symbols: List[str]) -> List[Dict]:
        """
        Get trading signals for multiple symbols
        
        Args:
            symbols: List of symbols to analyze
            
        Returns:
            List of trading signals
        """
        try:
            logger.info(f"Generating portfolio signals for {len(symbols)} symbols")
            
            signals = []
            
            for symbol in symbols:
                try:
                    signal = self.generate_trading_signal(symbol)
                    signals.append(signal)
                except Exception as e:
                    logger.error(f"Error generating signal for {symbol}: {str(e)}")
                    # Add fallback signal
                    signals.append(self._generate_fallback_signal(symbol))
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating portfolio signals: {str(e)}")
            return []
    
    def save_signals(self, filepath: str):
        """Save current signals to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.last_signals, f, indent=2, default=str)
            logger.info(f"Signals saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving signals: {str(e)}")
    
    def load_signals(self, filepath: str):
        """Load signals from file"""
        try:
            with open(filepath, 'r') as f:
                self.last_signals = json.load(f)
            logger.info(f"Signals loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading signals: {str(e)}")

# Example usage and testing
if __name__ == "__main__":
    # Test the trading bot
    bot = AITradingBot()
    
    # Test with a single symbol
    symbol = 'AAPL'
    
    print(f"Testing AI Trading Bot with {symbol}")
    print("=" * 50)
    
    # Collect data
    print("\n1. Collecting market data...")
    market_data = bot.collect_market_data(symbol, period='6mo')
    
    if not market_data.empty:
        print(f"✓ Collected {len(market_data)} market data points")
        
        # Collect news
        print("\n2. Collecting news data...")
        news_data = bot.collect_news_data(symbol, limit=10)
        print(f"✓ Collected {len(news_data)} news articles")
        
        # Generate trading signal
        print("\n3. Generating trading signal...")
        signal = bot.generate_trading_signal(symbol)
        
        print(f"\n✓ Generated {signal['signal'].upper()} signal")
        print(f"Confidence: {signal['confidence']:.1%}")
        print(f"Risk Level: {signal['risk_level']}")
        
        # Print explanation
        print("\n4. Signal Explanation:")
        print(signal['explanation'])
        
    else:
        print("✗ Failed to collect market data")
    
    # Test with multiple symbols
    print("\n\n" + "=" * 50)
    print("Testing portfolio analysis...")
    
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    portfolio_signals = bot.get_portfolio_signals(symbols)
    
    print(f"\nGenerated signals for {len(portfolio_signals)} symbols:")
    for signal in portfolio_signals:
        print(f"\n{signal['symbol']}: {signal['signal'].upper()} (Confidence: {signal['confidence']:.1%})")
    
    # Save signals
    bot.save_signals('trading_signals.json')
    print("\n✓ Signals saved to trading_signals.json")
