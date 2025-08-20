"""
Example Usage Script for AI Trading Assistant
Demonstrates all key features and capabilities
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from trading_bot import AITradingBot
from data_collector import MarketDataCollector, NewsCollector
from feature_engineering import TechnicalIndicators, DataPreprocessor, FeatureSelector
from sentiment_analysis import FinancialSentimentAnalyzer
from prediction_model import LSTMPredictor

def main():
    """Main example function demonstrating all features"""
    
    print("🤖 AI Trading Assistant - Example Usage")
    print("=" * 50)
    
    # Example 1: Basic Trading Bot Usage
    print("\n📊 Example 1: Basic Trading Bot Usage")
    print("-" * 30)
    basic_trading_example()
    
    # Example 2: Technical Analysis
    print("\n🔧 Example 2: Technical Analysis")
    print("-" * 30)
    technical_analysis_example()
    
    # Example 3: Sentiment Analysis
    print("\n📰 Example 3: Sentiment Analysis")
    print("-" * 30)
    sentiment_analysis_example()
    
    # Example 4: LSTM Model Training
    print("\n🧠 Example 4: LSTM Model Training")
    print("-" * 30)
    lstm_training_example()
    
    # Example 5: Portfolio Analysis
    print("\n💼 Example 5: Portfolio Analysis")
    print("-" * 30)
    portfolio_analysis_example()
    
    # Example 6: Advanced Configuration
    print("\n⚙️ Example 6: Advanced Configuration")
    print("-" * 30)
    advanced_configuration_example()
    
    print("\n✅ All examples completed successfully!")
    print("\n🚀 To run the web interface:")
    print("   Flask: python app.py")
    print("   Streamlit: streamlit run streamlit_app.py")

def basic_trading_example():
    """Demonstrate basic trading bot functionality"""
    try:
        print("Initializing AI Trading Bot...")
        bot = AITradingBot()
        
        # Collect market data for Apple
        print("Collecting market data for AAPL...")
        market_data = bot.collect_market_data('AAPL', period='6mo')
        
        if not market_data.empty:
            print(f"✓ Collected {len(market_data)} data points")
            print(f"Latest price: ${market_data['Close'].iloc[-1]:.2f}")
            
            # Collect news data
            print("Collecting news data...")
            news_data = bot.collect_news_data('AAPL', limit=10)
            print(f"✓ Collected {len(news_data)} news articles")
            
            # Generate trading signal
            print("Generating trading signal...")
            signal = bot.generate_trading_signal('AAPL')
            
            print(f"\n🎯 Trading Signal Generated:")
            print(f"Symbol: {signal['symbol']}")
            print(f"Signal: {signal['signal'].upper()}")
            print(f"Confidence: {signal['confidence']:.1%}")
            print(f"Risk Level: {signal['risk_level']}")
            
            # Show explanation
            print(f"\n📖 Signal Explanation:")
            print(signal['explanation'][:200] + "...")
            
        else:
            print("✗ Failed to collect market data")
            
    except Exception as e:
        print(f"Error in basic trading example: {str(e)}")

def technical_analysis_example():
    """Demonstrate technical indicator calculations"""
    try:
        print("Calculating technical indicators...")
        
        # Get sample data
        collector = MarketDataCollector()
        data = collector.get_stock_data('GOOGL', period='3mo')
        
        if not data.empty:
            # Calculate indicators
            ti = TechnicalIndicators()
            data_with_indicators = ti.calculate_all_indicators(data)
            
            print(f"✓ Calculated {len(data_with_indicators.columns) - 5} technical indicators")
            
            # Show some key indicators
            latest = data_with_indicators.iloc[-1]
            
            print(f"\n📊 Key Technical Indicators (Latest):")
            print(f"Close Price: ${latest['Close']:.2f}")
            
            if 'SMA_20' in data_with_indicators.columns:
                print(f"SMA 20: ${latest['SMA_20']:.2f}")
            if 'SMA_50' in data_with_indicators.columns:
                print(f"SMA 50: ${latest['SMA_50']:.2f}")
            if 'RSI' in data_with_indicators.columns:
                rsi = latest['RSI']
                status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                print(f"RSI: {rsi:.2f} ({status})")
            if 'MACD' in data_with_indicators.columns:
                print(f"MACD: {latest['MACD']:.4f}")
            
            # Feature selection
            print("\n🔍 Feature Selection:")
            selector = FeatureSelector(method='correlation')
            selected_features = selector.select_features(data_with_indicators, n_features=10)
            print(f"Top 10 features: {selected_features[:5]}...")
            
        else:
            print("✗ Failed to get market data")
            
    except Exception as e:
        print(f"Error in technical analysis example: {str(e)}")

def sentiment_analysis_example():
    """Demonstrate sentiment analysis functionality"""
    try:
        print("Initializing sentiment analyzer...")
        analyzer = FinancialSentimentAnalyzer()
        
        # Test individual text analysis
        test_texts = [
            "Apple stock surges 5% on strong earnings report",
            "Market crashes as investors panic over economic data",
            "Analysts upgrade Tesla target price on positive outlook",
            "Company reports disappointing quarterly results"
        ]
        
        print("\n📝 Testing sentiment analysis on sample texts:")
        for i, text in enumerate(test_texts, 1):
            result = analyzer.analyze_text(text)
            print(f"\n{i}. Text: {text}")
            print(f"   Sentiment: {result['sentiment']}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Keywords: {result['keywords_found'][:3]}")
        
        # Test news batch analysis
        print("\n📰 Testing news batch analysis:")
        test_news = [
            {'title': 'Stock market rallies on positive economic data', 'source': 'Test'},
            {'title': 'Tech stocks fall amid earnings concerns', 'source': 'Test'},
            {'title': 'Federal Reserve signals potential rate cuts', 'source': 'Test'}
        ]
        
        analyzed_news = analyzer.analyze_news_batch(test_news)
        print(f"✓ Analyzed {len(analyzed_news)} news articles")
        
        # Get market sentiment summary
        summary = analyzer.get_market_sentiment_summary(analyzed_news)
        print(f"\n📊 Market Sentiment Summary:")
        print(f"Overall Sentiment: {summary['overall_sentiment']}")
        print(f"Sentiment Score: {summary['sentiment_score']:.3f}")
        print(f"Confidence: {summary['confidence']:.3f}")
        
    except Exception as e:
        print(f"Error in sentiment analysis example: {str(e)}")

def lstm_training_example():
    """Demonstrate LSTM model training"""
    try:
        print("Setting up LSTM predictor...")
        lstm = LSTMPredictor(lookback_days=20, prediction_days=1)
        
        # Get sample data
        collector = MarketDataCollector()
        data = collector.get_stock_data('MSFT', period='1y')
        
        if not data.empty:
            # Calculate technical indicators
            ti = TechnicalIndicators()
            data_with_indicators = ti.calculate_all_indicators(data)
            
            if not data_with_indicators.empty:
                # Prepare features
                print("Preparing features for LSTM...")
                features, targets = lstm.prepare_data(data_with_indicators, target_col='Close')
                
                if len(features) > 0:
                    print(f"✓ Prepared {len(features)} feature sequences")
                    print(f"Feature shape: {features.shape}")
                    print(f"Target shape: {targets.shape}")
                    
                    # Scale features
                    preprocessor = DataPreprocessor()
                    scaled_features = preprocessor.scale_features(features, fit=True)
                    
                    # Split data
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(
                        scaled_features, targets, test_size=0.2, random_state=42
                    )
                    
                    print(f"Training set: {X_train.shape[0]} samples")
                    print(f"Test set: {X_test.shape[0]} samples")
                    
                    # Train model (with reduced epochs for demo)
                    print("\n🚀 Training LSTM model (reduced epochs for demo)...")
                    lstm.epochs = 10  # Reduce for demo
                    history = lstm.train(X_train, y_train)
                    
                    if lstm.is_trained:
                        print("✓ Model training completed!")
                        
                        # Evaluate model
                        test_metrics = lstm.evaluate(X_test, y_test)
                        print(f"\n📊 Model Performance:")
                        print(f"R² Score: {test_metrics['r2']:.4f}")
                        print(f"RMSE: {test_metrics['rmse']:.4f}")
                        print(f"MAE: {test_metrics['mae']:.4f}")
                        
                        # Make predictions
                        sample_predictions = lstm.predict(X_test[:5])
                        print(f"\n🔮 Sample Predictions:")
                        for i, (pred, actual) in enumerate(zip(sample_predictions.flatten(), y_test[:5])):
                            print(f"  {i+1}. Predicted: ${pred:.2f}, Actual: ${actual:.2f}")
                    else:
                        print("✗ Model training failed")
                else:
                    print("✗ No features prepared")
            else:
                print("✗ Failed to calculate technical indicators")
        else:
            print("✗ Failed to get market data")
            
    except Exception as e:
        print(f"Error in LSTM training example: {str(e)}")

def portfolio_analysis_example():
    """Demonstrate portfolio analysis functionality"""
    try:
        print("Setting up portfolio analysis...")
        bot = AITradingBot()
        
        # Define portfolio symbols
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
        print(f"Analyzing portfolio: {', '.join(symbols)}")
        
        # Collect data for all symbols
        portfolio_data = {}
        for symbol in symbols:
            print(f"Collecting data for {symbol}...")
            data = bot.collect_market_data(symbol, period='3mo')
            if not data.empty:
                portfolio_data[symbol] = data
                print(f"✓ {symbol}: {len(data)} data points")
            else:
                print(f"✗ {symbol}: Failed to collect data")
        
        if portfolio_data:
            # Generate portfolio signals
            print("\n🎯 Generating portfolio signals...")
            portfolio_signals = bot.get_portfolio_signals(list(portfolio_data.keys()))
            
            print(f"\n📊 Portfolio Analysis Results:")
            print(f"Total Symbols: {len(portfolio_signals)}")
            
            # Count signal types
            buy_count = sum(1 for s in portfolio_signals if s.get('signal') == 'buy')
            sell_count = sum(1 for s in portfolio_signals if s.get('signal') == 'sell')
            hold_count = sum(1 for s in portfolio_signals if s.get('signal') == 'hold')
            
            print(f"Buy Signals: {buy_count}")
            print(f"Sell Signals: {sell_count}")
            print(f"Hold Signals: {hold_count}")
            
            # Calculate average confidence
            confidences = [s.get('confidence', 0) for s in portfolio_signals]
            avg_confidence = np.mean(confidences)
            print(f"Average Confidence: {avg_confidence:.1%}")
            
            # Show individual signals
            print(f"\n📈 Individual Signals:")
            for signal in portfolio_signals:
                symbol = signal.get('symbol', 'Unknown')
                signal_type = signal.get('signal', 'hold').upper()
                confidence = signal.get('confidence', 0)
                risk = signal.get('risk_level', 'Unknown')
                
                print(f"  {symbol}: {signal_type} (Confidence: {confidence:.1%}, Risk: {risk})")
            
            # Save portfolio signals
            print(f"\n💾 Saving portfolio signals...")
            bot.save_signals('portfolio_signals.json')
            print("✓ Portfolio signals saved to portfolio_signals.json")
            
        else:
            print("✗ No portfolio data available")
            
    except Exception as e:
        print(f"Error in portfolio analysis example: {str(e)}")

def advanced_configuration_example():
    """Demonstrate advanced configuration options"""
    try:
        print("Setting up advanced configuration...")
        
        # Custom configuration
        custom_config = {
            'lookback_days': 60,
            'prediction_days': 3,
            'technical_confidence_weight': 0.7,
            'sentiment_confidence_weight': 0.3,
            'min_confidence_threshold': 0.7,
            'update_frequency_hours': 2,
            'max_news_articles': 30,
            'feature_selection_method': 'pca',
            'n_selected_features': 30
        }
        
        print("Custom configuration:")
        for key, value in custom_config.items():
            print(f"  {key}: {value}")
        
        # Initialize bot with custom config
        bot = AITradingBot(config=custom_config)
        print(f"\n✓ Bot initialized with custom configuration")
        
        # Test with a symbol
        print("\n🧪 Testing custom configuration...")
        symbol = 'NVDA'
        
        # Collect data
        data = bot.collect_market_data(symbol, period='6mo')
        if not data.empty:
            print(f"✓ Collected {len(data)} data points for {symbol}")
            
            # Generate signal with custom config
            signal = bot.generate_trading_signal(symbol)
            
            print(f"\n🎯 Signal with Custom Config:")
            print(f"Symbol: {signal['symbol']}")
            print(f"Signal: {signal['signal'].upper()}")
            print(f"Confidence: {signal['confidence']:.1%}")
            print(f"Risk Level: {signal['risk_level']}")
            
            # Show configuration impact
            tech_conf = signal.get('technical_analysis', {}).get('confidence', 0)
            sent_conf = signal.get('sentiment_analysis', {}).get('confidence', 0)
            
            print(f"\n⚖️ Configuration Impact:")
            print(f"Technical Weight: {custom_config['technical_confidence_weight']}")
            print(f"Sentiment Weight: {custom_config['sentiment_confidence_weight']}")
            print(f"Technical Confidence: {tech_conf:.1%}")
            print(f"Sentiment Confidence: {sent_conf:.1%}")
            
        else:
            print(f"✗ Failed to collect data for {symbol}")
            
    except Exception as e:
        print(f"Error in advanced configuration example: {str(e)}")

def performance_benchmark():
    """Run performance benchmarks"""
    print("\n⚡ Performance Benchmark")
    print("-" * 30)
    
    import time
    
    # Benchmark data collection
    print("Benchmarking data collection...")
    start_time = time.time()
    
    collector = MarketDataCollector()
    data = collector.get_stock_data('AAPL', period='1y')
    
    collection_time = time.time() - start_time
    print(f"Data collection: {collection_time:.2f} seconds")
    
    # Benchmark technical indicators
    if not data.empty:
        print("Benchmarking technical indicators...")
        start_time = time.time()
        
        ti = TechnicalIndicators()
        data_with_indicators = ti.calculate_all_indicators(data)
        
        indicators_time = time.time() - start_time
        print(f"Technical indicators: {indicators_time:.2f} seconds")
        print(f"Indicators calculated: {len(data_with_indicators.columns) - 5}")
    
    # Benchmark sentiment analysis
    print("Benchmarking sentiment analysis...")
    start_time = time.time()
    
    analyzer = FinancialSentimentAnalyzer()
    test_text = "Apple stock surges on strong earnings report"
    result = analyzer.analyze_text(test_text)
    
    sentiment_time = time.time() - start_time
    print(f"Sentiment analysis: {sentiment_time:.2f} seconds")
    
    print(f"\n📊 Performance Summary:")
    print(f"Data Collection: {collection_time:.2f}s")
    print(f"Technical Indicators: {indicators_time:.2f}s")
    print(f"Sentiment Analysis: {sentiment_time:.2f}s")
    print(f"Total: {collection_time + indicators_time + sentiment_time:.2f}s")

if __name__ == "__main__":
    try:
        main()
        
        # Optional: Run performance benchmark
        run_benchmark = input("\n🔍 Run performance benchmark? (y/n): ").lower().strip()
        if run_benchmark == 'y':
            performance_benchmark()
        
        print("\n🎉 Example usage completed!")
        print("\n💡 Next steps:")
        print("   1. Explore the web interfaces (Flask/Streamlit)")
        print("   2. Customize configurations for your needs")
        print("   3. Add more symbols to your portfolio")
        print("   4. Experiment with different time periods")
        print("   5. Save and analyze your trading signals")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Example interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running examples: {str(e)}")
        print("This might be due to missing dependencies or network issues.")
        print("Please check the requirements.txt and ensure all packages are installed.")
