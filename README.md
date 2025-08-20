# 🤖 AI Trading Assistant

A comprehensive, AI-powered trading assistant for stock and cryptocurrency markets that combines technical analysis, sentiment analysis, and machine learning to provide intelligent trading signals with detailed explanations.

## 🎯 Project Overview

This AI Trading Assistant is designed to be both a powerful trading tool and an educational platform. When it generates BUY/SELL/HOLD signals, it provides comprehensive explanations of the reasoning behind each decision, helping users understand:

- **Technical Analysis**: Moving averages, RSI, MACD, Bollinger Bands, and more
- **Market Sentiment**: News sentiment analysis using FinBERT
- **AI Predictions**: LSTM neural network price forecasting
- **Risk Assessment**: Confidence-based risk level classification

## 🏗️ Architecture

The application consists of three core modules:

### 1. Data Pipeline (`data_collector.py`)
- **Market Data**: Fetches OHLCV data from yfinance (stocks) and Binance API (cryptocurrencies)
- **News Data**: Web scraping from financial news sources (Yahoo Finance, Google News)
- **Real-time Updates**: Configurable update frequency

### 2. AI & Machine Learning (`prediction_model.py`, `sentiment_analysis.py`)
- **Sentiment Analysis**: FinBERT model for financial text sentiment classification
- **Price Prediction**: LSTM neural networks for time series forecasting
- **Feature Engineering**: 25+ technical indicators and advanced feature selection
- **Model Training**: Automated training with validation and performance metrics

### 3. Educational Engine (`trading_bot.py`)
- **Signal Generation**: Combines technical and sentiment analysis
- **Detailed Explanations**: Human-readable reasoning for every signal
- **Risk Assessment**: Confidence-based risk classification
- **Portfolio Management**: Multi-symbol tracking and analysis

## 🚀 Features

### Core Capabilities
- **Multi-Asset Support**: Stocks and cryptocurrencies
- **Real-time Analysis**: Live market data and news sentiment
- **Technical Indicators**: 25+ advanced technical indicators
- **AI Predictions**: LSTM-based price forecasting
- **Sentiment Analysis**: FinBERT-powered news sentiment
- **Educational Insights**: Detailed explanations for every signal

### User Interface
- **Web Dashboard**: Flask-based web application
- **Streamlit Alternative**: Modern, interactive Streamlit app
- **Responsive Design**: Mobile-friendly interface
- **Real-time Updates**: Live signal refresh capabilities

### Data Management
- **Export/Import**: Save and load trading signals
- **Historical Data**: Configurable time periods
- **Portfolio Tracking**: Multi-symbol management
- **Performance Metrics**: Training and testing statistics

## 📋 Requirements

### System Requirements
- Python 3.9+
- 8GB+ RAM (for LSTM training)
- Stable internet connection
- GPU recommended (for faster model training)

### Python Dependencies
```
pandas==2.1.4
numpy==1.24.3
yfinance==0.2.28
requests==2.31.0
beautifulsoup4==4.12.2
scikit-learn==1.3.2
tensorflow==2.15.0
transformers==4.36.2
torch==2.1.2
flask==3.0.0
plotly==5.17.0
streamlit==1.29.0
python-binance==1.0.19
ta==0.10.2
nltk==3.8.1
textblob==0.17.1
```

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd AI_Trading
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

### 4. Download NLTK Data
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
```

## 🚀 Usage

### Quick Start

#### 1. Basic Usage (Python Script)
```python
from trading_bot import AITradingBot

# Initialize the bot
bot = AITradingBot()

# Add symbols to track
bot.collect_market_data('AAPL', period='6mo')
bot.collect_news_data('AAPL', limit=20)

# Generate trading signal
signal = bot.generate_trading_signal('AAPL')
print(f"Signal: {signal['signal']}")
print(f"Confidence: {signal['confidence']:.1%}")
print(f"Explanation: {signal['explanation']}")
```

#### 2. Web Dashboard (Flask)
```bash
python app.py
```
Open your browser and navigate to `http://localhost:5000`

#### 3. Streamlit App
```bash
streamlit run streamlit_app.py
```

### Advanced Usage

#### Portfolio Analysis
```python
# Analyze multiple symbols
symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
portfolio_signals = bot.get_portfolio_signals(symbols)

for signal in portfolio_signals:
    print(f"{signal['symbol']}: {signal['signal']} ({signal['confidence']:.1%})")
```

#### Custom Configuration
```python
config = {
    'lookback_days': 60,
    'technical_confidence_weight': 0.7,
    'sentiment_confidence_weight': 0.3,
    'n_selected_features': 30
}

bot = AITradingBot(config=config)
```

#### Model Training and Evaluation
```python
# Train LSTM model
features, targets = bot.lstm_predictor.prepare_data(market_data)
history = bot.lstm_predictor.train(features, targets)

# Evaluate performance
test_metrics = bot.lstm_predictor.evaluate(test_features, test_targets)
print(f"R² Score: {test_metrics['r2']:.4f}")
```

## 📊 Technical Indicators

The system calculates over 25 technical indicators:

### Trend Indicators
- Simple Moving Averages (SMA 5, 10, 20, 50, 200)
- Exponential Moving Averages (EMA 12, 26, 50)
- MACD (Moving Average Convergence Divergence)
- Parabolic SAR
- ADX (Average Directional Index)
- Golden Cross/Death Cross detection

### Momentum Indicators
- RSI (Relative Strength Index)
- Stochastic Oscillator
- Williams %R
- ROC (Rate of Change)
- CCI (Commodity Channel Index)
- Money Flow Index

### Volatility Indicators
- Bollinger Bands (Upper, Middle, Lower, Width, Position)
- ATR (Average True Range)
- Keltner Channels

### Volume Indicators
- Volume SMA
- On Balance Volume (OBV)
- Volume Rate of Change
- Chaikin Money Flow
- VWAP (Volume Weighted Average Price)

### Support/Resistance
- Pivot Points
- Support/Resistance levels (R1, S1, R2, S2)
- Price position relative to levels

## 🧠 AI Models

### LSTM Neural Network
- **Architecture**: Multi-layer LSTM with dropout and batch normalization
- **Input**: Time series sequences of technical indicators
- **Output**: Price predictions for next 1-7 days
- **Training**: Adam optimizer with early stopping and learning rate reduction
- **Validation**: 20% test split with performance metrics

### FinBERT Sentiment Analysis
- **Model**: Pre-trained FinBERT for financial text
- **Fallback**: TextBlob for cases where FinBERT fails
- **Features**: Financial keyword extraction and market term identification
- **Output**: Positive/Negative/Neutral sentiment with confidence scores

### Feature Selection
- **Methods**: Correlation, PCA, and variance-based selection
- **Configurable**: Number of features (default: 25)
- **Dynamic**: Adapts to available data and indicators

## 📈 Signal Generation

### Signal Types
- **BUY**: Strong bullish indicators with positive sentiment
- **SELL**: Strong bearish indicators with negative sentiment
- **HOLD**: Mixed signals or insufficient confidence

### Confidence Levels
- **High (80%+)**: Strong technical and sentiment alignment
- **Medium (60-79%)**: Good alignment with some uncertainty
- **Low (<60%)**: Mixed or weak signals

### Risk Assessment
- **Low Risk**: High confidence signals (>80%)
- **Medium Risk**: Medium confidence signals (60-80%)
- **High Risk**: Low confidence signals (<60%)

## 🔧 Configuration

### Bot Configuration
```python
default_config = {
    'lookback_days': 30,                    # Days to look back for analysis
    'prediction_days': 1,                   # Days to predict ahead
    'technical_confidence_weight': 0.6,     # Weight for technical analysis
    'sentiment_confidence_weight': 0.4,     # Weight for sentiment analysis
    'min_confidence_threshold': 0.6,        # Minimum confidence for signals
    'update_frequency_hours': 4,            # Data update frequency
    'max_news_articles': 20,                # Maximum news articles to analyze
    'feature_selection_method': 'correlation', # Feature selection method
    'n_selected_features': 25               # Number of features to select
}
```

### Model Hyperparameters
```python
lstm_config = {
    'lstm_units': [128, 64, 32],           # LSTM layer sizes
    'dropout_rate': 0.2,                    # Dropout for regularization
    'learning_rate': 0.001,                 # Adam optimizer learning rate
    'batch_size': 32,                       # Training batch size
    'epochs': 100,                          # Maximum training epochs
    'validation_split': 0.2                 # Validation data split
}
```

## 📱 Web Interface

### Flask Dashboard
- **Home**: Project overview and quick start guide
- **Dashboard**: Symbol management and signal display
- **Analysis**: Detailed technical and sentiment analysis
- **Settings**: Bot configuration and data management
- **About**: Project information and disclaimers

### Streamlit Alternative
- **Modern UI**: Clean, responsive interface
- **Interactive Charts**: Plotly-powered visualizations
- **Real-time Updates**: Live data refresh capabilities
- **Mobile Friendly**: Responsive design for all devices

## 📊 Data Sources

### Market Data
- **Stocks**: Yahoo Finance (yfinance library)
- **Cryptocurrencies**: Binance API
- **Intervals**: 1m, 5m, 15m, 30m, 1h, 1d, 1w, 1m
- **Periods**: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max

### News Sources
- **Yahoo Finance**: Financial news and market updates
- **Google News**: General financial news
- **Real-time**: Continuous news monitoring
- **Sentiment**: AI-powered sentiment analysis

## 🚨 Disclaimer

**⚠️ IMPORTANT: This tool is for educational purposes only.**

- **Not Financial Advice**: This tool does not provide financial advice
- **Do Your Own Research**: Always conduct thorough research before trading
- **Consult Professionals**: Consider consulting with a financial advisor
- **Risk Warning**: Trading involves risk of loss
- **Past Performance**: Past performance doesn't guarantee future results
- **Educational Tool**: Use as a learning platform for trading education

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Areas for Contribution
- Additional technical indicators
- New sentiment analysis models
- Enhanced visualization features
- Performance optimizations
- Documentation improvements

## 📚 Learning Resources

This tool is designed to help you learn about:

- **Technical Analysis**: Understanding market indicators and patterns
- **Market Sentiment**: How news affects market movements
- **Machine Learning**: AI applications in finance
- **Risk Management**: Portfolio analysis and risk assessment
- **Trading Education**: Practical application of financial concepts

## 🔍 Troubleshooting

### Common Issues

#### 1. Model Training Fails
```bash
# Check available memory
free -h  # Linux/Mac
# or
wmic computersystem get TotalPhysicalMemory  # Windows

# Reduce batch size or lookback days
config = {'batch_size': 16, 'lookback_days': 20}
```

#### 2. News Scraping Issues
```python
# Use alternative news sources
news_collector = NewsCollector()
news = news_collector.scrape_google_news("AAPL")
```

#### 3. Memory Issues
```python
# Reduce feature count
config = {'n_selected_features': 15}
```

#### 4. API Rate Limits
```python
# Add delays between requests
import time
time.sleep(1)  # 1 second delay
```

### Performance Optimization
- Use GPU for LSTM training
- Reduce lookback period for faster analysis
- Limit number of tracked symbols
- Use feature selection to reduce dimensionality

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **FinBERT**: ProsusAI for the financial sentiment analysis model
- **yfinance**: Yahoo Finance data access
- **Binance API**: Cryptocurrency market data
- **Technical Analysis Library**: Technical indicators implementation
- **Open Source Community**: All contributors and maintainers

## 📞 Support

For support and questions:

- **Issues**: Create an issue on GitHub
- **Discussions**: Use GitHub Discussions
- **Documentation**: Check the wiki and examples
- **Community**: Join our community forum

---

**Happy Trading and Learning! 🚀📈**

Remember: The best investment you can make is in your own education and understanding of the markets.
