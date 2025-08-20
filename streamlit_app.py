"""
Streamlit Web Application for AI Trading Assistant
Alternative to Flask app for users who prefer Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Optional

# Import our custom modules
from trading_bot import AITradingBot
from data_collector import MarketDataCollector
from feature_engineering import TechnicalIndicators

# Configure Streamlit page
st.set_page_config(
    page_title="AI Trading Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize trading bot
@st.cache_resource
def get_trading_bot():
    """Get cached trading bot instance"""
    return AITradingBot()

trading_bot = get_trading_bot()

# Initialize session state
if 'symbols' not in st.session_state:
    st.session_state.symbols = []
if 'signals' not in st.session_state:
    st.session_state.signals = {}
if 'market_data' not in st.session_state:
    st.session_state.market_data = {}

def main():
    """Main Streamlit application"""
    
    # Sidebar
    st.sidebar.title("🤖 AI Trading Assistant")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["🏠 Home", "📊 Dashboard", "📈 Analysis", "⚙️ Settings", "ℹ️ About"]
    )
    
    # Page routing
    if page == "🏠 Home":
        home_page()
    elif page == "📊 Dashboard":
        dashboard_page()
    elif page == "📈 Analysis":
        analysis_page()
    elif page == "⚙️ Settings":
        settings_page()
    elif page == "ℹ️ About":
        about_page()

def home_page():
    """Home page content"""
    st.title("🤖 AI Trading Assistant")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Your Intelligent Trading Companion
        
        This AI-powered tool combines cutting-edge technology to provide comprehensive trading analysis:
        
        🔹 **Technical Analysis**: Advanced indicators and pattern recognition
        🔹 **Sentiment Analysis**: Real-time news sentiment using FinBERT
        🔹 **Machine Learning**: LSTM neural networks for price prediction
        🔹 **Educational Insights**: Detailed explanations for every trading signal
        
        ### How It Works
        
        1. **Add Symbols**: Enter stock or cryptocurrency symbols to track
        2. **AI Analysis**: Our AI analyzes market data and news sentiment
        3. **Get Signals**: Receive BUY/SELL/HOLD recommendations with confidence levels
        4. **Learn**: Understand the reasoning behind each signal with detailed explanations
        
        ### Getting Started
        
        Navigate to the Dashboard to begin adding symbols and receiving AI-powered trading signals.
        """)
        
        if st.button("🚀 Go to Dashboard", type="primary"):
            st.switch_page("📊 Dashboard")
    
    with col2:
        st.markdown("""
        ### Quick Stats
        
        **Tracked Symbols**: {}
        **Active Signals**: {}
        **Last Updated**: {}
        """.format(
            len(st.session_state.symbols),
            len(st.session_state.signals),
            datetime.now().strftime("%Y-%m-%d %H:%M") if st.session_state.signals else "Never"
        ))
        
        st.markdown("""
        ### Disclaimer
        
        ⚠️ **This tool is for educational purposes only.**
        
        Always conduct your own research and consider consulting with a financial advisor before making investment decisions.
        """)

def dashboard_page():
    """Dashboard page content"""
    st.title("📊 Trading Dashboard")
    st.markdown("---")
    
    # Add symbol section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        new_symbol = st.text_input(
            "Add New Symbol",
            placeholder="Enter symbol (e.g., AAPL, GOOGL, BTCUSDT)",
            help="Enter a stock or cryptocurrency symbol to track"
        )
    
    with col2:
        if st.button("➕ Add Symbol", type="primary"):
            if new_symbol:
                add_symbol(new_symbol.upper().strip())
                st.success(f"Added {new_symbol.upper()} successfully!")
                st.rerun()
    
    st.markdown("---")
    
    # Portfolio summary
    if st.session_state.symbols:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Symbols", len(st.session_state.symbols))
        
        with col2:
            buy_count = sum(1 for s in st.session_state.signals.values() if s.get('signal') == 'buy')
            st.metric("Buy Signals", buy_count, delta="Bullish")
        
        with col3:
            sell_count = sum(1 for s in st.session_state.signals.values() if s.get('signal') == 'sell')
            st.metric("Sell Signals", sell_count, delta="Bearish")
        
        with col4:
            hold_count = sum(1 for s in st.session_state.signals.values() if s.get('signal') == 'hold')
            st.metric("Hold Signals", hold_count, delta="Neutral")
        
        st.markdown("---")
        
        # Refresh signals button
        if st.button("🔄 Refresh All Signals", type="secondary"):
            with st.spinner("Refreshing signals..."):
                refresh_all_signals()
                st.success("Signals refreshed successfully!")
                st.rerun()
        
        # Signals display
        st.subheader("📊 Current Trading Signals")
        
        for symbol in st.session_state.symbols:
            if symbol in st.session_state.signals:
                signal = st.session_state.signals[symbol]
                display_signal_card(symbol, signal)
            else:
                # Generate signal if not exists
                with st.spinner(f"Generating signal for {symbol}..."):
                    try:
                        signal = trading_bot.generate_trading_signal(symbol)
                        st.session_state.signals[symbol] = signal
                        display_signal_card(symbol, signal)
                    except Exception as e:
                        st.error(f"Error generating signal for {symbol}: {str(e)}")
    else:
        st.info("No symbols tracked yet. Add some symbols to get started!")
        
        # Sample symbols suggestion
        st.markdown("### 💡 Try these popular symbols:")
        col1, col2, col3 = st.columns(3)
        
        sample_symbols = [['AAPL', 'GOOGL', 'MSFT'], ['TSLA', 'AMZN', 'NVDA'], ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']]
        
        for i, col in enumerate([col1, col2, col3]):
            with col:
                for symbol in sample_symbols[i]:
                    if st.button(f"➕ {symbol}", key=f"sample_{symbol}"):
                        add_symbol(symbol)
                        st.success(f"Added {symbol} successfully!")
                        st.rerun()

def analysis_page():
    """Analysis page content"""
    st.title("📈 Detailed Analysis")
    st.markdown("---")
    
    if not st.session_state.symbols:
        st.warning("No symbols tracked. Please add symbols in the Dashboard first.")
        return
    
    # Symbol selector
    selected_symbol = st.selectbox(
        "Select Symbol for Analysis",
        st.session_state.symbols,
        help="Choose a symbol to view detailed analysis"
    )
    
    if selected_symbol:
        st.subheader(f"📊 Analysis for {selected_symbol}")
        
        # Get market data
        if selected_symbol not in st.session_state.market_data:
            with st.spinner(f"Fetching market data for {selected_symbol}..."):
                market_data = trading_bot.collect_market_data(selected_symbol, period='6mo')
                if not market_data.empty:
                    st.session_state.market_data[selected_symbol] = market_data
        
        if selected_symbol in st.session_state.market_data:
            market_data = st.session_state.market_data[selected_symbol]
            
            # Price chart
            st.subheader("💰 Price Chart")
            fig = create_price_chart(market_data, selected_symbol)
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical indicators
            st.subheader("🔧 Technical Indicators")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Calculate Technical Indicators"):
                    with st.spinner("Calculating indicators..."):
                        ti = TechnicalIndicators()
                        data_with_indicators = ti.calculate_all_indicators(market_data)
                        
                        if not data_with_indicators.empty:
                            st.session_state.market_data[selected_symbol] = data_with_indicators
                            st.success("Technical indicators calculated!")
                            st.rerun()
            
            # Display indicators if available
            if 'SMA_20' in market_data.columns:
                display_technical_indicators(market_data)
            
            # News sentiment
            st.subheader("📰 News Sentiment")
            
            if st.button("Fetch Latest News"):
                with st.spinner("Fetching news..."):
                    news_data = trading_bot.collect_news_data(selected_symbol, limit=10)
                    if news_data:
                        display_news_sentiment(news_data)
                    else:
                        st.warning("No news data available")
            
            # Current signal
            if selected_symbol in st.session_state.signals:
                st.subheader("🎯 Current Trading Signal")
                signal = st.session_state.signals[selected_symbol]
                display_detailed_signal(signal)
        else:
            st.error(f"Unable to fetch market data for {selected_symbol}")

def settings_page():
    """Settings page content"""
    st.title("⚙️ Bot Settings")
    st.markdown("---")
    
    st.subheader("🤖 AI Model Configuration")
    
    with st.form("config_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            lookback_days = st.number_input(
                "Lookback Days",
                min_value=10,
                max_value=100,
                value=trading_bot.config['lookback_days'],
                help="Number of days to look back for technical analysis"
            )
            
            prediction_days = st.number_input(
                "Prediction Days",
                min_value=1,
                max_value=7,
                value=trading_bot.config['prediction_days'],
                help="Number of days to predict ahead"
            )
        
        with col2:
            tech_weight = st.slider(
                "Technical Analysis Weight",
                min_value=0.0,
                max_value=1.0,
                value=trading_bot.config['technical_confidence_weight'],
                step=0.1,
                help="Weight given to technical analysis vs sentiment analysis"
            )
            
            sent_weight = 1.0 - tech_weight
            st.metric("Sentiment Analysis Weight", f"{sent_weight:.1f}")
        
        n_features = st.number_input(
            "Number of Selected Features",
            min_value=10,
            max_value=50,
            value=trading_bot.config['n_selected_features'],
            help="Number of technical indicators to use for prediction"
        )
        
        if st.form_submit_button("💾 Save Settings"):
            # Update config
            trading_bot.config.update({
                'lookback_days': lookback_days,
                'prediction_days': prediction_days,
                'technical_confidence_weight': tech_weight,
                'sentiment_confidence_weight': sent_weight,
                'n_selected_features': n_features
            })
            
            st.success("Settings saved successfully!")
    
    st.markdown("---")
    
    # Data management
    st.subheader("🗄️ Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Save Current Signals"):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'signals_{timestamp}.json'
            
            try:
                trading_bot.save_signals(filename)
                st.success(f"Signals saved to {filename}")
            except Exception as e:
                st.error(f"Error saving signals: {str(e)}")
    
    with col2:
        uploaded_file = st.file_uploader(
            "Load Signals from File",
            type=['json'],
            help="Upload a previously saved signals file"
        )
        
        if uploaded_file is not None:
            try:
                signals_data = json.load(uploaded_file)
                trading_bot.last_signals = signals_data
                st.session_state.signals = signals_data
                st.success("Signals loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading signals: {str(e)}")

def about_page():
    """About page content"""
    st.title("ℹ️ About AI Trading Assistant")
    st.markdown("---")
    
    st.markdown("""
    ## 🤖 What is AI Trading Assistant?
    
    AI Trading Assistant is a comprehensive, AI-powered trading analysis tool that combines multiple advanced technologies to provide intelligent trading recommendations.
    
    ## 🧠 Core Technologies
    
    ### 1. Data Collection & Processing
    - **Market Data**: Real-time OHLCV data from yfinance (stocks) and Binance API (cryptocurrencies)
    - **News Data**: Web scraping from financial news sources with sentiment analysis
    
    ### 2. Technical Analysis
    - **Moving Averages**: SMA, EMA with Golden/Death Cross detection
    - **Momentum Indicators**: RSI, MACD, Stochastic Oscillator, Williams %R
    - **Volatility Indicators**: Bollinger Bands, ATR, Keltner Channels
    - **Volume Indicators**: OBV, VWAP, Volume Rate of Change
    
    ### 3. AI & Machine Learning
    - **Sentiment Analysis**: FinBERT model for financial text sentiment
    - **Price Prediction**: LSTM neural networks for time series forecasting
    - **Feature Selection**: Correlation, PCA, and variance-based feature selection
    
    ### 4. Educational Engine
    - **Signal Explanations**: Detailed reasoning behind every trading recommendation
    - **Risk Assessment**: Confidence-based risk level classification
    - **Learning Resources**: Understanding of technical and fundamental factors
    
    ## 🎯 Key Features
    
    - **Real-time Analysis**: Live market data and news sentiment
    - **Multi-asset Support**: Stocks and cryptocurrencies
    - **Portfolio Management**: Track multiple symbols simultaneously
    - **Web Dashboard**: User-friendly interface for analysis and monitoring
    - **Export/Import**: Save and load trading signals
    
    ## ⚠️ Important Disclaimer
    
    **This tool is for educational purposes only.**
    
    - Not financial advice
    - Always conduct your own research
    - Consider consulting with a financial advisor
    - Past performance doesn't guarantee future results
    - Trading involves risk of loss
    
    ## 🔧 Technical Requirements
    
    - Python 3.9+
    - TensorFlow/PyTorch for LSTM models
    - Transformers library for FinBERT
    - Streamlit for web interface
    
    ## 📚 Learning Resources
    
    This tool is designed to help you learn about:
    - Technical analysis and indicators
    - Market sentiment and news impact
    - Machine learning in finance
    - Risk management and portfolio analysis
    
    ## 🚀 Getting Started
    
    1. Add symbols to track in the Dashboard
    2. Let the AI analyze market data and news
    3. Review trading signals and explanations
    4. Learn from detailed analysis breakdowns
    5. Use as a learning tool for trading education
    """)

def add_symbol(symbol: str):
    """Add a new symbol to track"""
    if symbol not in st.session_state.symbols:
        st.session_state.symbols.append(symbol)
        
        # Collect market data
        market_data = trading_bot.collect_market_data(symbol, period='6mo')
        if not market_data.empty:
            st.session_state.market_data[symbol] = market_data

def refresh_all_signals():
    """Refresh all trading signals"""
    for symbol in st.session_state.symbols:
        try:
            signal = trading_bot.generate_trading_signal(symbol)
            st.session_state.signals[symbol] = signal
        except Exception as e:
            st.error(f"Error refreshing signal for {symbol}: {str(e)}")

def display_signal_card(symbol: str, signal: Dict):
    """Display a trading signal card"""
    signal_type = signal.get('signal', 'hold')
    confidence = signal.get('confidence', 0.0)
    
    # Color coding
    if signal_type == 'buy':
        color = "🟢"
        bg_color = "rgba(40, 167, 69, 0.1)"
    elif signal_type == 'sell':
        color = "🔴"
        bg_color = "rgba(220, 53, 69, 0.1)"
    else:
        color = "🟡"
        bg_color = "rgba(108, 117, 125, 0.1)"
    
    # Create card
    with st.container():
        st.markdown(f"""
        <div style="
            background-color: {bg_color};
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid {'#28a745' if signal_type == 'buy' else '#dc3545' if signal_type == 'sell' else '#6c757d'};
            margin: 1rem 0;
        ">
            <h4>{color} {symbol}: {signal_type.upper()}</h4>
            <p><strong>Confidence:</strong> {confidence:.1%}</p>
            <p><strong>Risk Level:</strong> {signal.get('risk_level', 'Unknown')}</p>
            <p><strong>Last Updated:</strong> {signal.get('timestamp', 'Unknown')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Expandable explanation
        with st.expander("📖 View Detailed Explanation"):
            if 'explanation' in signal:
                st.text(signal['explanation'])
            else:
                st.info("Explanation not available")

def create_price_chart(data: pd.DataFrame, symbol: str) -> go.Figure:
    """Create price chart with Plotly"""
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Volume bars
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        name='Volume',
        yaxis='y2',
        opacity=0.3
    ))
    
    # Layout
    fig.update_layout(
        title=f'{symbol} Price Chart',
        xaxis_title='Date',
        yaxis_title='Price',
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        height=500
    )
    
    return fig

def display_technical_indicators(data: pd.DataFrame):
    """Display technical indicators"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Moving Averages")
        if 'SMA_20' in data.columns:
            st.metric("SMA 20", f"${data['SMA_20'].iloc[-1]:.2f}")
        if 'SMA_50' in data.columns:
            st.metric("SMA 50", f"${data['SMA_50'].iloc[-1]:.2f}")
    
    with col2:
        st.subheader("📈 Momentum Indicators")
        if 'RSI' in data.columns:
            rsi = data['RSI'].iloc[-1]
            st.metric("RSI", f"{rsi:.2f}", 
                     delta="Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral")
        if 'MACD' in data.columns:
            st.metric("MACD", f"{data['MACD'].iloc[-1]:.4f}")

def display_news_sentiment(news_data: List[Dict]):
    """Display news sentiment analysis"""
    st.subheader("📰 Recent News Articles")
    
    for i, article in enumerate(news_data[:5]):  # Show first 5 articles
        sentiment = article.get('sentiment_analysis', {}).get('title_sentiment', 'neutral')
        confidence = article.get('sentiment_analysis', {}).get('title_confidence', 0.0)
        
        # Sentiment emoji
        sentiment_emoji = "🟢" if sentiment == 'positive' else "🔴" if sentiment == 'negative' else "🟡"
        
        st.markdown(f"""
        **{sentiment_emoji} {article.get('title', 'No title')}**
        
        Source: {article.get('source', 'Unknown')} | 
        Sentiment: {sentiment.title()} ({confidence:.1%})
        
        ---
        """)

def display_detailed_signal(signal: Dict):
    """Display detailed trading signal information"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Signal", signal.get('signal', 'Unknown').upper())
        st.metric("Confidence", f"{signal.get('confidence', 0):.1%}")
        st.metric("Risk Level", signal.get('risk_level', 'Unknown'))
    
    with col2:
        st.metric("Technical Weight", f"{signal.get('technical_analysis', {}).get('confidence', 0):.1%}")
        st.metric("Sentiment Weight", f"{signal.get('sentiment_analysis', {}).get('confidence', 0):.1%}")
    
    # Technical analysis details
    if 'technical_analysis' in signal:
        tech = signal['technical_analysis']
        st.subheader("🔧 Technical Analysis Details")
        
        if 'indicators' in tech:
            st.write("**Key Indicators:**")
            for indicator, score in tech['indicators'].items():
                emoji = "✅" if score > 0 else "❌" if score < 0 else "➖"
                st.write(f"{emoji} {indicator.replace('_', ' ').title()}: {score:.2f}")
    
    # Sentiment analysis details
    if 'sentiment_analysis' in signal:
        sent = signal['sentiment_analysis']
        st.subheader("📰 Sentiment Analysis Details")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Overall Sentiment", sent.get('sentiment', 'Unknown').title())
            st.metric("Sentiment Score", f"{sent.get('sentiment_score', 0):.3f}")
        
        with col2:
            st.metric("Articles Analyzed", sent.get('article_count', 0))
            if 'sentiment_distribution' in sent:
                dist = sent['sentiment_distribution']
                st.write(f"**Distribution:** Positive: {dist.get('positive', 0)}, Negative: {dist.get('negative', 0)}, Neutral: {dist.get('neutral', 0)}")

if __name__ == "__main__":
    main()
