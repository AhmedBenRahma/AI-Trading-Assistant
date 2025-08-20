"""
Flask Web Application for AI Trading Assistant
Provides web-based dashboard for trading signals and analysis
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.utils
import json
import logging
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional

# Import our custom modules
from trading_bot import AITradingBot
from data_collector import MarketDataCollector
from feature_engineering import TechnicalIndicators

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'ai_trading_assistant_secret_key_2024'

# Initialize trading bot
trading_bot = AITradingBot()

# Global variables for storing data
current_symbols = []
current_signals = {}
market_data_cache = {}

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Trading dashboard"""
    return render_template('dashboard.html', 
                         symbols=current_symbols,
                         signals=current_signals)

@app.route('/analysis/<symbol>')
def analysis(symbol):
    """Detailed analysis page for a specific symbol"""
    try:
        # Get market data
        if symbol not in market_data_cache:
            market_data = trading_bot.collect_market_data(symbol, period='6mo')
            if not market_data.empty:
                market_data_cache[symbol] = market_data
        
        # Get current signal
        signal = current_signals.get(symbol, {})
        
        return render_template('analysis.html', 
                             symbol=symbol,
                             signal=signal,
                             has_data=symbol in market_data_cache)
    except Exception as e:
        logger.error(f"Error in analysis page: {str(e)}")
        flash(f"Error loading analysis for {symbol}: {str(e)}", 'error')
        return redirect(url_for('dashboard'))

@app.route('/api/add_symbol', methods=['POST'])
def add_symbol():
    """Add a new symbol to track"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper().strip()
        
        if not symbol:
            return jsonify({'success': False, 'error': 'Symbol is required'})
        
        # Validate symbol format
        if len(symbol) < 1 or len(symbol) > 10:
            return jsonify({'success': False, 'error': 'Invalid symbol format'})
        
        # Check if symbol already exists
        if symbol in current_symbols:
            return jsonify({'success': False, 'error': 'Symbol already exists'})
        
        # Try to collect data for the symbol
        market_data = trading_bot.collect_market_data(symbol, period='6mo')
        if market_data.empty:
            return jsonify({'success': False, 'error': f'Unable to fetch data for {symbol}'})
        
        # Add to current symbols
        current_symbols.append(symbol)
        market_data_cache[symbol] = market_data
        
        # Generate initial signal
        try:
            signal = trading_bot.generate_trading_signal(symbol)
            current_signals[symbol] = signal
        except Exception as e:
            logger.warning(f"Could not generate signal for {symbol}: {str(e)}")
            # Create placeholder signal
            current_signals[symbol] = {
                'symbol': symbol,
                'signal': 'hold',
                'confidence': 0.0,
                'explanation': f'Analysis pending for {symbol}'
            }
        
        return jsonify({
            'success': True, 
            'symbol': symbol,
            'message': f'Added {symbol} successfully'
        })
        
    except Exception as e:
        logger.error(f"Error adding symbol: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/remove_symbol', methods=['POST'])
def remove_symbol():
    """Remove a symbol from tracking"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper().strip()
        
        if symbol in current_symbols:
            current_symbols.remove(symbol)
        
        if symbol in current_signals:
            del current_signals[symbol]
        
        if symbol in market_data_cache:
            del market_data_cache[symbol]
        
        return jsonify({
            'success': True,
            'message': f'Removed {symbol} successfully'
        })
        
    except Exception as e:
        logger.error(f"Error removing symbol: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/refresh_signals', methods=['POST'])
def refresh_signals():
    """Refresh trading signals for all tracked symbols"""
    try:
        refreshed_signals = []
        
        for symbol in current_symbols:
            try:
                # Refresh market data
                market_data = trading_bot.collect_market_data(symbol, period='6mo')
                if not market_data.empty:
                    market_data_cache[symbol] = market_data
                
                # Generate new signal
                signal = trading_bot.generate_trading_signal(symbol)
                current_signals[symbol] = signal
                refreshed_signals.append(signal)
                
            except Exception as e:
                logger.error(f"Error refreshing signal for {symbol}: {str(e)}")
                # Keep existing signal if available
                if symbol in current_signals:
                    refreshed_signals.append(current_signals[symbol])
        
        return jsonify({
            'success': True,
            'signals': refreshed_signals,
            'message': f'Refreshed {len(refreshed_signals)} signals'
        })
        
    except Exception as e:
        logger.error(f"Error refreshing signals: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_chart_data/<symbol>')
def get_chart_data(symbol):
    """Get chart data for a symbol"""
    try:
        if symbol not in market_data_cache:
            return jsonify({'success': False, 'error': 'No data available'})
        
        data = market_data_cache[symbol]
        
        # Calculate technical indicators
        ti = TechnicalIndicators()
        data_with_indicators = ti.calculate_all_indicators(data)
        
        if data_with_indicators.empty:
            return jsonify({'success': False, 'error': 'Failed to calculate indicators'})
        
        # Prepare chart data
        dates = data_with_indicators.index.strftime('%Y-%m-%d').tolist()
        
        chart_data = {
            'dates': dates,
            'close': data_with_indicators['Close'].tolist(),
            'volume': data_with_indicators['Volume'].tolist()
        }
        
        # Add technical indicators if available
        if 'SMA_20' in data_with_indicators.columns:
            chart_data['sma_20'] = data_with_indicators['SMA_20'].tolist()
        
        if 'SMA_50' in data_with_indicators.columns:
            chart_data['sma_50'] = data_with_indicators['SMA_50'].tolist()
        
        if 'RSI' in data_with_indicators.columns:
            chart_data['rsi'] = data_with_indicators['RSI'].tolist()
        
        if 'MACD' in data_with_indicators.columns:
            chart_data['macd'] = data_with_indicators['MACD'].tolist()
        
        return jsonify({
            'success': True,
            'data': chart_data
        })
        
    except Exception as e:
        logger.error(f"Error getting chart data: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_news/<symbol>')
def get_news(symbol):
    """Get news data for a symbol"""
    try:
        news_data = trading_bot.collect_news_data(symbol, limit=10)
        
        # Format news for display
        formatted_news = []
        for article in news_data:
            formatted_news.append({
                'title': article.get('title', ''),
                'source': article.get('source', ''),
                'timestamp': article.get('timestamp', ''),
                'sentiment': article.get('sentiment_analysis', {}).get('title_sentiment', 'neutral'),
                'confidence': article.get('sentiment_analysis', {}).get('title_confidence', 0.0)
            })
        
        return jsonify({
            'success': True,
            'news': formatted_news
        })
        
    except Exception as e:
        logger.error(f"Error getting news: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/portfolio_analysis')
def portfolio_analysis():
    """Get portfolio-wide analysis"""
    try:
        if not current_symbols:
            return jsonify({'success': False, 'error': 'No symbols tracked'})
        
        # Get portfolio signals
        portfolio_signals = trading_bot.get_portfolio_signals(current_symbols)
        
        # Calculate portfolio metrics
        total_signals = len(portfolio_signals)
        buy_signals = sum(1 for s in portfolio_signals if s.get('signal') == 'buy')
        sell_signals = sum(1 for s in portfolio_signals if s.get('signal') == 'sell')
        hold_signals = sum(1 for s in portfolio_signals if s.get('signal') == 'hold')
        
        avg_confidence = np.mean([s.get('confidence', 0) for s in portfolio_signals])
        
        portfolio_summary = {
            'total_symbols': total_signals,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'hold_signals': hold_signals,
            'average_confidence': round(avg_confidence, 3),
            'signals': portfolio_signals
        }
        
        return jsonify({
            'success': True,
            'portfolio': portfolio_summary
        })
        
    except Exception as e:
        logger.error(f"Error in portfolio analysis: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/save_signals')
def save_signals():
    """Save current signals to file"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'signals_{timestamp}.json'
        
        trading_bot.save_signals(filename)
        
        return jsonify({
            'success': True,
            'message': f'Signals saved to {filename}',
            'filename': filename
        })
        
    except Exception as e:
        logger.error(f"Error saving signals: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/load_signals', methods=['POST'])
def load_signals():
    """Load signals from file"""
    try:
        data = request.get_json()
        filename = data.get('filename', '')
        
        if not filename or not os.path.exists(filename):
            return jsonify({'success': False, 'error': 'File not found'})
        
        trading_bot.load_signals(filename)
        
        # Update current signals
        global current_signals
        current_signals = trading_bot.last_signals
        
        return jsonify({
            'success': True,
            'message': f'Signals loaded from {filename}',
            'signals': current_signals
        })
        
    except Exception as e:
        logger.error(f"Error loading signals: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/settings')
def settings():
    """Settings page"""
    return render_template('settings.html', config=trading_bot.config)

@app.route('/api/update_config', methods=['POST'])
def update_config():
    """Update bot configuration"""
    try:
        data = request.get_json()
        
        # Update config
        for key, value in data.items():
            if key in trading_bot.config:
                trading_bot.config[key] = value
        
        return jsonify({
            'success': True,
            'message': 'Configuration updated successfully'
        })
        
    except Exception as e:
        logger.error(f"Error updating config: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

# Create templates directory and basic templates
def create_templates():
    """Create basic HTML templates"""
    templates_dir = 'templates'
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
    
    # Base template
    base_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}AI Trading Assistant{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        .signal-buy { color: #28a745; font-weight: bold; }
        .signal-sell { color: #dc3545; font-weight: bold; }
        .signal-hold { color: #6c757d; font-weight: bold; }
        .confidence-high { color: #28a745; }
        .confidence-medium { color: #ffc107; }
        .confidence-low { color: #dc3545; }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/">
                <i class="fas fa-robot"></i> AI Trading Assistant
            </a>
            <div class="navbar-nav">
                <a class="nav-link" href="/dashboard">Dashboard</a>
                <a class="nav-link" href="/settings">Settings</a>
                <a class="nav-link" href="/about">About</a>
            </div>
        </div>
    </nav>
    
    <div class="container mt-4">
        {% block content %}{% endblock %}
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    {% block scripts %}{% endblock %}
</body>
</html>'''
    
    # Index template
    index_template = '''{% extends "base.html" %}
{% block title %}AI Trading Assistant - Home{% endblock %}
{% block content %}
<div class="row">
    <div class="col-md-8">
        <div class="card">
            <div class="card-header">
                <h2><i class="fas fa-chart-line"></i> Welcome to AI Trading Assistant</h2>
            </div>
            <div class="card-body">
                <p class="lead">Your intelligent companion for stock and cryptocurrency trading analysis.</p>
                <p>This AI-powered tool combines:</p>
                <ul>
                    <li><strong>Technical Analysis:</strong> Advanced indicators and pattern recognition</li>
                    <li><strong>Sentiment Analysis:</strong> Real-time news sentiment using FinBERT</li>
                    <li><strong>Machine Learning:</strong> LSTM neural networks for price prediction</li>
                    <li><strong>Educational Insights:</strong> Detailed explanations for every trading signal</li>
                </ul>
                <a href="/dashboard" class="btn btn-primary btn-lg">
                    <i class="fas fa-tachometer-alt"></i> Go to Dashboard
                </a>
            </div>
        </div>
    </div>
    <div class="col-md-4">
        <div class="card">
            <div class="card-header">
                <h4><i class="fas fa-info-circle"></i> Quick Start</h4>
            </div>
            <div class="card-body">
                <ol>
                    <li>Add stock symbols to track</li>
                    <li>Let AI analyze market data</li>
                    <li>Review trading signals</li>
                    <li>Learn from detailed explanations</li>
                </ol>
            </div>
        </div>
    </div>
</div>
{% endblock %}'''
    
    # Dashboard template
    dashboard_template = '''{% extends "base.html" %}
{% block title %}Trading Dashboard{% endblock %}
{% block content %}
<div class="row mb-4">
    <div class="col-md-8">
        <h2><i class="fas fa-tachometer-alt"></i> Trading Dashboard</h2>
    </div>
    <div class="col-md-4 text-end">
        <button class="btn btn-success" onclick="refreshSignals()">
            <i class="fas fa-sync-alt"></i> Refresh Signals
        </button>
    </div>
</div>

<div class="row mb-4">
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-plus"></i> Add Symbol</h5>
            </div>
            <div class="card-body">
                <div class="input-group">
                    <input type="text" class="form-control" id="newSymbol" placeholder="Enter symbol (e.g., AAPL)">
                    <button class="btn btn-primary" onclick="addSymbol()">Add</button>
                </div>
            </div>
        </div>
    </div>
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-chart-pie"></i> Portfolio Summary</h5>
            </div>
            <div class="card-body">
                <div id="portfolioSummary">Loading...</div>
            </div>
        </div>
    </div>
</div>

<div class="row">
    <div class="col-12">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-signal"></i> Trading Signals</h5>
            </div>
            <div class="card-body">
                <div id="signalsContainer">
                    {% if signals %}
                        {% for symbol, signal in signals.items() %}
                        <div class="alert alert-info">
                            <strong>{{ symbol }}:</strong> 
                            <span class="signal-{{ signal.signal }}">{{ signal.signal.upper() }}</span>
                            (Confidence: <span class="confidence-{{ 'high' if signal.confidence > 0.7 else 'medium' if signal.confidence > 0.5 else 'low' }}">{{ "%.1f"|format(signal.confidence * 100) }}%</span>)
                            <a href="/analysis/{{ symbol }}" class="btn btn-sm btn-outline-primary float-end">View Analysis</a>
                        </div>
                        {% endfor %}
                    {% else %}
                        <p>No symbols tracked yet. Add some symbols to get started!</p>
                    {% endif %}
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
function addSymbol() {
    const symbol = document.getElementById('newSymbol').value.trim().toUpperCase();
    if (!symbol) return;
    
    fetch('/api/add_symbol', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({symbol: symbol})
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            location.reload();
        } else {
            alert('Error: ' + data.error);
        }
    });
}

function refreshSignals() {
    fetch('/api/refresh_signals', {method: 'POST'})
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            location.reload();
        } else {
            alert('Error: ' + data.error);
        }
    });
}

// Load portfolio summary
fetch('/api/portfolio_analysis')
.then(response => response.json())
.then(data => {
    if (data.success) {
        const portfolio = data.portfolio;
        document.getElementById('portfolioSummary').innerHTML = `
            <p><strong>Total Symbols:</strong> ${portfolio.total_symbols}</p>
            <p><strong>Buy Signals:</strong> <span class="signal-buy">${portfolio.buy_signals}</span></p>
            <p><strong>Sell Signals:</strong> <span class="signal-sell">${portfolio.sell_signals}</span></p>
            <p><strong>Hold Signals:</strong> <span class="signal-hold">${portfolio.hold_signals}</span></p>
            <p><strong>Avg Confidence:</strong> ${(portfolio.average_confidence * 100).toFixed(1)}%</p>
        `;
    }
});
</script>
{% endblock %}'''
    
    # Write templates
    with open(os.path.join(templates_dir, 'base.html'), 'w') as f:
        f.write(base_template)
    
    with open(os.path.join(templates_dir, 'index.html'), 'w') as f:
        f.write(index_template)
    
    with open(os.path.join(templates_dir, 'dashboard.html'), 'w') as f:
        f.write(dashboard_template)
    
    # Create other basic templates
    templates = {
        'analysis.html': '''{% extends "base.html" %}
{% block title %}Analysis - {{ symbol }}{% endblock %}
{% block content %}
<h2>Analysis for {{ symbol }}</h2>
{% if has_data %}
<div id="chartContainer"></div>
<div id="signalDetails"></div>
{% else %}
<p>No data available for {{ symbol }}</p>
{% endif %}
{% endblock %}''',
        
        'settings.html': '''{% extends "base.html" %}
{% block title %}Settings{% endblock %}
{% block content %}
<h2>Bot Settings</h2>
<form id="configForm">
    <div class="mb-3">
        <label class="form-label">Lookback Days</label>
        <input type="number" class="form-control" name="lookback_days" value="{{ config.lookback_days }}">
    </div>
    <button type="submit" class="btn btn-primary">Save Settings</button>
</form>
{% endblock %}''',
        
        'about.html': '''{% extends "base.html" %}
{% block title %}About{% endblock %}
{% block content %}
<h2>About AI Trading Assistant</h2>
<p>This is an educational tool that demonstrates AI-powered trading analysis.</p>
<p><strong>Disclaimer:</strong> This tool is for educational purposes only. Always do your own research.</p>
{% endblock %}''',
        
        '404.html': '''{% extends "base.html" %}
{% block title %}Page Not Found{% endblock %}
{% block content %}
<h2>404 - Page Not Found</h2>
<p>The page you're looking for doesn't exist.</p>
{% endblock %}''',
        
        '500.html': '''{% extends "base.html" %}
{% block title %}Server Error{% endblock %}
{% block content %}
<h2>500 - Server Error</h2>
<p>Something went wrong on our end. Please try again later.</p>
{% endblock %}'''
    }
    
    for filename, content in templates.items():
        with open(os.path.join(templates_dir, filename), 'w') as f:
            f.write(content)

if __name__ == '__main__':
    # Create templates if they don't exist
    create_templates()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
