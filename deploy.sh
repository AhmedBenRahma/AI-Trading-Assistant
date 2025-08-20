#!/bin/bash

# AI Trading Assistant Deployment Script
# This script sets up the project on a new machine

echo "🚀 AI Trading Assistant - Deployment Script"
echo "============================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9+ first."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $python_version detected. Python $required_version+ is required."
    exit 1
fi

echo "✅ Python $python_version detected"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Install in development mode
echo "🔧 Installing in development mode..."
pip install -e .

# Download NLTK data
echo "📖 Downloading NLTK data..."
python3 -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    print('✅ NLTK data downloaded successfully')
except Exception as e:
    print(f'⚠️ Warning: Could not download NLTK data: {e}')
"

echo ""
echo "🎉 Deployment completed successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. Activate virtual environment: source .venv/bin/activate"
echo "   2. Test installation: python example_usage.py"
echo "   3. Run Flask app: python app.py"
echo "   4. Run Streamlit app: streamlit run streamlit_app.py"
echo ""
echo "💡 To activate virtual environment on this machine:"
echo "   source .venv/bin/activate"
echo ""
echo "🔗 For more information, see README.md"
