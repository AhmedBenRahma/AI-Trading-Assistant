@echo off
REM AI Trading Assistant Deployment Script for Windows
REM This script sets up the project on a new Windows machine

echo 🚀 AI Trading Assistant - Deployment Script
echo =============================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.9+ first.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set python_version=%%i
echo ✅ Python %python_version% detected

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv .venv

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo 📚 Installing dependencies...
pip install -r requirements.txt

REM Install in development mode
echo 🔧 Installing in development mode...
pip install -e .

REM Download NLTK data
echo 📖 Downloading NLTK data...
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('vader_lexicon', quiet=True); print('✅ NLTK data downloaded successfully')"

echo.
echo 🎉 Deployment completed successfully!
echo.
echo 📋 Next steps:
echo    1. Activate virtual environment: .venv\Scripts\activate.bat
echo    2. Test installation: python example_usage.py
echo    3. Run Flask app: python app.py
echo    4. Run Streamlit app: streamlit run streamlit_app.py
echo.
echo 💡 To activate virtual environment on this machine:
echo    .venv\Scripts\activate.bat
echo.
echo 🔗 For more information, see README.md
pause
