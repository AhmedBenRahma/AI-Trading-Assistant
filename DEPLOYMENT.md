# 🚀 AI Trading Assistant - Deployment Guide

This guide will help you deploy the AI Trading Assistant on different machines and operating systems.

## 📋 Prerequisites

- **Python 3.9+** installed on your system
- **Git** for cloning the repository
- **Internet connection** for downloading dependencies
- **8GB+ RAM** recommended for LSTM model training

## 🖥️ Operating System Support

- ✅ **macOS** (10.14+)
- ✅ **Ubuntu/Debian** (18.04+)
- ✅ **CentOS/RHEL** (7+)
- ✅ **Windows** (10+)
- ✅ **WSL** (Windows Subsystem for Linux)

## 🚀 Quick Deployment

### **Option 1: Automated Scripts (Recommended)**

#### **macOS/Linux**
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/AI-Trading-Assistant.git
cd AI-Trading-Assistant

# Make script executable and run
chmod +x deploy.sh
./deploy.sh
```

#### **Windows**
```cmd
# Clone the repository
git clone https://github.com/YOUR_USERNAME/AI-Trading-Assistant.git
cd AI-Trading-Assistant

# Run the batch file
deploy.bat
```

### **Option 2: Manual Setup**

#### **1. Clone Repository**
```bash
git clone https://github.com/YOUR_USERNAME/AI-Trading-Assistant.git
cd AI-Trading-Assistant
```

#### **2. Create Virtual Environment**
```bash
# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

#### **3. Install Dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

#### **4. Download NLTK Data**
```bash
python -c "
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)
print('NLTK data downloaded successfully')
"
```

## 🔧 Platform-Specific Instructions

### **macOS**

#### **Install Python (if not already installed)**
```bash
# Using Homebrew
brew install python@3.11

# Or download from python.org
# https://www.python.org/downloads/macos/
```

#### **Install Git (if not already installed)**
```bash
brew install git
```

### **Ubuntu/Debian**

#### **Install System Dependencies**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv git build-essential
```

#### **Install Python 3.11 (if needed)**
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-pip
```

### **CentOS/RHEL**

#### **Install System Dependencies**
```bash
sudo yum update
sudo yum install python3 python3-pip python3-venv git gcc
```

### **Windows**

#### **Install Python**
1. Download from [python.org](https://www.python.org/downloads/)
2. Check "Add Python to PATH" during installation
3. Install Git from [git-scm.com](https://git-scm.com/)

#### **Using WSL (Recommended for Windows)**
```bash
# Install WSL2
wsl --install

# Restart and open WSL
# Follow Ubuntu instructions above
```

## 📦 Dependency Management

### **Virtual Environment Best Practices**
```bash
# Always activate before working
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows

# Deactivate when done
deactivate
```

### **Updating Dependencies**
```bash
# Update all packages
pip list --outdated
pip install --upgrade package_name

# Update requirements.txt
pip freeze > requirements.txt
```

### **Troubleshooting Dependencies**
```bash
# Clear pip cache
pip cache purge

# Reinstall specific package
pip uninstall package_name
pip install package_name

# Check package compatibility
pip check
```

## 🧪 Testing the Installation

### **1. Basic Test**
```bash
# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows

# Test imports
python -c "
import pandas, numpy, yfinance
import tensorflow, transformers
print('✅ All packages imported successfully!')
"
```

### **2. Run Example Script**
```bash
python example_usage.py
```

### **3. Test Web Interfaces**
```bash
# Flask app
python app.py

# Streamlit app
streamlit run streamlit_app.py
```

## 🔄 Updating the Project

### **Pull Latest Changes**
```bash
# Activate virtual environment first
source .venv/bin/activate

# Pull latest changes
git pull origin main

# Update dependencies if needed
pip install -r requirements.txt
```

### **Reset to Clean State**
```bash
# Remove virtual environment
rm -rf .venv

# Recreate and reinstall
./deploy.sh  # macOS/Linux
# or
deploy.bat   # Windows
```

## 🚨 Common Issues and Solutions

### **1. Memory Issues**
```bash
# Reduce batch size in prediction_model.py
lstm.batch_size = 16  # Default is 32

# Reduce lookback days
config['lookback_days'] = 20  # Default is 30
```

### **2. GPU Issues**
```bash
# Force CPU usage
export CUDA_VISIBLE_DEVICES=""
# or
set CUDA_VISIBLE_DEVICES=""  # Windows
```

### **3. Network Issues**
```bash
# Use alternative package sources
pip install -i https://pypi.org/simple/ package_name

# Or use conda
conda install package_name
```

### **4. Permission Issues**
```bash
# Fix script permissions
chmod +x deploy.sh

# Fix virtual environment permissions
chmod -R 755 .venv/
```

## 📱 Running on Different Machines

### **Local Development Machine**
```bash
# Full development setup
pip install -e .[dev]
```

### **Production Server**
```bash
# Production dependencies only
pip install -r requirements.txt
```

### **Cloud Platforms**

#### **Google Colab**
```bash
# Install directly in notebook
!pip install git+https://github.com/YOUR_USERNAME/AI-Trading-Assistant.git
```

#### **AWS/GCP/Azure**
```bash
# Clone and deploy as usual
git clone https://github.com/YOUR_USERNAME/AI-Trading-Assistant.git
cd AI-Trading-Assistant
./deploy.sh
```

## 🔐 Security Considerations

### **API Keys and Secrets**
- Never commit API keys to Git
- Use environment variables for sensitive data
- Add `*.env` to `.gitignore`

### **Virtual Environment Security**
- Keep virtual environment private
- Don't share `.venv` directory
- Use different environments for different projects

## 📊 Performance Optimization

### **System Requirements**
- **RAM**: 8GB minimum, 16GB+ recommended
- **CPU**: 4+ cores recommended
- **Storage**: 10GB+ free space
- **GPU**: Optional but recommended for LSTM training

### **Optimization Tips**
```bash
# Use GPU if available
pip install tensorflow[gpu]

# Optimize for your CPU
pip install tensorflow-cpu

# Use smaller models for testing
config['n_selected_features'] = 15  # Default is 25
```

## 🔗 Useful Commands

### **Development Commands**
```bash
# Run tests
python -m pytest

# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

### **Deployment Commands**
```bash
# Create distribution
python setup.py sdist bdist_wheel

# Install from distribution
pip install dist/ai_trading_assistant-1.0.0.tar.gz
```

## 📞 Getting Help

### **Check Logs**
```bash
# Python errors
python script.py 2>&1 | tee error.log

# Package issues
pip install -v package_name
```

### **Common Resources**
- [Python Documentation](https://docs.python.org/)
- [TensorFlow Guide](https://www.tensorflow.org/guide)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Flask Documentation](https://flask.palletsprojects.com/)

### **Project Issues**
- Check the [GitHub Issues](https://github.com/YOUR_USERNAME/AI-Trading-Assistant/issues)
- Review the [README.md](README.md)
- Check the [example_usage.py](example_usage.py) file

## 🎯 Next Steps After Deployment

1. **Test Basic Functionality**: Run `python example_usage.py`
2. **Explore Web Interfaces**: Try both Flask and Streamlit apps
3. **Add Your Symbols**: Start tracking your favorite stocks/crypto
4. **Customize Settings**: Adjust configuration in `trading_bot.py`
5. **Learn the System**: Read through the detailed explanations
6. **Contribute**: Fork the repository and make improvements

---

**Happy Trading and Learning! 🚀📈**

Remember: This tool is for educational purposes. Always do your own research and consider consulting with a financial advisor.
