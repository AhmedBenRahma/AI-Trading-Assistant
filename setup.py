"""
Setup script for AI Trading Assistant
Install with: pip install -e .
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-trading-assistant",
    version="1.0.0",
    author="AI Trading Assistant Team",
    author_email="your-email@example.com",
    description="A comprehensive AI-powered trading assistant for stock and cryptocurrency markets",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/AI-Trading-Assistant",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "black",
            "flake8",
            "pytest",
            "pytest-cov",
            "mypy",
            "pre-commit",
        ],
        "web": [
            "flask",
            "streamlit",
            "plotly",
        ],
        "ml": [
            "tensorflow",
            "torch",
            "transformers",
            "scikit-learn",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-trading=example_usage:main",
            "ai-trading-flask=app:main",
            "ai-trading-streamlit=streamlit_app:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="trading, finance, ai, machine-learning, sentiment-analysis, technical-analysis",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/AI-Trading-Assistant/issues",
        "Source": "https://github.com/yourusername/AI-Trading-Assistant",
        "Documentation": "https://github.com/yourusername/AI-Trading-Assistant#readme",
    },
)
