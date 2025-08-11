"""Setup configuration for the forecasting package."""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="forcasting-pkg",
    version="0.1.0",
    author="Forcasting Package Contributors", 
    author_email="forcasting-pkg@example.com",
    description="A comprehensive forecasting and technical analysis toolkit for financial markets",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/example/forcasting-pkg",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "full": [
            "statsmodels>=0.13.0",
            "matplotlib>=3.5.0",
            "plotly>=5.0.0",
            "jupyter>=1.0.0",
            "seaborn>=0.11.0",
        ],
        "visualization": [
            "matplotlib>=3.5.0",
            "plotly>=5.0.0",
            "seaborn>=0.11.0",
        ],
        "stats": [
            "statsmodels>=0.13.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "forcasting-cli=forcasting_pkg.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "forcasting_pkg": [
            "examples/*.py",
            "examples/*.ipynb",
        ],
    },
    keywords=[
        "forecasting",
        "finance", 
        "trading",
        "technical-analysis",
        "arima",
        "machine-learning",
        "time-series",
        "stocks",
        "cryptocurrency",
        "market-analysis",
    ],
    project_urls={
        "Bug Reports": "https://github.com/example/forcasting-pkg/issues",
        "Source": "https://github.com/example/forcasting-pkg",
        "Documentation": "https://forcasting-pkg.readthedocs.io/",
    },
)