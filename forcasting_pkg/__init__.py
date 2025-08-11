"""
Forcasting Package - A comprehensive forecasting and technical analysis toolkit.

This package provides tools for financial forecasting, technical analysis,
and data visualization optimized for real-world applications.
"""

__version__ = "0.1.0"
__author__ = "Forcasting Package Contributors"
__email__ = "forcasting-pkg@example.com"

from .forecasting import ForecastingEngine
from .analysis import TechnicalAnalyzer
from .data import DataSource
from .models import ForecastResult, TechnicalIndicators

__all__ = [
    "ForecastingEngine",
    "TechnicalAnalyzer", 
    "DataSource",
    "ForecastResult",
    "TechnicalIndicators",
]