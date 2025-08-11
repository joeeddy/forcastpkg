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

# Crypto module (optional import to handle missing dependencies)
try:
    from .crypto import CryptoBreakoutPipeline, MEXCDataSource, BreakoutDetector
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    CryptoBreakoutPipeline = None
    MEXCDataSource = None
    BreakoutDetector = None

__all__ = [
    "ForecastingEngine",
    "TechnicalAnalyzer", 
    "DataSource",
    "ForecastResult",
    "TechnicalIndicators",
]

# Add crypto components if available
if CRYPTO_AVAILABLE:
    __all__.extend([
        "CryptoBreakoutPipeline",
        "MEXCDataSource", 
        "BreakoutDetector",
    ])