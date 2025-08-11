"""
Crypto breakout detection and forecasting module.

This module provides functionality for cryptocurrency breakout detection
and integration with forecasting models, specifically focused on MEXC exchange data.
"""

from .data_ingestion import MEXCDataSource
from .breakout_detection import BreakoutDetector
from .pipeline import CryptoBreakoutPipeline

__all__ = [
    "MEXCDataSource",
    "BreakoutDetector", 
    "CryptoBreakoutPipeline",
]