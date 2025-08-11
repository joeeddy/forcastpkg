"""Examples module for the forecasting package."""

# Import main example functions for easy access
from .basic_usage import (
    basic_forecasting_example,
    technical_analysis_example,
    model_comparison_example,
    crypto_forecasting_example,
    ensemble_forecasting_example,
    visualization_example
)

from .advanced_usage import (
    advanced_data_source_example,
    custom_model_integration_example,
    portfolio_analysis_example,
    backtesting_example,
    configuration_example,
    export_import_example
)

__all__ = [
    # Basic examples
    "basic_forecasting_example",
    "technical_analysis_example", 
    "model_comparison_example",
    "crypto_forecasting_example",
    "ensemble_forecasting_example",
    "visualization_example",
    
    # Advanced examples
    "advanced_data_source_example",
    "custom_model_integration_example",
    "portfolio_analysis_example",
    "backtesting_example",
    "configuration_example",
    "export_import_example"
]