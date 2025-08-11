# Forcasting Package

A comprehensive Python package for financial forecasting and technical analysis, optimized for real-world applications.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/forcasting-pkg.svg)](https://pypi.org/project/forcasting-pkg/)

## 🚀 Features

### Forecasting Models
- **ARIMA**: Advanced AutoRegressive Integrated Moving Average with automatic parameter selection
- **Linear Regression**: Polynomial regression with trend decomposition
- **Moving Averages**: Simple, Exponential, and Adaptive moving average forecasting
- **Ensemble Methods**: Combine multiple models for improved accuracy

### Cryptocurrency Analytics
- **MEXC Exchange Integration**: Real-time data fetching from MEXC cryptocurrency exchange
- **Breakout Detection**: Automated detection of price breakouts and volume spikes
- **Crypto Forecasting Pipeline**: End-to-end workflow for crypto analysis and prediction
- **Multi-Symbol Scanning**: Analyze multiple cryptocurrencies simultaneously

### Technical Analysis
- **Indicators**: RSI, MACD, Bollinger Bands, Stochastic, Williams %R, CCI
- **Signal Generation**: Automated trading signal generation with confidence levels
- **Trend Analysis**: Multi-timeframe trend identification and strength measurement

### Data Sources & Visualization
- **Multi-Source Data**: Yahoo Finance integration with fallback mechanisms
- **Interactive Charts**: Plotly and Matplotlib support for publication-ready visualizations
- **Real-time Data**: Current price fetching and historical data management

### Developer-Friendly Features
- **CLI Interface**: Command-line tools for quick analysis and automation
- **Extensible Architecture**: Easy integration of custom models and data sources
- **Type Safety**: Full Pydantic model validation and type hints
- **Comprehensive Testing**: Unit tests and example notebooks included

## 📦 Installation

### Basic Installation
```bash
pip install forcasting-pkg
```

### Full Installation (with all optional dependencies)
```bash
pip install forcasting-pkg[full]
```

### Crypto Features Installation
```bash
pip install forcasting-pkg[crypto]
```

### Development Installation
```bash
git clone https://github.com/example/forcasting-pkg.git
cd forcasting-pkg
pip install -e .[dev]
```

## 🔧 Quick Start

### Basic Forecasting
```python
from forcasting_pkg import ForecastingEngine
from forcasting_pkg.data import get_historical_data

# Get historical data
data = get_historical_data("AAPL", days=90)

# Create forecasting engine
engine = ForecastingEngine()

# Generate 30-day ARIMA forecast
forecast = engine.forecast(data, model="arima", symbol="AAPL", days=30)

print(f"Model: {forecast.model_type}")
print(f"Accuracy: {forecast.model_accuracy:.2%}")
print(f"Next day prediction: ${forecast.forecast_points[0].predicted_value:.2f}")
```

### Cryptocurrency Breakout Detection
```python
from forcasting_pkg.crypto import CryptoBreakoutPipeline

# Create crypto breakout detection pipeline
pipeline = CryptoBreakoutPipeline(use_mock_data=True)  # Set False for real data

# Run complete analysis on top cryptocurrencies  
results = pipeline.run_full_pipeline(
    symbol_limit=20,      # Analyze top 20 by volume
    historical_days=30,   # 30 days of data
    forecast_days=7,      # 7-day forecasts
    min_signal_strength=0.6,  # Only strong signals
    max_breakout_candidates=5  # Forecast top 5 candidates
)

# Display results
print(f"Found {len(results['breakout_signals'])} breakout signals")
print(f"Generated {len(results['forecasts'])} forecasts")

# Get pipeline summary with CryptoBreakoutPipeline
summary = pipeline.get_pipeline_summary(results)
print(f"Average signal strength: {summary['signal_summary']['avg_strength']:.3f}")
```

### Technical Analysis
```python
from forcasting_pkg import TechnicalAnalyzer
from forcasting_pkg.data import get_historical_data

# Get data and analyze
data = get_historical_data("TSLA", days=60)
analyzer = TechnicalAnalyzer()
analysis = analyzer.analyze(data, "TSLA")

print(f"RSI: {analysis.technical_indicators.rsi:.2f}")
print(f"Signal: {analysis.signal_strength}")
print(f"Confidence: {analysis.confidence:.2%}")
```

### Cryptocurrency CLI Tools
```bash
# Run crypto breakout detection and forecasting
python scripts/run_crypto_breakouts.py --symbols 50 --days 30 --forecast 7

# Use mock data for testing
python scripts/run_crypto_breakouts.py --mock --symbols 10 --candidates 3

# Export results to CSV
python scripts/run_crypto_breakouts.py --symbols 30 --output crypto_analysis_results/
```

### Technical Analysis
```python
from forcasting_pkg.visualization import plot_forecast, plot_technical_indicators

# Plot forecast
plot_forecast(data, forecast, symbol="AAPL", save_path="forecast.png")

# Plot technical indicators
plot_technical_indicators(data, analysis.technical_indicators, 
                         symbol="AAPL", save_path="technical.png")
```

## 🖥️ Command Line Interface

### Generate Forecasts
```bash
# ARIMA forecast for Apple stock
forcasting-cli forecast AAPL --model arima --days 30 --plot forecast.png

# Bitcoin forecast with multiple models comparison
forcasting-cli compare BTC --crypto --models arima,linear,moving_average

# Technical analysis with interactive plot
forcasting-cli analyze MSFT --plot analysis.html --interactive

# Crypto breakout detection and forecasting
python scripts/run_crypto_breakouts.py --symbols 20 --forecast 7 --output results/
```

### Available Commands
```bash
forcasting-cli forecast    # Generate forecasts
forcasting-cli analyze     # Technical analysis  
forcasting-cli compare     # Compare models
forcasting-cli info        # Package information
```

## 📊 Supported Models

### Forecasting Models

| Model | Description | Best For | Parameters |
|-------|-------------|----------|------------|
| **ARIMA** | AutoRegressive Integrated Moving Average | Stationary time series, trending data | Auto-selected (p,d,q) |
| **Linear** | Polynomial regression with trend | Linear trends, simple patterns | Degree (1-5) |
| **Moving Average** | Simple/Exponential/Adaptive MA | Smooth data, short-term forecasts | Window size |
| **Ensemble** | Combination of multiple models | Robust predictions, uncertain markets | Model weights |

### Technical Indicators

| Indicator | Range | Signals |
|-----------|-------|---------|
| **RSI** | 0-100 | Overbought (>70), Oversold (<30) |
| **MACD** | Unbounded | Line crossovers, trend changes |
| **Bollinger Bands** | Price-based | Breakouts, mean reversion |
| **Stochastic** | 0-100 | Momentum, reversal points |

## 🔌 Extensibility

### Custom Data Sources
```python
from forcasting_pkg.data import DataSource
from forcasting_pkg.models import MarketData

class CustomDataSource(DataSource):
    def get_historical_data(self, symbol, days, data_type):
        # Your custom data fetching logic
        return [MarketData(...)]
    
    def get_current_price(self, symbol, data_type):
        # Your custom price fetching logic
        return price

# Register with data manager
from forcasting_pkg.data import default_data_source
default_data_source.add_source(CustomDataSource(), priority=0)
```

### Custom Forecasting Models
```python
from forcasting_pkg.forecasting import BaseForecaster
from forcasting_pkg.models import ForecastResult

class CustomForecaster(BaseForecaster):
    def fit(self, data, symbol):
        # Your model training logic
        self.is_fitted = True
        return self
    
    def predict(self, steps):
        # Your prediction logic
        return ForecastResult(...)

# Register with engine
engine = ForecastingEngine()
engine.add_custom_model("custom", CustomForecaster)
```

## 📈 Examples & Tutorials

### Jupyter Notebooks
- [Basic Forecasting Tutorial](examples/basic_forecasting.ipynb)
- [Technical Analysis Guide](examples/technical_analysis.ipynb)
- [Model Comparison Study](examples/model_comparison.ipynb)
- [Custom Model Development](examples/custom_models.ipynb)

### Python Scripts
- [Automated Trading Signals](examples/trading_signals.py)
- [Portfolio Analysis](examples/portfolio_analysis.py)
- [Backtesting Framework](examples/backtesting.py)

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v --cov=forcasting_pkg
```

Run specific test categories:
```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Performance tests
pytest tests/performance/ -v
```

## 📚 Documentation

### API Reference
- [Forecasting Engine](docs/api/forecasting.md)
- [Technical Analysis](docs/api/analysis.md)
- [Data Sources](docs/api/data.md)
- [Visualization](docs/api/visualization.md)

### Guides
- [Getting Started](docs/guides/getting-started.md)
- [Model Selection](docs/guides/model-selection.md)
- [Performance Optimization](docs/guides/performance.md)
- [Production Deployment](docs/guides/deployment.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/example/forcasting-pkg.git
cd forcasting-pkg
pip install -e .[dev]
pre-commit install
```

### Code Quality
```bash
# Format code
black forcasting_pkg/

# Lint code  
flake8 forcasting_pkg/

# Type checking
mypy forcasting_pkg/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of excellent open-source libraries: pandas, scikit-learn, matplotlib, plotly
- Inspired by financial analysis tools and academic research in time series forecasting
- Special thanks to the Python community for providing robust numerical computing foundations

## 📞 Support

- **Documentation**: [https://forcasting-pkg.readthedocs.io/](https://forcasting-pkg.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/example/forcasting-pkg/issues)
- **Discussions**: [GitHub Discussions](https://github.com/example/forcasting-pkg/discussions)
- **Email**: forcasting-pkg@example.com

---

**Disclaimer**: This package is for educational and research purposes. Always do your own research before making investment decisions. Past performance does not guarantee future results.