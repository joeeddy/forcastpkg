"""Visualization tools for forecasting and technical analysis."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple, TYPE_CHECKING

# Type imports that might not be available at runtime
if TYPE_CHECKING:
    try:
        from matplotlib.figure import Figure
    except ImportError:
        Figure = Any
    try:
        import plotly.graph_objects as go
    except ImportError:
        go = Any
from datetime import datetime, timedelta

from ..models import ForecastResult, MarketData, TechnicalIndicators

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure as MPLFigure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    MPLFigure = Any

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
    PlotlyFigure = go.Figure
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    PlotlyFigure = Any


class ForecastVisualizer:
    """Visualization tools for forecasting results and market data."""
    
    def __init__(self, style: str = "default", figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            style: Plotting style ('default', 'dark', 'seaborn', etc.)
            figsize: Figure size for matplotlib plots
        """
        self.style = style
        self.figsize = figsize
        
        if MATPLOTLIB_AVAILABLE:
            plt.style.use(style if style != "default" else "seaborn-v0_8" if "seaborn" in plt.style.available else "default")
    
    def plot_forecast(
        self, 
        historical_data: Union[List[MarketData], pd.DataFrame],
        forecast_result: ForecastResult,
        symbol: str = None,
        save_path: Optional[str] = None,
        show_confidence: bool = True,
        interactive: bool = False
    ) -> Optional[Union[MPLFigure, PlotlyFigure]]:
        """
        Plot historical data with forecast predictions.
        
        Args:
            historical_data: Historical market data
            forecast_result: Forecast results to plot
            symbol: Symbol name for title
            save_path: Path to save the plot
            show_confidence: Whether to show confidence intervals
            interactive: Use plotly for interactive plot
            
        Returns:
            Figure object or None if plotting libraries not available
        """
        if interactive and PLOTLY_AVAILABLE:
            return self._plot_forecast_plotly(historical_data, forecast_result, symbol, save_path, show_confidence)
        elif MATPLOTLIB_AVAILABLE:
            return self._plot_forecast_matplotlib(historical_data, forecast_result, symbol, save_path, show_confidence)
        else:
            print("No plotting libraries available. Install matplotlib or plotly for visualization.")
            return None
    
    def _plot_forecast_matplotlib(
        self, 
        historical_data: Union[List[MarketData], pd.DataFrame],
        forecast_result: ForecastResult,
        symbol: str = None,
        save_path: Optional[str] = None,
        show_confidence: bool = True
    ) -> MPLFigure:
        """Create forecast plot using matplotlib."""
        # Prepare historical data
        if isinstance(historical_data, list):
            hist_df = pd.DataFrame([item.dict() for item in historical_data])
        else:
            hist_df = historical_data.copy()
        
        hist_df['date'] = pd.to_datetime(hist_df['date'])
        
        # Prepare forecast data
        forecast_dates = [point.date for point in forecast_result.forecast_points]
        forecast_values = [point.predicted_value for point in forecast_result.forecast_points]
        
        if show_confidence:
            conf_lower = [point.confidence_interval_lower for point in forecast_result.forecast_points]
            conf_upper = [point.confidence_interval_upper for point in forecast_result.forecast_points]
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot historical data
        ax.plot(hist_df['date'], hist_df['close'], label='Historical Price', color='blue', linewidth=2)
        
        # Plot forecast
        ax.plot(forecast_dates, forecast_values, label=f'Forecast ({forecast_result.model_type})', 
                color='red', linewidth=2, linestyle='--')
        
        # Plot confidence intervals
        if show_confidence and all(conf_lower) and all(conf_upper):
            ax.fill_between(forecast_dates, conf_lower, conf_upper, 
                          alpha=0.3, color='red', label='Confidence Interval')
        
        # Formatting
        symbol_name = symbol or forecast_result.symbol
        ax.set_title(f'{symbol_name} - Forecast ({forecast_result.forecast_period_days} days)\n'
                    f'Model: {forecast_result.model_type} | '
                    f'Accuracy: {forecast_result.model_accuracy:.2%}' if forecast_result.model_accuracy else '')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_forecast_plotly(
        self, 
        historical_data: Union[List[MarketData], pd.DataFrame],
        forecast_result: ForecastResult,
        symbol: str = None,
        save_path: Optional[str] = None,
        show_confidence: bool = True
    ) -> PlotlyFigure:
        """Create interactive forecast plot using plotly."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for interactive plots. Install with: pip install plotly")
            
        # Prepare historical data
        if isinstance(historical_data, list):
            hist_df = pd.DataFrame([item.dict() for item in historical_data])
        else:
            hist_df = historical_data.copy()
        
        hist_df['date'] = pd.to_datetime(hist_df['date'])
        
        # Prepare forecast data
        forecast_dates = [point.date for point in forecast_result.forecast_points]
        forecast_values = [point.predicted_value for point in forecast_result.forecast_points]
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=hist_df['date'],
            y=hist_df['close'],
            mode='lines',
            name='Historical Price',
            line=dict(color='blue', width=2)
        ))
        
        # Forecast line
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            mode='lines',
            name=f'Forecast ({forecast_result.model_type})',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Confidence intervals
        if show_confidence:
            conf_lower = [point.confidence_interval_lower for point in forecast_result.forecast_points if point.confidence_interval_lower is not None]
            conf_upper = [point.confidence_interval_upper for point in forecast_result.forecast_points if point.confidence_interval_upper is not None]
            
            if conf_lower and conf_upper:
                # Upper bound
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=conf_upper,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Lower bound with fill
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=conf_lower,
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(255, 0, 0, 0.3)',
                    name='Confidence Interval',
                    hoverinfo='skip'
                ))
        
        # Layout
        symbol_name = symbol or forecast_result.symbol
        title = f'{symbol_name} - Forecast ({forecast_result.forecast_period_days} days)<br>'
        title += f'Model: {forecast_result.model_type}'
        if forecast_result.model_accuracy:
            title += f' | Accuracy: {forecast_result.model_accuracy:.2%}'
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified',
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_technical_indicators(
        self, 
        data: Union[List[MarketData], pd.DataFrame],
        indicators: TechnicalIndicators,
        symbol: str = "Unknown",
        save_path: Optional[str] = None,
        interactive: bool = False
    ) -> Optional[Union[MPLFigure, PlotlyFigure]]:
        """
        Plot technical indicators alongside price data.
        
        Args:
            data: Historical market data
            indicators: Technical indicators to plot
            symbol: Symbol name
            save_path: Path to save the plot
            interactive: Use plotly for interactive plot
            
        Returns:
            Figure object or None
        """
        if interactive and PLOTLY_AVAILABLE:
            return self._plot_technical_plotly(data, indicators, symbol, save_path)
        elif MATPLOTLIB_AVAILABLE:
            return self._plot_technical_matplotlib(data, indicators, symbol, save_path)
        else:
            print("No plotting libraries available.")
            return None
    
    def _plot_technical_matplotlib(
        self, 
        data: Union[List[MarketData], pd.DataFrame],
        indicators: TechnicalIndicators,
        symbol: str,
        save_path: Optional[str] = None
    ) -> MPLFigure:
        """Plot technical indicators using matplotlib."""
        # Prepare data
        if isinstance(data, list):
            df = pd.DataFrame([item.dict() for item in data])
        else:
            df = data.copy()
        
        df['date'] = pd.to_datetime(df['date'])
        
        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(self.figsize[0], self.figsize[1] * 1.2), sharex=True)
        
        # Price plot with moving averages
        axes[0].plot(df['date'], df['close'], label='Close Price', color='black', linewidth=2)
        
        if indicators.moving_average_20:
            axes[0].axhline(y=indicators.moving_average_20, color='blue', linestyle='--', label='MA 20')
        if indicators.moving_average_50:
            axes[0].axhline(y=indicators.moving_average_50, color='orange', linestyle='--', label='MA 50')
        
        # Bollinger Bands
        if all([indicators.bollinger_upper, indicators.bollinger_lower, indicators.bollinger_middle]):
            axes[0].axhline(y=indicators.bollinger_upper, color='gray', linestyle=':', alpha=0.7, label='BB Upper')
            axes[0].axhline(y=indicators.bollinger_lower, color='gray', linestyle=':', alpha=0.7, label='BB Lower')
            axes[0].axhline(y=indicators.bollinger_middle, color='gray', linestyle='-', alpha=0.7, label='BB Middle')
        
        axes[0].set_title(f'{symbol} - Technical Analysis')
        axes[0].set_ylabel('Price')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # RSI plot
        if indicators.rsi:
            axes[1].axhline(y=indicators.rsi, color='purple', linewidth=2, label=f'RSI: {indicators.rsi:.1f}')
            axes[1].axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
            axes[1].axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
            axes[1].set_ylim(0, 100)
        
        axes[1].set_ylabel('RSI')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # MACD plot
        if indicators.macd:
            axes[2].axhline(y=indicators.macd, color='blue', linewidth=2, label=f'MACD: {indicators.macd:.3f}')
            if indicators.macd_signal:
                axes[2].axhline(y=indicators.macd_signal, color='red', linewidth=2, label=f'Signal: {indicators.macd_signal:.3f}')
            if indicators.macd_histogram:
                axes[2].axhline(y=indicators.macd_histogram, color='green', linewidth=1, label=f'Histogram: {indicators.macd_histogram:.3f}')
        
        axes[2].set_ylabel('MACD')
        axes[2].set_xlabel('Date')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_technical_plotly(
        self, 
        data: Union[List[MarketData], pd.DataFrame],
        indicators: TechnicalIndicators,
        symbol: str,
        save_path: Optional[str] = None
    ) -> PlotlyFigure:
        """Plot technical indicators using plotly."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for interactive plots. Install with: pip install plotly")
            
        # Prepare data
        if isinstance(data, list):
            df = pd.DataFrame([item.dict() for item in data])
        else:
            df = data.copy()
        
        df['date'] = pd.to_datetime(df['date'])
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=['Price & Moving Averages', 'RSI', 'MACD'],
            vertical_spacing=0.1
        )
        
        # Price data
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['close'],
            mode='lines', name='Close Price',
            line=dict(color='black', width=2)
        ), row=1, col=1)
        
        # Moving averages
        if indicators.moving_average_20:
            fig.add_hline(y=indicators.moving_average_20, line_dash="dash", line_color="blue", row=1, col=1)
        if indicators.moving_average_50:
            fig.add_hline(y=indicators.moving_average_50, line_dash="dash", line_color="orange", row=1, col=1)
        
        # RSI
        if indicators.rsi:
            fig.add_hline(y=indicators.rsi, line_color="purple", line_width=3, row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        if indicators.macd:
            fig.add_hline(y=indicators.macd, line_color="blue", line_width=3, row=3, col=1)
            if indicators.macd_signal:
                fig.add_hline(y=indicators.macd_signal, line_color="red", line_width=2, row=3, col=1)
        
        fig.update_layout(
            title=f'{symbol} - Technical Analysis',
            showlegend=True,
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_model_comparison(
        self,
        historical_data: Union[List[MarketData], pd.DataFrame],
        forecast_results: Dict[str, ForecastResult],
        symbol: str = "Unknown",
        save_path: Optional[str] = None,
        interactive: bool = False
    ) -> Optional[Union[MPLFigure, PlotlyFigure]]:
        """
        Plot comparison of multiple forecasting models.
        
        Args:
            historical_data: Historical market data
            forecast_results: Dictionary of model names to forecast results
            symbol: Symbol name
            save_path: Path to save the plot
            interactive: Use plotly for interactive plot
            
        Returns:
            Figure object or None
        """
        if not forecast_results:
            print("No forecast results to compare")
            return None
        
        if interactive and PLOTLY_AVAILABLE:
            return self._plot_comparison_plotly(historical_data, forecast_results, symbol, save_path)
        elif MATPLOTLIB_AVAILABLE:
            return self._plot_comparison_matplotlib(historical_data, forecast_results, symbol, save_path)
        else:
            print("No plotting libraries available.")
            return None
    
    def _plot_comparison_matplotlib(
        self,
        historical_data: Union[List[MarketData], pd.DataFrame],
        forecast_results: Dict[str, ForecastResult],
        symbol: str,
        save_path: Optional[str] = None
    ) -> MPLFigure:
        """Plot model comparison using matplotlib."""
        # Prepare historical data
        if isinstance(historical_data, list):
            hist_df = pd.DataFrame([item.dict() for item in historical_data])
        else:
            hist_df = historical_data.copy()
        
        hist_df['date'] = pd.to_datetime(hist_df['date'])
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot historical data
        ax.plot(hist_df['date'], hist_df['close'], label='Historical Price', color='black', linewidth=2)
        
        # Colors for different models
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        # Plot each forecast
        for i, (model_name, forecast) in enumerate(forecast_results.items()):
            color = colors[i % len(colors)]
            
            forecast_dates = [point.date for point in forecast.forecast_points]
            forecast_values = [point.predicted_value for point in forecast.forecast_points]
            
            accuracy_text = f" (Acc: {forecast.model_accuracy:.2%})" if forecast.model_accuracy else ""
            ax.plot(forecast_dates, forecast_values, label=f'{model_name}{accuracy_text}', 
                   color=color, linewidth=2, linestyle='--')
        
        ax.set_title(f'{symbol} - Model Comparison')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_comparison_plotly(
        self,
        historical_data: Union[List[MarketData], pd.DataFrame],
        forecast_results: Dict[str, ForecastResult],
        symbol: str,
        save_path: Optional[str] = None
    ) -> PlotlyFigure:
        """Plot model comparison using plotly."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for interactive plots. Install with: pip install plotly")
            
        # Prepare historical data
        if isinstance(historical_data, list):
            hist_df = pd.DataFrame([item.dict() for item in historical_data])
        else:
            hist_df = historical_data.copy()
        
        hist_df['date'] = pd.to_datetime(hist_df['date'])
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=hist_df['date'],
            y=hist_df['close'],
            mode='lines',
            name='Historical Price',
            line=dict(color='black', width=2)
        ))
        
        # Colors for different models
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        # Plot each forecast
        for i, (model_name, forecast) in enumerate(forecast_results.items()):
            color = colors[i % len(colors)]
            
            forecast_dates = [point.date for point in forecast.forecast_points]
            forecast_values = [point.predicted_value for point in forecast.forecast_points]
            
            accuracy_text = f" (Acc: {forecast.model_accuracy:.2%})" if forecast.model_accuracy else ""
            
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_values,
                mode='lines',
                name=f'{model_name}{accuracy_text}',
                line=dict(color=color, width=2, dash='dash')
            ))
        
        fig.update_layout(
            title=f'{symbol} - Model Comparison',
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified',
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig


# Convenience functions
def plot_forecast(
    historical_data: Union[List[MarketData], pd.DataFrame],
    forecast_result: ForecastResult,
    symbol: str = None,
    save_path: Optional[str] = None,
    interactive: bool = False
) -> Optional[Union[MPLFigure, PlotlyFigure]]:
    """Convenience function to plot forecast."""
    visualizer = ForecastVisualizer()
    return visualizer.plot_forecast(historical_data, forecast_result, symbol, save_path, interactive=interactive)


def plot_technical_indicators(
    data: Union[List[MarketData], pd.DataFrame],
    indicators: TechnicalIndicators,
    symbol: str = "Unknown",
    save_path: Optional[str] = None,
    interactive: bool = False
) -> Optional[Union[MPLFigure, PlotlyFigure]]:
    """Convenience function to plot technical indicators."""
    visualizer = ForecastVisualizer()
    return visualizer.plot_technical_indicators(data, indicators, symbol, save_path, interactive=interactive)


__all__ = [
    "ForecastVisualizer",
    "plot_forecast",
    "plot_technical_indicators"
]