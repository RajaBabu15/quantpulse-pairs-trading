"""
Visualization Module
==================

Professional trading dashboards, charts, and visualization tools for
quantitative analysis and portfolio monitoring.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.stats import gaussian_kde

# Try to import plotly for interactive charts
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Interactive charts will be disabled.")

from .analytics import PerformanceAnalytics, RiskMetrics, PerformanceAttribution
from .backtesting import BacktestResults

import logging
logger = logging.getLogger(__name__)

class TradingDashboard:
    """Professional trading dashboard with comprehensive visualizations."""
    
    def __init__(self, results: BacktestResults, benchmark_data: pd.DataFrame = None):
        print(f"🔄 ENTERING TradingDashboard.__init__() at {datetime.now().strftime('%H:%M:%S')}")
        
        self.results = results
        self.benchmark_data = benchmark_data
        self.analytics = PerformanceAnalytics(results, benchmark_data)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Color scheme
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'background': '#f8f9fa',
            'text': '#212529'
        }
        
        print(f"✅ EXITING TradingDashboard.__init__() at {datetime.now().strftime('%H:%M:%S')}")
    
    def create_performance_overview(self, figsize=(16, 12)) -> plt.Figure:
        """Create comprehensive performance overview dashboard."""
        print(f"🔄 ENTERING create_performance_overview() at {datetime.now().strftime('%H:%M:%S')}")
        
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Portfolio Value Chart (Top row, spans 3 columns)
        ax1 = fig.add_subplot(gs[0, :3])
        self._plot_portfolio_value(ax1)
        
        # 2. Key Metrics Table (Top right)
        ax2 = fig.add_subplot(gs[0, 3])
        self._plot_key_metrics_table(ax2)
        
        # 3. Monthly Returns Heatmap (Second row, spans 2 columns)
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_monthly_returns_heatmap(ax3)
        
        # 4. Drawdown Chart (Second row, right 2 columns)
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_drawdown_chart(ax4)
        
        # 5. Rolling Metrics (Third row, left 2 columns)
        ax5 = fig.add_subplot(gs[2, :2])
        self._plot_rolling_metrics(ax5)
        
        # 6. Trade Analysis (Third row, right 2 columns) 
        ax6 = fig.add_subplot(gs[2, 2:])
        self._plot_trade_analysis(ax6)
        
        # 7. Return Distribution (Bottom left)
        ax7 = fig.add_subplot(gs[3, :2])
        self._plot_return_distribution(ax7)
        
        # 8. Risk Metrics (Bottom right)
        ax8 = fig.add_subplot(gs[3, 2:])
        self._plot_risk_metrics(ax8)
        
        plt.suptitle('QuantPulse Trading Performance Dashboard', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        print(f"✅ EXITING create_performance_overview() at {datetime.now().strftime('%H:%M:%S')}")
        return fig
    
    def create_interactive_dashboard(self) -> Optional[Any]:
        """Create interactive dashboard using Plotly."""
        print(f"🔄 ENTERING create_interactive_dashboard() at {datetime.now().strftime('%H:%M:%S')}")
        
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available for interactive dashboard")
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Portfolio Value', 'Drawdown', 
                          'Rolling Sharpe Ratio', 'Trade PnL Distribution',
                          'Monthly Returns', 'Risk Metrics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )
        
        # 1. Portfolio Value
        portfolio_df = self.results.portfolio_df
        fig.add_trace(
            go.Scatter(
                x=portfolio_df.index,
                y=portfolio_df['portfolio_value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color=self.colors['primary'], width=2)
            ),
            row=1, col=1
        )
        
        # Add benchmark if available
        if self.benchmark_data is not None:
            # Normalize benchmark to same starting value
            bench_normalized = (self.benchmark_data / self.benchmark_data.iloc[0] * 
                              portfolio_df['portfolio_value'].iloc[0])
            fig.add_trace(
                go.Scatter(
                    x=bench_normalized.index,
                    y=bench_normalized.values,
                    mode='lines',
                    name='Benchmark',
                    line=dict(color=self.colors['secondary'], width=2, dash='dash')
                ),
                row=1, col=1
            )
        
        # 2. Drawdown
        returns = portfolio_df['portfolio_value'].pct_change().fillna(0)
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown,
                mode='lines',
                name='Drawdown',
                fill='tonexty',
                line=dict(color=self.colors['danger']),
                fillcolor='rgba(214, 39, 40, 0.3)'
            ),
            row=1, col=2
        )
        
        # 3. Rolling Sharpe Ratio
        rolling_metrics = self.analytics.rolling_metrics()
        if not rolling_metrics.empty:
            fig.add_trace(
                go.Scatter(
                    x=rolling_metrics.index,
                    y=rolling_metrics['rolling_sharpe'],
                    mode='lines',
                    name='Rolling Sharpe',
                    line=dict(color=self.colors['success'])
                ),
                row=2, col=1
            )
        
        # 4. Trade PnL Distribution
        if self.results.trades:
            trade_pnls = [t.pnl for t in self.results.trades]
            fig.add_trace(
                go.Histogram(
                    x=trade_pnls,
                    nbinsx=50,
                    name='Trade PnL',
                    marker_color=self.colors['primary'],
                    opacity=0.7
                ),
                row=2, col=2
            )
        
        # 5. Monthly Returns Heatmap
        monthly_table = self.analytics.monthly_performance_table()
        if not monthly_table.empty:
            fig.add_trace(
                go.Heatmap(
                    z=monthly_table.values * 100,  # Convert to percentage
                    x=monthly_table.columns,
                    y=monthly_table.index,
                    colorscale='RdYlGn',
                    text=np.round(monthly_table.values * 100, 2),
                    texttemplate="%{text}%",
                    textfont={"size": 10},
                    name='Monthly Returns (%)'
                ),
                row=3, col=1
            )
        
        # 6. Risk Metrics Bar Chart
        risk_metrics = self.analytics.calculate_advanced_risk_metrics()
        metrics_names = ['Sharpe Ratio', 'Calmar Ratio', 'Sterling Ratio', 'Sortino Ratio']
        metrics_values = [
            self.results.sharpe_ratio,
            risk_metrics.calmar_ratio,
            risk_metrics.sterling_ratio,
            self._calculate_sortino_ratio()  # Helper method
        ]
        
        fig.add_trace(
            go.Bar(
                x=metrics_names,
                y=metrics_values,
                name='Risk Metrics',
                marker_color=[self.colors['primary'] if v > 0 else self.colors['danger'] 
                             for v in metrics_values]
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="QuantPulse Interactive Trading Dashboard",
            title_x=0.5,
            height=900,
            showlegend=True,
            template='plotly_white'
        )
        
        print(f"✅ EXITING create_interactive_dashboard() at {datetime.now().strftime('%H:%M:%S')}")
        return fig
    
    def plot_pairs_analysis(self, pair_data: Dict[str, pd.DataFrame], 
                          figsize=(15, 10)) -> plt.Figure:
        """Plot pairs trading analysis."""
        print(f"🔄 ENTERING plot_pairs_analysis() at {datetime.now().strftime('%H:%M:%S')}")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Pairs Trading Analysis', fontsize=16, fontweight='bold')
        
        # Get first pair for demonstration
        if not pair_data:
            return fig
        
        symbol1, symbol2 = list(pair_data.keys())[:2]
        data1 = pair_data[symbol1]
        data2 = pair_data[symbol2]
        
        # 1. Price Series
        ax1 = axes[0, 0]
        ax1.plot(data1.index, data1['Close'], label=symbol1, color=self.colors['primary'])
        ax1_twin = ax1.twinx()
        ax1_twin.plot(data2.index, data2['Close'], label=symbol2, 
                     color=self.colors['secondary'])
        ax1.set_title('Price Series')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # 2. Spread Analysis
        ax2 = axes[0, 1]
        # Calculate spread (simplified)
        common_dates = data1.index.intersection(data2.index)
        spread = (data1.loc[common_dates]['Close'] - 
                 data2.loc[common_dates]['Close'])
        
        ax2.plot(common_dates, spread, color=self.colors['success'])
        ax2.axhline(y=spread.mean(), color='red', linestyle='--', alpha=0.7)
        ax2.axhline(y=spread.mean() + 2*spread.std(), color='orange', 
                   linestyle='--', alpha=0.7)
        ax2.axhline(y=spread.mean() - 2*spread.std(), color='orange', 
                   linestyle='--', alpha=0.7)
        ax2.set_title('Price Spread')
        ax2.legend(['Spread', 'Mean', '+2σ', '-2σ'])
        
        # 3. Correlation Analysis
        ax3 = axes[1, 0]
        returns1 = data1['Close'].pct_change().dropna()
        returns2 = data2['Close'].pct_change().dropna()
        common_returns = pd.concat([returns1, returns2], axis=1, 
                                 keys=[symbol1, symbol2]).dropna()
        
        ax3.scatter(common_returns.iloc[:, 0], common_returns.iloc[:, 1], 
                   alpha=0.6, color=self.colors['primary'])
        ax3.set_xlabel(f'{symbol1} Returns')
        ax3.set_ylabel(f'{symbol2} Returns')
        ax3.set_title('Returns Correlation')
        
        # Calculate and display correlation
        correlation = common_returns.corr().iloc[0, 1]
        ax3.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax3.transAxes, bbox=dict(boxstyle="round", 
                facecolor='white', alpha=0.8))
        
        # 4. Rolling Correlation
        ax4 = axes[1, 1]
        rolling_corr = common_returns.rolling(30).corr().iloc[0::2, 1]
        ax4.plot(rolling_corr.index, rolling_corr.values, 
                color=self.colors['info'])
        ax4.set_title('30-Day Rolling Correlation')
        ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        print(f"✅ EXITING plot_pairs_analysis() at {datetime.now().strftime('%H:%M:%S')}")
        return fig
    
    def plot_signal_analysis(self, signals_df: pd.DataFrame, 
                           figsize=(14, 8)) -> plt.Figure:
        """Plot trading signals analysis."""
        print(f"🔄 ENTERING plot_signal_analysis() at {datetime.now().strftime('%H:%M:%S')}")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Trading Signals Analysis', fontsize=16, fontweight='bold')
        
        # 1. Signal Strength Distribution
        ax1 = axes[0, 0]
        if 'signal_strength' in signals_df.columns:
            ax1.hist(signals_df['signal_strength'].dropna(), bins=50, 
                    alpha=0.7, color=self.colors['primary'])
            ax1.set_title('Signal Strength Distribution')
            ax1.set_xlabel('Signal Strength')
            ax1.set_ylabel('Frequency')
            ax1.axvline(x=signals_df['signal_strength'].mean(), color='red', 
                       linestyle='--', label='Mean')
            ax1.legend()
        
        # 2. Signals Over Time
        ax2 = axes[0, 1]
        if 'signal' in signals_df.columns:
            long_signals = signals_df[signals_df['signal'] > 0]
            short_signals = signals_df[signals_df['signal'] < 0]
            
            ax2.scatter(long_signals.index, long_signals['signal'], 
                       color=self.colors['success'], alpha=0.6, 
                       label='Long Signals', s=30)
            ax2.scatter(short_signals.index, short_signals['signal'], 
                       color=self.colors['danger'], alpha=0.6, 
                       label='Short Signals', s=30)
            ax2.set_title('Signals Over Time')
            ax2.legend()
        
        # 3. Signal vs Returns
        ax3 = axes[1, 0]
        if 'signal_strength' in signals_df.columns and 'returns' in signals_df.columns:
            ax3.scatter(signals_df['signal_strength'], signals_df['returns'], 
                       alpha=0.6, color=self.colors['primary'])
            ax3.set_xlabel('Signal Strength')
            ax3.set_ylabel('Next Period Return')
            ax3.set_title('Signal Strength vs Future Returns')
            
            # Add trend line
            z = np.polyfit(signals_df['signal_strength'].dropna(), 
                          signals_df['returns'].dropna(), 1)
            p = np.poly1d(z)
            ax3.plot(signals_df['signal_strength'], 
                    p(signals_df['signal_strength']), 
                    "r--", alpha=0.8)
        
        # 4. Signal Performance by Strength Buckets
        ax4 = axes[1, 1]
        if 'signal_strength' in signals_df.columns and 'returns' in signals_df.columns:
            # Create strength buckets
            signals_df_copy = signals_df.copy()
            signals_df_copy['strength_bucket'] = pd.cut(
                signals_df_copy['signal_strength'], 
                bins=5, labels=['Very Weak', 'Weak', 'Medium', 'Strong', 'Very Strong']
            )
            
            bucket_performance = signals_df_copy.groupby('strength_bucket')['returns'].mean()
            
            colors = [self.colors['danger'] if x < 0 else self.colors['success'] 
                     for x in bucket_performance.values]
            ax4.bar(range(len(bucket_performance)), bucket_performance.values, 
                   color=colors, alpha=0.7)
            ax4.set_xticks(range(len(bucket_performance)))
            ax4.set_xticklabels(bucket_performance.index, rotation=45)
            ax4.set_title('Performance by Signal Strength')
            ax4.set_ylabel('Average Return')
        
        plt.tight_layout()
        
        print(f"✅ EXITING plot_signal_analysis() at {datetime.now().strftime('%H:%M:%S')}")
        return fig
    
    def _plot_portfolio_value(self, ax):
        """Plot portfolio value chart."""
        portfolio_df = self.results.portfolio_df
        ax.plot(portfolio_df.index, portfolio_df['portfolio_value'], 
               color=self.colors['primary'], linewidth=2, label='Portfolio')
        
        if self.benchmark_data is not None:
            # Normalize benchmark
            bench_normalized = (self.benchmark_data / self.benchmark_data.iloc[0] * 
                              portfolio_df['portfolio_value'].iloc[0])
            ax.plot(bench_normalized.index, bench_normalized.values, 
                   color=self.colors['secondary'], linewidth=2, 
                   linestyle='--', label='Benchmark')
        
        ax.set_title('Portfolio Value Over Time', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    def _plot_key_metrics_table(self, ax):
        """Plot key metrics table."""
        ax.axis('off')
        
        # Key metrics
        metrics_data = [
            ['Total Return', f"{self.results.total_return:.2%}"],
            ['Annual Return', f"{self.results.annualized_return:.2%}"],
            ['Volatility', f"{self.results.volatility:.2%}"],
            ['Sharpe Ratio', f"{self.results.sharpe_ratio:.2f}"],
            ['Max Drawdown', f"{self.results.max_drawdown:.2%}"],
            ['Win Rate', f"{self.results.win_rate:.2%}"],
            ['Total Trades', f"{self.results.total_trades:,}"],
        ]
        
        table = ax.table(cellText=metrics_data,
                        colLabels=['Metric', 'Value'],
                        cellLoc='left',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Style the table
        for i in range(len(metrics_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4472C4')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    if i % 2 == 0:
                        cell.set_facecolor('#F2F2F2')
                    else:
                        cell.set_facecolor('white')
        
        ax.set_title('Key Performance Metrics', fontweight='bold', pad=20)
    
    def _plot_monthly_returns_heatmap(self, ax):
        """Plot monthly returns heatmap."""
        monthly_table = self.analytics.monthly_performance_table()
        
        if monthly_table.empty:
            ax.text(0.5, 0.5, 'No monthly data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Monthly Returns Heatmap')
            return
        
        # Create heatmap
        im = ax.imshow(monthly_table.values * 100, cmap='RdYlGn', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(monthly_table.columns)))
        ax.set_yticks(np.arange(len(monthly_table.index)))
        ax.set_xticklabels(monthly_table.columns)
        ax.set_yticklabels(monthly_table.index)
        
        # Add text annotations
        for i in range(len(monthly_table.index)):
            for j in range(len(monthly_table.columns)):
                value = monthly_table.iloc[i, j] * 100
                if not np.isnan(value):
                    text = ax.text(j, i, f'{value:.1f}%', ha="center", va="center",
                                  color="white" if abs(value) > 2 else "black", fontsize=8)
        
        ax.set_title('Monthly Returns Heatmap (%)', fontweight='bold')
        plt.colorbar(im, ax=ax, label='Return (%)')
    
    def _plot_drawdown_chart(self, ax):
        """Plot drawdown chart."""
        portfolio_df = self.results.portfolio_df
        returns = portfolio_df['portfolio_value'].pct_change().fillna(0)
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        ax.fill_between(drawdown.index, drawdown, 0, color=self.colors['danger'], 
                       alpha=0.3, label='Drawdown')
        ax.plot(drawdown.index, drawdown, color=self.colors['danger'], linewidth=1)
        
        ax.set_title('Portfolio Drawdown', fontweight='bold')
        ax.set_ylabel('Drawdown %')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax.grid(True, alpha=0.3)
    
    def _plot_rolling_metrics(self, ax):
        """Plot rolling metrics."""
        rolling_metrics = self.analytics.rolling_metrics()
        
        if rolling_metrics.empty:
            ax.text(0.5, 0.5, 'Insufficient data for rolling metrics', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Rolling Metrics')
            return
        
        ax2 = ax.twinx()
        
        # Rolling Sharpe on left axis
        line1 = ax.plot(rolling_metrics.index, rolling_metrics['rolling_sharpe'], 
                       color=self.colors['primary'], label='Sharpe Ratio')
        ax.set_ylabel('Sharpe Ratio', color=self.colors['primary'])
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Rolling volatility on right axis
        line2 = ax2.plot(rolling_metrics.index, rolling_metrics['rolling_volatility'], 
                        color=self.colors['secondary'], label='Volatility')
        ax2.set_ylabel('Volatility', color=self.colors['secondary'])
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        
        ax.set_title('Rolling Performance Metrics', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_trade_analysis(self, ax):
        """Plot trade analysis."""
        if not self.results.trades:
            ax.text(0.5, 0.5, 'No trades available', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title('Trade Analysis')
            return
        
        trade_pnls = [t.pnl for t in self.results.trades]
        winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
        losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
        
        # Trade PnL distribution
        ax.hist(winning_trades, bins=30, alpha=0.7, color=self.colors['success'], 
               label=f'Winning ({len(winning_trades)})')
        ax.hist(losing_trades, bins=30, alpha=0.7, color=self.colors['danger'], 
               label=f'Losing ({len(losing_trades)})')
        
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax.set_title('Trade PnL Distribution', fontweight='bold')
        ax.set_xlabel('Trade PnL ($)')
        ax.set_ylabel('Frequency')
        ax.legend()
    
    def _plot_return_distribution(self, ax):
        """Plot return distribution."""
        portfolio_df = self.results.portfolio_df
        returns = portfolio_df['portfolio_value'].pct_change().dropna()
        
        # Histogram
        ax.hist(returns, bins=50, alpha=0.7, color=self.colors['primary'], 
               density=True, label='Actual Returns')
        
        # Normal distribution overlay
        mu, sigma = returns.mean(), returns.std()
        x = np.linspace(returns.min(), returns.max(), 100)
        normal_pdf = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        ax.plot(x, normal_pdf, 'r--', label='Normal Distribution', linewidth=2)
        
        ax.set_title('Daily Returns Distribution', fontweight='bold')
        ax.set_xlabel('Daily Return')
        ax.set_ylabel('Density')
        ax.legend()
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    def _plot_risk_metrics(self, ax):
        """Plot risk metrics bar chart."""
        risk_metrics = self.analytics.calculate_advanced_risk_metrics()
        
        metrics = {
            'Sharpe': self.results.sharpe_ratio,
            'Calmar': risk_metrics.calmar_ratio,
            'Sterling': risk_metrics.sterling_ratio,
            'Sortino': self._calculate_sortino_ratio()
        }
        
        colors = [self.colors['success'] if v > 0 else self.colors['danger'] 
                 for v in metrics.values()]
        
        bars = ax.bar(metrics.keys(), metrics.values(), color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.set_title('Risk-Adjusted Performance Metrics', fontweight='bold')
        ax.set_ylabel('Ratio')
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics.values()):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height > 0 else -0.1),
                   f'{value:.2f}', ha='center', va='bottom' if height > 0 else 'top')
    
    def _calculate_sortino_ratio(self) -> float:
        """Calculate Sortino ratio."""
        portfolio_df = self.results.portfolio_df
        returns = portfolio_df['portfolio_value'].pct_change().dropna()
        
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return np.inf
        
        downside_volatility = downside_returns.std()
        excess_return = returns.mean() - 0.02/252  # 2% risk-free rate
        
        return excess_return / downside_volatility * np.sqrt(252) if downside_volatility > 0 else 0
    
    def save_dashboard(self, filepath: str, dpi: int = 300):
        """Save dashboard to file."""
        print(f"🔄 ENTERING save_dashboard({filepath}) at {datetime.now().strftime('%H:%M:%S')}")
        
        fig = self.create_performance_overview()
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        logger.info(f"Dashboard saved to {filepath}")
        
        print(f"✅ EXITING save_dashboard() at {datetime.now().strftime('%H:%M:%S')}")

class PerformanceCharts:
    """Specialized performance charts for different aspects of trading."""
    
    def __init__(self, results: BacktestResults):
        print(f"🔄 ENTERING PerformanceCharts.__init__() at {datetime.now().strftime('%H:%M:%S')}")
        
        self.results = results
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8'
        }
        
        print(f"✅ EXITING PerformanceCharts.__init__() at {datetime.now().strftime('%H:%M:%S')}")
    
    def plot_equity_curve_with_trades(self, figsize=(14, 8)) -> plt.Figure:
        """Plot equity curve with trade markers."""
        print(f"🔄 ENTERING plot_equity_curve_with_trades() at {datetime.now().strftime('%H:%M:%S')}")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot equity curve
        portfolio_df = self.results.portfolio_df
        ax.plot(portfolio_df.index, portfolio_df['portfolio_value'], 
               color=self.colors['primary'], linewidth=2, label='Portfolio Value')
        
        # Mark trade entry and exit points
        if self.results.trades:
            entry_dates = [t.entry_date for t in self.results.trades]
            exit_dates = [t.exit_date for t in self.results.trades if t.exit_date]
            
            # Entry points
            for date in entry_dates:
                if date in portfolio_df.index:
                    value = portfolio_df.loc[date, 'portfolio_value']
                    ax.scatter(date, value, color=self.colors['success'], 
                             marker='^', s=50, alpha=0.7)
            
            # Exit points
            for date in exit_dates:
                if date in portfolio_df.index:
                    value = portfolio_df.loc[date, 'portfolio_value']
                    ax.scatter(date, value, color=self.colors['danger'], 
                             marker='v', s=50, alpha=0.7)
        
        ax.set_title('Portfolio Equity Curve with Trade Markers', fontweight='bold', fontsize=14)
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.grid(True, alpha=0.3)
        ax.legend(['Portfolio Value', 'Trade Entry', 'Trade Exit'])
        
        # Format dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        print(f"✅ EXITING plot_equity_curve_with_trades() at {datetime.now().strftime('%H:%M:%S')}")
        return fig
    
    def plot_underwater_curve(self, figsize=(12, 6)) -> plt.Figure:
        """Plot underwater (drawdown) curve."""
        print(f"🔄 ENTERING plot_underwater_curve() at {datetime.now().strftime('%H:%M:%S')}")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        portfolio_df = self.results.portfolio_df
        returns = portfolio_df['portfolio_value'].pct_change().fillna(0)
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        ax.fill_between(drawdown.index, drawdown * 100, 0, 
                       color=self.colors['danger'], alpha=0.3)
        ax.plot(drawdown.index, drawdown * 100, 
               color=self.colors['danger'], linewidth=2)
        
        # Mark maximum drawdown point
        max_dd_idx = drawdown.idxmin()
        max_dd_value = drawdown.min() * 100
        ax.scatter(max_dd_idx, max_dd_value, color='red', s=100, 
                  marker='o', zorder=5, label=f'Max DD: {max_dd_value:.1f}%')
        
        ax.set_title('Underwater Curve (Drawdown)', fontweight='bold', fontsize=14)
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        print(f"✅ EXITING plot_underwater_curve() at {datetime.now().strftime('%H:%M:%S')}")
        return fig
    
    def plot_rolling_correlation_matrix(self, data: Dict[str, pd.DataFrame], 
                                      window: int = 30, figsize=(12, 8)) -> plt.Figure:
        """Plot rolling correlation matrix heatmap."""
        print(f"🔄 ENTERING plot_rolling_correlation_matrix() at {datetime.now().strftime('%H:%M:%S')}")
        
        if len(data) < 2:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'Need at least 2 assets for correlation matrix', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Calculate returns for all assets
        returns_data = {}
        for symbol, df in data.items():
            returns_data[symbol] = df['Close'].pct_change().dropna()
        
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()
        
        # Calculate rolling correlations
        rolling_corr = returns_df.rolling(window).corr()
        
        # Get unique dates for correlation matrices
        dates = rolling_corr.index.get_level_values(0).unique()[window-1:]
        
        # Create subplots for different time periods
        n_plots = min(6, len(dates) // (len(dates) // 6) if len(dates) > 6 else len(dates))
        selected_dates = dates[::len(dates)//n_plots][:n_plots]
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        symbols = list(data.keys())
        
        for i, date in enumerate(selected_dates):
            if i >= len(axes):
                break
            
            corr_matrix = rolling_corr.loc[date]
            
            im = axes[i].imshow(corr_matrix.values, cmap='RdYlBu_r', 
                              vmin=-1, vmax=1, aspect='auto')
            
            axes[i].set_xticks(range(len(symbols)))
            axes[i].set_yticks(range(len(symbols)))
            axes[i].set_xticklabels(symbols, rotation=45)
            axes[i].set_yticklabels(symbols)
            axes[i].set_title(f'{date.strftime("%Y-%m-%d")}')
            
            # Add correlation values
            for row in range(len(symbols)):
                for col in range(len(symbols)):
                    value = corr_matrix.iloc[row, col]
                    text_color = 'white' if abs(value) > 0.5 else 'black'
                    axes[i].text(col, row, f'{value:.2f}', ha="center", va="center",
                               color=text_color, fontsize=8)
        
        # Remove empty subplots
        for i in range(len(selected_dates), len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle(f'{window}-Day Rolling Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=axes[:len(selected_dates)], shrink=0.8, aspect=20)
        cbar.set_label('Correlation Coefficient')
        
        print(f"✅ EXITING plot_rolling_correlation_matrix() at {datetime.now().strftime('%H:%M:%S')}")
        return fig

def create_report_pdf(results: BacktestResults, filepath: str, 
                     benchmark_data: pd.DataFrame = None):
    """Create comprehensive PDF report."""
    print(f"🔄 ENTERING create_report_pdf({filepath}) at {datetime.now().strftime('%H:%M:%S')}")
    
    try:
        from matplotlib.backends.backend_pdf import PdfPages
        
        dashboard = TradingDashboard(results, benchmark_data)
        charts = PerformanceCharts(results)
        
        with PdfPages(filepath) as pdf:
            # Page 1: Main Dashboard
            fig1 = dashboard.create_performance_overview()
            pdf.savefig(fig1, bbox_inches='tight')
            plt.close(fig1)
            
            # Page 2: Equity Curve with Trades
            fig2 = charts.plot_equity_curve_with_trades()
            pdf.savefig(fig2, bbox_inches='tight')
            plt.close(fig2)
            
            # Page 3: Underwater Curve
            fig3 = charts.plot_underwater_curve()
            pdf.savefig(fig3, bbox_inches='tight')  
            plt.close(fig3)
        
        logger.info(f"PDF report created: {filepath}")
        
    except ImportError:
        logger.warning("PDF creation requires matplotlib. Saving individual charts instead.")
        
        # Save individual charts as PNG
        base_path = filepath.replace('.pdf', '')
        dashboard.save_dashboard(f"{base_path}_dashboard.png")
        
        fig2 = charts.plot_equity_curve_with_trades()
        fig2.savefig(f"{base_path}_equity_curve.png", dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        fig3 = charts.plot_underwater_curve()
        fig3.savefig(f"{base_path}_drawdown.png", dpi=300, bbox_inches='tight')
        plt.close(fig3)
    
    print(f"✅ EXITING create_report_pdf() at {datetime.now().strftime('%H:%M:%S')}")
