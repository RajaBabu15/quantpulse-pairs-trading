"""
Performance Analytics Module
===========================

Advanced performance analytics with risk attribution, regime analysis,
and comprehensive portfolio metrics.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import jarque_bera, normaltest

from .backtesting import BacktestResults, Trade

import logging
logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics."""
    var_95: float
    cvar_95: float
    max_drawdown: float
    max_dd_duration: int
    calmar_ratio: float
    sterling_ratio: float
    pain_index: float
    ulcer_index: float
    skewness: float
    kurtosis: float
    jarque_bera_pvalue: float
    tail_ratio: float
    
@dataclass  
class PerformanceAttribution:
    """Performance attribution analysis."""
    alpha: float
    beta: float
    tracking_error: float
    information_ratio: float
    up_capture: float
    down_capture: float
    capture_ratio: float
    selectivity: float
    timing: float

class PerformanceAnalytics:
    """Comprehensive performance analytics engine."""
    
    def __init__(self, results: BacktestResults, benchmark_data: pd.DataFrame = None):
        print(f"🔄 ENTERING PerformanceAnalytics.__init__() at {datetime.now().strftime('%H:%M:%S')}")
        
        self.results = results
        self.benchmark_data = benchmark_data
        self.portfolio_df = results.portfolio_df
        
        # Calculate returns if not already available
        if 'returns' not in self.portfolio_df.columns:
            self.portfolio_df['returns'] = self.portfolio_df['portfolio_value'].pct_change().fillna(0)
        
        print(f"✅ EXITING PerformanceAnalytics.__init__() at {datetime.now().strftime('%H:%M:%S')}")
    
    def calculate_advanced_risk_metrics(self) -> RiskMetrics:
        """Calculate advanced risk metrics."""
        print(f"🔄 ENTERING calculate_advanced_risk_metrics() at {datetime.now().strftime('%H:%M:%S')}")
        
        returns = self.portfolio_df['returns'].values
        portfolio_values = self.portfolio_df['portfolio_value'].values
        
        # Value at Risk and Conditional Value at Risk
        var_95 = np.percentile(returns, 5)
        tail_returns = returns[returns <= var_95]
        cvar_95 = np.mean(tail_returns) if len(tail_returns) > 0 else var_95
        
        # Drawdown analysis
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        
        max_drawdown = np.min(drawdowns)
        
        # Maximum drawdown duration
        dd_duration = self._calculate_drawdown_duration(drawdowns)
        max_dd_duration = np.max(dd_duration) if len(dd_duration) > 0 else 0
        
        # Calmar ratio
        annual_return = (cumulative_returns[-1] ** (252 / len(returns))) - 1
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sterling ratio (using average drawdown)
        avg_drawdown = np.mean(np.abs(drawdowns[drawdowns < 0])) if np.any(drawdowns < 0) else 0.01
        sterling_ratio = annual_return / avg_drawdown if avg_drawdown != 0 else 0
        
        # Pain index (average drawdown)
        pain_index = np.mean(np.abs(drawdowns))
        
        # Ulcer index
        ulcer_index = np.sqrt(np.mean(drawdowns ** 2))
        
        # Distribution metrics
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Normality test
        try:
            _, jb_pvalue = jarque_bera(returns)
        except:
            jb_pvalue = 0.0
        
        # Tail ratio (95th percentile / 5th percentile)
        tail_ratio = np.percentile(returns, 95) / abs(np.percentile(returns, 5)) if np.percentile(returns, 5) != 0 else 0
        
        risk_metrics = RiskMetrics(
            var_95=var_95,
            cvar_95=cvar_95,
            max_drawdown=max_drawdown,
            max_dd_duration=max_dd_duration,
            calmar_ratio=calmar_ratio,
            sterling_ratio=sterling_ratio,
            pain_index=pain_index,
            ulcer_index=ulcer_index,
            skewness=skewness,
            kurtosis=kurtosis,
            jarque_bera_pvalue=jb_pvalue,
            tail_ratio=tail_ratio
        )
        
        print(f"✅ EXITING calculate_advanced_risk_metrics() at {datetime.now().strftime('%H:%M:%S')}")
        return risk_metrics
    
    def performance_attribution(self, benchmark_returns: pd.Series = None) -> PerformanceAttribution:
        """Calculate performance attribution vs benchmark."""
        print(f"🔄 ENTERING performance_attribution() at {datetime.now().strftime('%H:%M:%S')}")
        
        if benchmark_returns is None:
            if self.benchmark_data is None:
                logger.warning("No benchmark data available for attribution")
                return self._empty_attribution()
            benchmark_returns = self.benchmark_data.pct_change().dropna()
        
        # Align returns
        portfolio_returns = self.portfolio_df['returns']
        common_index = portfolio_returns.index.intersection(benchmark_returns.index)
        
        if len(common_index) < 30:
            logger.warning("Insufficient overlapping data for attribution")
            return self._empty_attribution()
        
        port_ret = portfolio_returns.loc[common_index]
        bench_ret = benchmark_returns.loc[common_index]
        
        # Basic attribution metrics
        try:
            # Alpha and Beta
            covariance_matrix = np.cov(port_ret, bench_ret)
            beta = covariance_matrix[0, 1] / np.var(bench_ret) if np.var(bench_ret) > 0 else 0
            alpha = np.mean(port_ret) - beta * np.mean(bench_ret)
            
            # Tracking error
            tracking_error = np.std(port_ret - bench_ret) * np.sqrt(252)
            
            # Information ratio
            excess_return = np.mean(port_ret - bench_ret) * 252
            information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
            
            # Up/Down capture ratios
            up_periods = bench_ret > 0
            down_periods = bench_ret < 0
            
            if np.any(up_periods):
                up_capture = np.mean(port_ret[up_periods]) / np.mean(bench_ret[up_periods])
            else:
                up_capture = 0
            
            if np.any(down_periods):
                down_capture = np.mean(port_ret[down_periods]) / np.mean(bench_ret[down_periods])
            else:
                down_capture = 0
            
            capture_ratio = up_capture / abs(down_capture) if down_capture != 0 else 0
            
            # Selectivity and timing (simplified Treynor-Mazuy model)
            # This is a simplified version - full implementation would require regression
            selectivity = alpha  # Simplified
            timing = 0.0  # Would require more complex calculation
            
            attribution = PerformanceAttribution(
                alpha=alpha * 252,  # Annualized
                beta=beta,
                tracking_error=tracking_error,
                information_ratio=information_ratio,
                up_capture=up_capture,
                down_capture=down_capture,
                capture_ratio=capture_ratio,
                selectivity=selectivity * 252,
                timing=timing
            )
            
            print(f"✅ EXITING performance_attribution() at {datetime.now().strftime('%H:%M:%S')}")
            return attribution
        
        except Exception as e:
            logger.error(f"Error in performance attribution: {e}")
            return self._empty_attribution()
    
    def rolling_metrics(self, window: int = 63) -> pd.DataFrame:
        """Calculate rolling performance metrics."""
        print(f"🔄 ENTERING rolling_metrics({window}) at {datetime.now().strftime('%H:%M:%S')}")
        
        returns = self.portfolio_df['returns']
        
        rolling_metrics = pd.DataFrame(index=returns.index)
        
        # Rolling return
        rolling_metrics['rolling_return'] = returns.rolling(window).sum()
        
        # Rolling volatility  
        rolling_metrics['rolling_volatility'] = returns.rolling(window).std() * np.sqrt(252)
        
        # Rolling Sharpe ratio
        excess_returns = returns - 0.02/252  # Assume 2% risk-free rate
        rolling_metrics['rolling_sharpe'] = (
            excess_returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
        )
        
        # Rolling max drawdown
        rolling_metrics['rolling_max_dd'] = (
            returns.rolling(window).apply(self._rolling_max_drawdown, raw=False)
        )
        
        # Rolling Sortino ratio
        def rolling_sortino(x):
            if len(x) < 10:
                return np.nan
            downside_returns = x[x < 0]
            if len(downside_returns) == 0:
                return np.inf
            downside_volatility = np.std(downside_returns)
            return np.mean(x) / downside_volatility * np.sqrt(252) if downside_volatility > 0 else np.nan
        
        rolling_metrics['rolling_sortino'] = returns.rolling(window).apply(rolling_sortino)
        
        # Rolling VaR
        rolling_metrics['rolling_var_95'] = returns.rolling(window).quantile(0.05)
        
        print(f"✅ EXITING rolling_metrics() at {datetime.now().strftime('%H:%M:%S')}")
        return rolling_metrics
    
    def trade_analysis(self) -> Dict[str, Any]:
        """Analyze individual trade performance."""
        print(f"🔄 ENTERING trade_analysis() at {datetime.now().strftime('%H:%M:%S')}")
        
        if not self.results.trades:
            logger.warning("No trades available for analysis")
            return {}
        
        trades = self.results.trades
        
        # Basic trade statistics
        pnls = [t.pnl for t in trades]
        returns = [t.return_pct for t in trades]
        holding_periods = [t.holding_days for t in trades]
        
        # Performance by holding period buckets
        short_trades = [t for t in trades if t.holding_days <= 5]
        medium_trades = [t for t in trades if 5 < t.holding_days <= 20]
        long_trades = [t for t in trades if t.holding_days > 20]
        
        # Performance by signal strength
        strong_signals = [t for t in trades if t.signal_strength > 2.0]
        weak_signals = [t for t in trades if t.signal_strength <= 2.0]
        
        # Symbol pair analysis
        pair_performance = {}
        for trade in trades:
            pair_key = f"{trade.symbol1}-{trade.symbol2}"
            if pair_key not in pair_performance:
                pair_performance[pair_key] = {'trades': 0, 'total_pnl': 0, 'wins': 0}
            
            pair_performance[pair_key]['trades'] += 1
            pair_performance[pair_key]['total_pnl'] += trade.pnl
            if trade.pnl > 0:
                pair_performance[pair_key]['wins'] += 1
        
        # Add win rates
        for pair_key in pair_performance:
            pair_performance[pair_key]['win_rate'] = (
                pair_performance[pair_key]['wins'] / pair_performance[pair_key]['trades']
            )
        
        analysis = {
            'total_trades': len(trades),
            'avg_pnl': np.mean(pnls),
            'median_pnl': np.median(pnls),
            'pnl_std': np.std(pnls),
            'avg_return': np.mean(returns),
            'return_std': np.std(returns),
            'avg_holding_period': np.mean(holding_periods),
            'holding_period_std': np.std(holding_periods),
            
            'short_term_trades': {
                'count': len(short_trades),
                'avg_pnl': np.mean([t.pnl for t in short_trades]) if short_trades else 0,
                'win_rate': len([t for t in short_trades if t.pnl > 0]) / len(short_trades) if short_trades else 0
            },
            
            'medium_term_trades': {
                'count': len(medium_trades),
                'avg_pnl': np.mean([t.pnl for t in medium_trades]) if medium_trades else 0,
                'win_rate': len([t for t in medium_trades if t.pnl > 0]) / len(medium_trades) if medium_trades else 0
            },
            
            'long_term_trades': {
                'count': len(long_trades),
                'avg_pnl': np.mean([t.pnl for t in long_trades]) if long_trades else 0,
                'win_rate': len([t for t in long_trades if t.pnl > 0]) / len(long_trades) if long_trades else 0
            },
            
            'strong_signal_trades': {
                'count': len(strong_signals),
                'avg_pnl': np.mean([t.pnl for t in strong_signals]) if strong_signals else 0,
                'win_rate': len([t for t in strong_signals if t.pnl > 0]) / len(strong_signals) if strong_signals else 0
            },
            
            'weak_signal_trades': {
                'count': len(weak_signals),
                'avg_pnl': np.mean([t.pnl for t in weak_signals]) if weak_signals else 0,
                'win_rate': len([t for t in weak_signals if t.pnl > 0]) / len(weak_signals) if weak_signals else 0
            },
            
            'pair_performance': pair_performance,
            
            'best_trade': max(trades, key=lambda t: t.pnl) if trades else None,
            'worst_trade': min(trades, key=lambda t: t.pnl) if trades else None,
            'longest_trade': max(trades, key=lambda t: t.holding_days) if trades else None,
        }
        
        print(f"✅ EXITING trade_analysis() at {datetime.now().strftime('%H:%M:%S')}")
        return analysis
    
    def monthly_performance_table(self) -> pd.DataFrame:
        """Create monthly performance table."""
        print(f"🔄 ENTERING monthly_performance_table() at {datetime.now().strftime('%H:%M:%S')}")
        
        monthly_returns = self.results.monthly_returns
        
        if monthly_returns.empty:
            return pd.DataFrame()
        
        # Convert to monthly table format
        monthly_returns.index = pd.to_datetime(monthly_returns.index.to_timestamp())
        monthly_returns_df = monthly_returns.to_frame('return')
        monthly_returns_df['year'] = monthly_returns_df.index.year
        monthly_returns_df['month'] = monthly_returns_df.index.month
        
        # Pivot table
        monthly_table = monthly_returns_df.pivot_table(
            values='return', 
            index='year', 
            columns='month', 
            aggfunc='first'
        )
        
        # Add column names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_table.columns = [month_names[i-1] for i in monthly_table.columns]
        
        # Add annual totals
        monthly_table['Year Total'] = (monthly_table + 1).prod(axis=1) - 1
        
        print(f"✅ EXITING monthly_performance_table() at {datetime.now().strftime('%H:%M:%S')}")
        return monthly_table
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        print(f"🔄 ENTERING generate_report() at {datetime.now().strftime('%H:%M:%S')}")
        
        # Calculate all metrics
        risk_metrics = self.calculate_advanced_risk_metrics()
        attribution = self.performance_attribution()
        rolling_stats = self.rolling_metrics()
        trade_analysis = self.trade_analysis()
        monthly_table = self.monthly_performance_table()
        
        report = {
            'summary': {
                'total_return': self.results.total_return,
                'annualized_return': self.results.annualized_return,
                'volatility': self.results.volatility,
                'sharpe_ratio': self.results.sharpe_ratio,
                'max_drawdown': self.results.max_drawdown,
                'total_trades': self.results.total_trades,
                'win_rate': self.results.win_rate,
            },
            
            'risk_metrics': risk_metrics,
            'performance_attribution': attribution,
            'trade_analysis': trade_analysis,
            'monthly_performance': monthly_table,
            'rolling_metrics': rolling_stats,
            
            'portfolio_data': self.portfolio_df,
            'trades': self.results.trades,
            
            'report_generated': datetime.now(),
            'data_period': {
                'start': self.portfolio_df.index[0],
                'end': self.portfolio_df.index[-1],
                'days': len(self.portfolio_df)
            }
        }
        
        print(f"✅ EXITING generate_report() at {datetime.now().strftime('%H:%M:%S')}")
        return report
    
    def _calculate_drawdown_duration(self, drawdowns: np.ndarray) -> List[int]:
        """Calculate drawdown durations."""
        durations = []
        current_duration = 0
        
        for dd in drawdowns:
            if dd < 0:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0
        
        # Add final duration if still in drawdown
        if current_duration > 0:
            durations.append(current_duration)
        
        return durations
    
    def _rolling_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate rolling maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _empty_attribution(self) -> PerformanceAttribution:
        """Return empty attribution object."""
        return PerformanceAttribution(
            alpha=0, beta=0, tracking_error=0, information_ratio=0,
            up_capture=0, down_capture=0, capture_ratio=0,
            selectivity=0, timing=0
        )

class RegimeAnalysis:
    """Market regime detection and analysis."""
    
    def __init__(self, data: Dict[str, pd.DataFrame]):
        print(f"🔄 ENTERING RegimeAnalysis.__init__() at {datetime.now().strftime('%H:%M:%S')}")
        
        self.data = data
        self.regimes = {}
        
        print(f"✅ EXITING RegimeAnalysis.__init__() at {datetime.now().strftime('%H:%M:%S')}")
    
    def detect_volatility_regimes(self, window: int = 30) -> Dict[str, pd.Series]:
        """Detect volatility regimes using rolling volatility."""
        print(f"🔄 ENTERING detect_volatility_regimes() at {datetime.now().strftime('%H:%M:%S')}")
        
        vol_regimes = {}
        
        for symbol, df in self.data.items():
            returns = df['Close'].pct_change().dropna()
            rolling_vol = returns.rolling(window).std() * np.sqrt(252)
            
            # Define regimes based on quantiles
            vol_25 = rolling_vol.quantile(0.25)
            vol_75 = rolling_vol.quantile(0.75)
            
            regime = pd.Series(index=rolling_vol.index, dtype='object')
            regime[rolling_vol <= vol_25] = 'low_vol'
            regime[(rolling_vol > vol_25) & (rolling_vol <= vol_75)] = 'medium_vol'
            regime[rolling_vol > vol_75] = 'high_vol'
            
            vol_regimes[symbol] = regime
        
        self.regimes['volatility'] = vol_regimes
        
        print(f"✅ EXITING detect_volatility_regimes() at {datetime.now().strftime('%H:%M:%S')}")
        return vol_regimes
    
    def detect_trend_regimes(self, window: int = 50) -> Dict[str, pd.Series]:
        """Detect trend regimes using moving averages."""
        print(f"🔄 ENTERING detect_trend_regimes() at {datetime.now().strftime('%H:%M:%S')}")
        
        trend_regimes = {}
        
        for symbol, df in self.data.items():
            price = df['Close']
            ma_short = price.rolling(window//2).mean()
            ma_long = price.rolling(window).mean()
            
            regime = pd.Series(index=price.index, dtype='object')
            regime[ma_short > ma_long] = 'uptrend'
            regime[ma_short <= ma_long] = 'downtrend'
            
            # Add sideways regime based on slope
            ma_slope = ma_long.pct_change(window//4).rolling(5).mean()
            slope_threshold = 0.001  # 0.1% slope threshold
            
            sideways_mask = abs(ma_slope) < slope_threshold
            regime[sideways_mask] = 'sideways'
            
            trend_regimes[symbol] = regime
        
        self.regimes['trend'] = trend_regimes
        
        print(f"✅ EXITING detect_trend_regimes() at {datetime.now().strftime('%H:%M:%S')}")
        return trend_regimes
    
    def analyze_regime_performance(self, results: BacktestResults) -> Dict[str, Any]:
        """Analyze performance by market regime."""
        print(f"🔄 ENTERING analyze_regime_performance() at {datetime.now().strftime('%H:%M:%S')}")
        
        if not self.regimes:
            self.detect_volatility_regimes()
            self.detect_trend_regimes()
        
        regime_performance = {}
        
        # Analyze by volatility regimes
        if 'volatility' in self.regimes:
            vol_performance = {}
            portfolio_returns = results.portfolio_df['returns'] if 'returns' in results.portfolio_df else results.portfolio_df['portfolio_value'].pct_change()
            
            # For simplicity, use first symbol's regime (could be aggregated)
            first_symbol = list(self.regimes['volatility'].keys())[0]
            vol_regime = self.regimes['volatility'][first_symbol]
            
            # Align dates
            common_index = portfolio_returns.index.intersection(vol_regime.index)
            aligned_returns = portfolio_returns.loc[common_index]
            aligned_regime = vol_regime.loc[common_index]
            
            for regime_type in ['low_vol', 'medium_vol', 'high_vol']:
                regime_mask = aligned_regime == regime_type
                regime_returns = aligned_returns[regime_mask]
                
                if len(regime_returns) > 0:
                    vol_performance[regime_type] = {
                        'count': len(regime_returns),
                        'total_return': (1 + regime_returns).prod() - 1,
                        'avg_return': regime_returns.mean(),
                        'volatility': regime_returns.std() * np.sqrt(252),
                        'sharpe': regime_returns.mean() / regime_returns.std() * np.sqrt(252) if regime_returns.std() > 0 else 0,
                        'win_rate': len(regime_returns[regime_returns > 0]) / len(regime_returns)
                    }
            
            regime_performance['volatility'] = vol_performance
        
        # Analyze by trend regimes
        if 'trend' in self.regimes:
            trend_performance = {}
            
            first_symbol = list(self.regimes['trend'].keys())[0]
            trend_regime = self.regimes['trend'][first_symbol]
            
            common_index = portfolio_returns.index.intersection(trend_regime.index)
            aligned_returns = portfolio_returns.loc[common_index]
            aligned_regime = trend_regime.loc[common_index]
            
            for regime_type in ['uptrend', 'downtrend', 'sideways']:
                regime_mask = aligned_regime == regime_type
                regime_returns = aligned_returns[regime_mask]
                
                if len(regime_returns) > 0:
                    trend_performance[regime_type] = {
                        'count': len(regime_returns),
                        'total_return': (1 + regime_returns).prod() - 1,
                        'avg_return': regime_returns.mean(),
                        'volatility': regime_returns.std() * np.sqrt(252),
                        'sharpe': regime_returns.mean() / regime_returns.std() * np.sqrt(252) if regime_returns.std() > 0 else 0,
                        'win_rate': len(regime_returns[regime_returns > 0]) / len(regime_returns)
                    }
            
            regime_performance['trend'] = trend_performance
        
        print(f"✅ EXITING analyze_regime_performance() at {datetime.now().strftime('%H:%M:%S')}")
        return regime_performance

class AttributionAnalysis:
    """Performance attribution to individual factors."""
    
    def __init__(self, results: BacktestResults):
        print(f"🔄 ENTERING AttributionAnalysis.__init__() at {datetime.now().strftime('%H:%M:%S')}")
        
        self.results = results
        self.trades = results.trades
        
        print(f"✅ EXITING AttributionAnalysis.__init__() at {datetime.now().strftime('%H:%M:%S')}")
    
    def signal_strength_attribution(self) -> Dict[str, Any]:
        """Analyze performance attribution by signal strength."""
        print(f"🔄 ENTERING signal_strength_attribution() at {datetime.now().strftime('%H:%M:%S')}")
        
        if not self.trades:
            return {}
        
        # Group trades by signal strength buckets
        strength_buckets = {
            'weak': [],      # < 1.5
            'moderate': [],  # 1.5 - 2.5
            'strong': [],    # 2.5 - 3.5
            'very_strong': [] # > 3.5
        }
        
        for trade in self.trades:
            strength = trade.signal_strength
            if strength < 1.5:
                strength_buckets['weak'].append(trade)
            elif strength < 2.5:
                strength_buckets['moderate'].append(trade)
            elif strength < 3.5:
                strength_buckets['strong'].append(trade)
            else:
                strength_buckets['very_strong'].append(trade)
        
        # Calculate metrics for each bucket
        attribution = {}
        
        for bucket, trades in strength_buckets.items():
            if trades:
                pnls = [t.pnl for t in trades]
                attribution[bucket] = {
                    'trade_count': len(trades),
                    'total_pnl': sum(pnls),
                    'avg_pnl': np.mean(pnls),
                    'win_rate': len([p for p in pnls if p > 0]) / len(pnls),
                    'profit_factor': (sum([p for p in pnls if p > 0]) / 
                                    abs(sum([p for p in pnls if p < 0]))) if any(p < 0 for p in pnls) else np.inf,
                    'contribution_pct': sum(pnls) / sum([t.pnl for t in self.trades]) if sum([t.pnl for t in self.trades]) != 0 else 0
                }
        
        print(f"✅ EXITING signal_strength_attribution() at {datetime.now().strftime('%H:%M:%S')}")
        return attribution
    
    def pair_attribution(self) -> Dict[str, Any]:
        """Analyze performance attribution by trading pairs."""
        print(f"🔄 ENTERING pair_attribution() at {datetime.now().strftime('%H:%M:%S')}")
        
        pair_performance = {}
        
        for trade in self.trades:
            pair_key = f"{trade.symbol1}-{trade.symbol2}"
            
            if pair_key not in pair_performance:
                pair_performance[pair_key] = {
                    'trades': [],
                    'total_pnl': 0,
                    'trade_count': 0,
                    'wins': 0
                }
            
            pair_performance[pair_key]['trades'].append(trade)
            pair_performance[pair_key]['total_pnl'] += trade.pnl
            pair_performance[pair_key]['trade_count'] += 1
            if trade.pnl > 0:
                pair_performance[pair_key]['wins'] += 1
        
        # Calculate additional metrics
        total_pnl = sum([t.pnl for t in self.trades])
        
        for pair_key in pair_performance:
            pair_data = pair_performance[pair_key]
            pair_data['avg_pnl'] = pair_data['total_pnl'] / pair_data['trade_count']
            pair_data['win_rate'] = pair_data['wins'] / pair_data['trade_count']
            pair_data['contribution_pct'] = pair_data['total_pnl'] / total_pnl if total_pnl != 0 else 0
            
            # Calculate pair-specific metrics
            pnls = [t.pnl for t in pair_data['trades']]
            pair_data['volatility'] = np.std(pnls) if len(pnls) > 1 else 0
            pair_data['sharpe'] = pair_data['avg_pnl'] / pair_data['volatility'] if pair_data['volatility'] > 0 else 0
        
        # Sort by contribution
        sorted_pairs = sorted(pair_performance.items(), key=lambda x: x[1]['contribution_pct'], reverse=True)
        
        print(f"✅ EXITING pair_attribution() at {datetime.now().strftime('%H:%M:%S')}")
        return dict(sorted_pairs)
    
    def temporal_attribution(self) -> Dict[str, Any]:
        """Analyze performance attribution over time."""
        print(f"🔄 ENTERING temporal_attribution() at {datetime.now().strftime('%H:%M:%S')}")
        
        if not self.trades:
            return {}
        
        # Group trades by time periods
        trades_by_month = {}
        trades_by_quarter = {}
        trades_by_year = {}
        
        for trade in self.trades:
            month_key = trade.entry_date.strftime('%Y-%m')
            quarter_key = f"{trade.entry_date.year}-Q{(trade.entry_date.month-1)//3 + 1}"
            year_key = str(trade.entry_date.year)
            
            # Month
            if month_key not in trades_by_month:
                trades_by_month[month_key] = []
            trades_by_month[month_key].append(trade)
            
            # Quarter
            if quarter_key not in trades_by_quarter:
                trades_by_quarter[quarter_key] = []
            trades_by_quarter[quarter_key].append(trade)
            
            # Year
            if year_key not in trades_by_year:
                trades_by_year[year_key] = []
            trades_by_year[year_key].append(trade)
        
        def calculate_period_metrics(trades_dict):
            metrics = {}
            for period, trades in trades_dict.items():
                pnls = [t.pnl for t in trades]
                metrics[period] = {
                    'trade_count': len(trades),
                    'total_pnl': sum(pnls),
                    'avg_pnl': np.mean(pnls),
                    'win_rate': len([p for p in pnls if p > 0]) / len(pnls),
                    'best_trade': max(pnls),
                    'worst_trade': min(pnls)
                }
            return metrics
        
        temporal_attribution = {
            'monthly': calculate_period_metrics(trades_by_month),
            'quarterly': calculate_period_metrics(trades_by_quarter),
            'yearly': calculate_period_metrics(trades_by_year)
        }
        
        print(f"✅ EXITING temporal_attribution() at {datetime.now().strftime('%H:%M:%S')}")
        return temporal_attribution
