"""
Professional Backtesting Engine
==============================

Advanced backtesting system with walk-forward analysis, regime detection,
and comprehensive performance analytics.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .strategies import BaseStrategy, TradingSignal, SignalType
from .execution import TradingEngine, OrderSide, OrderType, RiskLimits

import logging
logger = logging.getLogger(__name__)

@dataclass 
class BacktestConfig:
    """Backtesting configuration."""
    initial_capital: float = 100000
    commission: float = 0.001
    slippage: float = 0.0005
    risk_free_rate: float = 0.02
    benchmark_symbol: str = "SPY"
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    max_positions: int = 10
    position_sizing: str = "equal_weight"  # equal_weight, risk_parity, kelly
    
@dataclass
class Trade:
    """Individual trade record."""
    entry_date: datetime
    exit_date: datetime
    symbol1: str
    symbol2: str
    side: str  # long/short
    entry_price1: float
    entry_price2: float
    exit_price1: float
    exit_price2: float
    quantity1: float
    quantity2: float
    pnl: float
    return_pct: float
    holding_days: int
    max_dd_during_trade: float
    commission: float
    signal_strength: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BacktestResults:
    """Comprehensive backtest results."""
    # Performance metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Portfolio metrics
    portfolio_df: pd.DataFrame
    trades: List[Trade]
    monthly_returns: pd.Series
    drawdown_series: pd.Series
    
    # Risk metrics
    var_95: float
    cvar_95: float
    skewness: float
    kurtosis: float
    
    # Additional metrics
    metadata: Dict[str, Any] = field(default_factory=dict)

class Backtester:
    """Professional backtesting engine."""
    
    def __init__(self, config: BacktestConfig = None):
        print(f"🔄 ENTERING Backtester.__init__() at {datetime.now().strftime('%H:%M:%S')}")
        
        self.config = config or BacktestConfig()
        self.risk_limits = RiskLimits(
            max_position_size=0.2,
            max_daily_loss=0.05,
            max_drawdown=0.20
        )
        
        # Results storage
        self.results = None
        self.portfolio_history = []
        self.trade_history = []
        self.active_positions = {}
        
        print(f"✅ EXITING Backtester.__init__() at {datetime.now().strftime('%H:%M:%S')}")
    
    def run(self, strategy: BaseStrategy, data: Dict[str, pd.DataFrame], 
           start_date: str = None, end_date: str = None) -> BacktestResults:
        """Run backtest with given strategy and data."""
        print(f"🔄 ENTERING run() backtest at {datetime.now().strftime('%H:%M:%S')}")
        
        # Initialize trading engine
        engine = TradingEngine(
            initial_capital=self.config.initial_capital,
            risk_limits=self.risk_limits
        )
        
        # Get date range
        all_dates = self._get_common_date_range(data, start_date, end_date)
        
        if len(all_dates) < 30:
            raise ValueError("Insufficient data for backtesting")
        
        logger.info(f"Backtesting from {all_dates[0]} to {all_dates[-1]} ({len(all_dates)} days)")
        
        # Generate signals
        signals = strategy.generate_signals(data)
        signals_by_date = self._group_signals_by_date(signals)
        
        logger.info(f"Generated {len(signals)} signals across {len(signals_by_date)} dates")
        
        # Initialize portfolio tracking
        portfolio_values = []
        benchmark_values = []
        
        # Get benchmark data if available
        benchmark_data = data.get(self.config.benchmark_symbol)
        
        # Main backtest loop
        for i, current_date in enumerate(all_dates):
            try:
                # Get current prices
                current_prices = {}
                for symbol, df in data.items():
                    if current_date in df.index:
                        current_prices[symbol] = df.loc[current_date, 'Close']
                
                # Update market data
                engine.update_market_data(current_prices)
                
                # Process signals for this date
                if current_date in signals_by_date:
                    for signal in signals_by_date[current_date]:
                        self._process_signal(engine, signal, current_prices)
                
                # Check for position exits
                self._check_position_exits(engine, current_prices, current_date)
                
                # Record portfolio state
                portfolio_summary = engine.get_portfolio_summary()
                portfolio_value = (portfolio_summary['portfolio_metrics']['total_value'] + 
                                 self.config.initial_capital + 
                                 portfolio_summary['portfolio_metrics']['realized_pnl'])
                
                portfolio_values.append({
                    'date': current_date,
                    'portfolio_value': portfolio_value,
                    'positions': len(portfolio_summary['positions']),
                    'unrealized_pnl': portfolio_summary['portfolio_metrics']['unrealized_pnl'],
                    'realized_pnl': portfolio_summary['portfolio_metrics']['realized_pnl']
                })
                
                # Benchmark value
                if benchmark_data is not None and current_date in benchmark_data.index:
                    benchmark_price = benchmark_data.loc[current_date, 'Close']
                    benchmark_return = benchmark_price / benchmark_data.iloc[0]['Close']
                    benchmark_values.append(self.config.initial_capital * benchmark_return)
                else:
                    benchmark_values.append(self.config.initial_capital)
                
                # Periodic logging
                if i % 50 == 0:
                    logger.info(f"Processed {i+1}/{len(all_dates)} days, Portfolio: ${portfolio_value:,.2f}")
            
            except Exception as e:
                logger.error(f"Error processing date {current_date}: {e}")
                continue
        
        # Build results
        portfolio_df = pd.DataFrame(portfolio_values)
        portfolio_df.set_index('date', inplace=True)
        
        # Calculate performance metrics
        results = self._calculate_performance_metrics(portfolio_df, benchmark_values)
        
        logger.info(f"Backtest completed: {results.total_return:.2%} return, {results.sharpe_ratio:.2f} Sharpe")
        
        print(f"✅ EXITING run() backtest at {datetime.now().strftime('%H:%M:%S')}")
        return results
    
    def _get_common_date_range(self, data: Dict[str, pd.DataFrame], 
                              start_date: str = None, end_date: str = None) -> List[datetime]:
        """Get common date range across all data."""
        print(f"🔄 ENTERING _get_common_date_range() at {datetime.now().strftime('%H:%M:%S')}")
        
        # Find common dates
        common_dates = None
        for symbol, df in data.items():
            if common_dates is None:
                common_dates = set(df.index)
            else:
                common_dates = common_dates.intersection(set(df.index))
        
        if not common_dates:
            raise ValueError("No common dates found across data")
        
        # Sort dates
        sorted_dates = sorted(list(common_dates))
        
        # Filter by date range if specified
        if start_date:
            start_dt = pd.to_datetime(start_date)
            sorted_dates = [d for d in sorted_dates if d >= start_dt]
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            sorted_dates = [d for d in sorted_dates if d <= end_dt]
        
        print(f"✅ EXITING _get_common_date_range() [{len(sorted_dates)} dates] at {datetime.now().strftime('%H:%M:%S')}")
        return sorted_dates
    
    def _group_signals_by_date(self, signals: List[TradingSignal]) -> Dict[datetime, List[TradingSignal]]:
        """Group signals by date."""
        signals_by_date = {}
        
        for signal in signals:
            date = signal.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            if date not in signals_by_date:
                signals_by_date[date] = []
            signals_by_date[date].append(signal)
        
        return signals_by_date
    
    def _process_signal(self, engine: TradingEngine, signal: TradingSignal, prices: Dict[str, float]):
        """Process a trading signal."""
        print(f"🔄 ENTERING _process_signal({signal.symbol1}-{signal.symbol2}) at {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            # Check if we already have this pair
            pair_key = f"{signal.symbol1}-{signal.symbol2}"
            if pair_key in self.active_positions:
                logger.debug(f"Already have position in {pair_key}")
                return
            
            # Check if prices are available
            if signal.symbol1 not in prices or signal.symbol2 not in prices:
                logger.warning(f"Prices not available for {pair_key}")
                return
            
            # Calculate position sizes
            portfolio_summary = engine.get_portfolio_summary()
            current_value = (portfolio_summary['portfolio_metrics']['total_value'] + 
                           self.config.initial_capital + 
                           portfolio_summary['portfolio_metrics']['realized_pnl'])
            
            # Use equal weight sizing by default
            position_value = current_value * 0.1  # 10% per position
            
            # Calculate quantities
            price1 = prices[signal.symbol1]
            price2 = prices[signal.symbol2]
            
            quantity1 = position_value / (price1 * (1 + signal.hedge_ratio))
            quantity2 = quantity1 * signal.hedge_ratio
            
            # Submit orders based on signal type
            if signal.signal_type == SignalType.BUY:
                # Long signal1, short signal2
                order1_id = engine.submit_trade(signal.symbol1, OrderSide.BUY, quantity1)
                order2_id = engine.submit_trade(signal.symbol2, OrderSide.SELL, quantity2)
            elif signal.signal_type == SignalType.SELL:
                # Short signal1, long signal2  
                order1_id = engine.submit_trade(signal.symbol1, OrderSide.SELL, quantity1)
                order2_id = engine.submit_trade(signal.symbol2, OrderSide.BUY, quantity2)
            else:
                return  # HOLD signal
            
            # Track active position
            self.active_positions[pair_key] = {
                'signal': signal,
                'entry_date': signal.timestamp,
                'order1_id': order1_id,
                'order2_id': order2_id,
                'quantity1': quantity1,
                'quantity2': quantity2,
                'entry_price1': price1,
                'entry_price2': price2
            }
            
            logger.info(f"Opened position: {pair_key} - {signal.signal_type.name}")
            
            print(f"✅ EXITING _process_signal() at {datetime.now().strftime('%H:%M:%S')}")
        
        except Exception as e:
            logger.error(f"Error processing signal for {signal.symbol1}-{signal.symbol2}: {e}")
    
    def _check_position_exits(self, engine: TradingEngine, prices: Dict[str, float], current_date: datetime):
        """Check for position exit conditions."""
        print(f"🔄 ENTERING _check_position_exits() [{len(self.active_positions)} positions] at {datetime.now().strftime('%H:%M:%S')}")
        
        positions_to_close = []
        
        for pair_key, position in self.active_positions.items():
            try:
                signal = position['signal']
                
                # Check if prices are available
                if signal.symbol1 not in prices or signal.symbol2 not in prices:
                    continue
                
                current_price1 = prices[signal.symbol1]
                current_price2 = prices[signal.symbol2]
                
                # Calculate current spread and z-score
                current_spread = current_price1 - signal.hedge_ratio * current_price2
                entry_spread = position['entry_price1'] - signal.hedge_ratio * position['entry_price2']
                
                # Simple mean reversion exit: close when spread returns to mean
                # This is simplified - in practice, you'd use the same z-score logic as entry
                spread_change = (current_spread - entry_spread) / abs(entry_spread)
                
                # Exit conditions
                should_exit = False
                exit_reason = ""
                
                # Profit target
                if signal.take_profit and abs(spread_change) > signal.take_profit:
                    should_exit = True
                    exit_reason = "profit_target"
                
                # Stop loss
                elif signal.stop_loss and abs(spread_change) > signal.stop_loss:
                    should_exit = True
                    exit_reason = "stop_loss"
                
                # Time-based exit (max holding period)
                elif (current_date - position['entry_date']).days > 30:  # 30 day max hold
                    should_exit = True
                    exit_reason = "time_exit"
                
                # Mean reversion exit (simplified)
                elif abs(spread_change) < 0.001:  # Spread returned to entry level
                    should_exit = True
                    exit_reason = "mean_reversion"
                
                if should_exit:
                    self._close_position(engine, pair_key, position, current_price1, current_price2, 
                                       current_date, exit_reason)
                    positions_to_close.append(pair_key)
            
            except Exception as e:
                logger.error(f"Error checking exit for {pair_key}: {e}")
        
        # Remove closed positions
        for pair_key in positions_to_close:
            del self.active_positions[pair_key]
        
        if positions_to_close:
            logger.info(f"Closed {len(positions_to_close)} positions")
        
        print(f"✅ EXITING _check_position_exits() at {datetime.now().strftime('%H:%M:%S')}")
    
    def _close_position(self, engine: TradingEngine, pair_key: str, position: Dict, 
                       exit_price1: float, exit_price2: float, exit_date: datetime, exit_reason: str):
        """Close a pairs trading position."""
        print(f"🔄 ENTERING _close_position({pair_key}) at {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            signal = position['signal']
            
            # Submit closing orders (reverse of opening)
            if signal.signal_type == SignalType.BUY:
                # Close long signal1, short signal2
                engine.submit_trade(signal.symbol1, OrderSide.SELL, position['quantity1'])
                engine.submit_trade(signal.symbol2, OrderSide.BUY, position['quantity2'])
            else:
                # Close short signal1, long signal2
                engine.submit_trade(signal.symbol1, OrderSide.BUY, position['quantity1'])  
                engine.submit_trade(signal.symbol2, OrderSide.SELL, position['quantity2'])
            
            # Calculate trade P&L
            if signal.signal_type == SignalType.BUY:
                pnl1 = (exit_price1 - position['entry_price1']) * position['quantity1']
                pnl2 = (position['entry_price2'] - exit_price2) * position['quantity2']  # Short position
            else:
                pnl1 = (position['entry_price1'] - exit_price1) * position['quantity1']  # Short position
                pnl2 = (exit_price2 - position['entry_price2']) * position['quantity2']
            
            total_pnl = pnl1 + pnl2
            
            # Calculate return percentage
            entry_value = position['entry_price1'] * position['quantity1'] + position['entry_price2'] * position['quantity2']
            return_pct = total_pnl / entry_value if entry_value > 0 else 0
            
            # Create trade record
            trade = Trade(
                entry_date=position['entry_date'],
                exit_date=exit_date,
                symbol1=signal.symbol1,
                symbol2=signal.symbol2,
                side=signal.signal_type.name,
                entry_price1=position['entry_price1'],
                entry_price2=position['entry_price2'],
                exit_price1=exit_price1,
                exit_price2=exit_price2,
                quantity1=position['quantity1'],
                quantity2=position['quantity2'],
                pnl=total_pnl,
                return_pct=return_pct,
                holding_days=(exit_date - position['entry_date']).days,
                max_dd_during_trade=0.0,  # Would need to track this
                commission=0.0,  # Calculated by engine
                signal_strength=signal.strength,
                metadata={'exit_reason': exit_reason, 'hedge_ratio': signal.hedge_ratio}
            )
            
            self.trade_history.append(trade)
            
            logger.info(f"Closed position {pair_key}: P&L=${total_pnl:.2f} ({return_pct:.2%}) - {exit_reason}")
            
            print(f"✅ EXITING _close_position() at {datetime.now().strftime('%H:%M:%S')}")
        
        except Exception as e:
            logger.error(f"Error closing position {pair_key}: {e}")
    
    def _calculate_performance_metrics(self, portfolio_df: pd.DataFrame, 
                                     benchmark_values: List[float]) -> BacktestResults:
        """Calculate comprehensive performance metrics."""
        print(f"🔄 ENTERING _calculate_performance_metrics() at {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            # Portfolio returns
            portfolio_values = portfolio_df['portfolio_value'].values
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            
            # Basic metrics
            total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
            n_days = len(portfolio_values)
            annualized_return = (1 + total_return) ** (252 / n_days) - 1
            volatility = np.std(returns) * np.sqrt(252)
            
            # Risk metrics
            downside_returns = returns[returns < 0]
            downside_volatility = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
            
            sharpe_ratio = (annualized_return - self.config.risk_free_rate) / volatility if volatility > 0 else 0
            sortino_ratio = (annualized_return - self.config.risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
            
            # Drawdown
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdowns)
            
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Trade statistics
            winning_trades = len([t for t in self.trade_history if t.pnl > 0])
            losing_trades = len([t for t in self.trade_history if t.pnl < 0])
            total_trades = len(self.trade_history)
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            if winning_trades > 0:
                avg_win = np.mean([t.pnl for t in self.trade_history if t.pnl > 0])
            else:
                avg_win = 0
            
            if losing_trades > 0:
                avg_loss = abs(np.mean([t.pnl for t in self.trade_history if t.pnl < 0]))
            else:
                avg_loss = 0
            
            profit_factor = avg_win / avg_loss if avg_loss > 0 else np.inf
            
            # Risk metrics
            if len(returns) > 0:
                var_95 = np.percentile(returns, 5)
                cvar_95 = np.mean(returns[returns <= var_95]) if np.any(returns <= var_95) else var_95
                skewness = stats.skew(returns)
                kurtosis = stats.kurtosis(returns)
            else:
                var_95 = cvar_95 = skewness = kurtosis = 0
            
            # Monthly returns
            portfolio_df['month'] = portfolio_df.index.to_period('M')
            monthly_values = portfolio_df.groupby('month')['portfolio_value'].last()
            monthly_returns = monthly_values.pct_change().dropna()
            
            # Create results
            results = BacktestResults(
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                calmar_ratio=calmar_ratio,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_factor=profit_factor,
                portfolio_df=portfolio_df,
                trades=self.trade_history,
                monthly_returns=monthly_returns,
                drawdown_series=pd.Series(drawdowns, index=portfolio_df.index[1:]),
                var_95=var_95,
                cvar_95=cvar_95,
                skewness=skewness,
                kurtosis=kurtosis,
                metadata={
                    'benchmark_return': (benchmark_values[-1] - benchmark_values[0]) / benchmark_values[0] if benchmark_values else 0,
                    'excess_return': total_return - ((benchmark_values[-1] - benchmark_values[0]) / benchmark_values[0] if benchmark_values else 0)
                }
            )
            
            print(f"✅ EXITING _calculate_performance_metrics() at {datetime.now().strftime('%H:%M:%S')}")
            return results
        
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            # Return minimal results on error
            return BacktestResults(
                total_return=0, annualized_return=0, volatility=0, sharpe_ratio=0,
                sortino_ratio=0, max_drawdown=0, calmar_ratio=0, total_trades=0,
                winning_trades=0, losing_trades=0, win_rate=0, avg_win=0, avg_loss=0,
                profit_factor=0, portfolio_df=portfolio_df, trades=[], 
                monthly_returns=pd.Series(), drawdown_series=pd.Series(),
                var_95=0, cvar_95=0, skewness=0, kurtosis=0
            )

class WalkForwardOptimizer:
    """Walk-forward optimization for strategy parameters."""
    
    def __init__(self, backtester: Backtester):
        print(f"🔄 ENTERING WalkForwardOptimizer.__init__() at {datetime.now().strftime('%H:%M:%S')}")
        
        self.backtester = backtester
        self.optimization_history = []
        
        print(f"✅ EXITING WalkForwardOptimizer.__init__() at {datetime.now().strftime('%H:%M:%S')}")
    
    def optimize(self, strategy_class, data: Dict[str, pd.DataFrame], 
                parameter_ranges: Dict[str, List], 
                optimization_window: int = 252,  # 1 year
                forward_window: int = 63,       # 3 months
                objective: str = "sharpe_ratio") -> Dict[str, Any]:
        """Run walk-forward optimization."""
        print(f"🔄 ENTERING optimize() walk-forward at {datetime.now().strftime('%H:%M:%S')}")
        
        # Get date range
        all_dates = self.backtester._get_common_date_range(data)
        
        if len(all_dates) < optimization_window + forward_window:
            raise ValueError("Insufficient data for walk-forward optimization")
        
        optimization_results = []
        
        # Walk-forward loop
        start_idx = 0
        while start_idx + optimization_window + forward_window <= len(all_dates):
            # Define periods
            opt_start = all_dates[start_idx]
            opt_end = all_dates[start_idx + optimization_window - 1]
            forward_start = all_dates[start_idx + optimization_window]
            forward_end = all_dates[min(start_idx + optimization_window + forward_window - 1, len(all_dates) - 1)]
            
            logger.info(f"Optimizing {opt_start} to {opt_end}, Forward testing {forward_start} to {forward_end}")
            
            # Optimize parameters on in-sample data
            best_params = self._optimize_parameters(
                strategy_class, data, parameter_ranges, objective,
                opt_start.strftime('%Y-%m-%d'), opt_end.strftime('%Y-%m-%d')
            )
            
            # Test on out-of-sample data
            strategy = strategy_class(best_params)
            forward_results = self.backtester.run(
                strategy, data, 
                forward_start.strftime('%Y-%m-%d'), forward_end.strftime('%Y-%m-%d')
            )
            
            optimization_results.append({
                'optimization_period': (opt_start, opt_end),
                'forward_period': (forward_start, forward_end),
                'best_parameters': best_params,
                'forward_results': forward_results,
                'forward_sharpe': forward_results.sharpe_ratio,
                'forward_return': forward_results.total_return
            })
            
            # Move forward
            start_idx += forward_window
        
        # Analyze results
        forward_sharpes = [r['forward_sharpe'] for r in optimization_results]
        forward_returns = [r['forward_return'] for r in optimization_results]
        
        summary = {
            'optimization_results': optimization_results,
            'avg_forward_sharpe': np.mean(forward_sharpes),
            'avg_forward_return': np.mean(forward_returns),
            'sharpe_std': np.std(forward_sharpes),
            'return_std': np.std(forward_returns),
            'hit_rate': len([s for s in forward_sharpes if s > 0]) / len(forward_sharpes)
        }
        
        logger.info(f"Walk-forward completed: Avg Sharpe={summary['avg_forward_sharpe']:.2f}, Hit Rate={summary['hit_rate']:.2%}")
        
        print(f"✅ EXITING optimize() walk-forward at {datetime.now().strftime('%H:%M:%S')}")
        return summary
    
    def _optimize_parameters(self, strategy_class, data: Dict[str, pd.DataFrame], 
                           parameter_ranges: Dict[str, List], objective: str,
                           start_date: str, end_date: str):
        """Optimize strategy parameters using grid search."""
        print(f"🔄 ENTERING _optimize_parameters() at {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            from itertools import product
            from .strategies import StrategyParameters
            
            # Generate parameter combinations
            param_names = list(parameter_ranges.keys())
            param_values = list(parameter_ranges.values())
            
            best_score = -np.inf
            best_params = None
            
            total_combinations = np.prod([len(vals) for vals in param_values])
            logger.info(f"Testing {total_combinations} parameter combinations")
            
            for i, param_combo in enumerate(product(*param_values)):
                try:
                    # Create parameters object
                    param_dict = dict(zip(param_names, param_combo))
                    
                    # Convert to StrategyParameters if needed
                    if hasattr(strategy_class, '__init__'):
                        params = StrategyParameters(**param_dict)
                    else:
                        params = param_dict
                    
                    # Create strategy and run backtest
                    strategy = strategy_class(params)
                    results = self.backtester.run(strategy, data, start_date, end_date)
                    
                    # Get objective score
                    score = getattr(results, objective, 0)
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                    
                    if i % 10 == 0:
                        logger.debug(f"Tested {i}/{total_combinations} combinations, best {objective}={best_score:.3f}")
                
                except Exception as e:
                    logger.warning(f"Error testing parameters {param_combo}: {e}")
                    continue
            
            logger.info(f"Best parameters found: {objective}={best_score:.3f}")
            
            print(f"✅ EXITING _optimize_parameters() at {datetime.now().strftime('%H:%M:%S')}")
            return best_params
            
        except Exception as e:
            logger.error(f"Error in parameter optimization: {e}")
            # Return default parameters
            from .strategies import StrategyParameters
            return StrategyParameters()
