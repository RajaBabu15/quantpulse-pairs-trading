#!/usr/bin/env python3
"""
QuantPulse L2 Loss Optimization System
=====================================

Advanced optimization system using L2 regularization for pairs trading strategies.
Features:
- L2 regularized parameter optimization
- Cross-validation with time series splits
- Bayesian hyperparameter optimization
- Risk-adjusted objective functions
- Portfolio-level optimization
- Advanced regularization techniques

Author: QuantPulse Trading Systems
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
import optuna
from functools import partial
import time
import json

# Import our trading system
from run import HFTPairsTrader, PairsTrader, get_optimal_config, HFT_AVAILABLE
from performance_analyzer import generate_random_pairs, QuantPulsePerformanceAnalyzer

# ============================================================================
# L2 REGULARIZED OPTIMIZATION FRAMEWORK
# ============================================================================

class L2OptimizedPairsTrader:
    """
    Pairs trader with L2 regularized parameter optimization
    """
    
    def __init__(self, symbol1, symbol2, regularization_strength=0.001, use_hft=True):
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.reg_strength = regularization_strength
        self.use_hft = use_hft and HFT_AVAILABLE
        
        # Optimization bounds
        self.param_bounds = {
            'lookback': (5, 60),
            'z_entry': (0.5, 4.0),
            'z_exit': (0.1, 2.0),
            'position_size': (1000, 50000),
            'transaction_cost': (0.0001, 0.005)
        }
        
        # Best parameters
        self.best_params = None
        self.optimization_history = []
        self.cv_scores = []
        
    def objective_function(self, params, prices_train, prices_val, return_details=False):
        """
        L2 regularized objective function for parameter optimization
        
        Args:
            params: [lookback, z_entry, z_exit, position_size, transaction_cost]
            prices_train: Training price data
            prices_val: Validation price data
            return_details: Whether to return detailed metrics
        """
        lookback, z_entry, z_exit, position_size, transaction_cost = params
        
        # Ensure parameters are within bounds
        lookback = max(5, min(60, int(lookback)))
        z_entry = max(0.5, min(4.0, abs(z_entry)))
        z_exit = max(0.1, min(2.0, abs(z_exit)))
        position_size = max(1000, min(50000, abs(position_size)))
        transaction_cost = max(0.0001, min(0.005, abs(transaction_cost)))
        
        # Ensure logical parameter relationships
        if z_exit >= z_entry:
            z_exit = z_entry * 0.5
        
        try:
            # Create trader with current parameters
            if self.use_hft:
                trader = HFTPairsTrader(
                    self.symbol1, self.symbol2,
                    lookback=lookback,
                    z_entry=z_entry,
                    z_exit=z_exit,
                    position_size=position_size,
                    transaction_cost=transaction_cost
                )
            else:
                trader = PairsTrader(
                    self.symbol1, self.symbol2,
                    lookback=lookback,
                    z_entry=z_entry,
                    z_exit=z_exit,
                    position_size=position_size,
                    transaction_cost=transaction_cost
                )
            
            # Train on training data
            if hasattr(trader, 'backtest_hft') and self.use_hft:
                train_results = trader.backtest_hft_with_data(prices_train, hybrid_mode=True)
            else:
                train_results = trader.backtest_with_data(prices_train)
            
            # Validate on validation data
            trader_val = type(trader)(
                self.symbol1, self.symbol2,
                lookback=lookback,
                z_entry=z_entry,
                z_exit=z_exit,
                position_size=position_size,
                transaction_cost=transaction_cost
            )
            
            if hasattr(trader_val, 'backtest_hft') and self.use_hft:
                val_results = trader_val.backtest_hft_with_data(prices_val, hybrid_mode=True)
            else:
                val_results = trader_val.backtest_with_data(prices_val)
            
            # Calculate primary metrics
            train_pnl = train_results.get('final_pnl', 0)
            val_pnl = val_results.get('final_pnl', 0)
            train_sharpe = train_results.get('sharpe_ratio', 0)
            val_sharpe = val_results.get('sharpe_ratio', 0)
            train_win_rate = train_results.get('win_rate', 0)
            val_win_rate = val_results.get('win_rate', 0)
            train_max_dd = train_results.get('max_drawdown', 0)
            val_max_dd = val_results.get('max_drawdown', 0)
            
            # Risk-adjusted return (primary objective)
            val_risk_adj_return = val_pnl / (abs(val_max_dd) + 1e6)  # Avoid division by zero
            
            # L2 Regularization terms
            param_vector = np.array([
                lookback / 60,  # Normalize
                z_entry / 4,
                z_exit / 2,
                position_size / 50000,
                transaction_cost / 0.005
            ])
            
            l2_penalty = self.reg_strength * np.sum(param_vector**2)
            
            # Overfitting penalty (difference between train and validation performance)
            overfitting_penalty = abs(train_sharpe - val_sharpe) * 0.1
            
            # Combine objectives (maximize validation performance, minimize L2 penalty)
            objective = val_risk_adj_return - l2_penalty - overfitting_penalty
            
            if return_details:
                return {
                    'objective': objective,
                    'val_pnl': val_pnl,
                    'val_sharpe': val_sharpe,
                    'val_win_rate': val_win_rate,
                    'val_max_dd': val_max_dd,
                    'val_risk_adj_return': val_risk_adj_return,
                    'train_pnl': train_pnl,
                    'train_sharpe': train_sharpe,
                    'l2_penalty': l2_penalty,
                    'overfitting_penalty': overfitting_penalty,
                    'params': {
                        'lookback': lookback,
                        'z_entry': z_entry,
                        'z_exit': z_exit,
                        'position_size': position_size,
                        'transaction_cost': transaction_cost
                    }
                }
            
            return -objective  # Minimize (negative of maximize)
            
        except Exception as e:
            # Return high penalty for invalid parameter combinations
            return 1e6
    
    def optimize_with_cross_validation(self, prices, n_splits=3, method='bayesian', n_trials=100):
        """
        Optimize parameters using time series cross-validation with L2 regularization
        """
        print(f"🔧 Optimizing {self.symbol1}-{self.symbol2} with L2 regularization...")
        print(f"📊 Method: {method.upper()}, Splits: {n_splits}, Trials: {n_trials}")
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_results = []
        
        if method == 'bayesian':
            # Bayesian optimization with Optuna
            def optuna_objective(trial):
                lookback = trial.suggest_int('lookback', *self.param_bounds['lookback'])
                z_entry = trial.suggest_float('z_entry', *self.param_bounds['z_entry'])
                z_exit = trial.suggest_float('z_exit', *self.param_bounds['z_exit'])
                position_size = trial.suggest_int('position_size', *self.param_bounds['position_size'])
                transaction_cost = trial.suggest_float('transaction_cost', *self.param_bounds['transaction_cost'])
                
                params = [lookback, z_entry, z_exit, position_size, transaction_cost]
                
                # Cross-validation
                cv_scores = []
                for train_idx, val_idx in tscv.split(prices):
                    prices_train = prices.iloc[train_idx]
                    prices_val = prices.iloc[val_idx]
                    
                    score = self.objective_function(params, prices_train, prices_val)
                    cv_scores.append(-score)  # Convert back to maximize
                
                return np.mean(cv_scores)
            
            study = optuna.create_study(direction='maximize')
            study.optimize(optuna_objective, n_trials=n_trials, show_progress_bar=True)
            
            best_params = [
                study.best_params['lookback'],
                study.best_params['z_entry'],
                study.best_params['z_exit'],
                study.best_params['position_size'],
                study.best_params['transaction_cost']
            ]
            
            # Store optimization history
            self.optimization_history = [
                {
                    'trial': i,
                    'value': trial.value,
                    'params': trial.params
                }
                for i, trial in enumerate(study.trials)
            ]
            
        elif method == 'differential_evolution':
            # Differential Evolution optimization
            bounds = [
                self.param_bounds['lookback'],
                self.param_bounds['z_entry'],
                self.param_bounds['z_exit'],
                self.param_bounds['position_size'],
                self.param_bounds['transaction_cost']
            ]
            
            def de_objective(params):
                cv_scores = []
                for train_idx, val_idx in tscv.split(prices):
                    prices_train = prices.iloc[train_idx]
                    prices_val = prices.iloc[val_idx]
                    
                    score = self.objective_function(params, prices_train, prices_val)
                    cv_scores.append(score)
                
                return np.mean(cv_scores)
            
            result = differential_evolution(de_objective, bounds, maxiter=n_trials//10, seed=42)
            best_params = result.x
            
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Final validation on full dataset splits
        final_cv_scores = []
        for train_idx, val_idx in tscv.split(prices):
            prices_train = prices.iloc[train_idx]
            prices_val = prices.iloc[val_idx]
            
            result = self.objective_function(best_params, prices_train, prices_val, return_details=True)
            final_cv_scores.append(result)
        
        self.best_params = {
            'lookback': int(best_params[0]),
            'z_entry': best_params[1],
            'z_exit': best_params[2],
            'position_size': int(best_params[3]),
            'transaction_cost': best_params[4]
        }
        
        self.cv_scores = final_cv_scores
        
        # Summary statistics
        avg_val_pnl = np.mean([s['val_pnl'] for s in final_cv_scores])
        avg_val_sharpe = np.mean([s['val_sharpe'] for s in final_cv_scores])
        avg_val_win_rate = np.mean([s['val_win_rate'] for s in final_cv_scores])
        
        print(f"✅ Optimization Complete!")
        print(f"📈 Average Validation P&L: ${avg_val_pnl:,.2f}")
        print(f"📊 Average Validation Sharpe: {avg_val_sharpe:.3f}")
        print(f"🎯 Average Validation Win Rate: {avg_val_win_rate:.1%}")
        print(f"🔧 Best Parameters: {self.best_params}")
        
        return self.best_params, self.cv_scores

# ============================================================================
# PORTFOLIO L2 OPTIMIZATION
# ============================================================================

class PortfolioL2Optimizer:
    """
    Portfolio-level L2 optimization for multiple pairs
    """
    
    def __init__(self, pairs_data, regularization_strength=0.001):
        self.pairs_data = pairs_data  # Dict of {pair_name: price_data}
        self.reg_strength = regularization_strength
        self.optimized_pairs = {}
        self.portfolio_results = {}
        
    def optimize_all_pairs(self, method='bayesian', n_trials=50, n_splits=3):
        """
        Optimize all pairs using L2 regularization
        """
        print("🚀 PORTFOLIO L2 OPTIMIZATION")
        print("=" * 50)
        print(f"📊 Optimizing {len(self.pairs_data)} pairs")
        print(f"🔧 Method: {method.upper()}")
        print(f"⚡ Regularization: {self.reg_strength}")
        print()
        
        start_time = time.time()
        
        for i, (pair_name, prices) in enumerate(self.pairs_data.items(), 1):
            print(f"📈 {i}/{len(self.pairs_data)}: Optimizing {pair_name}")
            
            # Extract symbols
            symbol1, symbol2 = pair_name.split('-')
            
            try:
                # Create optimizer
                optimizer = L2OptimizedPairsTrader(
                    symbol1, symbol2,
                    regularization_strength=self.reg_strength
                )
                
                # Optimize with cross-validation
                best_params, cv_scores = optimizer.optimize_with_cross_validation(
                    prices, n_splits=n_splits, method=method, n_trials=n_trials
                )
                
                self.optimized_pairs[pair_name] = {
                    'optimizer': optimizer,
                    'best_params': best_params,
                    'cv_scores': cv_scores,
                    'avg_val_pnl': np.mean([s['val_pnl'] for s in cv_scores]),
                    'avg_val_sharpe': np.mean([s['val_sharpe'] for s in cv_scores]),
                    'avg_val_win_rate': np.mean([s['val_win_rate'] for s in cv_scores])
                }
                
                print(f"✅ {pair_name}: ${self.optimized_pairs[pair_name]['avg_val_pnl']:,.2f} avg P&L")
                
            except Exception as e:
                print(f"❌ {pair_name}: Failed - {e}")
                continue
            
            print()
        
        total_time = time.time() - start_time
        
        print(f"✅ Portfolio optimization complete!")
        print(f"⏱️ Total time: {total_time:.2f}s")
        print(f"✅ Successfully optimized: {len(self.optimized_pairs)}/{len(self.pairs_data)} pairs")
        
        # Portfolio summary
        self.analyze_portfolio()
        
        return self.optimized_pairs
    
    def analyze_portfolio(self):
        """
        Analyze optimized portfolio performance
        """
        if not self.optimized_pairs:
            print("❌ No optimized pairs to analyze")
            return
        
        print("\n📊 L2 OPTIMIZED PORTFOLIO ANALYSIS")
        print("=" * 40)
        
        # Aggregate metrics
        total_avg_pnl = sum(pair['avg_val_pnl'] for pair in self.optimized_pairs.values())
        total_pairs = len(self.optimized_pairs)
        profitable_pairs = sum(1 for pair in self.optimized_pairs.values() if pair['avg_val_pnl'] > 0)
        
        avg_sharpe = np.mean([pair['avg_val_sharpe'] for pair in self.optimized_pairs.values()])
        avg_win_rate = np.mean([pair['avg_val_win_rate'] for pair in self.optimized_pairs.values()])
        
        print(f"💰 Total Average Portfolio P&L: ${total_avg_pnl:,.2f}")
        print(f"📊 Profitable Pairs: {profitable_pairs}/{total_pairs} ({profitable_pairs/total_pairs:.1%})")
        print(f"📈 Average Sharpe Ratio: {avg_sharpe:.3f}")
        print(f"🎯 Average Win Rate: {avg_win_rate:.1%}")
        print()
        
        # Top performers
        sorted_pairs = sorted(self.optimized_pairs.items(), 
                            key=lambda x: x[1]['avg_val_pnl'], reverse=True)
        
        print("🏆 TOP 5 L2 OPTIMIZED PERFORMERS")
        print("-" * 35)
        for i, (pair_name, data) in enumerate(sorted_pairs[:5], 1):
            print(f"{i}. {pair_name}: ${data['avg_val_pnl']:,.2f} "
                  f"(Sharpe: {data['avg_val_sharpe']:.3f}, "
                  f"Win: {data['avg_val_win_rate']:.1%})")
        
        print()
        
        # Parameter analysis
        self.analyze_optimal_parameters()
        
        # Store results
        self.portfolio_results = {
            'total_avg_pnl': total_avg_pnl,
            'profitable_pairs': profitable_pairs,
            'total_pairs': total_pairs,
            'profitability_rate': profitable_pairs / total_pairs,
            'avg_sharpe': avg_sharpe,
            'avg_win_rate': avg_win_rate,
            'top_performers': sorted_pairs[:10]
        }
    
    def analyze_optimal_parameters(self):
        """
        Analyze the distribution of optimal parameters across pairs
        """
        print("🔍 OPTIMAL PARAMETER ANALYSIS")
        print("-" * 30)
        
        # Collect all parameters
        params_data = []
        for pair_name, data in self.optimized_pairs.items():
            params = data['best_params'].copy()
            params['pair'] = pair_name
            params['avg_pnl'] = data['avg_val_pnl']
            params_data.append(params)
        
        if not params_data:
            return
        
        df = pd.DataFrame(params_data)
        
        # Parameter statistics
        param_stats = df[['lookback', 'z_entry', 'z_exit', 'position_size', 'transaction_cost']].describe()
        
        print("📊 Parameter Distribution Summary:")
        print(param_stats.round(4))
        print()
        
        # Correlation with performance
        correlations = df[['lookback', 'z_entry', 'z_exit', 'position_size', 'transaction_cost', 'avg_pnl']].corr()['avg_pnl'].sort_values(ascending=False)
        
        print("🔗 Parameter-Performance Correlations:")
        for param, corr in correlations.items():
            if param != 'avg_pnl':
                print(f"  {param}: {corr:.3f}")
        print()
    
    def create_optimization_plots(self, save_plots=True):
        """
        Create comprehensive optimization analysis plots
        """
        if not self.optimized_pairs:
            print("❌ No optimized pairs to plot")
            return
        
        # Set up plotting
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Collect data
        pair_names = list(self.optimized_pairs.keys())
        avg_pnls = [self.optimized_pairs[pair]['avg_val_pnl'] for pair in pair_names]
        avg_sharpes = [self.optimized_pairs[pair]['avg_val_sharpe'] for pair in pair_names]
        avg_win_rates = [self.optimized_pairs[pair]['avg_val_win_rate'] for pair in pair_names]
        
        # Parameter data
        params_data = []
        for pair_name, data in self.optimized_pairs.items():
            params = data['best_params'].copy()
            params['avg_pnl'] = data['avg_val_pnl']
            params_data.append(params)
        
        params_df = pd.DataFrame(params_data)
        
        # 1. Portfolio Performance Distribution
        ax1 = fig.add_subplot(gs[0, 0:2])
        ax1.hist([p/1000 for p in avg_pnls], bins=20, alpha=0.7, edgecolor='black')
        ax1.set_title('📊 L2 Optimized P&L Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Average P&L ($K)')
        ax1.set_ylabel('Number of Pairs')
        ax1.axvline(np.mean(avg_pnls)/1000, color='red', linestyle='--', 
                   label=f'Mean: ${np.mean(avg_pnls)/1000:.1f}K')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Sharpe Ratio vs P&L
        ax2 = fig.add_subplot(gs[0, 2:4])
        scatter = ax2.scatter([p/1000 for p in avg_pnls], avg_sharpes, 
                             c=avg_win_rates, cmap='RdYlGn', alpha=0.7, s=60)
        ax2.set_title('📈 Risk-Adjusted Returns', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Average P&L ($K)')
        ax2.set_ylabel('Sharpe Ratio')
        plt.colorbar(scatter, ax=ax2, label='Win Rate')
        ax2.grid(True, alpha=0.3)
        
        # 3. Parameter Distributions
        param_names = ['lookback', 'z_entry', 'z_exit', 'position_size', 'transaction_cost']
        for i, param in enumerate(param_names):
            row = (i // 2) + 1
            col = (i % 2) * 2
            ax = fig.add_subplot(gs[row, col:col+2])
            
            values = params_df[param].values
            if param == 'position_size':
                values = values / 1000  # Convert to thousands
                ax.set_xlabel(f'{param.replace("_", " ").title()} ($K)')
            else:
                ax.set_xlabel(param.replace("_", " ").title())
            
            ax.hist(values, bins=15, alpha=0.7, edgecolor='black')
            ax.set_title(f'🔧 Optimal {param.replace("_", " ").title()} Distribution', 
                        fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of Pairs')
            ax.axvline(np.mean(values), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(values):.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. Performance vs Regularization
        ax6 = fig.add_subplot(gs[3, :2])
        profitable_pairs = [p for p in avg_pnls if p > 0]
        losing_pairs = [p for p in avg_pnls if p <= 0]
        
        ax6.bar(['Profitable', 'Losing'], [len(profitable_pairs), len(losing_pairs)], 
               color=['green', 'red'], alpha=0.7)
        ax6.set_title('🎯 L2 Optimization Success Rate', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Number of Pairs')
        
        for i, (label, count) in enumerate([('Profitable', len(profitable_pairs)), 
                                          ('Losing', len(losing_pairs))]):
            ax6.text(i, count + 0.1, f'{count}\n({count/len(avg_pnls):.1%})', 
                    ha='center', va='bottom', fontweight='bold')
        
        ax6.grid(True, alpha=0.3, axis='y')
        
        # 5. Top Performers
        ax7 = fig.add_subplot(gs[3, 2:])
        top_pairs = sorted(zip(pair_names, avg_pnls), key=lambda x: x[1], reverse=True)[:10]
        names = [pair.replace('-', '\nvs\n') for pair, _ in top_pairs]
        values = [pnl/1000 for _, pnl in top_pairs]
        
        bars = ax7.bar(range(len(names)), values, color='skyblue', alpha=0.8)
        ax7.set_title('🏆 Top 10 L2 Optimized Pairs', fontsize=14, fontweight='bold')
        ax7.set_xlabel('Pair Rank')
        ax7.set_ylabel('Average P&L ($K)')
        ax7.set_xticks(range(len(names)))
        ax7.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        
        # Color bars by performance
        max_val = max(values)
        for bar, val in zip(bars, values):
            bar.set_color(plt.cm.Greens(0.4 + 0.6 * val / max_val))
        
        ax7.grid(True, alpha=0.3, axis='y')
        
        # Overall title
        fig.suptitle(f'QuantPulse L2 Optimization Analysis - {len(self.optimized_pairs)} Pairs\n'
                    f'Total Portfolio: ${sum(avg_pnls):,.2f} | '
                    f'Success Rate: {len(profitable_pairs)/len(avg_pnls):.1%} | '
                    f'L2 Regularization: {self.reg_strength}',
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_plots:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'quantpulse_l2_optimization_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"📊 L2 optimization plots saved: {filename}")
        
        plt.show()
    
    def save_results(self, filename=None):
        """
        Save optimization results to JSON
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'l2_optimization_results_{timestamp}.json'
        
        # Prepare data for JSON serialization
        results = {
            'optimization_config': {
                'regularization_strength': self.reg_strength,
                'total_pairs': len(self.pairs_data),
                'successful_pairs': len(self.optimized_pairs)
            },
            'portfolio_summary': self.portfolio_results,
            'optimized_pairs': {}
        }
        
        for pair_name, data in self.optimized_pairs.items():
            results['optimized_pairs'][pair_name] = {
                'best_params': data['best_params'],
                'avg_val_pnl': data['avg_val_pnl'],
                'avg_val_sharpe': data['avg_val_sharpe'],
                'avg_val_win_rate': data['avg_val_win_rate']
            }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {filename}")
        return filename

# ============================================================================
# ENHANCED TRADERS WITH L2 OPTIMIZATION
# ============================================================================

def add_l2_methods_to_traders():
    """
    Add L2 optimization methods to existing trader classes
    """
    
    def backtest_with_data(self, prices):
        """Backtest with provided price data"""
        if len(prices) < self.lookback + 10:
            raise ValueError("Not enough data for backtest")
        
        # Calculate spread using provided data
        spread, z_score = self.calculate_spread_stats(prices)
        
        # Reset trading state
        self.position = 0
        self.pnl = 0
        self.trades = []
        self.equity_curve = []
        
        # Trading loop (same as original but with provided data)
        for i in range(self.lookback, len(prices)):
            date = prices.index[i]
            current_spread = spread.iloc[i]
            current_z = z_score.iloc[i]
            
            if pd.isna(current_z):
                continue
                
            # Trading logic (same as parent backtest method)
            if self.position == 0:
                if current_z > self.z_entry:
                    self.position = -1
                    self.entry_price = current_spread
                    cost = self.position_size * self.transaction_cost
                    self.pnl -= cost
                    self.trades.append({
                        'date': date,
                        'action': 'SHORT_ENTRY',
                        'spread': current_spread,
                        'z_score': current_z,
                        'cost': cost
                    })
                elif current_z < -self.z_entry:
                    self.position = 1
                    self.entry_price = current_spread
                    cost = self.position_size * self.transaction_cost
                    self.pnl -= cost
                    self.trades.append({
                        'date': date,
                        'action': 'LONG_ENTRY',
                        'spread': current_spread,
                        'z_score': current_z,
                        'cost': cost
                    })
            else:
                exit_signal = False
                reason = ''
                
                if abs(current_z) < self.z_exit:
                    exit_signal = True
                    reason = 'MEAN_REVERSION'
                elif (self.position == 1 and current_z < -self.z_entry * 1.5) or \
                     (self.position == -1 and current_z > self.z_entry * 1.5):
                    exit_signal = True
                    reason = 'STOP_LOSS'
                
                if exit_signal:
                    spread_change = current_spread - self.entry_price
                    trade_pnl = self.position * spread_change * self.position_size
                    cost = self.position_size * self.transaction_cost
                    
                    self.pnl += trade_pnl - cost
                    
                    self.trades.append({
                        'date': date,
                        'action': f'{reason}_EXIT',
                        'spread': current_spread,
                        'z_score': current_z,
                        'trade_pnl': trade_pnl,
                        'cost': cost,
                        'total_pnl': self.pnl
                    })
                    
                    self.position = 0
                    self.entry_price = 0
            
            # Record equity curve
            unrealized_pnl = 0
            if self.position != 0:
                spread_change = current_spread - self.entry_price
                unrealized_pnl = self.position * spread_change * self.position_size
            
            self.equity_curve.append({
                'date': date,
                'realized_pnl': self.pnl,
                'unrealized_pnl': unrealized_pnl,
                'total_pnl': self.pnl + unrealized_pnl,
                'position': self.position,
                'spread': current_spread,
                'z_score': current_z
            })
        
        return self.analyze_results()
    
    def backtest_hft_with_data(self, prices, hybrid_mode=True):
        """HFT backtest with provided price data"""
        # Similar to backtest_hft but uses provided prices instead of downloading
        if len(prices) < self.lookback + 10:
            raise ValueError("Not enough data for backtest")
        
        # Calculate traditional spread
        spread, z_score_traditional = self.calculate_spread_stats(prices)
        
        # Convert to HFT format
        hft_batches = self._convert_to_hft_format(prices)
        
        # Reset trading state
        self.position = 0
        self.pnl = 0
        self.trades = []
        self.equity_curve = []
        
        # Track signals
        hft_signals = []
        traditional_signals = []
        
        # Price/return history
        prices1_history = []
        prices2_history = []
        returns1_history = []
        returns2_history = []
        
        # Trading loop (simplified version of original)
        for i in range(self.lookback, len(prices)):
            date = prices.index[i]
            current_spread = spread.iloc[i]
            current_z_traditional = z_score_traditional.iloc[i]
            
            if pd.isna(current_z_traditional):
                continue
            
            # Build history
            prices1_history.append(prices['price1'].iloc[i])
            prices2_history.append(prices['price2'].iloc[i])
            
            if len(prices1_history) > 1:
                ret1 = (prices1_history[-1] / prices1_history[-2]) - 1
                ret2 = (prices2_history[-1] / prices2_history[-2]) - 1
                returns1_history.append(ret1)
                returns2_history.append(ret2)
            
            # Get HFT signal
            if self.use_hft_strategies and hft_batches and i - self.lookback < len(hft_batches):
                batch_prices, batch_symbols = hft_batches[i - self.lookback]
                hft_result = self._process_hft_batch(batch_prices, batch_symbols)
                current_z_hft = hft_result['z_score']
                
                # Advanced strategies signal (simplified)
                bid = prices['price1'].iloc[i] * 0.9995
                ask = prices['price1'].iloc[i] * 1.0005
                volume = 1000
                
                try:
                    advanced_signal, _ = self._get_advanced_signal(
                        prices1_history, prices2_history, returns1_history, returns2_history,
                        bid, ask, volume
                    )
                except:
                    advanced_signal = 0
            else:
                current_z_hft = current_z_traditional
                advanced_signal = 0
            
            # Combine signals
            if hybrid_mode:
                final_z_score = 0.7 * current_z_traditional + 0.3 * current_z_hft
                current_z = final_z_score
            else:
                current_z = current_z_hft if self.use_hft_strategies else current_z_traditional
            
            # Store signals
            hft_signals.append(current_z_hft)
            traditional_signals.append(current_z_traditional)
            
            # Trading logic (same as parent)
            if self.position == 0:
                if current_z > self.z_entry:
                    self.position = -1
                    self.entry_price = current_spread
                    cost = self.position_size * self.transaction_cost
                    self.pnl -= cost
                    
                    self.trades.append({
                        'date': date,
                        'action': 'HFT_SHORT_ENTRY',
                        'spread': current_spread,
                        'z_score': current_z,
                        'cost': cost
                    })
                elif current_z < -self.z_entry:
                    self.position = 1
                    self.entry_price = current_spread
                    cost = self.position_size * self.transaction_cost
                    self.pnl -= cost
                    
                    self.trades.append({
                        'date': date,
                        'action': 'HFT_LONG_ENTRY',
                        'spread': current_spread,
                        'z_score': current_z,
                        'cost': cost
                    })
            else:
                exit_signal = False
                reason = ''
                
                if abs(current_z) < self.z_exit:
                    exit_signal = True
                    reason = 'MEAN_REVERSION'
                elif (self.position == 1 and current_z < -self.z_entry * 1.5) or \
                     (self.position == -1 and current_z > self.z_entry * 1.5):
                    exit_signal = True
                    reason = 'STOP_LOSS'
                
                if exit_signal:
                    spread_change = current_spread - self.entry_price
                    trade_pnl = self.position * spread_change * self.position_size
                    cost = self.position_size * self.transaction_cost
                    
                    self.pnl += trade_pnl - cost
                    
                    self.trades.append({
                        'date': date,
                        'action': f'{reason}_EXIT',
                        'spread': current_spread,
                        'z_score': current_z,
                        'trade_pnl': trade_pnl,
                        'cost': cost,
                        'total_pnl': self.pnl
                    })
                    
                    self.position = 0
                    self.entry_price = 0
            
            # Record equity curve
            unrealized_pnl = 0
            if self.position != 0:
                spread_change = current_spread - self.entry_price
                unrealized_pnl = self.position * spread_change * self.position_size
            
            self.equity_curve.append({
                'date': date,
                'realized_pnl': self.pnl,
                'unrealized_pnl': unrealized_pnl,
                'total_pnl': self.pnl + unrealized_pnl,
                'position': self.position,
                'spread': current_spread,
                'z_score': current_z
            })
        
        # Store signals
        self.hft_signals = hft_signals
        self.traditional_signals = traditional_signals
        
        return self.analyze_hft_results()
    
    # Add methods to both classes
    from run import PairsTrader, HFTPairsTrader
    PairsTrader.backtest_with_data = backtest_with_data
    if HFT_AVAILABLE:
        HFTPairsTrader.backtest_hft_with_data = backtest_hft_with_data

# Add the methods
add_l2_methods_to_traders()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function for L2 optimization
    """
    print("🚀 QUANTPULSE L2 OPTIMIZATION SYSTEM")
    print("=" * 60)
    
    # Generate test data (using previous analysis results if available)
    print("📊 Generating pairs for L2 optimization...")
    pairs = generate_random_pairs(20, sector_bias=True)  # Smaller set for detailed optimization
    
    # Collect price data
    pairs_data = {}
    
    for symbol1, symbol2 in pairs[:10]:  # First 10 pairs for demonstration
        pair_name = f"{symbol1}-{symbol2}"
        print(f"📥 Downloading data for {pair_name}")
        
        try:
            # Create temporary trader to get data
            temp_trader = PairsTrader(symbol1, symbol2)
            prices = temp_trader.get_data('2020-01-01', '2024-12-31')
            pairs_data[pair_name] = prices
            print(f"✅ {pair_name}: {len(prices)} data points")
        except Exception as e:
            print(f"❌ {pair_name}: Failed - {e}")
            continue
    
    if not pairs_data:
        print("❌ No valid pairs data collected")
        return
    
    print(f"\n✅ Collected data for {len(pairs_data)} pairs")
    
    # Run L2 optimization
    optimizer = PortfolioL2Optimizer(
        pairs_data, 
        regularization_strength=0.01  # Moderate regularization
    )
    
    # Optimize portfolio
    results = optimizer.optimize_all_pairs(
        method='bayesian',  # Use Bayesian optimization
        n_trials=30,       # Reduced for demo (increase for production)
        n_splits=3         # Time series cross-validation splits
    )
    
    # Create analysis plots
    optimizer.create_optimization_plots(save_plots=True)
    
    # Save results
    results_file = optimizer.save_results()
    
    print(f"\n🎉 L2 OPTIMIZATION COMPLETE!")
    print(f"📊 Results saved to: {results_file}")
    print(f"📈 Plots generated and saved")
    
    return optimizer, results

if __name__ == "__main__":
    optimizer, results = main()
