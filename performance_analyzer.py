#!/usr/bin/env python3
"""
QuantPulse Performance Analyzer
==============================

Comprehensive performance measurement system for the QuantPulse Pairs Trading System.
Features:
- Function-level performance profiling
- Extended time range backtesting (2020-2024)
- Random pair generation from major indices
- Advanced plotting and visualization
- Statistical performance analysis

Author: QuantPulse Trading Systems
"""

import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf
from functools import wraps
import cProfile
import pstats
import io
from contextlib import contextmanager
import warnings
warnings.filterwarnings('ignore')

# Import our trading system
from run import PairsTrader, HFTPairsTrader, get_optimal_config, HFT_AVAILABLE

# ============================================================================
# PERFORMANCE MEASUREMENT DECORATORS
# ============================================================================

def profile_function(func):
    """Decorator to profile function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        pr.disable()
        
        # Store profiling data
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        
        execution_time = end_time - start_time
        
        # Store in global profiling data
        if not hasattr(wrapper, 'profile_data'):
            wrapper.profile_data = []
        
        wrapper.profile_data.append({
            'function': func.__name__,
            'execution_time': execution_time,
            'timestamp': datetime.now(),
            'profile_stats': s.getvalue()
        })
        
        print(f"⏱️  {func.__name__}: {execution_time:.4f}s")
        return result
    
    return wrapper

@contextmanager
def timer(operation_name):
    """Context manager for timing operations"""
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print(f"⏱️  {operation_name}: {end - start:.4f}s")

# ============================================================================
# RANDOM PAIR GENERATOR
# ============================================================================

# Major stock indices for random pair selection
MAJOR_STOCKS = {
    'sp500_large': [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
        'UNH', 'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'ABBV', 'PFE',
        'BAC', 'KO', 'AVGO', 'PEP', 'WMT', 'DIS', 'TMO', 'COST', 'MRK',
        'ABT', 'VZ', 'ADBE', 'NFLX', 'XOM', 'CRM', 'ACN', 'NKE', 'ORCL',
        'MDT', 'TXN', 'QCOM', 'HON', 'NEE', 'LIN', 'UPS', 'PM', 'T',
        'IBM', 'LOW', 'AMD', 'INTC', 'BA', 'GS', 'CAT', 'C', 'RTX'
    ],
    
    'sp500_mid': [
        'MS', 'WFC', 'DE', 'SPGI', 'BLK', 'AXP', 'NOW', 'SYK', 'TGT',
        'BKNG', 'MMM', 'ISRG', 'PLD', 'MDLZ', 'ZTS', 'LRCX', 'ADI',
        'GILD', 'CB', 'MO', 'TJX', 'CI', 'CCI', 'SO', 'DUK', 'CSX',
        'ICE', 'PNC', 'AON', 'USB', 'CME', 'EOG', 'CL', 'APD', 'MCO'
    ],
    
    'tech_focused': [
        'AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'NVDA', 'TSLA', 'NFLX',
        'ADBE', 'CRM', 'ORCL', 'AMD', 'INTC', 'QCOM', 'TXN', 'AVGO',
        'NOW', 'LRCX', 'ADI', 'KLAC', 'MRVL', 'MU', 'AMAT', 'SNPS'
    ],
    
    'financials': [
        'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'USB', 'PNC',
        'BLK', 'CB', 'AON', 'MCO', 'ICE', 'CME', 'SPGI', 'TFC', 'COF'
    ],
    
    'healthcare': [
        'UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'MDT',
        'ISRG', 'ZTS', 'GILD', 'CVS', 'CI', 'HUM', 'ANTM', 'BSX'
    ],
    
    'consumer': [
        'AMZN', 'WMT', 'HD', 'PG', 'KO', 'PEP', 'DIS', 'COST',
        'NKE', 'TGT', 'MDLZ', 'MO', 'TJX', 'CL', 'SBUX', 'MCD'
    ]
}

def generate_random_pairs(num_pairs=32, sector_bias=True, avoid_duplicates=True):
    """
    Generate random stock pairs for testing
    
    Args:
        num_pairs: Number of pairs to generate
        sector_bias: If True, bias towards same-sector pairs
        avoid_duplicates: Ensure no symbol appears twice
    """
    pairs = []
    used_symbols = set()
    
    all_sectors = list(MAJOR_STOCKS.keys())
    
    for i in range(num_pairs):
        attempts = 0
        max_attempts = 100
        
        while attempts < max_attempts:
            if sector_bias and random.random() > 0.3:  # 70% chance for same sector
                sector = random.choice(all_sectors)
                sector_stocks = MAJOR_STOCKS[sector]
                
                if len(sector_stocks) >= 2:
                    available = [s for s in sector_stocks if s not in used_symbols]
                    if len(available) >= 2:
                        symbol1, symbol2 = random.sample(available, 2)
                    else:
                        # Fall back to any sector
                        all_stocks = [s for stocks in MAJOR_STOCKS.values() for s in stocks]
                        available = [s for s in all_stocks if s not in used_symbols]
                        if len(available) >= 2:
                            symbol1, symbol2 = random.sample(available, 2)
                        else:
                            break
                else:
                    continue
            else:
                # Random cross-sector pairs
                all_stocks = [s for stocks in MAJOR_STOCKS.values() for s in stocks]
                available = [s for s in all_stocks if s not in used_symbols]
                
                if len(available) >= 2:
                    symbol1, symbol2 = random.sample(available, 2)
                else:
                    break
            
            # Check if this pair makes sense (basic validation)
            if symbol1 != symbol2:
                pairs.append((symbol1, symbol2))
                if avoid_duplicates:
                    used_symbols.update([symbol1, symbol2])
                break
            
            attempts += 1
        
        if attempts >= max_attempts:
            print(f"⚠️ Could only generate {len(pairs)} pairs (target: {num_pairs})")
            break
    
    return pairs

# ============================================================================
# ENHANCED PERFORMANCE TRADER
# ============================================================================

class PerformancePairsTrader(HFTPairsTrader if HFT_AVAILABLE else PairsTrader):
    """Extended trader with performance measurement capabilities"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.performance_metrics = {}
        self.function_timings = {}
    
    @profile_function
    def get_data_timed(self, start_date, end_date, interval='1d'):
        """Timed version of get_data"""
        return super().get_data(start_date, end_date, interval)
    
    @profile_function  
    def calculate_spread_stats_timed(self, prices):
        """Timed version of calculate_spread_stats"""
        return super().calculate_spread_stats(prices)
    
    @profile_function
    def backtest_timed(self, start_date, end_date, interval='1d'):
        """Timed version of backtest"""
        if hasattr(super(), 'backtest_hft'):
            return super().backtest_hft(start_date, end_date, interval, hybrid_mode=True)
        else:
            return super().backtest(start_date, end_date, interval)

# ============================================================================
# COMPREHENSIVE PERFORMANCE ANALYZER
# ============================================================================

class QuantPulsePerformanceAnalyzer:
    """Main performance analysis class"""
    
    def __init__(self):
        self.results = []
        self.performance_data = {}
        self.start_time = None
        self.total_execution_time = 0
        
    def run_comprehensive_analysis(self, 
                                 num_pairs=32, 
                                 start_date='2020-01-01', 
                                 end_date='2024-12-31',
                                 save_plots=True):
        """
        Run comprehensive performance analysis
        
        Args:
            num_pairs: Number of random pairs to test
            start_date: Start date for backtesting (extended range)
            end_date: End date for backtesting
            save_plots: Whether to save plots to disk
        """
        
        print("🚀 QUANTPULSE COMPREHENSIVE PERFORMANCE ANALYSIS")
        print("=" * 60)
        print(f"📊 Testing {num_pairs} random pairs")
        print(f"📅 Extended Period: {start_date} to {end_date}")
        print(f"⚡ HFT Available: {'✅ YES' if HFT_AVAILABLE else '❌ NO'}")
        print()
        
        self.start_time = time.time()
        
        # Generate random pairs
        with timer("Random pair generation"):
            pairs = generate_random_pairs(num_pairs, sector_bias=True)
        
        print(f"✅ Generated {len(pairs)} random pairs")
        print()
        
        # Test each pair
        successful_tests = 0
        failed_tests = 0
        
        for i, (symbol1, symbol2) in enumerate(pairs, 1):
            print(f"📊 Testing {i}/{len(pairs)}: {symbol1} vs {symbol2}")
            
            try:
                # Get optimal config for this pair
                config = get_optimal_config(symbol1, symbol2)
                
                # Create performance trader
                trader = PerformancePairsTrader(
                    symbol1, symbol2,
                    lookback=config['lookback'],
                    z_entry=config['z_entry'],
                    z_exit=config['z_exit'],
                    position_size=config['position_size'],
                    transaction_cost=config['transaction_cost']
                )
                
                # Run timed backtest
                with timer(f"Backtest {symbol1}-{symbol2}"):
                    results = trader.backtest_timed(start_date, end_date)
                
                if 'error' not in results:
                    results['pair'] = f"{symbol1}-{symbol2}"
                    results['symbol1'] = symbol1
                    results['symbol2'] = symbol2
                    results['config_used'] = config
                    
                    self.results.append(results)
                    successful_tests += 1
                    
                    print(f"✅ Profit: ${results['final_pnl']:,.2f} | "
                          f"Trades: {results['num_trades']} | "
                          f"Win Rate: {results['win_rate']:.1%}")
                else:
                    failed_tests += 1
                    print(f"❌ Failed: {results['error']}")
                    
            except Exception as e:
                failed_tests += 1
                print(f"❌ Error: {e}")
            
            print()
        
        self.total_execution_time = time.time() - self.start_time
        
        print(f"✅ Analysis Complete!")
        print(f"📊 Successful: {successful_tests}/{len(pairs)} pairs")
        print(f"❌ Failed: {failed_tests}/{len(pairs)} pairs")
        print(f"⏱️ Total Time: {self.total_execution_time:.2f}s")
        print()
        
        # Analyze results
        self.analyze_performance()
        
        # Create visualizations
        if save_plots:
            self.create_comprehensive_plots()
        
        return self.results
    
    def analyze_performance(self):
        """Analyze performance metrics"""
        if not self.results:
            print("❌ No results to analyze")
            return
        
        df = pd.DataFrame(self.results)
        
        print("📈 PERFORMANCE ANALYSIS")
        print("=" * 30)
        
        # Basic statistics
        total_profit = df['final_pnl'].sum()
        profitable_pairs = (df['final_pnl'] > 0).sum()
        total_pairs = len(df)
        avg_profit = df['final_pnl'].mean()
        median_profit = df['final_pnl'].median()
        
        print(f"💰 Total Portfolio Profit: ${total_profit:,.2f}")
        print(f"📊 Profitable Pairs: {profitable_pairs}/{total_pairs} ({profitable_pairs/total_pairs:.1%})")
        print(f"📈 Average Profit: ${avg_profit:,.2f}")
        print(f"📊 Median Profit: ${median_profit:,.2f}")
        print()
        
        # Performance distribution
        print("🎯 PROFIT DISTRIBUTION")
        print("-" * 20)
        profit_ranges = [
            (100000, "🚀 > $100K"),
            (50000, "💰 $50K-$100K"), 
            (10000, "📈 $10K-$50K"),
            (1000, "✅ $1K-$10K"),
            (0, "⚖️ $0-$1K"),
            (-float('inf'), "❌ Losses")
        ]
        
        for threshold, label in profit_ranges:
            if threshold == -float('inf'):
                count = (df['final_pnl'] < 0).sum()
            elif threshold == 0:
                count = ((df['final_pnl'] >= 0) & (df['final_pnl'] < 1000)).sum()
            elif threshold == 1000:
                count = ((df['final_pnl'] >= 1000) & (df['final_pnl'] < 10000)).sum()
            elif threshold == 10000:
                count = ((df['final_pnl'] >= 10000) & (df['final_pnl'] < 50000)).sum()
            elif threshold == 50000:
                count = ((df['final_pnl'] >= 50000) & (df['final_pnl'] < 100000)).sum()
            else:
                count = (df['final_pnl'] >= threshold).sum()
            
            print(f"{label}: {count} pairs ({count/total_pairs:.1%})")
        
        print()
        
        # Top performers
        print("🏆 TOP 5 PERFORMERS")
        print("-" * 20)
        top_5 = df.nlargest(5, 'final_pnl')
        for idx, row in top_5.iterrows():
            print(f"🥇 {row['pair']}: ${row['final_pnl']:,.2f} "
                  f"({row['num_trades']} trades, {row['win_rate']:.1%} win rate)")
        
        print()
        
        # Bottom performers  
        print("⚠️ BOTTOM 5 PERFORMERS")
        print("-" * 22)
        bottom_5 = df.nsmallest(5, 'final_pnl')
        for idx, row in bottom_5.iterrows():
            print(f"⚠️ {row['pair']}: ${row['final_pnl']:,.2f} "
                  f"({row['num_trades']} trades, {row['win_rate']:.1%} win rate)")
        
        print()
        
        # Risk metrics
        print("⚖️ RISK METRICS")
        print("-" * 15)
        print(f"📊 Sharpe Ratio (avg): {df['sharpe_ratio'].mean():.3f}")
        print(f"📉 Max Drawdown (avg): ${df['max_drawdown'].mean():,.2f}")
        print(f"🎯 Win Rate (avg): {df['win_rate'].mean():.1%}")
        print(f"💹 Volatility (profit): ${df['final_pnl'].std():,.2f}")
        
        # Store summary statistics
        self.performance_data = {
            'total_profit': total_profit,
            'profitable_pairs': profitable_pairs,
            'total_pairs': total_pairs,
            'profitability_rate': profitable_pairs / total_pairs,
            'avg_profit': avg_profit,
            'median_profit': median_profit,
            'profit_std': df['final_pnl'].std(),
            'avg_sharpe': df['sharpe_ratio'].mean(),
            'avg_win_rate': df['win_rate'].mean(),
            'execution_time': self.total_execution_time
        }
    
    def create_comprehensive_plots(self):
        """Create comprehensive visualization plots"""
        if not self.results:
            print("❌ No results to plot")
            return
            
        df = pd.DataFrame(self.results)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create comprehensive figure with subplots
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Profit Distribution Histogram
        ax1 = fig.add_subplot(gs[0, 0:2])
        profits = df['final_pnl'] / 1000  # Convert to thousands
        ax1.hist(profits, bins=30, alpha=0.7, edgecolor='black')
        ax1.set_title('📊 Profit Distribution (All Pairs)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Profit ($K)')
        ax1.set_ylabel('Number of Pairs')
        ax1.axvline(profits.mean(), color='red', linestyle='--', label=f'Mean: ${profits.mean():.1f}K')
        ax1.axvline(profits.median(), color='green', linestyle='--', label=f'Median: ${profits.median():.1f}K')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Profitability by Sector (if we can infer sectors)
        ax2 = fig.add_subplot(gs[0, 2:4])
        profitable = (df['final_pnl'] > 0).astype(int)
        sectors = []
        for _, row in df.iterrows():
            symbol1, symbol2 = row['symbol1'], row['symbol2']
            # Simple sector classification
            if any(s in ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD'] for s in [symbol1, symbol2]):
                sectors.append('Tech')
            elif any(s in ['JPM', 'BAC', 'GS', 'MS', 'WFC', 'C'] for s in [symbol1, symbol2]):
                sectors.append('Finance')
            elif any(s in ['KO', 'PEP', 'WMT', 'TGT', 'PG'] for s in [symbol1, symbol2]):
                sectors.append('Consumer')
            elif any(s in ['XOM', 'CVX', 'COP', 'EOG'] for s in [symbol1, symbol2]):
                sectors.append('Energy')
            else:
                sectors.append('Mixed')
        
        df['sector'] = sectors
        sector_stats = df.groupby('sector').agg({
            'final_pnl': ['mean', 'count'],
            'win_rate': 'mean'
        }).round(2)
        
        sector_profits = df.groupby('sector')['final_pnl'].mean() / 1000
        bars = ax2.bar(sector_profits.index, sector_profits.values)
        ax2.set_title('💼 Average Profit by Sector', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Average Profit ($K)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Color bars by profitability
        for bar, profit in zip(bars, sector_profits.values):
            bar.set_color('green' if profit > 0 else 'red')
            bar.set_alpha(0.7)
        
        ax2.grid(True, alpha=0.3)
        
        # 3. Scatter: Trades vs Profit
        ax3 = fig.add_subplot(gs[1, 0:2])
        scatter = ax3.scatter(df['num_trades'], df['final_pnl']/1000, 
                             c=df['win_rate'], cmap='RdYlGn', alpha=0.6, s=60)
        ax3.set_title('📈 Trading Activity vs Profitability', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Number of Trades')
        ax3.set_ylabel('Profit ($K)')
        plt.colorbar(scatter, ax=ax3, label='Win Rate')
        ax3.grid(True, alpha=0.3)
        
        # 4. Win Rate Distribution
        ax4 = fig.add_subplot(gs[1, 2:4])
        ax4.hist(df['win_rate'], bins=20, alpha=0.7, edgecolor='black')
        ax4.set_title('🎯 Win Rate Distribution', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Win Rate')
        ax4.set_ylabel('Number of Pairs')
        ax4.axvline(df['win_rate'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["win_rate"].mean():.1%}')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Risk-Return Scatter (Sharpe Ratio vs Profit)
        ax5 = fig.add_subplot(gs[2, 0:2])
        ax5.scatter(df['sharpe_ratio'], df['final_pnl']/1000, alpha=0.6, s=60)
        ax5.set_title('⚖️ Risk-Adjusted Returns', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Sharpe Ratio')
        ax5.set_ylabel('Profit ($K)')
        ax5.grid(True, alpha=0.3)
        
        # Add quadrant lines
        ax5.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax5.axvline(0, color='black', linestyle='-', alpha=0.3)
        
        # 6. Max Drawdown vs Profit
        ax6 = fig.add_subplot(gs[2, 2:4])
        ax6.scatter(-df['max_drawdown']/1000, df['final_pnl']/1000, alpha=0.6, s=60)
        ax6.set_title('📉 Drawdown vs Profit', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Max Drawdown ($K)')
        ax6.set_ylabel('Profit ($K)')
        ax6.grid(True, alpha=0.3)
        
        # 7. Top 10 Performers Bar Chart
        ax7 = fig.add_subplot(gs[3, :])
        top_10 = df.nlargest(10, 'final_pnl')
        bars = ax7.bar(range(len(top_10)), top_10['final_pnl']/1000)
        ax7.set_title('🏆 Top 10 Performing Pairs', fontsize=14, fontweight='bold')
        ax7.set_xlabel('Pair Rank')
        ax7.set_ylabel('Profit ($K)')
        ax7.set_xticks(range(len(top_10)))
        ax7.set_xticklabels(top_10['pair'], rotation=45, ha='right')
        
        # Color bars by performance
        for i, (bar, profit) in enumerate(zip(bars, top_10['final_pnl'])):
            if i < 3:  # Top 3 in gold gradient
                bar.set_color(plt.cm.Oranges(0.8 - i*0.2))
            else:
                bar.set_color(plt.cm.Greens(0.7))
        
        ax7.grid(True, alpha=0.3, axis='y')
        
        # Add overall title
        fig.suptitle(f'QuantPulse Performance Analysis - {len(df)} Random Pairs\n'
                    f'Total Portfolio: ${df["final_pnl"].sum():,.2f} | '
                    f'Profitable: {(df["final_pnl"] > 0).sum()}/{len(df)} pairs | '
                    f'Execution Time: {self.total_execution_time:.2f}s',
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save the comprehensive plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'quantpulse_performance_analysis_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Comprehensive plot saved: {filename}")
        
        # Create additional performance summary plot
        self.create_performance_summary_plot(df, timestamp)
        
        plt.show()
    
    def create_performance_summary_plot(self, df, timestamp):
        """Create a separate performance summary plot"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('QuantPulse Performance Summary Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Cumulative Profit Timeline (simulated)
        profits = df['final_pnl'].values
        cumulative = np.cumsum(profits)
        ax1.plot(range(1, len(cumulative)+1), cumulative/1000, linewidth=2, color='darkgreen')
        ax1.fill_between(range(1, len(cumulative)+1), cumulative/1000, alpha=0.3, color='lightgreen')
        ax1.set_title('💰 Cumulative Portfolio Growth', fontweight='bold')
        ax1.set_xlabel('Pair Number')
        ax1.set_ylabel('Cumulative Profit ($K)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Performance Metrics Radar Chart (simplified as bar chart)
        metrics = ['Profitability Rate', 'Avg Win Rate', 'Avg Sharpe', 'Risk Score']
        values = [
            (df['final_pnl'] > 0).mean(),
            df['win_rate'].mean(),
            max(0, min(1, (df['sharpe_ratio'].mean() + 2) / 4)),  # Normalize Sharpe
            max(0, min(1, 1 - (-df['max_drawdown'].mean() / df['final_pnl'].std())))  # Risk score
        ]
        
        bars = ax2.bar(metrics, values, color=['gold', 'lightblue', 'lightcoral', 'lightgreen'])
        ax2.set_title('📊 Key Performance Metrics', fontweight='bold')
        ax2.set_ylabel('Score (0-1)')
        ax2.set_ylim(0, 1)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        for bar, value in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Profit vs Loss Pairs
        profitable_count = (df['final_pnl'] > 0).sum()
        loss_count = len(df) - profitable_count
        
        ax3.pie([profitable_count, loss_count], 
               labels=[f'Profitable\n({profitable_count})', f'Loss\n({loss_count})'],
               colors=['lightgreen', 'lightcoral'],
               autopct='%1.1f%%',
               startangle=90)
        ax3.set_title('🎯 Profit vs Loss Distribution', fontweight='bold')
        
        # 4. Execution Performance
        execution_data = {
            'Total Pairs': len(df),
            'Execution Time (s)': self.total_execution_time,
            'Pairs per Second': len(df) / self.total_execution_time,
            'Avg Time per Pair (s)': self.total_execution_time / len(df)
        }
        
        y_pos = range(len(execution_data))
        values = list(execution_data.values())
        ax4.barh(y_pos, values, color='skyblue')
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(execution_data.keys())
        ax4.set_title('⚡ Execution Performance', fontweight='bold')
        ax4.set_xlabel('Value')
        
        for i, v in enumerate(values):
            ax4.text(v + max(values) * 0.01, i, f'{v:.2f}', 
                    va='center', ha='left', fontweight='bold')
        
        plt.tight_layout()
        
        # Save performance summary
        summary_filename = f'quantpulse_summary_{timestamp}.png'
        plt.savefig(summary_filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📈 Performance summary saved: {summary_filename}")
        
        plt.show()

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """Main execution function"""
    analyzer = QuantPulsePerformanceAnalyzer()
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis(
        num_pairs=32,
        start_date='2020-01-01',  # Extended 5-year range
        end_date='2024-12-31',
        save_plots=True
    )
    
    print("\n🎉 ANALYSIS COMPLETE!")
    print(f"📊 Results saved and plots generated")
    print(f"💾 Check current directory for PNG files")
    
    return analyzer, results

if __name__ == "__main__":
    analyzer, results = main()
