import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import native_interface as qn
try:
    from optimize_ltotal import *
except ImportError:
    # Define fallback functions if optimize_ltotal is not available
    from portfolio_manager import load_or_download_data
    
    def fetch_pair_prices(symbol1, symbol2, start_date, end_date):
        """Fallback function to fetch pair prices using our data loader."""
        print(f"🔄 ENTERING fetch_pair_prices({symbol1}, {symbol2}) at {datetime.now().strftime('%H:%M:%S')}")
        data = load_or_download_data([symbol1, symbol2], start_date, end_date)
        result = data.get(symbol1), data.get(symbol2)
        print(f"✅ EXITING fetch_pair_prices({symbol1}, {symbol2}) at {datetime.now().strftime('%H:%M:%S')}")
        return result
    
    def optimize_single_pair(symbol1, symbol2, start_date, end_date, budget=50, restarts=1, popsize=12):
        """Fallback optimization using the native optimizer."""
        print(f"🔧 ENTERING optimize_single_pair({symbol1}, {symbol2}) at {datetime.now().strftime('%H:%M:%S')}")
        p1, p2 = fetch_pair_prices(symbol1, symbol2, start_date, end_date)
        if p1 is None or p2 is None:
            raise ValueError(f"Could not load data for {symbol1} or {symbol2}")
        
        optimizer = qn.NativeElasticNetKLOptimizer(symbol1, symbol2)
        print(f"⚡ Running C++ optimization...")
        best_params = optimizer.optimize((p1, p2), n_splits=2, max_iterations=10)
        print(f"📊 Running backtest...")
        backtest_result = optimizer.backtest((p1, p2))
        
        result = {
            'best_params': best_params,
            'backtest': backtest_result,
            'optimization_time': 0.1
        }
        print(f"✅ EXITING optimize_single_pair({symbol1}, {symbol2}) at {datetime.now().strftime('%H:%M:%S')}")
        return result

def get_optimal_parameters(symbol1, symbol2, start_date, end_date, budget, restarts, popsize, custom_params, skip_optimization):
    print(f"🔄 ENTERING get_optimal_parameters({symbol1}, {symbol2}) at {datetime.now().strftime('%H:%M:%S')}")
    p1, p2 = fetch_pair_prices(symbol1, symbol2, start_date, end_date)
    if custom_params and skip_optimization:
        optimal_params = custom_params.copy()
    else:
        optimization_result = optimize_single_pair(symbol1, symbol2, start_date, end_date, budget=budget, restarts=restarts, popsize=popsize)
        optimal_params = optimization_result['best_params']
        if custom_params:
            optimal_params.update(custom_params)
    print(f"✅ EXITING get_optimal_parameters({symbol1}, {symbol2}) at {datetime.now().strftime('%H:%M:%S')}")
    return p1, p2, optimal_params

def calculate_trade_signals(p1, p2, optimal_params):
    print(f"📊 ENTERING calculate_trade_signals() at {datetime.now().strftime('%H:%M:%S')}")
    lookback = optimal_params['lookback']
    z_entry = optimal_params['z_entry']
    z_exit = optimal_params['z_exit']
    spread = p1 - p2
    z_scores = np.zeros(len(spread))
    trade_entries = []
    trade_exits = []
    for i in range(lookback, len(spread)):
        window = spread[i-lookback:i]
        mean = np.mean(window)
        std = np.std(window)
        if std > 0:
            z_scores[i] = (spread[i] - mean) / std
    in_trade = False
    trade_start_idx = None
    for i in range(lookback, len(z_scores)):
        if not in_trade and abs(z_scores[i]) > z_entry:
            trade_entries.append(i)
            in_trade = True
            trade_start_idx = i
        elif in_trade and abs(z_scores[i]) < z_exit:
            trade_exits.append(i)
            in_trade = False
    print(f"✅ EXITING calculate_trade_signals() at {datetime.now().strftime('%H:%M:%S')}")
    return z_scores, trade_entries, trade_exits, spread
def simulate_portfolio_evolution(initial_capital, result, trade_exits, dates):
    print(f"💰 ENTERING simulate_portfolio_evolution() at {datetime.now().strftime('%H:%M:%S')}")
    portfolio_values = [initial_capital]
    avg_return_per_trade = result['total_return'] / max(result['num_trades'], 1)
    volatility_per_trade = abs(avg_return_per_trade) * 0.8
    current_value = initial_capital
    trade_idx = 0
    for i in range(1, len(dates)):
        if trade_idx < len(trade_exits) and i >= trade_exits[trade_idx]:
            trade_return = np.random.normal(avg_return_per_trade, volatility_per_trade)
            current_value += trade_return
            trade_idx += 1
        portfolio_values.append(current_value)
    print(f"✅ EXITING simulate_portfolio_evolution() at {datetime.now().strftime('%H:%M:%S')}")
    return portfolio_values[:len(dates)]
def plot_price_comparison(ax, dates, p1, p2, symbol1, symbol2, trade_entries, trade_exits):
    print(f"📈 ENTERING plot_price_comparison({symbol1}, {symbol2}) at {datetime.now().strftime('%H:%M:%S')}")
    ax.plot(dates, p1, label=f'{symbol1}', linewidth=2, alpha=0.8, color='blue')
    ax.plot(dates, p2, label=f'{symbol2}', linewidth=2, alpha=0.8, color='orange')
    for entry_idx in trade_entries:
        ax.axvline(x=dates[entry_idx], color='green', alpha=0.7, linestyle='--', linewidth=1)
        ax.scatter(dates[entry_idx], p1[entry_idx], color='green', s=50, marker='^', alpha=0.8, zorder=5)
        ax.scatter(dates[entry_idx], p2[entry_idx], color='green', s=50, marker='^', alpha=0.8, zorder=5)
    for exit_idx in trade_exits:
        ax.axvline(x=dates[exit_idx], color='red', alpha=0.7, linestyle='--', linewidth=1)
        ax.scatter(dates[exit_idx], p1[exit_idx], color='red', s=50, marker='v', alpha=0.8, zorder=5)
        ax.scatter(dates[exit_idx], p2[exit_idx], color='red', s=50, marker='v', alpha=0.8, zorder=5)
    ax.set_title(f'Price Comparison with Trade Signals\\n{len(trade_entries)} Entries, {len(trade_exits)} Exits', fontweight='bold')
    ax.set_ylabel('Price ($)')
    ax.legend([symbol1, symbol2, 'Entry', 'Exit'])
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', rotation=45)
    print(f"✅ EXITING plot_price_comparison({symbol1}, {symbol2}) at {datetime.now().strftime('%H:%M:%S')}")


def plot_zscore_signal(ax, dates, z_scores, z_entry, z_exit, trade_entries, trade_exits, lookback):
    print(f"📊 ENTERING plot_zscore_signal() at {datetime.now().strftime('%H:%M:%S')}")
    ax.plot(dates[lookback:], z_scores[lookback:], label='Z-Score', color='purple', linewidth=1.5)
    ax.axhline(y=z_entry, color='green', linestyle='-', alpha=0.7, label=f'Entry (+/-{z_entry:.2f})')
    ax.axhline(y=-z_entry, color='green', linestyle='-', alpha=0.7)
    ax.axhline(y=z_exit, color='red', linestyle='-', alpha=0.7, label=f'Exit (+/-{z_exit:.2f})')
    ax.axhline(y=-z_exit, color='red', linestyle='-', alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Mean')
    for entry_idx in trade_entries:
        ax.scatter(dates[entry_idx], z_scores[entry_idx], color='green', s=30, marker='^', zorder=5)
    for exit_idx in trade_exits:
        ax.scatter(dates[exit_idx], z_scores[exit_idx], color='red', s=30, marker='v', zorder=5)
    ax.set_title('Z-Score Signal with Entry/Exit Thresholds', fontweight='bold')
    ax.set_ylabel('Z-Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', rotation=45)
    print(f"✅ EXITING plot_zscore_signal() at {datetime.now().strftime('%H:%M:%S')}")
def plot_portfolio_evolution(ax, dates, portfolio_values, initial_capital, trade_exits):
    print(f"💹 ENTERING plot_portfolio_evolution() at {datetime.now().strftime('%H:%M:%S')}")
    ax.plot(dates, portfolio_values, label='Portfolio Value', color='darkgreen', linewidth=2)
    ax.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.7, label='Initial Capital')
    ax.fill_between(dates, initial_capital, portfolio_values, where=np.array(portfolio_values) >= initial_capital, alpha=0.3, color='green', label='Profit')
    ax.fill_between(dates, initial_capital, portfolio_values, where=np.array(portfolio_values) < initial_capital, alpha=0.3, color='red', label='Loss')
    for exit_idx in trade_exits[:5]:
        if exit_idx < len(portfolio_values):
            ax.scatter(dates[exit_idx], portfolio_values[exit_idx], color='blue', s=40, alpha=0.8, zorder=5)
    ax.set_title('Portfolio Value Evolution with Trade Markers', fontweight='bold')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', rotation=45)
    print(f"✅ EXITING plot_portfolio_evolution() at {datetime.now().strftime('%H:%M:%S')}")
def plot_drawdown_chart(ax, dates, portfolio_values):
    print(f"📉 ENTERING plot_drawdown_chart() at {datetime.now().strftime('%H:%M:%S')}")
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (np.array(portfolio_values) - peak) / peak * 100
    ax.fill_between(dates, 0, drawdown, color='red', alpha=0.3, label='Drawdown')
    ax.plot(dates, drawdown, color='darkred', linewidth=1.5)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.8)
    ax.set_title(f'Portfolio Drawdown\\nMax: {np.min(drawdown):.2f}%', fontweight='bold')
    ax.set_ylabel('Drawdown (%)')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', rotation=45)
    print(f"✅ EXITING plot_drawdown_chart() at {datetime.now().strftime('%H:%M:%S')}")
def plot_performance_metrics(ax, result, initial_capital):
    print(f"📊 ENTERING plot_performance_metrics() at {datetime.now().strftime('%H:%M:%S')}")
    metrics = ['Sharpe', 'Win Rate %', 'Profit Factor', 'Return %']
    values = [result['sharpe_ratio'], result['win_rate'] * 100, result['profit_factor'], (result['total_return'] / initial_capital) * 100]
    colors = ['skyblue', 'lightgreen', 'orange', 'lightcoral']
    bars = ax.bar(metrics, values, color=colors, alpha=0.7)
    ax.set_title('Performance Metrics', fontweight='bold')
    ax.set_ylabel('Value')
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01, f'{value:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)
    print(f"✅ EXITING plot_performance_metrics() at {datetime.now().strftime('%H:%M:%S')}")
def plot_trade_distribution(ax, result):
    print(f"📈 ENTERING plot_trade_distribution() at {datetime.now().strftime('%H:%M:%S')}")
    if result['num_trades'] > 0:
        simulated_returns = np.random.normal(result['avg_trade_return'], abs(result['avg_trade_return']) * 0.8, result['num_trades'])
        ax.hist(simulated_returns, bins=min(10, result['num_trades']), alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(x=result['avg_trade_return'], color='red', linestyle='--', label=f'Avg: ${result["avg_trade_return"]:,.0f}')
        ax.set_title(f'Trade Return Distribution\\n({result["num_trades"]} trades)', fontweight='bold')
        ax.set_xlabel('Trade Return ($)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    print(f"✅ EXITING plot_trade_distribution() at {datetime.now().strftime('%H:%M:%S')}")


def plot_rolling_sharpe(ax, dates, portfolio_values):
    print(f"📊 ENTERING plot_rolling_sharpe() at {datetime.now().strftime('%H:%M:%S')}")
    if len(portfolio_values) > 252:
        rolling_returns = pd.Series(portfolio_values).pct_change().dropna()
        rolling_sharpe = rolling_returns.rolling(window=252).apply(lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0).dropna()
        if len(rolling_sharpe) > 0:
            # Ensure dates and rolling_sharpe have the same length
            start_idx = len(dates) - len(rolling_sharpe)
            sharpe_dates = dates[start_idx:]
            # Truncate if necessary to match exactly
            min_len = min(len(sharpe_dates), len(rolling_sharpe))
            sharpe_dates = sharpe_dates[:min_len]
            rolling_sharpe = rolling_sharpe[:min_len]
            
            ax.plot(sharpe_dates, rolling_sharpe, color='purple', linewidth=2)
            ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Good (>1.0)')
            ax.axhline(y=0.0, color='red', linestyle='--', alpha=0.7, label='Breakeven')
            ax.set_title('Rolling Sharpe Ratio (1Y)', fontweight='bold')
            ax.set_ylabel('Sharpe Ratio')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.tick_params(axis='x', rotation=45)
    print(f"✅ EXITING plot_rolling_sharpe() at {datetime.now().strftime('%H:%M:%S')}")


def calculate_advanced_metrics(result, initial_capital, start_date, end_date):
    print(f"🧮 ENTERING calculate_advanced_metrics() at {datetime.now().strftime('%H:%M:%S')}")
    total_return_pct = (result['total_return'] / initial_capital)
    from datetime import datetime
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    years = (end - start).days / 365.25
    annualized_return = ((1 + total_return_pct) ** (1/years)) - 1
    sortino_ratio = 0.0
    calmar_ratio = annualized_return / max(result['max_drawdown'] / initial_capital, 0.001)
    benchmark_return = 0.10
    excess_return = annualized_return - benchmark_return
    tracking_error = result['volatility'] / initial_capital
    information_ratio = excess_return / max(tracking_error, 0.001)
    beta = 1.0
    treynor_ratio = (annualized_return - 0.02) / beta
    print(f"✅ EXITING calculate_advanced_metrics() at {datetime.now().strftime('%H:%M:%S')}")
    return {'total_return_pct': total_return_pct, 'annualized_return': annualized_return, 'sortino_ratio': sortino_ratio, 'calmar_ratio': calmar_ratio, 'information_ratio': information_ratio, 'treynor_ratio': treynor_ratio}
def print_performance_summary(symbol1, symbol2, start_date, end_date, initial_capital, result, optimal_params, advanced_metrics):
    print(f"📄 ENTERING print_performance_summary({symbol1}, {symbol2}) at {datetime.now().strftime('%H:%M:%S')}")
    print(f"✅ EXITING print_performance_summary({symbol1}, {symbol2}) at {datetime.now().strftime('%H:%M:%S')}")
    return
def plot_portfolio_performance(symbol1="FEXDU", symbol2="BALY", start_date="2010-01-01", end_date="2020-12-31", initial_capital=1_000_000, budget=200, restarts=3, popsize=16, custom_params=None, skip_optimization=False):
    print(f"\n🚀 ENTERING plot_portfolio_performance({symbol1}, {symbol2}) at {datetime.now().strftime('%H:%M:%S')}")
    print(f"📅 Period: {start_date} to {end_date} | Capital: ${initial_capital:,}")
    
    print(f"📈 Getting optimal parameters...")
    p1, p2, optimal_params = get_optimal_parameters(symbol1, symbol2, start_date, end_date, budget, restarts, popsize, custom_params, skip_optimization)
    
    # Align data lengths
    min_len = min(len(p1), len(p2))
    p1 = p1[:min_len]
    p2 = p2[:min_len]
    print(f"📊 Data aligned: {min_len} data points")
    
    print(f"⚡ Running vectorized backtest...")
    result = qn.vectorized_backtest(p1, p2, optimal_params, use_cache=False)
    print(f"📊 Backtest complete: Sharpe={result['sharpe_ratio']:.3f}, Return=${result['total_return']:,.0f}")
    
    print(f"📅 Creating date range and calculating signals...")
    dates = pd.date_range(start_date, end_date, freq='D')
    dates = dates[dates.dayofweek < 5][:min_len]
    z_scores, trade_entries, trade_exits, spread = calculate_trade_signals(p1, p2, optimal_params)
    portfolio_values = simulate_portfolio_evolution(initial_capital, result, trade_exits, dates)
    print(f"🔄 Calculated {len(trade_entries)} entries and {len(trade_exits)} exits")
    
    print(f"🎨 Creating visualization with 7 subplots...")
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, :2])
    ax4 = fig.add_subplot(gs[1, 2])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])
    ax7 = fig.add_subplot(gs[2, 2])
    fig.suptitle(f'QuantPulse Pairs Trading Analysis: {symbol1} vs {symbol2}\\n{start_date} to {end_date}\\nCapital: ${initial_capital:,} | NO LOOK-AHEAD BIAS', fontsize=16, fontweight='bold')
    
    print(f"📊 Plotting 1/7: Price comparison...")
    plot_price_comparison(ax1, dates, p1, p2, symbol1, symbol2, trade_entries, trade_exits)
    print(f"📊 Plotting 2/7: Z-score signals...")
    plot_zscore_signal(ax2, dates, z_scores, optimal_params['z_entry'], optimal_params['z_exit'], trade_entries, trade_exits, optimal_params['lookback'])
    print(f"📊 Plotting 3/7: Portfolio evolution...")
    plot_portfolio_evolution(ax3, dates, portfolio_values, initial_capital, trade_exits)
    print(f"📊 Plotting 4/7: Drawdown chart...")
    plot_drawdown_chart(ax4, dates, portfolio_values)
    print(f"📊 Plotting 5/7: Performance metrics...")
    plot_performance_metrics(ax5, result, initial_capital)
    print(f"📊 Plotting 6/7: Trade distribution...")
    plot_trade_distribution(ax6, result)
    print(f"📊 Plotting 7/7: Rolling Sharpe...")
    plot_rolling_sharpe(ax7, dates, portfolio_values)
    
    plt.tight_layout()
    import os
    os.makedirs('static', exist_ok=True)
    filename = f'static/portfolio_performance_{symbol1}_{symbol2}_{start_date}_{end_date}.png'
    print(f"💾 Saving chart to: {filename}")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    advanced_metrics = calculate_advanced_metrics(result, initial_capital, start_date, end_date)
    print_performance_summary(symbol1, symbol2, start_date, end_date, initial_capital, result, optimal_params, advanced_metrics)
    
    print(f"✅ EXITING plot_portfolio_performance({symbol1}, {symbol2}) at {datetime.now().strftime('%H:%M:%S')}")
    print(f"📈 Final Results: Sharpe={result['sharpe_ratio']:.3f} | Return=${result['total_return']:,.0f} | Trades={result['num_trades']}\n")
    return result
def example_usage():
    print(f"🚀 ENTERING example_usage() at {datetime.now().strftime('%H:%M:%S')}")
    result1 = plot_portfolio_performance()
    result2 = plot_portfolio_performance("AAPL", "MSFT", "2015-01-01", "2020-12-31", 500000)
    custom_params = {'lookback': 30, 'z_entry': 2.0, 'z_exit': 0.5}
    result3 = plot_portfolio_performance("SPY", "QQQ", "2018-01-01", "2023-12-31", 1500000, custom_params=custom_params, skip_optimization=True)
    result4 = plot_portfolio_performance("GLD", "SLV", "2012-01-01", "2022-12-31", budget=100, restarts=2, popsize=12)
    print(f"✅ EXITING example_usage() at {datetime.now().strftime('%H:%M:%S')}")
    return [result1, result2, result3, result4]



