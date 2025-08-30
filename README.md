# Algorithmic Trading Backtester

A sophisticated event-driven backtesting system for statistical arbitrage and pairs trading strategies.

## 🚀 Features

- **Event-Driven Architecture**: Modular and extensible design
- **Statistical Arbitrage**: Advanced pairs trading with z-score analysis
- **Performance Optimization**: Numba JIT compilation for 30x speedup
- **Comprehensive Analytics**: Sharpe ratio, max drawdown, Sortino ratio
- **Interactive Visualizations**: Plotly-based charts and dashboards
- **Real Market Data**: Yahoo Finance integration

## 📁 Project Structure

```
quantpulse-pairs-trading/
├── trading_backtester.py    # Main backtesting engine
├── requirements.txt         # Python dependencies
├── src/                    # Source code modules
├── tests/                  # Test suite
├── docs/                   # Documentation
├── data/                   # Market data cache
└── results/                # Backtest results
```

## 🛠 Setup

### 1. Create Conda Environment
```bash
conda create -n quantpulse-trading python=3.10 -y
conda activate quantpulse-trading
```

### 2. Install Dependencies
```bash
conda install -c conda-forge numpy pandas matplotlib seaborn numba plotly yfinance -y
```

### 3. Run the Backtester
```bash
python trading_backtester.py
```

## 📊 Trading Strategy

The system implements **Statistical Arbitrage** using:

- **Pairs Selection**: ETF pairs (SPY/QQQ, IWM/DIA, GLD/TLT, SPY/EEM)
- **Signal Generation**: Z-score based mean reversion
- **Risk Management**: Position sizing, stop-losses, exposure limits
- **Performance Tracking**: Real-time P&L and risk metrics

## 🎯 Key Metrics

- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough loss
- **Win Rate**: Percentage of profitable trades
- **Sortino Ratio**: Downside risk adjustment
- **Calmar Ratio**: Return vs maximum drawdown

## 📈 Educational Value

This project demonstrates:

1. **Quantitative Finance**: Statistical arbitrage concepts
2. **Software Engineering**: Event-driven architecture
3. **Performance Optimization**: Numba JIT compilation
4. **Data Analysis**: Financial time series analysis
5. **Visualization**: Interactive financial charts

## 🔧 Configuration

Adjust parameters in the `Config` class:

```python
config = Config(
    initial_capital=100000,
    position_size=0.1,         # 10% per position
    z_score_entry=2.0,         # Entry threshold
    z_score_exit=0.5,          # Exit threshold
    max_positions=5,           # Position limit
    commission=0.001,          # 0.1% commission
    slippage=0.0005           # 0.05% slippage
)
```

## 📝 License

Educational project - feel free to learn and modify!
