"""
QuantPulse Pairs Trading System
===============================

A high-performance pairs trading system with C++ acceleration and ARM64 SIMD optimization.

Main Components:
- Data Management: Real-time and historical market data handling
- Strategy Engine: Pairs trading strategies with ML-enhanced signals
- Risk Management: Position sizing, drawdown control, and risk metrics
- Execution: Order management and trade execution simulation
- Analytics: Performance attribution and regime analysis
- Visualization: Professional trading charts and reports

Example Usage:
    >>> import quantpulse as qp
    >>> 
    >>> # Initialize trading system
    >>> engine = qp.TradingEngine()
    >>> 
    >>> # Load market data
    >>> data_manager = qp.DataManager()
    >>> data = data_manager.get_pairs_data(['AAPL', 'MSFT'], '2020-01-01', '2023-12-31')
    >>> 
    >>> # Run pairs trading strategy
    >>> strategy = qp.PairsTradingStrategy()
    >>> signals = strategy.generate_signals(data)
    >>> 
    >>> # Execute backtest
    >>> backtest = qp.Backtester(initial_capital=100000)
    >>> results = backtest.run(signals, data)
    >>> 
    >>> # Analyze performance
    >>> analytics = qp.PerformanceAnalytics(results)
    >>> analytics.generate_report()

For more information, see: https://github.com/RajaBabu15/quantpulse-pairs-trading
"""

from datetime import datetime
import logging

# Version information
__version__ = "2.1.0"
__author__ = "QuantPulse Team"
__email__ = "contact@quantpulse.com"
__license__ = "MIT"
__copyright__ = f"2023-2025 QuantPulse Team"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Core components - Lazy imports to avoid circular dependencies
def _setup_imports():
    """Setup module imports and validate dependencies."""
    import sys
    import warnings
    
    # Check Python version
    if sys.version_info < (3, 9):
        warnings.warn(
            "QuantPulse requires Python 3.9+. "
            "Some features may not work correctly.",
            RuntimeWarning
        )
    
    # Check critical dependencies
    try:
        import numpy as np
        import pandas as pd
        import scipy
    except ImportError as e:
        raise ImportError(
            f"Missing critical dependency: {e}. "
            "Please install with: pip install -r requirements.txt"
        )
    
    # Check C++ extensions
    try:
        from . import native_interface
    except ImportError:
        warnings.warn(
            "C++ extensions not available. "
            "Performance will be degraded. "
            "Please build with: python setup.py build_ext --inplace",
            RuntimeWarning
        )

# Initialize the package
_setup_imports()

# Core API - Import main classes and functions
from .data_manager import DataManager, MarketDataProvider
from .strategies import (
    PairsTradingStrategy,
    StatisticalArbitrageStrategy, 
    MeanReversionStrategy
)
from .execution import (
    TradingEngine,
    OrderManager,
    PositionManager,
    RiskManager
)
from .backtesting import Backtester, WalkForwardOptimizer
from .analytics import (
    PerformanceAnalytics,
    RiskMetrics,
    RegimeAnalysis,
    AttributionAnalysis
)
from .visualization import (
    TradingDashboard,
    PerformanceReports,
    RiskCharts
)
from .utils import (
    Logger,
    Config,
    Validators,
    TimeUtils
)

# Native interface (if available)
try:
    from . import native_interface
    HAS_NATIVE_EXTENSIONS = True
except ImportError:
    HAS_NATIVE_EXTENSIONS = False

# Configuration
class Config:
    """Global configuration settings."""
    
    # Data settings
    DATA_CACHE_DIR = "data/cache"
    DATA_PROVIDERS = ["yfinance", "alpha_vantage", "polygon"]
    
    # Trading settings
    DEFAULT_COMMISSION = 0.001  # 0.1% per trade
    DEFAULT_SLIPPAGE = 0.0005   # 0.05% slippage
    MAX_POSITION_SIZE = 0.1     # 10% max position
    
    # Performance settings
    RISK_FREE_RATE = 0.02       # 2% annual risk-free rate
    BENCHMARK_SYMBOL = "SPY"    # S&P 500 benchmark
    
    # System settings
    LOG_LEVEL = "INFO"
    PARALLEL_JOBS = -1          # Use all available cores
    CACHE_ENABLED = True

# Utility functions
def get_version():
    """Get QuantPulse version string."""
    return __version__

def get_system_info():
    """Get system information and capabilities."""
    import platform
    import multiprocessing
    
    info = {
        'version': __version__,
        'python_version': platform.python_version(),
        'platform': platform.system(),
        'architecture': platform.machine(),
        'cpu_count': multiprocessing.cpu_count(),
        'has_native_extensions': HAS_NATIVE_EXTENSIONS,
    }
    
    # Check for SIMD capabilities
    if platform.machine() in ['arm64', 'aarch64']:
        info['simd_support'] = 'ARM64 NEON'
    elif 'x86' in platform.machine().lower():
        info['simd_support'] = 'x86_64 AVX2'
    else:
        info['simd_support'] = 'Generic'
    
    return info

def validate_environment():
    """Validate that the environment is properly configured."""
    issues = []
    
    # Check dependencies
    try:
        import numpy
        if numpy.__version__ < "1.20.0":
            issues.append(f"NumPy {numpy.__version__} is outdated, please upgrade to 1.20.0+")
    except ImportError:
        issues.append("NumPy is required but not installed")
    
    try:
        import pandas
        if pandas.__version__ < "1.3.0":
            issues.append(f"Pandas {pandas.__version__} is outdated, please upgrade to 1.3.0+")
    except ImportError:
        issues.append("Pandas is required but not installed")
    
    # Check native extensions
    if not HAS_NATIVE_EXTENSIONS:
        issues.append("Native C++ extensions are not available, performance will be degraded")
    
    return issues

# Print startup message
def _print_startup_info():
    """Print startup information."""
    info = get_system_info()
    print(f"""
🚀 QuantPulse Pairs Trading System v{info['version']}
   Platform: {info['platform']} {info['architecture']}
   Python: {info['python_version']} | CPUs: {info['cpu_count']}
   SIMD: {info['simd_support']} | Native: {info['has_native_extensions']}
   
📊 Ready for quantitative trading analysis!
   Documentation: https://github.com/RajaBabu15/quantpulse-pairs-trading
""")

# Auto-print startup info
if __name__ != "__main__":
    import os
    if os.getenv("QUANTPULSE_QUIET") != "1":
        _print_startup_info()

# Export main API
__all__ = [
    # Core classes
    'DataManager',
    'MarketDataProvider',
    'PairsTradingStrategy',
    'StatisticalArbitrageStrategy',
    'MeanReversionStrategy',
    'TradingEngine',
    'OrderManager',
    'PositionManager',
    'RiskManager',
    'Backtester',
    'WalkForwardOptimizer',
    'PerformanceAnalytics',
    'RiskMetrics',
    'RegimeAnalysis',
    'AttributionAnalysis',
    'TradingDashboard',
    'PerformanceReports',
    'RiskCharts',
    
    # Utilities
    'Logger',
    'Config',
    'Validators',
    'TimeUtils',
    'get_version',
    'get_system_info',
    'validate_environment',
    
    # Constants
    'HAS_NATIVE_EXTENSIONS',
]
