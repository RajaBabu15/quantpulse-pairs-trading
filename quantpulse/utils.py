"""
Utilities Module
===============

Core utilities for configuration, logging, data validation, and helper functions
for the QuantPulse trading system.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import os
import json
import yaml
import logging
import logging.config
from pathlib import Path
import warnings
from functools import wraps
import time

import numpy as np
import pandas as pd

import logging
logger = logging.getLogger(__name__)

class ConfigManager:
    """Configuration management with environment variable support."""
    
    def __init__(self, config_path: Optional[str] = None):
        print(f"🔄 ENTERING ConfigManager.__init__({config_path}) at {datetime.now().strftime('%H:%M:%S')}")
        
        self.config_path = config_path or "config.yaml"
        self.config = {}
        self.env_prefix = "QUANTPULSE_"
        
        self._load_default_config()
        if os.path.exists(self.config_path):
            self._load_config_file()
        self._load_environment_variables()
        
        print(f"✅ EXITING ConfigManager.__init__() at {datetime.now().strftime('%H:%M:%S')}")
    
    def _load_default_config(self):
        """Load default configuration."""
        print(f"🔄 ENTERING _load_default_config() at {datetime.now().strftime('%H:%M:%S')}")
        
        self.config = {
            # Data Configuration
            'data': {
                'providers': {
                    'yahoo_finance': {
                        'enabled': True,
                        'timeout': 30,
                        'retry_attempts': 3
                    },
                    'alpha_vantage': {
                        'enabled': False,
                        'api_key': None,
                        'timeout': 30,
                        'retry_attempts': 3
                    }
                },
                'cache': {
                    'enabled': True,
                    'expiry_hours': 24,
                    'max_size_mb': 100
                },
                'validation': {
                    'min_data_points': 100,
                    'max_missing_pct': 5.0,
                    'price_change_threshold': 0.5  # 50% daily change threshold
                }
            },
            
            # Trading Configuration
            'trading': {
                'initial_capital': 100000,
                'max_positions': 10,
                'position_sizing': {
                    'method': 'equal_weight',  # 'equal_weight', 'volatility_adjusted', 'risk_parity'
                    'max_position_size': 0.1,  # 10% of portfolio per position
                    'min_position_size': 0.01  # 1% minimum
                },
                'risk_management': {
                    'max_portfolio_var': 0.02,  # 2% daily VaR
                    'max_correlation': 0.8,
                    'stop_loss': 0.05,  # 5% stop loss
                    'take_profit': 0.10,  # 10% take profit
                    'max_drawdown': 0.15  # 15% max drawdown
                }
            },
            
            # Strategy Configuration
            'strategies': {
                'pairs_trading': {
                    'lookback_period': 252,
                    'entry_threshold': 2.0,
                    'exit_threshold': 0.0,
                    'stop_loss': 3.0,
                    'min_half_life': 1,
                    'max_half_life': 252
                },
                'mean_reversion': {
                    'lookback_period': 20,
                    'entry_threshold': 2.0,
                    'exit_threshold': 0.0,
                    'bollinger_period': 20,
                    'bollinger_std': 2.0
                },
                'statistical_arbitrage': {
                    'cointegration_lookback': 252,
                    'signal_lookback': 20,
                    'entry_threshold': 2.0,
                    'exit_threshold': 0.0,
                    'hedge_ratio_update_freq': 20
                }
            },
            
            # Backtesting Configuration
            'backtesting': {
                'commission': 0.001,  # 0.1% commission
                'slippage': 0.0005,   # 0.05% slippage
                'benchmark': '^GSPC',  # S&P 500
                'start_date': '2020-01-01',
                'end_date': None,  # Current date if None
                'rebalance_frequency': 'daily',
                'walk_forward': {
                    'enabled': False,
                    'training_periods': 252,
                    'testing_periods': 63,
                    'step_size': 21
                }
            },
            
            # Logging Configuration
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file_logging': {
                    'enabled': True,
                    'filename': 'quantpulse.log',
                    'max_size_mb': 10,
                    'backup_count': 5
                }
            },
            
            # Performance Configuration
            'performance': {
                'parallel_processing': {
                    'enabled': True,
                    'n_jobs': -1  # Use all available cores
                },
                'optimization': {
                    'use_numba': True,
                    'use_cython': False
                }
            }
        }
        
        print(f"✅ EXITING _load_default_config() at {datetime.now().strftime('%H:%M:%S')}")
    
    def _load_config_file(self):
        """Load configuration from file."""
        print(f"🔄 ENTERING _load_config_file() at {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    file_config = yaml.safe_load(f)
                else:
                    file_config = json.load(f)
            
            # Merge with default config
            self._deep_merge(self.config, file_config)
            logger.info(f"Loaded configuration from {self.config_path}")
            
        except Exception as e:
            logger.warning(f"Failed to load config file {self.config_path}: {e}")
        
        print(f"✅ EXITING _load_config_file() at {datetime.now().strftime('%H:%M:%S')}")
    
    def _load_environment_variables(self):
        """Load configuration from environment variables."""
        print(f"🔄 ENTERING _load_environment_variables() at {datetime.now().strftime('%H:%M:%S')}")
        
        env_vars = {k: v for k, v in os.environ.items() if k.startswith(self.env_prefix)}
        
        for key, value in env_vars.items():
            # Remove prefix and convert to nested dict path
            config_key = key[len(self.env_prefix):].lower()
            config_path = config_key.split('_')
            
            # Set nested configuration
            current = self.config
            for path_part in config_path[:-1]:
                if path_part not in current:
                    current[path_part] = {}
                current = current[path_part]
            
            # Convert value types
            try:
                # Try to parse as JSON first (handles bools, numbers, lists, etc.)
                current[config_path[-1]] = json.loads(value)
            except json.JSONDecodeError:
                # If not valid JSON, treat as string
                current[config_path[-1]] = value
        
        if env_vars:
            logger.info(f"Loaded {len(env_vars)} environment variables")
        
        print(f"✅ EXITING _load_environment_variables() at {datetime.now().strftime('%H:%M:%S')}")
    
    def _deep_merge(self, base_dict: Dict, update_dict: Dict):
        """Deep merge two dictionaries."""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated path."""
        print(f"🔄 ENTERING get({key_path}) at {datetime.now().strftime('%H:%M:%S')}")
        
        keys = key_path.split('.')
        current = self.config
        
        try:
            for key in keys:
                current = current[key]
            
            print(f"✅ EXITING get() at {datetime.now().strftime('%H:%M:%S')}")
            return current
        except (KeyError, TypeError):
            print(f"✅ EXITING get() (default) at {datetime.now().strftime('%H:%M:%S')}")
            return default
    
    def set(self, key_path: str, value: Any):
        """Set configuration value by dot-separated path."""
        print(f"🔄 ENTERING set({key_path}, {value}) at {datetime.now().strftime('%H:%M:%S')}")
        
        keys = key_path.split('.')
        current = self.config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
        
        print(f"✅ EXITING set() at {datetime.now().strftime('%H:%M:%S')}")
    
    def save_config(self, filepath: Optional[str] = None):
        """Save current configuration to file."""
        print(f"🔄 ENTERING save_config({filepath}) at {datetime.now().strftime('%H:%M:%S')}")
        
        filepath = filepath or self.config_path
        
        try:
            with open(filepath, 'w') as f:
                if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
                else:
                    json.dump(self.config, f, indent=2)
            
            logger.info(f"Configuration saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
        
        print(f"✅ EXITING save_config() at {datetime.now().strftime('%H:%M:%S')}")

class LoggingManager:
    """Advanced logging configuration and management."""
    
    @staticmethod
    def setup_logging(config: Optional[Dict] = None, log_dir: str = "logs"):
        """Setup comprehensive logging configuration."""
        print(f"🔄 ENTERING LoggingManager.setup_logging() at {datetime.now().strftime('%H:%M:%S')}")
        
        # Default logging config
        if config is None:
            config = {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file_logging': {
                    'enabled': True,
                    'filename': 'quantpulse.log',
                    'max_size_mb': 10,
                    'backup_count': 5
                }
            }
        
        # Create log directory
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(exist_ok=True)
        
        # Configure logging
        logging_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                },
                'detailed': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': config.get('level', 'INFO'),
                    'formatter': 'standard',
                    'stream': 'ext://sys.stdout'
                }
            },
            'loggers': {
                'quantpulse': {
                    'level': config.get('level', 'INFO'),
                    'handlers': ['console'],
                    'propagate': False
                },
                'root': {
                    'level': 'WARNING',
                    'handlers': ['console']
                }
            }
        }
        
        # Add file logging if enabled
        if config.get('file_logging', {}).get('enabled', True):
            from logging.handlers import RotatingFileHandler
            
            log_file = log_dir_path / config['file_logging'].get('filename', 'quantpulse.log')
            max_bytes = config['file_logging'].get('max_size_mb', 10) * 1024 * 1024
            backup_count = config['file_logging'].get('backup_count', 5)
            
            logging_config['handlers']['file'] = {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': config.get('level', 'INFO'),
                'formatter': 'detailed',
                'filename': str(log_file),
                'maxBytes': max_bytes,
                'backupCount': backup_count
            }
            
            logging_config['loggers']['quantpulse']['handlers'].append('file')
        
        # Apply configuration
        logging.config.dictConfig(logging_config)
        
        # Set up warnings capture
        logging.captureWarnings(True)
        
        logger = logging.getLogger('quantpulse')
        logger.info("Logging system initialized")
        
        print(f"✅ EXITING LoggingManager.setup_logging() at {datetime.now().strftime('%H:%M:%S')}")

class DataValidator:
    """Data validation and quality checks."""
    
    @staticmethod
    def validate_price_data(df: pd.DataFrame, symbol: str = "Unknown", 
                           config: Optional[Dict] = None) -> Tuple[bool, List[str]]:
        """Validate price data quality."""
        print(f"🔄 ENTERING validate_price_data({symbol}) at {datetime.now().strftime('%H:%M:%S')}")
        
        errors = []
        warnings_list = []
        
        # Default validation config
        if config is None:
            config = {
                'min_data_points': 100,
                'max_missing_pct': 5.0,
                'price_change_threshold': 0.5,
                'required_columns': ['Open', 'High', 'Low', 'Close', 'Volume']
            }
        
        # Check if DataFrame is empty
        if df.empty:
            errors.append(f"{symbol}: DataFrame is empty")
            return False, errors
        
        # Check required columns
        required_cols = config.get('required_columns', ['Close'])
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"{symbol}: Missing required columns: {missing_cols}")
        
        # Check minimum data points
        min_points = config.get('min_data_points', 100)
        if len(df) < min_points:
            errors.append(f"{symbol}: Insufficient data points ({len(df)} < {min_points})")
        
        # Check for excessive missing data
        max_missing_pct = config.get('max_missing_pct', 5.0)
        for col in df.columns:
            if col in df.select_dtypes(include=[np.number]).columns:
                missing_pct = df[col].isna().mean() * 100
                if missing_pct > max_missing_pct:
                    warnings_list.append(f"{symbol}: {col} has {missing_pct:.1f}% missing data")
        
        # Check for extreme price changes
        threshold = config.get('price_change_threshold', 0.5)
        if 'Close' in df.columns:
            returns = df['Close'].pct_change().abs()
            extreme_changes = returns > threshold
            if extreme_changes.any():
                count = extreme_changes.sum()
                max_change = returns.max()
                warnings_list.append(f"{symbol}: {count} extreme price changes detected (max: {max_change:.1%})")
        
        # Check for price anomalies (High < Low, etc.)
        if all(col in df.columns for col in ['High', 'Low', 'Open', 'Close']):
            # High should be >= Low
            high_low_errors = df['High'] < df['Low']
            if high_low_errors.any():
                errors.append(f"{symbol}: {high_low_errors.sum()} instances where High < Low")
            
            # High should be >= Open and Close
            high_open_errors = df['High'] < df['Open']
            high_close_errors = df['High'] < df['Close']
            if high_open_errors.any() or high_close_errors.any():
                total_errors = high_open_errors.sum() + high_close_errors.sum()
                errors.append(f"{symbol}: {total_errors} instances where High < Open/Close")
            
            # Low should be <= Open and Close
            low_open_errors = df['Low'] > df['Open']
            low_close_errors = df['Low'] > df['Close']
            if low_open_errors.any() or low_close_errors.any():
                total_errors = low_open_errors.sum() + low_close_errors.sum()
                errors.append(f"{symbol}: {total_errors} instances where Low > Open/Close")
        
        # Check for zero or negative prices
        if 'Close' in df.columns:
            non_positive = df['Close'] <= 0
            if non_positive.any():
                errors.append(f"{symbol}: {non_positive.sum()} non-positive prices")
        
        # Check volume data
        if 'Volume' in df.columns:
            zero_volume = df['Volume'] == 0
            if zero_volume.mean() > 0.1:  # More than 10% zero volume
                warnings_list.append(f"{symbol}: {zero_volume.mean():.1%} of days have zero volume")
        
        # Log warnings
        for warning in warnings_list:
            logger.warning(warning)
        
        # Determine if data is valid
        is_valid = len(errors) == 0
        
        if errors:
            for error in errors:
                logger.error(error)
        else:
            logger.info(f"{symbol}: Data validation passed")
        
        print(f"✅ EXITING validate_price_data() at {datetime.now().strftime('%H:%M:%S')}")
        return is_valid, errors + warnings_list
    
    @staticmethod
    def clean_price_data(df: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """Clean price data by handling missing values and anomalies."""
        print(f"🔄 ENTERING clean_price_data() at {datetime.now().strftime('%H:%M:%S')}")
        
        cleaned_df = df.copy()
        
        # Remove duplicate dates
        if isinstance(cleaned_df.index, pd.DatetimeIndex):
            cleaned_df = cleaned_df[~cleaned_df.index.duplicated(keep='first')]
        
        # Handle missing values
        if method == 'forward_fill':
            cleaned_df = cleaned_df.fillna(method='ffill')
        elif method == 'backward_fill':
            cleaned_df = cleaned_df.fillna(method='bfill')
        elif method == 'interpolate':
            cleaned_df = cleaned_df.interpolate(method='time')
        elif method == 'drop':
            cleaned_df = cleaned_df.dropna()
        
        # Fix price anomalies
        if all(col in cleaned_df.columns for col in ['High', 'Low', 'Open', 'Close']):
            # Ensure High >= Low
            cleaned_df['High'] = np.maximum(cleaned_df['High'], cleaned_df['Low'])
            
            # Ensure High >= Open, Close and Low <= Open, Close
            cleaned_df['High'] = np.maximum(cleaned_df['High'], 
                                          np.maximum(cleaned_df['Open'], cleaned_df['Close']))
            cleaned_df['Low'] = np.minimum(cleaned_df['Low'],
                                         np.minimum(cleaned_df['Open'], cleaned_df['Close']))
        
        # Remove non-positive prices
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in ['Open', 'High', 'Low', 'Close']:
                cleaned_df = cleaned_df[cleaned_df[col] > 0]
        
        print(f"✅ EXITING clean_price_data() at {datetime.now().strftime('%H:%M:%S')}")
        return cleaned_df

class PerformanceTimer:
    """Performance timing and profiling utilities."""
    
    def __init__(self, name: str = "Operation"):
        print(f"🔄 ENTERING PerformanceTimer.__init__({name}) at {datetime.now().strftime('%H:%M:%S')}")
        
        self.name = name
        self.start_time = None
        self.end_time = None
        
        print(f"✅ EXITING PerformanceTimer.__init__() at {datetime.now().strftime('%H:%M:%S')}")
    
    def __enter__(self):
        print(f"🔄 ENTERING {self.name} at {datetime.now().strftime('%H:%M:%S')}")
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        duration = self.end_time - self.start_time
        print(f"✅ EXITING {self.name} at {datetime.now().strftime('%H:%M:%S')} (Duration: {duration:.4f}s)")
        logger.info(f"{self.name} completed in {duration:.4f} seconds")
    
    @property
    def duration(self) -> Optional[float]:
        """Get duration in seconds."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None

def timer(func: Callable = None, *, name: Optional[str] = None):
    """Decorator to time function execution."""
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapper(*args, **kwargs):
            timer_name = name or f"{f.__module__}.{f.__name__}"
            with PerformanceTimer(timer_name):
                return f(*args, **kwargs)
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)

class MemoryProfiler:
    """Memory usage profiling utilities."""
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage statistics."""
        print(f"🔄 ENTERING get_memory_usage() at {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            stats = {
                'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
                'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
                'memory_percent': process.memory_percent(),
                'cpu_percent': process.cpu_percent()
            }
            
            print(f"✅ EXITING get_memory_usage() at {datetime.now().strftime('%H:%M:%S')}")
            return stats
            
        except ImportError:
            logger.warning("psutil not available for memory profiling")
            return {}
    
    @staticmethod
    def log_memory_usage(prefix: str = "Memory Usage"):
        """Log current memory usage."""
        print(f"🔄 ENTERING log_memory_usage() at {datetime.now().strftime('%H:%M:%S')}")
        
        stats = MemoryProfiler.get_memory_usage()
        if stats:
            logger.info(f"{prefix}: RSS={stats['rss_mb']:.1f}MB, "
                       f"VMS={stats['vms_mb']:.1f}MB, "
                       f"Memory%={stats['memory_percent']:.1f}%, "
                       f"CPU%={stats['cpu_percent']:.1f}%")
        
        print(f"✅ EXITING log_memory_usage() at {datetime.now().strftime('%H:%M:%S')}")

def memory_profile(func: Callable = None, *, log_before: bool = True, log_after: bool = True):
    """Decorator to profile memory usage around function execution."""
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapper(*args, **kwargs):
            if log_before:
                MemoryProfiler.log_memory_usage(f"Before {f.__name__}")
            
            result = f(*args, **kwargs)
            
            if log_after:
                MemoryProfiler.log_memory_usage(f"After {f.__name__}")
            
            return result
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)

class DataFrameUtils:
    """Utility functions for DataFrame operations."""
    
    @staticmethod
    def safe_divide(numerator: Union[pd.Series, np.ndarray, float], 
                   denominator: Union[pd.Series, np.ndarray, float], 
                   fill_value: float = 0.0) -> Union[pd.Series, np.ndarray, float]:
        """Safely divide two arrays/series, handling division by zero."""
        print(f"🔄 ENTERING safe_divide() at {datetime.now().strftime('%H:%M:%S')}")
        
        if isinstance(denominator, (pd.Series, np.ndarray)):
            result = np.where(denominator != 0, numerator / denominator, fill_value)
            if isinstance(numerator, pd.Series):
                result = pd.Series(result, index=numerator.index)
        else:
            result = numerator / denominator if denominator != 0 else fill_value
        
        print(f"✅ EXITING safe_divide() at {datetime.now().strftime('%H:%M:%S')}")
        return result
    
    @staticmethod
    def rolling_apply_parallel(series: pd.Series, func: Callable, window: int, 
                             n_jobs: int = -1) -> pd.Series:
        """Apply rolling function with parallel processing."""
        print(f"🔄 ENTERING rolling_apply_parallel() at {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            from joblib import Parallel, delayed
            
            def apply_window(i):
                if i < window - 1:
                    return np.nan
                window_data = series.iloc[i-window+1:i+1]
                return func(window_data)
            
            results = Parallel(n_jobs=n_jobs)(
                delayed(apply_window)(i) for i in range(len(series))
            )
            
            result = pd.Series(results, index=series.index)
            
        except ImportError:
            # Fallback to regular rolling apply
            logger.warning("joblib not available, using regular rolling apply")
            result = series.rolling(window).apply(func)
        
        print(f"✅ EXITING rolling_apply_parallel() at {datetime.now().strftime('%H:%M:%S')}")
        return result
    
    @staticmethod
    def resample_to_frequency(df: pd.DataFrame, freq: str, 
                             agg_methods: Dict[str, str] = None) -> pd.DataFrame:
        """Resample DataFrame to different frequency."""
        print(f"🔄 ENTERING resample_to_frequency({freq}) at {datetime.now().strftime('%H:%M:%S')}")
        
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex for resampling")
        
        if agg_methods is None:
            # Default aggregation methods for common column names
            agg_methods = {
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }
        
        # Apply aggregation methods
        agg_dict = {}
        for col in df.columns:
            if col in agg_methods:
                agg_dict[col] = agg_methods[col]
            else:
                # Default to last for numerical columns, first for others
                if df[col].dtype in ['float64', 'int64']:
                    agg_dict[col] = 'last'
                else:
                    agg_dict[col] = 'first'
        
        result = df.resample(freq).agg(agg_dict)
        
        print(f"✅ EXITING resample_to_frequency() at {datetime.now().strftime('%H:%M:%S')}")
        return result

class MathUtils:
    """Mathematical utility functions."""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        print(f"🔄 ENTERING calculate_sharpe_ratio() at {datetime.now().strftime('%H:%M:%S')}")
        
        excess_returns = returns.mean() * 252 - risk_free_rate
        volatility = returns.std() * np.sqrt(252)
        
        sharpe = excess_returns / volatility if volatility > 0 else 0
        
        print(f"✅ EXITING calculate_sharpe_ratio() at {datetime.now().strftime('%H:%M:%S')}")
        return sharpe
    
    @staticmethod
    def calculate_max_drawdown(cumulative_returns: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """Calculate maximum drawdown and its duration."""
        print(f"🔄 ENTERING calculate_max_drawdown() at {datetime.now().strftime('%H:%M:%S')}")
        
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        
        # Find the peak before max drawdown
        peak_date = running_max.loc[:max_dd_date].idxmax()
        
        print(f"✅ EXITING calculate_max_drawdown() at {datetime.now().strftime('%H:%M:%S')}")
        return max_dd, peak_date, max_dd_date
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence: float = 0.05) -> float:
        """Calculate Value at Risk."""
        print(f"🔄 ENTERING calculate_var() at {datetime.now().strftime('%H:%M:%S')}")
        
        var = np.percentile(returns, confidence * 100)
        
        print(f"✅ EXITING calculate_var() at {datetime.now().strftime('%H:%M:%S')}")
        return var
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        print(f"🔄 ENTERING calculate_cvar() at {datetime.now().strftime('%H:%M:%S')}")
        
        var = MathUtils.calculate_var(returns, confidence)
        tail_returns = returns[returns <= var]
        cvar = tail_returns.mean() if len(tail_returns) > 0 else var
        
        print(f"✅ EXITING calculate_cvar() at {datetime.now().strftime('%H:%M:%S')}")
        return cvar
    
    @staticmethod
    def half_life_mean_reversion(prices: pd.Series) -> float:
        """Calculate half-life of mean reversion."""
        print(f"🔄 ENTERING half_life_mean_reversion() at {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            from statsmodels.tsa.stattools import adfuller
            from statsmodels.regression.linear_model import OLS
            
            # Calculate lagged differences
            lagged_prices = prices.shift(1)
            delta_prices = prices.diff()
            
            # Remove NaN values
            df_reg = pd.DataFrame({
                'delta': delta_prices,
                'lagged': lagged_prices
            }).dropna()
            
            # Run regression: Δp_t = α + β * p_{t-1} + ε_t
            model = OLS(df_reg['delta'], df_reg['lagged']).fit()
            beta = model.params.iloc[0]
            
            # Half-life calculation
            if beta < 0:
                half_life = -np.log(2) / np.log(1 + beta)
            else:
                half_life = np.inf
            
            print(f"✅ EXITING half_life_mean_reversion() at {datetime.now().strftime('%H:%M:%S')}")
            return half_life
            
        except ImportError:
            logger.warning("statsmodels not available for half-life calculation")
            return np.nan
        except Exception as e:
            logger.error(f"Error calculating half-life: {e}")
            return np.nan

def setup_quantpulse_environment(config_path: Optional[str] = None) -> ConfigManager:
    """Setup complete QuantPulse environment with configuration and logging."""
    print(f"🔄 ENTERING setup_quantpulse_environment() at {datetime.now().strftime('%H:%M:%S')}")
    
    # Load configuration
    config_manager = ConfigManager(config_path)
    
    # Setup logging
    logging_config = config_manager.get('logging', {})
    LoggingManager.setup_logging(logging_config)
    
    # Log system information
    logger = logging.getLogger('quantpulse')
    logger.info("QuantPulse environment initialized")
    
    # Log configuration summary
    logger.info(f"Initial capital: ${config_manager.get('trading.initial_capital', 100000):,}")
    logger.info(f"Data validation enabled: {config_manager.get('data.validation.min_data_points', 100) > 0}")
    logger.info(f"Parallel processing: {config_manager.get('performance.parallel_processing.enabled', True)}")
    
    # Log memory usage
    MemoryProfiler.log_memory_usage("System startup")
    
    print(f"✅ EXITING setup_quantpulse_environment() at {datetime.now().strftime('%H:%M:%S')}")
    return config_manager

# Global configuration instance
_global_config = None

def get_config() -> ConfigManager:
    """Get global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = ConfigManager()
    return _global_config
