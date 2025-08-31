"""
CFFI-based Python wrapper for high-performance C++ quantitative trading functions.
Provides seamless integration between Python and optimized SIMD/OpenMP C++ code.
"""

import os
import sys
import ctypes
import numpy as np
from ctypes import Structure, c_double, c_int, c_size_t, c_char_p, POINTER
from pathlib import Path
import warnings
import threading
import atexit
from typing import Tuple, Optional, Dict, Any, List

# Platform-specific library loading
if sys.platform == "win32":
    LIBRARY_EXT = ".dll"
    COMPILE_FLAGS = ["-O3", "-march=native", "-mavx2", "-fopenmp", "-std=c++17"]
elif sys.platform == "darwin":
    LIBRARY_EXT = ".dylib" 
    COMPILE_FLAGS = ["-O3", "-march=native", "-mavx2", "-Xpreprocessor", "-fopenmp", "-std=c++17"]
else:
    LIBRARY_EXT = ".so"
    COMPILE_FLAGS = ["-O3", "-march=native", "-mavx2", "-fopenmp", "-std=c++17"]

# C structures matching quantpulse_core.h
class BacktestResult(Structure):
    """C structure for backtest results."""
    _fields_ = [
        ("total_return", c_double),
        ("sharpe_ratio", c_double),
        ("max_drawdown", c_double),
        ("num_trades", c_int),
        ("win_rate", c_double),
        ("profit_factor", c_double),
        ("avg_trade_return", c_double),
        ("volatility", c_double)
    ]
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to Python dictionary."""
        return {
            'total_return': self.total_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'num_trades': self.num_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_trade_return': self.avg_trade_return,
            'volatility': self.volatility
        }

class TradingParameters(Structure):
    """C structure for trading parameters."""
    _fields_ = [
        ("lookback", c_int),
        ("z_entry", c_double),
        ("z_exit", c_double),
        ("position_size", c_int),
        ("transaction_cost", c_double),
        ("profit_target", c_double),
        ("stop_loss", c_double)
    ]

class QuantPulseNative:
    """
    High-performance native library interface for QuantPulse trading optimization.
    
    Provides SIMD-accelerated mathematical operations, parallel cross-validation,
    thread-safe caching, and optimized backtesting functionality.
    """
    
    def __init__(self, library_path: Optional[str] = None, auto_build: bool = True):
        """
        Initialize the native library interface.
        
        Args:
            library_path: Path to compiled shared library. If None, searches standard locations.
            auto_build: Whether to automatically build the library if not found.
        """
        self._lib = None
        self._library_path = None
        self._is_initialized = False
        self._lock = threading.Lock()
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
        
        if library_path:
            self._library_path = Path(library_path)
        else:
            self._library_path = self._find_or_build_library(auto_build)
        
        if self._library_path and self._library_path.exists():
            self._load_library()
        else:
            warnings.warn("Native library not found. Falling back to Python implementation.")
    
    def _find_or_build_library(self, auto_build: bool) -> Optional[Path]:
        """Find existing library or build if needed."""
        # Search in common locations
        search_paths = [
            Path.cwd() / "build" / f"quantpulse_core{LIBRARY_EXT}",
            Path.cwd() / f"quantpulse_core{LIBRARY_EXT}",
            Path(__file__).parent / "build" / f"quantpulse_core{LIBRARY_EXT}",
            Path(__file__).parent / f"quantpulse_core{LIBRARY_EXT}"
        ]
        
        for path in search_paths:
            if path.exists():
                return path
        
        if auto_build:
            return self._build_library()
        
        return None
    
    def _build_library(self) -> Optional[Path]:
        """
        Automatically build the C++ library using available compiler.
        """
        try:
            import subprocess
            
            # Detect available compiler
            compilers = ["g++", "clang++", "c++"]
            compiler = None
            
            for cmd in compilers:
                try:
                    subprocess.run([cmd, "--version"], capture_output=True, check=True)
                    compiler = cmd
                    break
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            
            if not compiler:
                warnings.warn("No suitable C++ compiler found. Cannot build native library.")
                return None
            
            # Source files
            source_dir = Path(__file__).parent / "csrc"
            if not source_dir.exists():
                warnings.warn(f"Source directory {source_dir} not found.")
                return None
            
            source_files = [
                source_dir / "parallel_cv.cpp",
                source_dir / "simd_ops.cpp", 
                source_dir / "optimization_cache.cpp"
            ]
            
            # Check if all source files exist
            missing_files = [f for f in source_files if not f.exists()]
            if missing_files:
                warnings.warn(f"Missing source files: {missing_files}")
                return None
            
            # Build directory
            build_dir = Path(__file__).parent / "build"
            build_dir.mkdir(exist_ok=True)
            
            output_path = build_dir / f"quantpulse_core{LIBRARY_EXT}"
            
            # Compile command
            cmd = [compiler, "-shared", "-fPIC"] + COMPILE_FLAGS + [
                "-I", str(source_dir),
                "-o", str(output_path)
            ] + [str(f) for f in source_files]
            
            # Add OpenMP linking
            if sys.platform == "darwin":
                cmd.extend(["-lomp"])
            else:
                cmd.extend(["-fopenmp"])
            
            print(f"Building native library with: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Build failed with error:\n{result.stderr}")
                return None
            
            if output_path.exists():
                print(f"Successfully built library: {output_path}")
                return output_path
            else:
                print("Build appeared successful but output file not found.")
                return None
                
        except Exception as e:
            warnings.warn(f"Failed to build native library: {e}")
            return None
    
    def _load_library(self):
        """Load the compiled shared library and set up function signatures."""
        try:
            # Load the library
            self._lib = ctypes.CDLL(str(self._library_path))
            
            # Set up function signatures
            self._setup_function_signatures()
            
            self._is_initialized = True
            print(f"Successfully loaded native library: {self._library_path}")
            
        except Exception as e:
            warnings.warn(f"Failed to load native library: {e}")
            self._lib = None
    
    def _setup_function_signatures(self):
        """Define function signatures for all C++ functions."""
        if not self._lib:
            return
        
        # SIMD operations
        self._lib.simd_vector_add.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), c_size_t]
        self._lib.simd_vector_add.restype = None
        
        self._lib.simd_vector_mean.argtypes = [POINTER(c_double), c_size_t]
        self._lib.simd_vector_mean.restype = c_double
        
        self._lib.simd_vector_std.argtypes = [POINTER(c_double), c_size_t, c_double]
        self._lib.simd_vector_std.restype = c_double
        
        # Spread and z-score calculations
        self._lib.calculate_spread_and_zscore.argtypes = [
            POINTER(c_double), POINTER(c_double), c_size_t, c_int,
            POINTER(c_double), POINTER(c_double)
        ]
        self._lib.calculate_spread_and_zscore.restype = None
        
        # Backtesting
        self._lib.vectorized_backtest.argtypes = [
            POINTER(c_double), POINTER(c_double), c_size_t,
            TradingParameters
        ]
        self._lib.vectorized_backtest.restype = BacktestResult
        
        # Cross-validation
        self._lib.parallel_cross_validation.argtypes = [
            POINTER(c_double), POINTER(c_double), c_size_t,
            POINTER(c_double), c_int, c_double, c_double, c_double
        ]
        self._lib.parallel_cross_validation.restype = c_double
        
        # Batch optimization
        self._lib.batch_parameter_optimization.argtypes = [
            POINTER(c_double), POINTER(c_double), c_size_t,
            POINTER(POINTER(c_double)), c_int, c_int,
            POINTER(BacktestResult)
        ]
        self._lib.batch_parameter_optimization.restype = None
        
        # Cache management
        self._lib.cached_vectorized_backtest.argtypes = [
            POINTER(c_double), POINTER(c_double), c_size_t, TradingParameters
        ]
        self._lib.cached_vectorized_backtest.restype = BacktestResult
        
        self._lib.cached_objective_evaluation.argtypes = [
            POINTER(c_double), c_size_t, POINTER(c_double), POINTER(c_double), c_size_t,
            c_double, c_double, c_double
        ]
        self._lib.cached_objective_evaluation.restype = c_double
        
        self._lib.print_cache_statistics.argtypes = []
        self._lib.print_cache_statistics.restype = None
        
        self._lib.clear_all_caches.argtypes = []
        self._lib.clear_all_caches.restype = None
        
        self._lib.warm_up_caches.argtypes = [POINTER(c_double), POINTER(c_double), c_size_t]
        self._lib.warm_up_caches.restype = None
    
    @property
    def is_available(self) -> bool:
        """Check if native library is available and loaded."""
        return self._is_initialized and self._lib is not None
    
    def _ensure_contiguous_array(self, arr: np.ndarray) -> np.ndarray:
        """Ensure array is C-contiguous and double precision."""
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr, dtype=np.float64)
        if not arr.flags.c_contiguous or arr.dtype != np.float64:
            arr = np.ascontiguousarray(arr, dtype=np.float64)
        return arr
    
    def vector_add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """SIMD-accelerated vector addition (native-only)."""
        if not self.is_available:
            raise RuntimeError("Native library not available; vector_add requires native acceleration.")
        
        a = self._ensure_contiguous_array(a)
        b = self._ensure_contiguous_array(b)
        
        if len(a) != len(b):
            raise ValueError("Arrays must have the same length")
        
        result = np.empty_like(a)
        self._lib.simd_vector_add(
            a.ctypes.data_as(POINTER(c_double)),
            b.ctypes.data_as(POINTER(c_double)),
            result.ctypes.data_as(POINTER(c_double)),
            len(a)
        )
        return result
    
    def vector_mean(self, arr: np.ndarray) -> float:
        """SIMD-accelerated vector mean calculation (native-only)."""
        if not self.is_available:
            raise RuntimeError("Native library not available; vector_mean requires native acceleration.")
        
        arr = self._ensure_contiguous_array(arr)
        return self._lib.simd_vector_mean(
            arr.ctypes.data_as(POINTER(c_double)),
            len(arr)
        )
    
    def vector_std(self, arr: np.ndarray, mean: Optional[float] = None) -> float:
        """SIMD-accelerated vector standard deviation calculation (native-only)."""
        if not self.is_available:
            raise RuntimeError("Native library not available; vector_std requires native acceleration.")
        
        arr = self._ensure_contiguous_array(arr)
        if mean is None:
            mean = self.vector_mean(arr)
        
        return self._lib.simd_vector_std(
            arr.ctypes.data_as(POINTER(c_double)),
            len(arr),
            mean
        )
    
    def calculate_spread_and_zscore(self, prices1: np.ndarray, prices2: np.ndarray, 
                                   lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate spread and z-scores with SIMD optimization and caching (native-only).
        """
        if not self.is_available:
            raise RuntimeError("Native library not available; calculate_spread_and_zscore requires native acceleration.")
        
        prices1 = self._ensure_contiguous_array(prices1)
        prices2 = self._ensure_contiguous_array(prices2)
        
        if len(prices1) != len(prices2):
            raise ValueError("Price arrays must have the same length")
        
        n = len(prices1)
        spread = np.empty(n, dtype=np.float64)
        z_scores = np.empty(n, dtype=np.float64)
        
        self._lib.calculate_spread_and_zscore(
            prices1.ctypes.data_as(POINTER(c_double)),
            prices2.ctypes.data_as(POINTER(c_double)),
            n, lookback,
            spread.ctypes.data_as(POINTER(c_double)),
            z_scores.ctypes.data_as(POINTER(c_double))
        )
        
        return spread, z_scores
    
    def vectorized_backtest(self, prices1: np.ndarray, prices2: np.ndarray,
                           params: Dict[str, float], use_cache: bool = True) -> Dict[str, float]:
        """Run vectorized backtest with optional caching (native-only)."""
        if not self.is_available:
            raise RuntimeError("Native library not available; vectorized_backtest requires native acceleration.")
        
        prices1 = self._ensure_contiguous_array(prices1)
        prices2 = self._ensure_contiguous_array(prices2)
        
        # Convert parameters
        trading_params = TradingParameters(
            lookback=int(params.get('lookback', 20)),
            z_entry=params.get('z_entry', 2.0),
            z_exit=params.get('z_exit', 0.5),
            position_size=int(params.get('position_size', 10000)),
            transaction_cost=params.get('transaction_cost', 0.001),
            profit_target=params.get('profit_target', 2.0),
            stop_loss=params.get('stop_loss', 1.0)
        )
        
        # Choose cached or uncached version
        backtest_func = (self._lib.cached_vectorized_backtest if use_cache 
                        else self._lib.vectorized_backtest)
        
        result = backtest_func(
            prices1.ctypes.data_as(POINTER(c_double)),
            prices2.ctypes.data_as(POINTER(c_double)),
            len(prices1),
            trading_params
        )
        
        return result.to_dict()
    
    def parallel_cross_validation(self, prices1: np.ndarray, prices2: np.ndarray,
                                 params: np.ndarray, n_folds: int = 3,
                                 l1_ratio: float = 0.7, alpha: float = 0.02,
                                 kl_weight: float = 0.15) -> float:
        """Run parallel cross-validation for parameter optimization (native-only)."""
        if not self.is_available:
            raise RuntimeError("Native library not available; parallel_cross_validation requires native acceleration.")
        
        prices1 = self._ensure_contiguous_array(prices1)
        prices2 = self._ensure_contiguous_array(prices2)
        params = self._ensure_contiguous_array(params)
        
        return self._lib.parallel_cross_validation(
            prices1.ctypes.data_as(POINTER(c_double)),
            prices2.ctypes.data_as(POINTER(c_double)),
            len(prices1),
            params.ctypes.data_as(POINTER(c_double)),
            n_folds, l1_ratio, alpha, kl_weight
        )
    
    def batch_parameter_optimization(self, prices1: np.ndarray, prices2: np.ndarray,
                                   parameter_sets: List[np.ndarray]) -> List[Dict[str, float]]:
        """Optimize multiple parameter sets in parallel (native-only)."""
        if not self.is_available:
            raise RuntimeError("Native library not available; batch_parameter_optimization requires native acceleration.")
        if not parameter_sets:
            return []
        
        prices1 = self._ensure_contiguous_array(prices1)
        prices2 = self._ensure_contiguous_array(prices2)
        
        # Prepare parameter array pointers
        n_sets = len(parameter_sets)
        param_ptrs = (POINTER(c_double) * n_sets)()
        param_arrays = []
        
        for i, params in enumerate(parameter_sets):
            params_array = self._ensure_contiguous_array(params)
            param_arrays.append(params_array)  # Keep reference
            param_ptrs[i] = params_array.ctypes.data_as(POINTER(c_double))
        
        # Prepare results array
        results = (BacktestResult * n_sets)()
        
        self._lib.batch_parameter_optimization(
            prices1.ctypes.data_as(POINTER(c_double)),
            prices2.ctypes.data_as(POINTER(c_double)),
            len(prices1),
            param_ptrs, n_sets, len(parameter_sets[0]),
            results
        )
        
        return [result.to_dict() for result in results]
    
    def warm_up_caches(self, prices1: np.ndarray, prices2: np.ndarray):
        """Warm up caches with common parameter combinations (native-only)."""
        if not self.is_available:
            raise RuntimeError("Native library not available; warm_up_caches requires native acceleration.")
        
        prices1 = self._ensure_contiguous_array(prices1)
        prices2 = self._ensure_contiguous_array(prices2)
        
        self._lib.warm_up_caches(
            prices1.ctypes.data_as(POINTER(c_double)),
            prices2.ctypes.data_as(POINTER(c_double)),
            len(prices1)
        )
    
    def print_cache_statistics(self):
        """Print cache hit rates and statistics."""
        if self.is_available:
            self._lib.print_cache_statistics()
    
    def clear_caches(self):
        """Clear all cached data."""
        if self.is_available:
            self._lib.clear_all_caches()
    
    def cleanup(self):
        """Clean up resources."""
        if self._lib:
            self.clear_caches()
            self._lib = None
        self._is_initialized = False

# Global instance for easy access
_native_instance = None

def get_native_instance() -> QuantPulseNative:
    """Get or create the global native instance."""
    global _native_instance
    if _native_instance is None:
        _native_instance = QuantPulseNative()
    return _native_instance

# Convenience functions that use the global instance
def is_native_available() -> bool:
    """Check if native acceleration is available."""
    return get_native_instance().is_available

def calculate_spread_and_zscore(prices1: np.ndarray, prices2: np.ndarray, 
                               lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate spread and z-scores using native acceleration if available."""
    return get_native_instance().calculate_spread_and_zscore(prices1, prices2, lookback)

def vectorized_backtest(prices1: np.ndarray, prices2: np.ndarray,
                       params: Dict[str, float], use_cache: bool = True) -> Dict[str, float]:
    """Run vectorized backtest using native acceleration if available."""
    return get_native_instance().vectorized_backtest(prices1, prices2, params, use_cache)

def parallel_cross_validation(prices1: np.ndarray, prices2: np.ndarray,
                             params: np.ndarray, n_folds: int = 3,
                             l1_ratio: float = 0.7, alpha: float = 0.02,
                             kl_weight: float = 0.15) -> float:
    """Run parallel cross-validation using native acceleration if available."""
    return get_native_instance().parallel_cross_validation(
        prices1, prices2, params, n_folds, l1_ratio, alpha, kl_weight
    )

# ============================================================================
# Native-accelerated ElasticNet + KL + RMSprop Optimizer (High-level)
# ============================================================================

class NativeRMSprop:
    """Lightweight RMSprop optimizer used with native objective evaluation."""
    def __init__(self, learning_rate: float = 0.01, decay: float = 0.9, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.squared_gradients = None
        self.history: List[Dict[str, Any]] = []

    def reset(self, size: int):
        import numpy as _np
        self.squared_gradients = _np.zeros(size)
        self.history = []

    def compute_gradient(self, params, objective_func, delta: float = 1e-5):
        import numpy as _np
        grad = _np.zeros_like(params)
        for i in range(len(params)):
            p_plus = params.copy(); p_minus = params.copy()
            p_plus[i] += delta; p_minus[i] -= delta
            grad[i] = (objective_func(p_plus) - objective_func(p_minus)) / (2 * delta)
        return grad

    def step(self, params, objective_func):
        import numpy as _np
        if self.squared_gradients is None:
            self.reset(len(params))
        grad = self.compute_gradient(params, objective_func)
        self.squared_gradients = (self.decay * self.squared_gradients + (1 - self.decay) * (grad ** 2))
        update = self.learning_rate * grad / (_np.sqrt(self.squared_gradients) + self.epsilon)
        new_params = params - update  # minimize objective
        # Track history at current params
        self.history.append({
            'params': params.copy(),
            'gradient': grad.copy(),
            'update': update.copy(),
            'objective': objective_func(params)
        })
        return new_params

class NativeElasticNetKLOptimizer:
    """
    High-level optimizer that leverages QuantPulse native library for:
    - Parallel cross-validation objective evaluation (includes ElasticNet + KL)
    - Vectorized backtesting for validation/testing
    - RMSprop-based parameter search in normalized space [0,1]
    """
    def __init__(self, symbol1: str, symbol2: str,
                 l1_ratio: float = 0.7, alpha: float = 0.02, kl_weight: float = 0.15,
                 optimization_mode: str = 'hybrid'):
        import numpy as _np
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.l1_ratio = float(l1_ratio)
        self.alpha = float(alpha)
        self.kl_weight = float(kl_weight)
        self.optimization_mode = optimization_mode
        self.native = get_native_instance()
        self.rmsprop = NativeRMSprop(learning_rate=0.01)
        # Parameter bounds mirror Python implementation
        self.param_bounds = {
            'lookback': (5, 60),
            'z_entry': (0.5, 4.0),
            'z_exit': (0.1, 2.0),
            'position_size': (1000, 50000),
            'transaction_cost': (0.0001, 0.005),
            'profit_target': (1.5, 5.0),
            'stop_loss': (0.5, 2.0)
        }
        self.best_params: Optional[Dict[str, Any]] = None
        self.cv_scores: Optional[List[float]] = None
        self.history: List[Dict[str, Any]] = []
        # Initial normalized params
        self._initial = _np.array([0.5, 0.5, 0.25, 0.6, 0.3, 0.5, 0.5], dtype=float)

    def _normalize(self, params):
        import numpy as _np
        names = list(self.param_bounds.keys())
        norm = _np.zeros(len(names))
        for i, n in enumerate(names):
            lo, hi = self.param_bounds[n]
            norm[i] = (params[i] - lo) / (hi - lo)
        return norm

    def _denormalize(self, norm):
        import numpy as _np
        names = list(self.param_bounds.keys())
        params = _np.zeros(len(names))
        for i, n in enumerate(names):
            lo, hi = self.param_bounds[n]
            params[i] = norm[i] * (hi - lo) + lo
        return params

    def _extract_arrays(self, prices) -> Tuple[np.ndarray, np.ndarray]:
        import numpy as _np
        try:
            import pandas as _pd  # type: ignore
        except Exception:
            _pd = None
        if _pd is not None and hasattr(prices, 'columns'):
            # Expect DataFrame with price1/price2
            p1 = prices['price1'].values.astype(float)
            p2 = prices['price2'].values.astype(float)
            return p1, p2
        # tuple/list of two arrays
        if isinstance(prices, (tuple, list)) and len(prices) == 2:
            p1 = _np.asarray(prices[0], dtype=float)
            p2 = _np.asarray(prices[1], dtype=float)
            return p1, p2
        raise ValueError("Unsupported price format. Pass DataFrame with ['price1','price2'] or (p1, p2) arrays.")

    def _cv_objective(self, prices1: np.ndarray, prices2: np.ndarray, norm_params: np.ndarray, n_splits: int) -> float:
        # Denormalize and call native CV. Native returns a score to maximize -> we minimize negative.
        denorm = self._denormalize(norm_params)
        score = self.native.parallel_cross_validation(
            prices1, prices2, denorm, n_folds=n_splits, l1_ratio=self.l1_ratio,
            alpha=self.alpha, kl_weight=self.kl_weight
        )
        return -float(score)  # minimize

    def optimize(self, prices, n_splits: int = 3, max_iterations: int = 25) -> Dict[str, Any]:
        import numpy as _np
        if not self.native.is_available:
            raise RuntimeError("Native acceleration is not available; cannot run native optimizer.")
        p1, p2 = self._extract_arrays(prices)
        current = self._initial.copy()
        best = current.copy()
        best_obj = float('inf')
        self.rmsprop.reset(len(current))
        self.history = []

        def objective_func(nparams):
            return self._cv_objective(p1, p2, nparams, n_splits)

        for it in range(int(max_iterations)):
            new_params = self.rmsprop.step(current, objective_func)
            new_params = _np.clip(new_params, 0.0, 1.0)
            cur_obj = objective_func(new_params)
            if cur_obj < best_obj:
                best_obj = cur_obj
                best = new_params.copy()
            current = new_params
            self.history.append({'iter': it + 1, 'objective': cur_obj})
            # Simple convergence check
            if len(self.rmsprop.history) > 10:
                recent = [h['objective'] for h in self.rmsprop.history[-10:]]
                if _np.std(recent) < 1e-6:
                    break

        final = self._denormalize(best)
        self.best_params = {
            'lookback': int(final[0]),
            'z_entry': float(final[1]),
            'z_exit': float(final[2]),
            'position_size': int(final[3]),
            'transaction_cost': float(final[4]),
            'profit_target': float(final[5]),
            'stop_loss': float(final[6])
        }
        return self.best_params

    def backtest(self, prices, use_cache: bool = True) -> Dict[str, float]:
        if not self.best_params:
            raise RuntimeError("Run optimize() first to obtain best parameters.")
        p1, p2 = self._extract_arrays(prices)
        return self.native.vectorized_backtest(p1, p2, self.best_params, use_cache=use_cache)

class NativeElasticNetKLPortfolioOptimizer:
    """Portfolio-level optimizer using native CV/backtest for each pair."""
    def __init__(self, l1_ratio: float = 0.7, alpha: float = 0.02, kl_weight: float = 0.15):
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.kl_weight = kl_weight
        self.results: Dict[str, Any] = {}

    def optimize_all(self, pairs_data: Dict[str, Any], n_splits: int = 3, max_iterations: int = 25) -> Dict[str, Any]:
        import time as _time
        for pair_name, prices in pairs_data.items():
            try:
                sym1, sym2 = pair_name.split('-') if '-' in pair_name else (None, None)
                if not sym1 or not sym2:
                    raise ValueError("pair_name must be like 'SYM1-SYM2'")
                opt = NativeElasticNetKLOptimizer(sym1, sym2, self.l1_ratio, self.alpha, self.kl_weight)
                t0 = _time.time()
                best = opt.optimize(prices, n_splits=n_splits, max_iterations=max_iterations)
                optimization_time = _time.time() - t0
                bt = opt.backtest(prices)
                self.results[pair_name] = {
                    'best_params': best,
                    'backtest': bt,
                    'optimizer_history': opt.history,
                    'optimization_time': optimization_time
                }
            except Exception as e:
                self.results[pair_name] = {'error': str(e)}
        return self.results

# Convenience factory

def create_native_elasticnet_optimizer(symbol1: str, symbol2: str,
                                       l1_ratio: float = 0.7, alpha: float = 0.02, kl_weight: float = 0.15,
                                       optimization_mode: str = 'hybrid') -> NativeElasticNetKLOptimizer:
    """Create a native optimizer for a given pair."""
    return NativeElasticNetKLOptimizer(symbol1, symbol2, l1_ratio, alpha, kl_weight, optimization_mode)

# ---------------------------------------------------------------------------
# Portfolio utilities: analysis and report generation (native-only)
# ---------------------------------------------------------------------------

def analyze_native_portfolio(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze portfolio-level statistics from native optimization results."""
    import numpy as _np
    successful = {k: v for k, v in results.items() if isinstance(v, dict) and 'backtest' in v}
    if not successful:
        return {'summary': {'total_pairs': 0, 'successful_optimizations': 0}}
    sharpes = [_v['backtest'].get('sharpe_ratio', 0.0) for _v in successful.values()]
    returns = [_v['backtest'].get('total_return', 0.0) for _v in successful.values()]
    drawdowns = [_v['backtest'].get('max_drawdown', 0.0) for _v in successful.values()]
    opt_times = [_v.get('optimization_time', 0.0) for _v in successful.values()]
    summary = {
        'total_pairs': len(results),
        'successful_optimizations': len(successful),
        'average_sharpe': float(_np.mean(sharpes)) if sharpes else 0.0,
        'average_total_return': float(_np.mean(returns)) if returns else 0.0,
        'average_max_drawdown': float(_np.mean(drawdowns)) if drawdowns else 0.0,
        'total_optimization_time': float(sum(opt_times)) if opt_times else 0.0
    }
    return {'summary': summary, 'pair_results': results}

def generate_native_performance_report(results: Dict[str, Any], output_file: Optional[str] = None) -> Dict[str, Any]:
    """Generate and optionally save a native optimization performance report (JSON)."""
    import json as _json
    import os as _os
    report = analyze_native_portfolio(results)
    if output_file:
        _os.makedirs('performance', exist_ok=True)
        if not output_file.startswith('performance/'):
            output_file = f'performance/{output_file}'
        with open(output_file, 'w') as f:
            _json.dump(report, f, indent=2, default=str)
    return report
