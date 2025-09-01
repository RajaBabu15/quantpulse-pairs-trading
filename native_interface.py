import numpy as np
from typing import Tuple, Dict, Any, List, Optional
import quantpulse_core_py as _core

class QuantPulseNative:
    def __init__(self):
        pass
    @property
    def is_available(self) -> bool:
        return True
    def _ensure_array(self, arr: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(arr, dtype=np.float64)
    def vector_add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return _core.simd_vector_add(self._ensure_array(a), self._ensure_array(b))
    def vector_mean(self, arr: np.ndarray) -> float:
        return float(_core.simd_vector_mean(self._ensure_array(arr)))
    def vector_std(self, arr: np.ndarray, mean: Optional[float] = None) -> float:
        arr = self._ensure_array(arr)
        if mean is None:
            mean = self.vector_mean(arr)
        return float(_core.simd_vector_std(arr, float(mean)))
    def calculate_spread_and_zscore(self, prices1: np.ndarray, prices2: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        spread, z = _core.calculate_spread_and_zscore(self._ensure_array(prices1), self._ensure_array(prices2), int(lookback))
        return np.asarray(spread), np.asarray(z)
    def vectorized_backtest(self, prices1: np.ndarray, prices2: np.ndarray, params: Dict[str, float], use_cache: bool = True) -> Dict[str, float]:
        return dict(_core.vectorized_backtest(self._ensure_array(prices1), self._ensure_array(prices2), params, bool(use_cache)))
    def parallel_cross_validation(self, prices1: np.ndarray, prices2: np.ndarray, params: np.ndarray, n_folds: int = 3, l1_ratio: float = 0.7, alpha: float = 0.02, kl_weight: float = 0.15) -> float:
        return float(_core.parallel_cross_validation(self._ensure_array(prices1), self._ensure_array(prices2), self._ensure_array(params), int(n_folds), float(l1_ratio), float(alpha), float(kl_weight)))
    def batch_parameter_optimization(self, prices1: np.ndarray, prices2: np.ndarray, parameter_sets: List[np.ndarray]) -> List[Dict[str, float]]:
        pysets = [self._ensure_array(p) for p in parameter_sets]
        results = _core.batch_parameter_optimization(self._ensure_array(prices1), self._ensure_array(prices2), pysets)
        return [dict(x) for x in results]
    def get_trade_returns(self, prices1: np.ndarray, prices2: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        return np.asarray(_core.backtest_trade_returns(self._ensure_array(prices1), self._ensure_array(prices2), params))
    def warm_up_caches(self, prices1: np.ndarray, prices2: np.ndarray):
        _core.warm_up_caches(self._ensure_array(prices1), self._ensure_array(prices2))
    def print_cache_statistics(self):
        _core.print_cache_statistics()
    def clear_caches(self):
        _core.clear_all_caches()

# Global instance
_native = QuantPulseNative()

# Convenience functions
def is_native_available() -> bool:
    """Check if native acceleration is available."""
    return _native.is_available

def calculate_spread_and_zscore(prices1: np.ndarray, prices2: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    return _native.calculate_spread_and_zscore(prices1, prices2, lookback)
def vectorized_backtest(prices1: np.ndarray, prices2: np.ndarray, params: Dict[str, float], use_cache: bool = True) -> Dict[str, float]:
    return _native.vectorized_backtest(prices1, prices2, params, use_cache)
def parallel_cross_validation(prices1: np.ndarray, prices2: np.ndarray, params: np.ndarray, n_folds: int = 3, l1_ratio: float = 0.7, alpha: float = 0.02, kl_weight: float = 0.15) -> float:
    return _native.parallel_cross_validation(prices1, prices2, params, n_folds, l1_ratio, alpha, kl_weight)

class NativeRMSprop:
    def __init__(self, lr: float = 0.001, rho: float = 0.9, epsilon: float = 1e-8):
        self.lr = lr
        self.rho = rho
        self.epsilon = epsilon
        self.cache = {}
        self.history = []
        
    def reset(self, param_size: int):
        """Reset the optimizer state."""
        self.cache = {}
        self.history = []
        
    def step(self, params: np.ndarray, objective_func) -> np.ndarray:
        """Perform one optimization step using numerical gradients."""
        # Improved finite difference gradient estimation
        eps = 1e-4  # More stable step size
        gradients = np.zeros_like(params)
        f0 = objective_func(params)
        
        for i in range(len(params)):
            # Adaptive step size based on parameter value
            adaptive_eps = max(eps, abs(params[i]) * eps)
            params_plus = params.copy()
            
            # Ensure bounds are respected when computing gradient
            if params_plus[i] + adaptive_eps > 1.0:
                params_plus[i] = max(0.0, params[i] - adaptive_eps)
                f_plus = objective_func(params_plus)
                gradients[i] = (f0 - f_plus) / adaptive_eps  # Backward difference
            else:
                params_plus[i] += adaptive_eps
                f_plus = objective_func(params_plus)
                gradients[i] = (f_plus - f0) / adaptive_eps  # Forward difference
        
        # Update using RMSprop
        updated = self.update(params, gradients)
        
        # Store in history
        self.history.append({'objective': f0, 'params': params.copy(), 'gradients': gradients.copy()})
        
        return updated
        
    def update(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        if 'cache' not in self.cache:
            self.cache['cache'] = np.zeros_like(params)
        cache = self.cache['cache']
        
        # Check for invalid gradients
        if np.any(np.isnan(gradients)) or np.any(np.isinf(gradients)):
            print("Warning: Invalid gradients detected, skipping update")
            return params
        
        cache = self.rho * cache + (1 - self.rho) * (gradients ** 2)
        
        # Avoid division by very small numbers
        denominator = np.sqrt(cache) + self.epsilon
        update_step = self.lr * gradients / denominator
        
        # Clip update step to prevent large jumps
        max_step = 0.1  # Maximum parameter change per step
        update_step = np.clip(update_step, -max_step, max_step)
        
        updated = params - update_step
        self.cache['cache'] = cache
        return updated

class NativeElasticNetKLOptimizer:
    def __init__(self, symbol1: str, symbol2: str, l1_ratio: float = 0.7, alpha: float = 0.02, kl_weight: float = 0.15, optimization_mode: str = 'hybrid'):
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.kl_weight = kl_weight
        self.optimization_mode = optimization_mode
        self.native = _native
        self.rmsprop = NativeRMSprop(lr=0.01)
        self.param_bounds = {'lookback': (5, 60), 'z_entry': (0.5, 4.0), 'z_exit': (0.1, 2.0), 'position_size': (1000, 50000), 'transaction_cost': (0.0001, 0.005), 'profit_target': (1.5, 5.0), 'stop_loss': (0.5, 2.0)}
        self.best_params = None
        self.history = []
        # Better initial parameters: mid-range values for most params
        self._initial = np.array([0.3, 0.6, 0.4, 0.2, 0.2, 0.4, 0.3])  # More conservative starting point
    def _denormalize(self, norm):
        names = list(self.param_bounds.keys())
        params = np.zeros(len(names))
        for i, n in enumerate(names):
            lo, hi = self.param_bounds[n]
            params[i] = norm[i] * (hi - lo) + lo
        return params
    def _extract_arrays(self, prices):
        if hasattr(prices, 'columns'):
            p1, p2 = prices['price1'].values.astype(float), prices['price2'].values.astype(float)
        elif isinstance(prices, (tuple, list)) and len(prices) == 2:
            p1, p2 = np.asarray(prices[0], dtype=float), np.asarray(prices[1], dtype=float)
        else:
            raise ValueError("Unsupported price format")
        
        # Comprehensive price data validation
        self._validate_prices(p1, p2)
        return p1, p2
    
    def _validate_prices(self, prices1, prices2):
        """Validate price data for trading analysis."""
        if len(prices1) != len(prices2):
            raise ValueError(f"Price arrays must have same length: {len(prices1)} vs {len(prices2)}")
        if len(prices1) < 50:
            raise ValueError(f"Insufficient data points: {len(prices1)}. Need at least 50 points.")
        if np.any(np.isnan(prices1)) or np.any(np.isnan(prices2)):
            raise ValueError("Price data contains NaN values")
        if np.any(np.isinf(prices1)) or np.any(np.isinf(prices2)):
            raise ValueError("Price data contains infinite values")
        if np.any(prices1 <= 0) or np.any(prices2 <= 0):
            raise ValueError("Price data contains non-positive values")
        if np.std(prices1) == 0 or np.std(prices2) == 0:
            raise ValueError("Price data has zero variance (constant prices)")
    
    def _validate_parameters(self, params):
        """Validate parameter consistency."""
        if params['z_exit'] >= params['z_entry']:
            raise ValueError(f"z_exit ({params['z_exit']:.3f}) must be less than z_entry ({params['z_entry']:.3f})")
        if params['profit_target'] <= params['stop_loss']:
            raise ValueError(f"profit_target ({params['profit_target']:.3f}) must be greater than stop_loss ({params['stop_loss']:.3f})")
        if params['lookback'] < 5:
            raise ValueError(f"lookback ({params['lookback']}) must be at least 5")
        if params['transaction_cost'] < 0:
            raise ValueError(f"transaction_cost ({params['transaction_cost']:.6f}) cannot be negative")
    def _cv_objective(self, prices1, prices2, norm_params, n_splits):
        denorm = self._denormalize(norm_params)
        score = self.native.parallel_cross_validation(prices1, prices2, denorm, n_folds=n_splits, l1_ratio=self.l1_ratio, alpha=self.alpha, kl_weight=self.kl_weight)
        return -float(score)
    def optimize(self, prices, n_splits: int = 3, max_iterations: int = 25):
        p1, p2 = self._extract_arrays(prices)
        current = self._initial.copy()
        best = current.copy()
        best_obj = float('inf')
        self.rmsprop.reset(len(current))
        self.history = []
        def objective_func(nparams):
            return self._cv_objective(p1, p2, nparams, n_splits)
        for it in range(max_iterations):
            new_params = self.rmsprop.step(current, objective_func)
            new_params = np.clip(new_params, 0.0, 1.0)
            cur_obj = objective_func(new_params)
            if cur_obj < best_obj:
                best_obj = cur_obj
                best = new_params.copy()
            current = new_params
            self.history.append({'iter': it + 1, 'objective': cur_obj})
            
            # Improved convergence check
            if len(self.rmsprop.history) > 15:  # Require more history
                recent = [h['objective'] for h in self.rmsprop.history[-10:]]
                recent_std = np.std(recent)
                recent_mean = np.abs(np.mean(recent))
                
                # Use relative convergence criterion
                rel_std = recent_std / (recent_mean + 1e-8)  # Avoid division by zero
                if rel_std < 1e-4:  # More reasonable threshold
                    print(f"Convergence achieved at iteration {it + 1} (relative std: {rel_std:.6f})")
                    break
                    
                # Also check for gradient norm
                if len(self.rmsprop.history) >= 5:
                    recent_grads = [np.linalg.norm(h['gradients']) for h in self.rmsprop.history[-5:]]
                    if np.mean(recent_grads) < 1e-6:
                        print(f"Gradient norm convergence at iteration {it + 1}")
                        break
        final = self._denormalize(best)
        self.best_params = {'lookback': int(final[0]), 'z_entry': float(final[1]), 'z_exit': float(final[2]), 'position_size': int(final[3]), 'transaction_cost': float(final[4]), 'profit_target': float(final[5]), 'stop_loss': float(final[6])}
        
        # Validate final parameters
        self._validate_parameters(self.best_params)
        
        return self.best_params
    def backtest(self, prices, use_cache: bool = True):
        if self.best_params is None:
            raise ValueError("Must run optimize() before backtest()")
        
        p1, p2 = self._extract_arrays(prices)
        
        # Re-validate parameters before backtesting
        self._validate_parameters(self.best_params)
        
        try:
            result = self.native.vectorized_backtest(p1, p2, self.best_params, use_cache=use_cache)
            
            # Validate backtest results
            if not isinstance(result, dict):
                raise ValueError("Backtest returned invalid result format")
            
            # Check for suspicious results
            if 'total_return' in result and np.isnan(result['total_return']):
                print("Warning: Backtest returned NaN total_return")
            if 'sharpe_ratio' in result and abs(result.get('sharpe_ratio', 0)) > 10:
                print(f"Warning: Suspiciously high Sharpe ratio: {result['sharpe_ratio']:.3f}")
                
            return result
            
        except Exception as e:
            print(f"Backtest failed: {str(e)}")
            return {'total_return': float('nan'), 'sharpe_ratio': float('nan'), 'max_drawdown': float('nan'), 'num_trades': 0, 'error': str(e)}

class NativeElasticNetKLPortfolioOptimizer:
    def __init__(self, l1_ratio: float = 0.7, alpha: float = 0.02, kl_weight: float = 0.15):
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.kl_weight = kl_weight
        self.results = {}
    def optimize_all(self, pairs_data, n_splits: int = 3, max_iterations: int = 25):
        import time
        failed_pairs = []
        
        for pair_name, prices in pairs_data.items():
            try:
                sym1, sym2 = pair_name.split('-')
                opt = NativeElasticNetKLOptimizer(sym1, sym2, self.l1_ratio, self.alpha, self.kl_weight)
                t0 = time.time()
                
                print(f"Optimizing {pair_name}...")
                best = opt.optimize(prices, n_splits=n_splits, max_iterations=max_iterations)
                optimization_time = time.time() - t0
                
                bt = opt.backtest(prices)
                self.results[pair_name] = {'best_params': best, 'backtest': bt, 'optimizer_history': opt.history, 'optimization_time': optimization_time}
                
                print(f"✓ {pair_name} completed in {optimization_time:.2f}s")
                
            except Exception as e:
                print(f"✗ Failed to optimize {pair_name}: {str(e)}")
                failed_pairs.append((pair_name, str(e)))
                self.results[pair_name] = {
                    'best_params': None, 
                    'backtest': {'total_return': float('nan'), 'sharpe_ratio': float('nan'), 'max_drawdown': float('nan')}, 
                    'optimizer_history': [], 
                    'optimization_time': 0,
                    'error': str(e)
                }
        
        if failed_pairs:
            print(f"\nWarning: {len(failed_pairs)} pairs failed to optimize:")
            for pair, error in failed_pairs:
                print(f"  - {pair}: {error}")
                
        return self.results
def create_native_elasticnet_optimizer(symbol1: str, symbol2: str, l1_ratio: float = 0.7, alpha: float = 0.02, kl_weight: float = 0.15, optimization_mode: str = 'hybrid'):
    return NativeElasticNetKLOptimizer(symbol1, symbol2, l1_ratio, alpha, kl_weight, optimization_mode)
def generate_native_performance_report(results, output_path: str = 'performance_report.json'):
    import json
    import numpy as np
    
    # Custom JSON encoder to handle NaN values
    class NaNEncoder(json.JSONEncoder):
        def encode(self, obj):
            if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                return 'null'
            return super().encode(obj)
    
    summary = {}
    total_pairs = len(results)
    successful_pairs = 0
    
    for pair, result in results.items():
        if 'backtest' in result and result['backtest'] is not None:
            bt = result['backtest']
            # Handle potential missing keys with proper defaults
            total_return = bt.get('total_return', float('nan'))
            sharpe_ratio = bt.get('sharpe_ratio', float('nan'))
            max_drawdown = bt.get('max_drawdown', float('nan'))
            
            summary[pair] = {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio, 
                'max_drawdown': max_drawdown,
                'valid': not (np.isnan(total_return) and np.isnan(sharpe_ratio))
            }
            
            if not np.isnan(total_return):
                successful_pairs += 1
        else:
            summary[pair] = {
                'total_return': float('nan'),
                'sharpe_ratio': float('nan'),
                'max_drawdown': float('nan'),
                'valid': False
            }
    
    report = {
        'summary': summary,
        'statistics': {
            'total_pairs': total_pairs,
            'successful_pairs': successful_pairs,
            'success_rate': successful_pairs / total_pairs if total_pairs > 0 else 0.0
        },
        'detailed_results': results
    }
    
    try:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, cls=NaNEncoder)
        print(f"Performance report saved to {output_path}")
    except Exception as e:
        print(f"Failed to save report: {str(e)}")
    
    return report
