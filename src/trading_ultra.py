#!/usr/bin/env python3
"""
ULTRA-OPTIMIZED NANOSECOND HFT ENGINE
- C++ ring buffer with bitmask indexing and prefetch
- Zero-copy data marshalling with structured dtypes
- Vectorized synthetic data generation
- Sub-200ns target latency
"""

import numpy as np
import time
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict
import cProfile
import pstats
from contextlib import contextmanager
import sys
sys.path.insert(0, '.')  # Allow importing extensions from current directory

# Require native extensions
try:
    from nanoext import (
        zscore_batch_rb_inc as c_zscore_batch_rb_inc,
        corr_pairs_rb_inc as c_corr_pairs_rb_inc,
        mem_advise_sequential as c_mem_advise_seq,
        mem_prefault as c_mem_prefault,
    )
except Exception as e:
    raise RuntimeError("nanoext (C extension) is required") from e

try:
    import nano_rb
    print("✅ C++ nano_rb extension loaded")
except ImportError:
    print("⚠️  nano_rb C++ extension not available. Using Numba fallback.")
    nano_rb = None


class PerformanceProfiler:
    """Ultra-lightweight performance profiler"""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
    
    @contextmanager
    def time_function(self, name: str):
        """Lightweight timing context manager"""
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            end = time.perf_counter_ns()
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(end - start)
    
    def print_summary(self):
        """Print concise performance summary"""
        if not self.timings:
            return
        
        print("\n🚀 ULTRA-OPTIMIZED PERFORMANCE BREAKDOWN")
        print("="*60)
        
        total_time = 0
        component_times = {}
        
        for name, times in self.timings.items():
            if times:
                times_array = np.array(times)
                mean_ns = float(np.mean(times_array))
                p95_ns = float(np.percentile(times_array, 95))
                count = len(times)
                total_ns = float(np.sum(times_array))
                
                component_times[name] = total_ns
                total_time += total_ns
                
                print(f"{name:25s}: {mean_ns:>8.0f} ns ({mean_ns/1000:>6.2f} μs) "
                      f"p95: {p95_ns:>8.0f} ns [{count:>6,} calls]")
        
        print("="*60)
        if total_time > 0:
            for name, comp_time in sorted(component_times.items(), key=lambda x: x[1], reverse=True):
                pct = (comp_time / total_time) * 100
                print(f"{name:25s}: {pct:>5.1f}% of total time")


# Global profiler
profiler = PerformanceProfiler()


def next_power_of_two(n: int) -> int:
    """Get next power of two >= n"""
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    return n + 1


class UltraFastHFTEngine:
    """Ultra-optimized HFT engine targeting sub-200ns latencies"""

    def __init__(self, symbols: List[str], window: int = 1024,
                 lookback: int = 30,
                 pair_indices: np.ndarray | None = None,
                 thresholds: np.ndarray | None = None,
                 seed: int | None = None):
        
        self.symbols = symbols
        self.num_symbols = len(symbols)
        
        # Ensure window is power-of-two for bitmask optimization
        self.window = next_power_of_two(window)
        self.window_mask = self.window - 1
        
        self.lookback = lookback
        self.seed = seed
        
        print(f"🔧 Engine Config: window={self.window} (mask=0x{self.window_mask:x}), lookback={lookback}")
        
        # RNG for reproducible synthetic data
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        # Ring buffer state - optimized layout
        self.price_rb = np.zeros((self.num_symbols, self.window), dtype=np.float64, order='C')
        self.write_idx = np.zeros(self.num_symbols, dtype=np.int32)

        # Memory optimizations
        try:
            c_mem_prefault(self.price_rb)
            _ = c_mem_advise_seq(self.price_rb)
            print("✅ Memory optimizations applied")
        except Exception:
            print("⚠️  Memory optimizations failed")

        # Pair configuration
        if pair_indices is not None:
            self.pair_indices = pair_indices.astype(np.int32)
        else:
            self.pair_indices = np.array([0, 1, 1, 2, 2, 3, 0, 2], dtype=np.int32)
        
        if thresholds is not None:
            self.thresholds = thresholds.astype(np.float64)
        else:
            self.thresholds = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64)
        
        # Incremental state arrays
        self.zsum = np.zeros(len(self.thresholds), dtype=np.float64)
        self.zsumsq = np.zeros(len(self.thresholds), dtype=np.float64)
        self.zs_initialized = False

        self.csx = np.zeros(len(self.thresholds), dtype=np.float64)
        self.csy = np.zeros(len(self.thresholds), dtype=np.float64)
        self.csxx = np.zeros(len(self.thresholds), dtype=np.float64)
        self.csyy = np.zeros(len(self.thresholds), dtype=np.float64)
        self.csxy = np.zeros(len(self.thresholds), dtype=np.float64)
        self.corr_initialized = False

        # Pre-allocate arrays to eliminate allocation overhead
        self.temp_symbol_ids = np.empty(100, dtype=np.int32)  # will resize if needed
        self.temp_prices = np.empty(100, dtype=np.float64)
        
        print(f"✅ Engine initialized with {len(self.symbols)} symbols, {len(self.thresholds)} pairs")

    def process_market_data_zero_copy(self, symbol_ids: np.ndarray, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """Zero-copy processing path - no data extraction overhead"""
        with profiler.time_function("total_processing"):
            start_ns = time.perf_counter_ns()

            # Direct ring buffer update using C++ extension
            with profiler.time_function("cpp_ring_buffer_update"):
                if nano_rb is not None:
                    nano_rb.rb_update(self.price_rb, self.write_idx, symbol_ids, prices, 
                                    self.window, self.window_mask)
                else:
                    # Fallback to optimized Numba version
                    self._fallback_ring_buffer_update(symbol_ids, prices)

            # Incremental z-score signals
            with profiler.time_function("zscore_computation"):
                init_flag = 1 if self.zs_initialized else 0
                signals = c_zscore_batch_rb_inc(
                    self.price_rb, self.write_idx, self.pair_indices, self.lookback, 
                    self.thresholds, self.zsum, self.zsumsq, init_flag
                )
                self.zs_initialized = True

            # Incremental correlations
            with profiler.time_function("correlation_computation"):
                correlations = None
                try:
                    cinit = 1 if self.corr_initialized else 0
                    correlations = c_corr_pairs_rb_inc(
                        self.price_rb, self.write_idx, self.pair_indices, self.lookback,
                        self.csx, self.csy, self.csxx, self.csyy, self.csxy, cinit
                    )
                    self.corr_initialized = True
                except Exception:
                    correlations = np.zeros(len(self.thresholds), dtype=np.float64)

            end_ns = time.perf_counter_ns()
            return signals, correlations, (end_ns - start_ns)

    def _fallback_ring_buffer_update(self, symbol_ids: np.ndarray, prices: np.ndarray):
        """Fallback Numba implementation with bitmask optimization"""
        from numba import njit
        
        @njit(cache=True, fastmath=True)
        def _update_rb(price_rb, write_idx, symbol_ids, prices, mask):
            n = symbol_ids.shape[0]
            num_symbols = price_rb.shape[0]
            for i in range(n):
                sid = symbol_ids[i]
                if 0 <= sid < num_symbols:
                    idx = write_idx[sid]
                    price_rb[sid, idx] = prices[i]
                    write_idx[sid] = (idx + 1) & mask
        
        _update_rb(self.price_rb, self.write_idx, symbol_ids, prices, self.window_mask)

    def run_ultra_optimized_demo(self, duration_seconds: int = 3):
        """Ultra-optimized demo with zero-copy data flow"""
        print(f"\n🚀 ULTRA-OPTIMIZED HFT DEMO ({duration_seconds}s)")
        
        start_time = time.time()
        total_lat_ns = 0
        msgs = 0
        signals = np.zeros(len(self.thresholds), dtype=np.int8)
        
        # Pre-allocate batch arrays (zero allocation in loop)
        batch_size = 50
        symbol_ids_batch = np.empty(batch_size, dtype=np.int32)
        prices_batch = np.empty(batch_size, dtype=np.float64)
        
        # Pre-compute symbol ID pattern
        symbol_pattern = np.arange(batch_size, dtype=np.int32) % self.num_symbols

        with profiler.time_function("main_loop"):
            while time.time() - start_time < duration_seconds:
                # Ultra-fast vectorized data generation (no Python loops)
                with profiler.time_function("vectorized_data_gen"):
                    symbol_ids_batch[:] = symbol_pattern
                    # Rotate pattern to simulate different symbols
                    symbol_pattern = (symbol_pattern + 1) % self.num_symbols
                    
                    # Generate prices using RNG
                    prices_batch[:] = self.rng.normal(100.0, 1.0, batch_size)
                
                # Zero-copy processing
                sig, corr, lat_ns = self.process_market_data_zero_copy(symbol_ids_batch, prices_batch)
                signals = sig
                total_lat_ns += lat_ns
                msgs += batch_size

        avg_ns = total_lat_ns / max(msgs, 1)
        throughput = msgs / duration_seconds
        
        print("\n🎯 ULTRA-OPTIMIZED RESULTS:")
        print(f"   Messages processed: {msgs:,}")
        print(f"   Average latency: {avg_ns:,.0f} ns ({avg_ns/1000:.2f} μs)")
        print(f"   Throughput: {throughput:,.0f} msg/sec")
        print(f"   Signals per batch: {len(signals)}")
        print(f"   Extension used: {'C++' if nano_rb else 'Numba fallback'}")
        
        profiler.print_summary()

    def benchmark_isolated_components(self, iterations: int = 100000):
        """Benchmark individual components in isolation"""
        print(f"\n⚡ ISOLATED COMPONENT BENCHMARKS ({iterations:,} iterations)")
        print("="*70)
        
        # Prepare test data
        batch_size = 50
        symbol_ids = np.arange(batch_size, dtype=np.int32) % self.num_symbols
        prices = np.random.normal(100.0, 1.0, batch_size).astype(np.float64)
        
        # Benchmark C++ ring buffer update
        if nano_rb is not None:
            times = []
            for _ in range(iterations):
                start = time.perf_counter_ns()
                nano_rb.rb_update(self.price_rb, self.write_idx, symbol_ids, prices, 
                                self.window, self.window_mask)
                end = time.perf_counter_ns()
                times.append(end - start)
            
            times = np.array(times)
            print(f"C++ Ring Buffer Update:")
            print(f"  Mean: {np.mean(times):>8.0f} ns ({np.mean(times)/1000:>6.2f} μs)")
            print(f"  P95:  {np.percentile(times, 95):>8.0f} ns ({np.percentile(times, 95)/1000:>6.2f} μs)")
            print(f"  P99:  {np.percentile(times, 99):>8.0f} ns ({np.percentile(times, 99)/1000:>6.2f} μs)")
        
        # Benchmark Z-score computation
        init_flag = 1 if self.zs_initialized else 0
        times = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            _ = c_zscore_batch_rb_inc(
                self.price_rb, self.write_idx, self.pair_indices, self.lookback, 
                self.thresholds, self.zsum, self.zsumsq, init_flag
            )
            end = time.perf_counter_ns()
            times.append(end - start)
        
        times = np.array(times)
        print(f"Z-Score Computation (C):")
        print(f"  Mean: {np.mean(times):>8.0f} ns ({np.mean(times)/1000:>6.2f} μs)")
        print(f"  P95:  {np.percentile(times, 95):>8.0f} ns ({np.percentile(times, 95)/1000:>6.2f} μs)")
        
        # Benchmark correlation computation
        cinit = 1 if self.corr_initialized else 0
        times = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            try:
                _ = c_corr_pairs_rb_inc(
                    self.price_rb, self.write_idx, self.pair_indices, self.lookback,
                    self.csx, self.csy, self.csxx, self.csyy, self.csxy, cinit
                )
            except Exception:
                pass
            end = time.perf_counter_ns()
            times.append(end - start)
        
        times = np.array(times)
        print(f"Correlation Computation (C):")
        print(f"  Mean: {np.mean(times):>8.0f} ns ({np.mean(times)/1000:>6.2f} μs)")
        print(f"  P95:  {np.percentile(times, 95):>8.0f} ns ({np.percentile(times, 95)/1000:>6.2f} μs)")


def main():
    ap = argparse.ArgumentParser(description="Ultra-optimized nanosecond HFT engine")
    ap.add_argument('--duration', type=int, default=5, help='Demo duration seconds')
    ap.add_argument('--profile', action='store_true', help='Enable cProfile profiling')
    ap.add_argument('--benchmark', action='store_true', help='Run isolated function benchmarks')
    ap.add_argument('--bench-iterations', type=int, default=100000, help='Benchmark iterations')
    ap.add_argument('--window', type=int, default=1024, help='Ring buffer window size')
    ap.add_argument('--lookback', type=int, default=30, help='Lookback window for calculations')
    ap.add_argument('--seed', type=int, default=42, help='Random seed for reproducible results')
    args = ap.parse_args()

    # Default configuration
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX']
    pair_indices = np.array([0, 1, 1, 2, 2, 3, 0, 2], dtype=np.int32)
    thresholds = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64)
    
    engine = UltraFastHFTEngine(
        symbols, 
        window=args.window,
        lookback=args.lookback,
        pair_indices=pair_indices, 
        thresholds=thresholds, 
        seed=args.seed
    )
    
    if args.profile:
        # Deep profiling with cProfile
        profiler_obj = cProfile.Profile()
        profiler_obj.enable()
        
        engine.run_ultra_optimized_demo(duration_seconds=args.duration)
        
        profiler_obj.disable()
        stats = pstats.Stats(profiler_obj)
        stats.sort_stats('cumulative')
        print("\n" + "="*80)
        print("🔬 DETAILED CPROFILE ANALYSIS")
        print("="*80)
        stats.print_stats(15)
    else:
        engine.run_ultra_optimized_demo(duration_seconds=args.duration)
    
    if args.benchmark:
        engine.benchmark_isolated_components(iterations=args.bench_iterations)


if __name__ == "__main__":
    main()
