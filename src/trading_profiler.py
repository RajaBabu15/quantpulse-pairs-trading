#!/usr/bin/env python3
"""
NANOSECOND-OPTIMIZED HFT with DETAILED PERFORMANCE PROFILING
- Single optimized path only
- Ring-buffer price updates (Numba)
- Incremental z-score via C extension
- Optional incremental pairwise correlation via C extension
- Detailed timing and profiling of each component
"""

import numpy as np
from numba import njit, prange
import time
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict
import cProfile
import pstats
from functools import wraps
from contextlib import contextmanager
import sys

# Require native extension (optimized path only)
try:
    from nanoext import (
        zscore_batch_rb_inc as c_zscore_batch_rb_inc,
        corr_pairs_rb_inc as c_corr_pairs_rb_inc,
        mem_advise_sequential as c_mem_advise_seq,
        mem_prefault as c_mem_prefault,
    )
except Exception as e:
    raise RuntimeError("nanoext (C extension) is required for the streamlined optimized path") from e


class PerformanceProfiler:
    """Performance profiler for HFT components"""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self.counters: Dict[str, int] = {}
    
    @contextmanager
    def time_function(self, name: str):
        """Context manager for timing functions"""
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            end = time.perf_counter_ns()
            elapsed_ns = end - start
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(elapsed_ns)
            self.counters[name] = self.counters.get(name, 0) + 1
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics"""
        stats = {}
        for name, times in self.timings.items():
            if times:
                times_array = np.array(times)
                stats[name] = {
                    'count': len(times),
                    'total_ns': float(np.sum(times_array)),
                    'mean_ns': float(np.mean(times_array)),
                    'median_ns': float(np.median(times_array)),
                    'std_ns': float(np.std(times_array)),
                    'min_ns': float(np.min(times_array)),
                    'max_ns': float(np.max(times_array)),
                    'p95_ns': float(np.percentile(times_array, 95)),
                    'p99_ns': float(np.percentile(times_array, 99))
                }
        return stats
    
    def print_detailed_report(self):
        """Print detailed performance report"""
        stats = self.get_stats()
        if not stats:
            print("No profiling data collected")
            return
        
        print("\n" + "="*80)
        print("🔍 DETAILED PERFORMANCE ANALYSIS")
        print("="*80)
        
        # Sort by total time spent
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['total_ns'], reverse=True)
        
        total_time_ns = sum(s['total_ns'] for s in stats.values())
        
        for name, stat in sorted_stats:
            pct = (stat['total_ns'] / total_time_ns) * 100 if total_time_ns > 0 else 0
            print(f"\n📊 {name}")
            print(f"   Count:        {stat['count']:>10,}")
            print(f"   Total time:   {stat['total_ns']:>10,.0f} ns ({pct:>5.1f}%)")
            print(f"   Mean:         {stat['mean_ns']:>10,.0f} ns ({stat['mean_ns']/1000:>6.2f} μs)")
            print(f"   Median:       {stat['median_ns']:>10,.0f} ns ({stat['median_ns']/1000:>6.2f} μs)")
            print(f"   Std Dev:      {stat['std_ns']:>10,.0f} ns")
            print(f"   Min:          {stat['min_ns']:>10,.0f} ns ({stat['min_ns']/1000:>6.2f} μs)")
            print(f"   Max:          {stat['max_ns']:>10,.0f} ns ({stat['max_ns']/1000:>6.2f} μs)")
            print(f"   P95:          {stat['p95_ns']:>10,.0f} ns ({stat['p95_ns']/1000:>6.2f} μs)")
            print(f"   P99:          {stat['p99_ns']:>10,.0f} ns ({stat['p99_ns']/1000:>6.2f} μs)")
            
            # Frequency analysis
            throughput = stat['count'] / (stat['total_ns'] / 1e9) if stat['total_ns'] > 0 else 0
            print(f"   Throughput:   {throughput:>10,.0f} calls/sec")
        
        print(f"\n🎯 TOTAL PROFILED TIME: {total_time_ns:,.0f} ns ({total_time_ns/1e9:.3f} sec)")
        print("="*80)


# Global profiler instance
profiler = PerformanceProfiler()


# O(1) per update ring-buffer writes (no shifting)
@njit(cache=True, fastmath=True, parallel=True)
def fast_price_update_ringbuffer(price_rb: np.ndarray, write_idx: np.ndarray,
                                symbol_ids: np.ndarray, prices: np.ndarray):
    n = len(symbol_ids)
    width = price_rb.shape[1]
    for i in prange(n):
        sid = symbol_ids[i]
        if 0 <= sid < price_rb.shape[0]:
            idx = write_idx[sid]
            price_rb[sid, idx] = prices[i]
            idx += 1
            if idx == width:
                idx = 0
            write_idx[sid] = idx


class NanosecondHFTEngine:
    """Ultra-low latency engine with detailed performance monitoring"""

    def __init__(self, symbols: List[str], window: int = 1024,
                 lookback: int = 30,
                 pair_indices: np.ndarray | None = None,
                 thresholds: np.ndarray | None = None,
                 seed: int | None = None):
        self.symbols = symbols
        self.num_symbols = len(symbols)
        self.window = window
        self.lookback = lookback
        self.seed = seed
        # RNG for reproducible synthetic data
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        # Ring buffer state
        self.price_rb = np.zeros((self.num_symbols, self.window), dtype=np.float64)
        self.write_idx = np.zeros(self.num_symbols, dtype=np.int32)

        # Best-effort memory tuning
        try:
            c_mem_prefault(self.price_rb)
            _ = c_mem_advise_seq(self.price_rb)
        except Exception:
            pass

        # Pair configuration and incremental state (example pairs)
        if pair_indices is not None:
            self.pair_indices = pair_indices.astype(np.int32)
        else:
            self.pair_indices = np.array([0, 1, 1, 2, 2, 3, 0, 2], dtype=np.int32)
        if thresholds is not None:
            self.thresholds = thresholds.astype(np.float64)
        else:
            self.thresholds = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64)
        self.zsum = np.zeros(len(self.thresholds), dtype=np.float64)
        self.zsumsq = np.zeros(len(self.thresholds), dtype=np.float64)
        self.zs_initialized = False

        # Optional incremental correlation state for the same pairs
        self.csx = np.zeros(len(self.thresholds), dtype=np.float64)
        self.csy = np.zeros(len(self.thresholds), dtype=np.float64)
        self.csxx = np.zeros(len(self.thresholds), dtype=np.float64)
        self.csyy = np.zeros(len(self.thresholds), dtype=np.float64)
        self.csxy = np.zeros(len(self.thresholds), dtype=np.float64)
        self.corr_initialized = False

    def process_market_data(self, market_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """Process one batch of ticks using the optimized path with detailed profiling"""
        with profiler.time_function("total_processing"):
            start_ns = time.perf_counter_ns()

            # Extract fields
            with profiler.time_function("data_extraction"):
                symbol_ids = market_data[:, 1].astype(np.int32)
                prices = market_data[:, 2].astype(np.float64)

            # Update ring buffer (in place)
            with profiler.time_function("ring_buffer_update"):
                fast_price_update_ringbuffer(self.price_rb, self.write_idx, symbol_ids, prices)

            # Incremental z-score signals
            with profiler.time_function("zscore_computation"):
                init_flag = 1 if self.zs_initialized else 0
                signals = c_zscore_batch_rb_inc(
                    self.price_rb, self.write_idx, self.pair_indices, self.lookback, self.thresholds,
                    self.zsum, self.zsumsq, init_flag
                )
                self.zs_initialized = True

            # Incremental correlations for current pairs
            correlations = None
            with profiler.time_function("correlation_computation"):
                try:
                    cinit = 1 if self.corr_initialized else 0
                    correlations = c_corr_pairs_rb_inc(
                        self.price_rb, self.write_idx, self.pair_indices, self.lookback,
                        self.csx, self.csy, self.csxx, self.csyy, self.csxy, cinit
                    )
                    self.corr_initialized = True
                except Exception:
                    pass

            end_ns = time.perf_counter_ns()
            return signals, (correlations if correlations is not None else np.zeros(len(self.thresholds), dtype=np.float64)), (end_ns - start_ns)

    def run_demo_with_profiling(self, duration_seconds: int = 3):
        print(f"\n🔥 Optimized path demo with detailed profiling ({duration_seconds}s)")
        start_time = time.time()
        total_lat_ns = 0
        msgs = 0
        signals = np.zeros(len(self.thresholds), dtype=np.int8)

        # Profiling data generation
        with profiler.time_function("data_generation"):
            while time.time() - start_time < duration_seconds:
                batch_size = 50
                with profiler.time_function("synthetic_data_creation"):
                    market_data = np.array([
                        [time.time_ns(), i % self.num_symbols, 100.0 + self.rng.normal(0, 1)]
                        for i in range(batch_size)
                    ])
                
                sig, corr, lat_ns = self.process_market_data(market_data)
                signals = sig  # keep last
                total_lat_ns += lat_ns
                msgs += batch_size

        avg_ns = total_lat_ns / max(msgs, 1)
        print("📊 Basic Results:")
        print(f"   Messages processed: {msgs:,}")
        print(f"   Average latency: {avg_ns:,.0f} ns ({avg_ns/1000:.2f} μs)")
        print(f"   Signals per batch: {len(signals)}")
        try:
            print(f"   Correlations per batch: {len(corr)} (sample: {float(corr[0]):+.4f})")
        except Exception:
            pass

        # Print detailed profiling report
        profiler.print_detailed_report()

    def benchmark_specific_functions(self, iterations: int = 10000):
        """Benchmark specific functions in isolation"""
        print(f"\n🎯 ISOLATED FUNCTION BENCHMARKS ({iterations:,} iterations)")
        print("="*60)
        
        # Prepare test data
        batch_size = 50
        market_data = np.array([
            [time.time_ns(), i % self.num_symbols, 100.0 + self.rng.normal(0, 1)]
            for i in range(batch_size)
        ])
        symbol_ids = market_data[:, 1].astype(np.int32)
        prices = market_data[:, 2].astype(np.float64)
        
        # Benchmark data extraction
        times = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            _ = market_data[:, 1].astype(np.int32)
            _ = market_data[:, 2].astype(np.float64)
            end = time.perf_counter_ns()
            times.append(end - start)
        
        times = np.array(times)
        print(f"Data Extraction:")
        print(f"  Mean: {np.mean(times):,.0f} ns ({np.mean(times)/1000:.2f} μs)")
        print(f"  P95:  {np.percentile(times, 95):,.0f} ns ({np.percentile(times, 95)/1000:.2f} μs)")
        
        # Benchmark ring buffer update
        times = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            fast_price_update_ringbuffer(self.price_rb, self.write_idx, symbol_ids, prices)
            end = time.perf_counter_ns()
            times.append(end - start)
        
        times = np.array(times)
        print(f"Ring Buffer Update:")
        print(f"  Mean: {np.mean(times):,.0f} ns ({np.mean(times)/1000:.2f} μs)")
        print(f"  P95:  {np.percentile(times, 95):,.0f} ns ({np.percentile(times, 95)/1000:.2f} μs)")
        
        # Benchmark Z-score computation
        times = []
        init_flag = 1 if self.zs_initialized else 0
        for _ in range(iterations):
            start = time.perf_counter_ns()
            _ = c_zscore_batch_rb_inc(
                self.price_rb, self.write_idx, self.pair_indices, self.lookback, self.thresholds,
                self.zsum, self.zsumsq, init_flag
            )
            end = time.perf_counter_ns()
            times.append(end - start)
        
        times = np.array(times)
        print(f"Z-Score Computation (C):")
        print(f"  Mean: {np.mean(times):,.0f} ns ({np.mean(times)/1000:.2f} μs)")
        print(f"  P95:  {np.percentile(times, 95):,.0f} ns ({np.percentile(times, 95)/1000:.2f} μs)")
        
        # Benchmark correlation computation
        times = []
        cinit = 1 if self.corr_initialized else 0
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
        print(f"  Mean: {np.mean(times):,.0f} ns ({np.mean(times)/1000:.2f} μs)")
        print(f"  P95:  {np.percentile(times, 95)::.0f} ns ({np.percentile(times, 95)/1000:.2f} μs)")
        

def main():
    ap = argparse.ArgumentParser(description="Nanosecond HFT engine with detailed profiling")
    ap.add_argument('--duration', type=int, default=3, help='Demo duration seconds')
    ap.add_argument('--profile', action='store_true', help='Enable cProfile profiling')
    ap.add_argument('--benchmark', action='store_true', help='Run isolated function benchmarks')
    ap.add_argument('--bench-iterations', type=int, default=10000, help='Benchmark iterations')
    ap.add_argument('--seed', type=int, help='Random seed for reproducible synthetic data')
    args = ap.parse_args()

    # Default configuration
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX']
    pair_indices = np.array([0, 1, 1, 2, 2, 3, 0, 2], dtype=np.int32)
    thresholds = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64)
    
    engine = NanosecondHFTEngine(symbols, pair_indices=pair_indices, thresholds=thresholds, seed=args.seed)
    
    if args.profile:
        # Use cProfile for deep profiling
        profiler_obj = cProfile.Profile()
        profiler_obj.enable()
        
        engine.run_demo_with_profiling(duration_seconds=args.duration)
        
        profiler_obj.disable()
        stats = pstats.Stats(profiler_obj)
        stats.sort_stats('cumulative')
        print("\n" + "="*80)
        print("🔬 CPROFILE ANALYSIS (Top 20 functions)")
        print("="*80)
        stats.print_stats(20)
    else:
        engine.run_demo_with_profiling(duration_seconds=args.duration)
    
    if args.benchmark:
        engine.benchmark_specific_functions(iterations=args.bench_iterations)


if __name__ == "__main__":
    main()
