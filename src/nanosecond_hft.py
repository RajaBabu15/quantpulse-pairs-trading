#!/usr/bin/env python3
"""
🚀 NANOSECOND-LEVEL HFT TRADING ENGINE (streamlined)
Keeps only three paths:
- Online main (demo loop)
- Optimized benchmark path (critical path metrics)
- Low latency signal path (batch z-score)
"""

import os
import time
from typing import List, Tuple, Optional

import numpy as np
from numba import njit, prange
import multiprocessing as mp
import ctypes

# ========================================
# Minimal data structures needed by benchmarks
# ========================================

class SymbolMapper:
    """Pre-computed symbol to integer mapping for O(1) array access"""
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.symbol_to_id = {symbol: idx for idx, symbol in enumerate(symbols)}
        self.id_array = np.arange(len(symbols), dtype=np.int32)

    def get_id_fast(self, symbol: str) -> int:
        return self.symbol_to_id.get(symbol, -1)


class LockFreeQueue:
    """Single-producer, single-consumer lock-free queue (simple variant)"""
    def __init__(self, capacity: int = 8192):
        self.capacity = 1 << (capacity - 1).bit_length()
        self.mask = self.capacity - 1
        self.buffer = mp.Array(ctypes.c_double, self.capacity)
        self.head = mp.Value(ctypes.c_uint64, 0)
        self.tail = mp.Value(ctypes.c_uint64, 0)

    def enqueue(self, value: float) -> bool:
        t = self.tail.value
        nt = (t + 1) & self.mask
        if nt == self.head.value:
            return False
        self.buffer[t] = value
        self.tail.value = nt
        return True

    def dequeue(self) -> Optional[float]:
        h = self.head.value
        if h == self.tail.value:
            return None
        v = self.buffer[h]
        self.head.value = (h + 1) & self.mask
        return v


# ========================================
# Low latency signal kernel (Numba)
# ========================================

@njit(cache=True, fastmath=True)
def fast_signal_batch_processing(price_matrix: np.ndarray,
                                symbol_pairs: np.ndarray,
                                lookback: int,
                                thresholds: np.ndarray) -> np.ndarray:
    n_pairs = symbol_pairs.shape[0]
    signals = np.zeros(n_pairs, dtype=np.int8)
    for pair_idx in range(n_pairs):
        i = symbol_pairs[pair_idx, 0]
        j = symbol_pairs[pair_idx, 1]
        thr = thresholds[pair_idx]
        p1 = price_matrix[i, :]
        p2 = price_matrix[j, :]
        n = len(p1)
        if n >= lookback:
            s = p1[-lookback:] - p2[-lookback:]
            m = np.mean(s)
            sd = np.std(s)
            if sd > 0.0:
                z = (p1[-1] - p2[-1] - m) / sd
                if z > thr:
                    signals[pair_idx] = 1
                elif z < -thr:
                    signals[pair_idx] = -1
    return signals


# ========================================
# Streamlined engine exposing three paths
# ========================================

class NanosecondHFTEngine:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.num_symbols = len(symbols)
        self.symbol_mapper = SymbolMapper(symbols)
        self.price_matrix = np.zeros((self.num_symbols, 1000), dtype=np.float64)

    def process_market_data_core(self, market_data: np.ndarray) -> Tuple[np.ndarray, int]:
        start = time.perf_counter_ns()
        ts = market_data[:, 0].astype(np.int64)
        sids = market_data[:, 1].astype(np.int32)
        prices = market_data[:, 2].astype(np.float64)
        for k, sid in enumerate(sids):
            if 0 <= sid < self.num_symbols:
                self.price_matrix[sid, :-1] = self.price_matrix[sid, 1:]
                self.price_matrix[sid, -1] = prices[k]
        pairs = np.array([[i, i+1] for i in range(min(self.num_symbols-1, 3))], dtype=np.int32)
        thresholds = np.full(len(pairs), 2.0)
        signals = fast_signal_batch_processing(self.price_matrix, pairs, 30, thresholds)
        lat = time.perf_counter_ns() - start
        return signals, lat

    def benchmark_critical_path(self, iterations: int = 100000) -> dict:
        print(f"\n🏁 CRITICAL PATH BENCHMARKS ({iterations:,} iterations)")
        print("=" * 60)
        test_symbols = self.symbols[:10]
        # 1) Dict lookup vs array indexing
        n_ops = iterations
        # dict lookup
        start = time.perf_counter_ns()
        for k in range(n_ops):
            _ = self.symbol_mapper.get_id_fast(test_symbols[k % len(test_symbols)])
        dict_time = time.perf_counter_ns() - start
        # array access
        arr = self.symbol_mapper.id_array
        start = time.perf_counter_ns()
        for k in range(n_ops):
            _ = arr[k % len(arr)]
        array_time = time.perf_counter_ns() - start
        # 2) Signal processing batch
        test_matrix = np.random.normal(100, 2, (len(test_symbols), 1000)).astype(np.float64)
        dummy_pairs = np.array([[i, i+1] for i in range(len(test_symbols)-1)], dtype=np.int32)
        dummy_thresholds = np.full(len(dummy_pairs), 2.0)
        runs = max(1, iterations // 10)
        start = time.perf_counter_ns()
        for _ in range(runs):
            _ = fast_signal_batch_processing(test_matrix, dummy_pairs, 30, dummy_thresholds)
        sig_time = time.perf_counter_ns() - start
        avg_sig = sig_time / runs
        # 3) Memory access
        start = time.perf_counter_ns()
        tmp = 0.0
        for k in range(n_ops):
            tmp += test_matrix[k % test_matrix.shape[0], k % test_matrix.shape[1]]
        mem_time = time.perf_counter_ns() - start
        # 4) Queue ops
        q = LockFreeQueue(1024)
        start = time.perf_counter_ns()
        for _ in range(n_ops):
            q.enqueue(1.0)
            _ = q.dequeue()
        q_time = time.perf_counter_ns() - start
        per_lookup = dict_time / n_ops
        per_array = array_time / n_ops
        per_mem = mem_time / n_ops
        per_queue = (q_time / n_ops) / 2.0
        total_critical = per_array + per_mem + per_queue
        max_tput = 1_000_000_000 / total_critical if total_critical > 0 else float('inf')
        print(f"Dictionary lookup: {per_lookup:.1f} ns/op")
        print(f"Array access:      {per_array:.1f} ns/op")
        print(f"Speedup (array/dict): {per_lookup/per_array:.2f}x")
        print(f"Signal processing: {avg_sig/1000:.2f} μs/batch")
        print(f"Memory access:     {per_mem:.1f} ns/op")
        print(f"Queue operations:  {per_queue:.1f} ns/op")
        print(f"\nCritical path:     {total_critical:.1f} ns")
        print(f"Max throughput:    {max_tput:,.0f} ops/sec")
        return {
            'dict_lookup_ns': per_lookup,
            'array_access_ns': per_array,
            'signal_processing_ns': avg_sig,
            'memory_access_ns': per_mem,
            'queue_operation_ns': per_queue,
            'critical_path_ns': total_critical,
            'max_throughput_ops': max_tput,
            'iterations': iterations,
        }

    def run_nanosecond_demo(self, duration_seconds: int = 3):
        print(f"\n🔥 RUNNING ONLINE DEMO ({duration_seconds}s)")
        start_time = time.time()
        total_lat = 0
        msgs = 0
        last_signals = None
        while time.time() - start_time < duration_seconds:
            batch = 50
            md = np.array([
                [time.time_ns(), i % self.num_symbols, 100.0 + np.random.normal(0, 1)]
                for i in range(batch)
            ], dtype=np.float64)
            sig, lat = self.process_market_data_core(md)
            last_signals = sig
            total_lat += lat
            msgs += batch
        avg_ns = total_lat / max(1, msgs)
        print("\n📊 ONLINE DEMO RESULTS:")
        print(f"   Messages processed: {msgs:,}")
        print(f"   Average latency: {avg_ns:,.0f} ns ({avg_ns/1000:.2f} μs)")
        print(f"   Signals per batch: {0 if last_signals is None else len(last_signals)}")


def main():
    print("🚀 NANOSECOND-LEVEL HFT OPTIMIZATIONS")
    print("⚡ Optimized critical path, streamlined online demo")
    print("=" * 60)
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX']
    engine = NanosecondHFTEngine(symbols)
    results = engine.benchmark_critical_path(iterations=50000)
    engine.run_nanosecond_demo(duration_seconds=3)
    print("\n🎯 SUMMARY:")
    print(f"   Critical path: {results['critical_path_ns']:.1f} ns")
    print(f"   Max throughput: {results['max_throughput_ops']:,.0f} ops/sec")


if __name__ == "__main__":
    main()
