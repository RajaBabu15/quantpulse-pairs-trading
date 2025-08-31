#!/usr/bin/env python3
"""
Ultra-Optimized HFT Pairs Trading Engine - Production Version
Minimal overhead implementation for maximum performance
"""

import sys
import numpy as np
import time

# Import the ultra-fast C++ runloop modules
try:
    import hft_core
    import hft_core_neon
    HAS_NEON = True
except ImportError:
    try:
        import hft_core
        HAS_NEON = False
    except ImportError:
        raise ImportError("Failed to import HFT core modules")

def create_trading_pairs(num_symbols, num_pairs):
    """Create trading pairs from symbol universe"""
    pairs = []
    np.random.seed(42)
    
    for i in range(num_pairs):
        s1 = np.random.randint(0, num_symbols)
        s2 = np.random.randint(0, num_symbols)
        if s1 != s2:
            pairs.extend([s1, s2])
        else:
            pairs.extend([i % num_symbols, (i + 1) % num_symbols])
    
    return np.array(pairs, dtype=np.int32)

def run_hft_engine(duration_seconds=5.0, batch_size=128, num_symbols=1000, 
                   num_pairs=200, window=1024, lookback=100):
    """Run the HFT trading engine with minimal overhead"""
    
    # Configuration
    window_mask = window - 1
    seed = 0x123456789abcdef
    
    # Initialize data structures
    price_rb = np.zeros((num_symbols, window), dtype=np.float64)
    write_idx = np.zeros(num_symbols, dtype=np.int32)
    pair_indices = create_trading_pairs(num_symbols, num_pairs)
    zsum = np.zeros(num_pairs, dtype=np.float64)
    zsumsq = np.zeros(num_pairs, dtype=np.float64)
    
    # Choose engine
    if HAS_NEON:
        run_func = hft_core_neon.run_loop_neon
    else:
        run_func = hft_core.run_loop
    
    # Execute
    stats = run_func(
        duration_seconds,
        batch_size,
        num_symbols,
        window,
        window_mask,
        lookback,
        price_rb,
        write_idx,
        pair_indices,
        zsum,
        zsumsq,
        seed=seed
    )
    
    return stats

if __name__ == "__main__":
    stats = run_hft_engine()
    print(f"Messages: {stats.total_messages}, Latency: {stats.wall_clock_avg_latency_ns:.1f}ns, Throughput: {stats.throughput_msg_sec/1e6:.1f}M/s")
