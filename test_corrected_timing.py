#!/usr/bin/env python3

import numpy as np
import time
import nanoext_runloop_corrected

def test_corrected_timing():
    print("🔬 Testing CORRECTED timing measurements with proper ARM64 calibration")
    print("=" * 80)
    
    # Configuration
    num_symbols = 1000
    window = 1024  # Power of 2
    window_mask = window - 1
    lookback = 100
    num_pairs = 100
    batch_size = 64
    test_duration = 2.0  # seconds
    
    # Initialize arrays
    price_rb = np.zeros((num_symbols, window), dtype=np.float64)
    write_idx = np.zeros(num_symbols, dtype=np.int32)
    
    # Create some trading pairs
    pair_indices = np.zeros(num_pairs * 2, dtype=np.int32)
    for i in range(num_pairs):
        pair_indices[2*i] = i % num_symbols
        pair_indices[2*i + 1] = (i + 1) % num_symbols
    
    # Initialize other arrays
    thresholds = np.full(num_pairs, 2.0, dtype=np.float64)
    zsum = np.zeros(num_pairs, dtype=np.float64)
    zsumsq = np.zeros(num_pairs, dtype=np.float64)
    csx = np.zeros(num_pairs, dtype=np.float64)
    csy = np.zeros(num_pairs, dtype=np.float64)
    csxx = np.zeros(num_pairs, dtype=np.float64)
    csyy = np.zeros(num_pairs, dtype=np.float64)
    csxy = np.zeros(num_pairs, dtype=np.float64)
    
    # Run the corrected timing test
    print(f"Running test for {test_duration} seconds...")
    print(f"Configuration: {num_symbols} symbols, {num_pairs} pairs, batch={batch_size}")
    
    wall_start = time.perf_counter()
    
    stats = nanoext_runloop_corrected.run_loop_corrected(
        test_duration,
        batch_size,
        num_symbols,
        window,
        window_mask,
        lookback,
        price_rb,
        write_idx,
        pair_indices,
        thresholds,
        zsum,
        zsumsq,
        csx,
        csy,
        csxx,
        csyy,
        csxy,
        seed=0x12345678abcdef,
        collect_histograms=True
    )
    
    wall_end = time.perf_counter()
    python_wall_duration = wall_end - wall_start
    
    print("\n🎯 VALIDATION SUMMARY:")
    print(f"   Python wall-clock: {python_wall_duration:.3f} s")
    print(f"   C++ wall-clock: {stats.wall_clock_duration_s:.3f} s")
    print(f"   Duration match: {abs(python_wall_duration - stats.wall_clock_duration_s) < 0.1}")
    
    print(f"\n   RDTSC latency: {stats.avg_latency_ns:.1f} ns/msg")
    print(f"   Wall-clock latency: {stats.wall_clock_avg_latency_ns:.1f} ns/msg")
    print(f"   Latency ratio: {stats.wall_clock_avg_latency_ns / stats.avg_latency_ns:.2f}x")
    
    # Sanity checks
    latency_reasonable = 50 <= stats.wall_clock_avg_latency_ns <= 50000  # 50ns to 50μs
    throughput_reasonable = stats.throughput_msg_sec > 10000  # At least 10K msg/sec
    
    print(f"\n✅ Validation Results:")
    print(f"   Latency reasonable (50ns-50μs): {latency_reasonable}")
    print(f"   Throughput reasonable (>10K/s): {throughput_reasonable}")
    print(f"   Overall timing FIXED: {latency_reasonable and throughput_reasonable}")
    
    return stats

if __name__ == "__main__":
    stats = test_corrected_timing()
