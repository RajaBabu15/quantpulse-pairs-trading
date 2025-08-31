#!/usr/bin/env python3
"""
NEON Performance Validation Script

This script compares the performance of the baseline HFT core against
the NEON-accelerated version to measure the SIMD acceleration factor.
"""

import sys
import numpy as np
import time
import hft_core
import hft_core_neon

def create_test_data(num_symbols, num_pairs, window, lookback):
    """Create test data structures for performance testing"""
    # Ring buffer for price history
    price_rb = np.zeros((num_symbols, window), dtype=np.float64)
    write_idx = np.zeros(num_symbols, dtype=np.int32)
    
    # Trading pairs
    pairs = []
    np.random.seed(42)
    for i in range(num_pairs):
        s1 = np.random.randint(0, num_symbols)
        s2 = np.random.randint(0, num_symbols)
        if s1 != s2:
            pairs.extend([s1, s2])
        else:
            pairs.extend([i % num_symbols, (i + 1) % num_symbols])
    
    pair_indices = np.array(pairs, dtype=np.int32)
    thresholds = np.full(num_pairs, 2.0, dtype=np.float64)
    
    # Z-score computation state
    zsum = np.zeros(num_pairs, dtype=np.float64)
    zsumsq = np.zeros(num_pairs, dtype=np.float64)
    
    # Correlation computation state
    csx = np.zeros(num_pairs, dtype=np.float64)
    csy = np.zeros(num_pairs, dtype=np.float64)
    csxx = np.zeros(num_pairs, dtype=np.float64)
    csyy = np.zeros(num_pairs, dtype=np.float64)
    csxy = np.zeros(num_pairs, dtype=np.float64)
    
    return {
        'price_rb': price_rb,
        'write_idx': write_idx,
        'pair_indices': pair_indices,
        'thresholds': thresholds,
        'zsum': zsum,
        'zsumsq': zsumsq,
        'csx': csx,
        'csy': csy,
        'csxx': csxx,
        'csyy': csyy,
        'csxy': csxy
    }

def run_performance_comparison():
    print("🔬 NEON PERFORMANCE VALIDATION")
    print("=" * 50)
    print()
    
    # Test configuration
    num_symbols = 1000
    window = 1024
    window_mask = window - 1
    lookback = 100
    num_pairs = 200
    batch_size = 128
    test_duration = 5.0  # seconds
    seed = 0x123456789abcdef
    
    print(f"📊 Test Configuration:")
    print(f"   Symbols: {num_symbols}")
    print(f"   Trading pairs: {num_pairs}")
    print(f"   Window size: {window}")
    print(f"   Batch size: {batch_size}")
    print(f"   Duration: {test_duration}s each")
    print()
    
    # Create test data
    print("🔧 Preparing test data...")
    test_data = create_test_data(num_symbols, num_pairs, window, lookback)
    print("✅ Test data ready")
    print()
    
    # Test 1: Baseline HFT Core
    print("🏃 Running BASELINE performance test...")
    
    # Fresh copies of data for baseline test
    baseline_data = {}
    for key, value in test_data.items():
        baseline_data[key] = value.copy()
    
    baseline_start = time.perf_counter()
    baseline_stats = hft_core.run_loop_corrected(
        test_duration,
        batch_size,
        num_symbols,
        window,
        window_mask,
        lookback,
        baseline_data['price_rb'],
        baseline_data['write_idx'],
        baseline_data['pair_indices'],
        baseline_data['thresholds'],
        baseline_data['zsum'],
        baseline_data['zsumsq'],
        baseline_data['csx'],
        baseline_data['csy'],
        baseline_data['csxx'],
        baseline_data['csyy'],
        baseline_data['csxy'],
        seed=seed,
        collect_histograms=True
    )
    baseline_end = time.perf_counter()
    baseline_wall_time = baseline_end - baseline_start
    
    print(f"✅ Baseline completed")
    print(f"   Messages: {baseline_stats.total_messages:,}")
    print(f"   Avg latency: {baseline_stats.wall_clock_avg_latency_ns:.1f} ns")
    print(f"   Throughput: {baseline_stats.throughput_msg_sec/1e6:.1f}M msg/sec")
    print()
    
    # Test 2: NEON-accelerated HFT Core
    print("🏃 Running NEON-ACCELERATED performance test...")
    
    # Fresh copies of data for NEON test
    neon_data = {}
    for key, value in test_data.items():
        neon_data[key] = value.copy()
    
    neon_start = time.perf_counter()
    neon_stats = hft_core_neon.run_loop_neon(
        test_duration,
        batch_size,
        num_symbols,
        window,
        window_mask,
        lookback,
        neon_data['price_rb'],
        neon_data['write_idx'],
        neon_data['pair_indices'],
        neon_data['thresholds'],
        neon_data['zsum'],
        neon_data['zsumsq'],
        neon_data['csx'],
        neon_data['csy'],
        neon_data['csxx'],
        neon_data['csyy'],
        neon_data['csxy'],
        seed=seed,
        collect_histograms=True
    )
    neon_end = time.perf_counter()
    neon_wall_time = neon_end - neon_start
    
    print(f"✅ NEON-accelerated completed")
    print(f"   Messages: {neon_stats.total_messages:,}")
    print(f"   Avg latency: {neon_stats.wall_clock_avg_latency_ns:.1f} ns") 
    print(f"   Throughput: {neon_stats.throughput_msg_sec/1e6:.1f}M msg/sec")
    print(f"   NEON operations: {neon_stats.neon_operations:,}")
    print()
    
    # Performance Comparison
    print("🎯 PERFORMANCE COMPARISON")
    print("=" * 30)
    
    latency_improvement = baseline_stats.wall_clock_avg_latency_ns / neon_stats.wall_clock_avg_latency_ns
    throughput_improvement = neon_stats.throughput_msg_sec / baseline_stats.throughput_msg_sec
    
    print(f"📈 Latency Improvement:")
    print(f"   Baseline: {baseline_stats.wall_clock_avg_latency_ns:.1f} ns")
    print(f"   NEON: {neon_stats.wall_clock_avg_latency_ns:.1f} ns")
    print(f"   Improvement: {latency_improvement:.2f}x faster")
    
    print(f"\n🚀 Throughput Improvement:")
    print(f"   Baseline: {baseline_stats.throughput_msg_sec/1e6:.1f}M msg/sec")
    print(f"   NEON: {neon_stats.throughput_msg_sec/1e6:.1f}M msg/sec")
    print(f"   Improvement: {throughput_improvement:.2f}x faster")
    
    print(f"\n⏱️  Wall Clock Comparison:")
    print(f"   Baseline duration: {baseline_wall_time:.3f}s")
    print(f"   NEON duration: {neon_wall_time:.3f}s")
    print(f"   Wall clock improvement: {baseline_wall_time/neon_wall_time:.2f}x")
    
    # NEON effectiveness analysis
    print(f"\n🔬 NEON Effectiveness Analysis:")
    if latency_improvement > 1.5:
        effectiveness = "🔥 EXCELLENT"
    elif latency_improvement > 1.2:
        effectiveness = "⚡ GOOD"
    elif latency_improvement > 1.05:
        effectiveness = "✅ MODERATE"
    else:
        effectiveness = "⚠️ MINIMAL"
    
    print(f"   NEON acceleration: {effectiveness}")
    print(f"   Expected on ARM64: 1.2-2.0x improvement")
    print(f"   Achieved: {latency_improvement:.2f}x improvement")
    
    # Message processing efficiency
    baseline_cycles_per_msg = baseline_stats.wall_clock_avg_latency_ns * 3.5  # ~3.5GHz M4
    neon_cycles_per_msg = neon_stats.wall_clock_avg_latency_ns * 3.5
    
    print(f"\n⚙️  CPU Efficiency:")
    print(f"   Baseline: ~{baseline_cycles_per_msg:.0f} cycles/msg")
    print(f"   NEON: ~{neon_cycles_per_msg:.0f} cycles/msg")
    print(f"   Cycle reduction: {baseline_cycles_per_msg - neon_cycles_per_msg:.0f} cycles/msg")
    
    # Final verdict
    print(f"\n🏆 FINAL VERDICT:")
    if latency_improvement >= 1.3 and neon_stats.wall_clock_avg_latency_ns < 15:
        verdict = "🔥 NEON acceleration is HIGHLY EFFECTIVE"
    elif latency_improvement >= 1.15 and neon_stats.wall_clock_avg_latency_ns < 20:
        verdict = "⚡ NEON acceleration is EFFECTIVE"  
    elif latency_improvement >= 1.05:
        verdict = "✅ NEON acceleration provides MODEST gains"
    else:
        verdict = "⚠️ NEON acceleration shows MINIMAL improvement"
    
    print(f"   {verdict}")
    print(f"   Recommend: {'NEON version' if latency_improvement >= 1.1 else 'Either version'} for production")
    
    return {
        'baseline_stats': baseline_stats,
        'neon_stats': neon_stats,
        'latency_improvement': latency_improvement,
        'throughput_improvement': throughput_improvement
    }

if __name__ == "__main__":
    try:
        results = run_performance_comparison()
        print(f"\n✅ NEON validation completed successfully!")
        print(f"🔥 NEON achieved {results['latency_improvement']:.2f}x latency improvement")
        print(f"🚀 NEON achieved {results['throughput_improvement']:.2f}x throughput improvement")
    except Exception as e:
        print(f"❌ Error during NEON validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
