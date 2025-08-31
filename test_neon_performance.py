#!/usr/bin/env python3
"""
NEON vs Scalar Performance Comparison
Measures performance improvement from NEON SIMD acceleration
"""

import numpy as np
import time
import nanoext_runloop_corrected  # Scalar version
import nanoext_runloop_neon      # NEON version

def setup_test_data(num_symbols=1000, num_pairs=200, batch_size=128):
    """Create identical test data for both engines"""
    window = 1024
    window_mask = window - 1
    lookback = 100
    
    # Initialize arrays
    price_rb = np.zeros((num_symbols, window), dtype=np.float64)
    write_idx = np.zeros(num_symbols, dtype=np.int32)
    
    # Create trading pairs (ensure even number for NEON vectorization)
    pair_indices = np.zeros(num_pairs * 2, dtype=np.int32)
    np.random.seed(42)
    for i in range(num_pairs):
        s1 = np.random.randint(0, num_symbols)
        s2 = np.random.randint(0, num_symbols)
        if s1 != s2:
            pair_indices[2*i] = s1
            pair_indices[2*i + 1] = s2
        else:
            pair_indices[2*i] = i % num_symbols
            pair_indices[2*i + 1] = (i + 1) % num_symbols
    
    # Initialize state arrays
    thresholds = np.full(num_pairs, 2.0, dtype=np.float64)
    zsum = np.zeros(num_pairs, dtype=np.float64)
    zsumsq = np.zeros(num_pairs, dtype=np.float64)
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
        'csxy': csxy,
        'window': window,
        'window_mask': window_mask,
        'lookback': lookback
    }

def run_performance_comparison():
    print("🔥 NEON vs SCALAR PERFORMANCE COMPARISON")
    print("=" * 60)
    print("🎯 Testing NEON SIMD acceleration on Apple M4")
    print()
    
    # Test configuration
    num_symbols = 1000
    num_pairs = 200  # Even number for optimal NEON vectorization
    batch_size = 128
    test_duration = 3.0  # seconds
    
    print(f"📊 Test Configuration:")
    print(f"   Symbols: {num_symbols}")
    print(f"   Trading pairs: {num_pairs} (optimized for NEON 2x vectorization)")
    print(f"   Batch size: {batch_size}")
    print(f"   Duration: {test_duration}s each")
    print()
    
    # Prepare identical test data
    test_data = setup_test_data(num_symbols, num_pairs, batch_size)
    
    # Test 1: Scalar (baseline) performance
    print("🔬 Testing SCALAR baseline performance...")
    
    scalar_data = {k: v.copy() if hasattr(v, 'copy') else v for k, v in test_data.items()}
    
    scalar_stats = nanoext_runloop_corrected.run_loop_corrected(
        test_duration,
        batch_size,
        num_symbols,
        scalar_data['window'],
        scalar_data['window_mask'],
        scalar_data['lookback'],
        scalar_data['price_rb'],
        scalar_data['write_idx'],
        scalar_data['pair_indices'],
        scalar_data['thresholds'],
        scalar_data['zsum'],
        scalar_data['zsumsq'],
        scalar_data['csx'],
        scalar_data['csy'],
        scalar_data['csxx'],
        scalar_data['csyy'],
        scalar_data['csxy'],
        seed=0x123456789abcdef,
        collect_histograms=False  # Skip for cleaner output
    )
    
    print("\n" + "="*50)
    
    # Test 2: NEON-accelerated performance
    print("🚀 Testing NEON-accelerated performance...")
    
    neon_data = {k: v.copy() if hasattr(v, 'copy') else v for k, v in test_data.items()}
    
    neon_stats = nanoext_runloop_neon.run_loop_neon(
        test_duration,
        batch_size,
        num_symbols,
        neon_data['window'],
        neon_data['window_mask'],
        neon_data['lookback'],
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
        seed=0x123456789abcdef,
        collect_histograms=False  # Skip for cleaner output
    )
    
    # Performance comparison analysis
    print("\n" + "="*50)
    print("🎯 PERFORMANCE COMPARISON RESULTS")
    print("=" * 50)
    
    # Latency comparison
    scalar_latency = scalar_stats.wall_clock_avg_latency_ns
    neon_latency = neon_stats.wall_clock_avg_latency_ns
    latency_improvement = scalar_latency / neon_latency
    
    print(f"⏱️  Latency Results:")
    print(f"   Scalar (baseline): {scalar_latency:>8.1f} ns")
    print(f"   NEON (optimized):  {neon_latency:>8.1f} ns")
    print(f"   Improvement:       {latency_improvement:>8.2f}x")
    print(f"   Reduction:         {((scalar_latency - neon_latency) / scalar_latency * 100):>7.1f}%")
    
    # Throughput comparison
    scalar_throughput = scalar_stats.throughput_msg_sec / 1e6
    neon_throughput = neon_stats.throughput_msg_sec / 1e6
    throughput_improvement = neon_throughput / scalar_throughput
    
    print(f"\n🚀 Throughput Results:")
    print(f"   Scalar (baseline): {scalar_throughput:>8.1f}M msg/sec")
    print(f"   NEON (optimized):  {neon_throughput:>8.1f}M msg/sec")
    print(f"   Improvement:       {throughput_improvement:>8.2f}x")
    print(f"   Increase:          {((neon_throughput - scalar_throughput) / scalar_throughput * 100):>7.1f}%")
    
    # NEON-specific metrics
    print(f"\n🔥 NEON Metrics:")
    print(f"   NEON operations:   {neon_stats.neon_operations:>8,}")
    print(f"   SIMD efficiency:   {neon_stats.neon_operations / (neon_stats.total_messages / batch_size):>8.1f} ops/batch")
    
    # Performance classification
    if latency_improvement >= 1.5:
        perf_class = "🔥 EXCELLENT"
    elif latency_improvement >= 1.2:
        perf_class = "✅ GOOD"
    elif latency_improvement >= 1.05:
        perf_class = "⚡ MARGINAL"
    else:
        perf_class = "⚠️  NO IMPROVEMENT"
    
    print(f"\n🏆 NEON Acceleration: {perf_class} ({latency_improvement:.2f}x speedup)")
    
    # System analysis
    cpu_freq_ghz = 3.5
    scalar_cycles = scalar_latency * cpu_freq_ghz
    neon_cycles = neon_latency * cpu_freq_ghz
    
    print(f"\n🎛️  System Analysis:")
    print(f"   Scalar CPU cycles/msg: ~{scalar_cycles:.0f}")
    print(f"   NEON CPU cycles/msg:   ~{neon_cycles:.0f}")
    print(f"   Cycle reduction:       {scalar_cycles - neon_cycles:.0f} cycles")
    
    # Validation
    timing_consistent = (
        0.9 <= scalar_stats.wall_clock_duration_s / test_duration <= 1.1 and
        0.9 <= neon_stats.wall_clock_duration_s / test_duration <= 1.1
    )
    
    print(f"\n🔍 Validation:")
    print(f"   Timing consistent: {'✅' if timing_consistent else '❌'}")
    print(f"   Both tests valid: {'✅' if timing_consistent else '❌'}")
    
    # Expected vs actual improvement
    expected_min = 1.2  # 20% improvement expected
    expected_max = 1.5  # 50% improvement expected
    
    meets_expectations = expected_min <= latency_improvement <= expected_max * 2
    print(f"   Meets expectations: {'✅' if meets_expectations else '⚠️'} (expected {expected_min:.1f}-{expected_max:.1f}x)")
    
    # Overall assessment
    success = timing_consistent and latency_improvement >= 1.1
    print(f"\n🎯 NEON Integration: {'✅ SUCCESS' if success else '❌ NEEDS WORK'}")
    
    if success:
        print(f"\n🚀 NEON acceleration achieved {latency_improvement:.2f}x speedup!")
        print(f"💫 New ultra-low latency: {neon_latency:.1f} ns per message")
    else:
        print(f"\n⚠️  NEON acceleration needs optimization or debugging")
    
    return {
        'scalar_latency_ns': scalar_latency,
        'neon_latency_ns': neon_latency,
        'improvement_factor': latency_improvement,
        'success': success
    }

if __name__ == "__main__":
    try:
        results = run_performance_comparison()
        
        if results['success']:
            print(f"\n✅ NEON integration completed successfully!")
            print(f"🔥 {results['improvement_factor']:.2f}x performance improvement achieved")
        else:
            print(f"\n⚠️  NEON integration needs further optimization")
            
    except Exception as e:
        print(f"❌ Error running NEON comparison: {e}")
        import traceback
        traceback.print_exc()
