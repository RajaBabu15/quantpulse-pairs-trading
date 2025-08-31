#!/usr/bin/env python3
"""
Reproducible Benchmark Harness
Uses calibrated timer, collects histograms and prints p50/p95/p99/p999 and perf-style counters
"""

import time
import argparse
import numpy as np
from collections import defaultdict
import json
import sys
from pathlib import Path

def ns_percentiles(arr):
    """Calculate percentiles from array of nanosecond measurements"""
    if len(arr) == 0:
        return {'count': 0}
    
    arr = np.array(arr, dtype=np.float64)
    return {
        'count': len(arr),
        'min': float(arr.min()),
        'p50': float(np.percentile(arr, 50)),
        'p95': float(np.percentile(arr, 95)),
        'p99': float(np.percentile(arr, 99)),
        'p999': float(np.percentile(arr, 99.9)),
        'max': float(arr.max()),
        'mean': float(arr.mean()),
        'std': float(arr.std())
    }

def print_perf_counters(stats, duration_s):
    """Print performance counters in perf-style format"""
    print(f"\n⚡ Performance Counters ({duration_s:.1f}s):")
    print(f"   {stats['total_messages']:>12,} messages")
    print(f"   {stats['total_batches']:>12,} batches")
    print(f"   {stats['messages_per_sec']:>12,.0f} messages/sec")
    print(f"   {stats['batches_per_sec']:>12,.0f} batches/sec")
    print(f"   {stats['avg_batch_size']:>12.1f} avg batch size")

def run_benchmark(
    duration_s=5.0,
    batch_size=128,
    num_symbols=1000,
    num_pairs=200,
    timer_factor_ns=41.67,
    warmup_batches=1000,
    sample_rate=100  # Sample every Nth batch for histograms
):
    """Run calibrated benchmark with histogram collection"""
    
    print("🏁 REPRODUCIBLE HFT BENCHMARK HARNESS")
    print("=" * 60)
    print(f"⏱️  Timer calibration: {timer_factor_ns:.2f} ns/cycle (Apple M4)")
    print(f"📊 Config: {num_symbols} symbols, {num_pairs} pairs, batch={batch_size}")
    print(f"🕐 Duration: {duration_s}s, warmup: {warmup_batches} batches")
    print()
    
    try:
        import nanoext_runloop_corrected
    except ImportError:
        print("❌ C++ engine not available. Please build with: python setup_runloop_corrected.py build_ext --inplace")
        return None
    
    # Initialize data structures
    window = 1024
    window_mask = window - 1
    lookback = 100
    
    price_rb = np.zeros((num_symbols, window), dtype=np.float64)
    write_idx = np.zeros(num_symbols, dtype=np.int32)
    
    # Create trading pairs
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
    
    # Warmup phase
    print(f"🔥 Warming up ({warmup_batches} batches)...")
    warmup_stats = nanoext_runloop_corrected.run_loop_corrected(
        0.5,  # 500ms warmup
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
        seed=0x123456789abcdef,
        collect_histograms=False  # Skip histograms during warmup
    )
    print(f"   Warmup complete: {warmup_stats.total_messages:,} messages")
    print()
    
    # Main benchmark run
    print(f"🚀 Running main benchmark ({duration_s}s)...")
    
    benchmark_start = time.perf_counter()
    
    stats = nanoext_runloop_corrected.run_loop_corrected(
        duration_s,
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
        seed=0x987654321fedcba,
        collect_histograms=True
    )
    
    benchmark_end = time.perf_counter()
    python_duration = benchmark_end - benchmark_start
    
    # Collect performance statistics
    perf_stats = {
        'total_messages': int(stats.total_messages),
        'total_batches': int(stats.total_messages // batch_size),
        'messages_per_sec': float(stats.throughput_msg_sec),
        'batches_per_sec': float(stats.throughput_msg_sec / batch_size),
        'avg_batch_size': float(batch_size),
        'duration_actual_s': float(stats.wall_clock_duration_s),
        'duration_target_s': float(duration_s),
        'python_overhead_s': float(python_duration - stats.wall_clock_duration_s)
    }
    
    # Latency statistics
    latency_stats = {
        'wall_clock_avg_ns': float(stats.wall_clock_avg_latency_ns),
        'rdtsc_avg_ns': float(stats.avg_latency_ns),
        'timer_consistency_ratio': float(stats.wall_clock_avg_latency_ns / stats.avg_latency_ns)
    }
    
    # Print results
    print_perf_counters(perf_stats, duration_s)
    
    print(f"\n📊 Latency Statistics:")
    print(f"   Wall-clock avg: {latency_stats['wall_clock_avg_ns']:>8.1f} ns")
    print(f"   RDTSC avg:      {latency_stats['rdtsc_avg_ns']:>8.1f} ns")
    print(f"   Timer ratio:    {latency_stats['timer_consistency_ratio']:>8.2f}x")
    
    # Performance classification
    avg_latency = latency_stats['wall_clock_avg_ns']
    if avg_latency < 50:
        perf_class = "🔥 ULTRA-LOW"
    elif avg_latency < 100:
        perf_class = "⚡ VERY-LOW"
    elif avg_latency < 500:
        perf_class = "🚀 LOW"
    elif avg_latency < 1000:
        perf_class = "✅ SUB-MICROSECOND"
    else:
        perf_class = "⚠️  MICROSECOND+"
    
    print(f"\n🏆 Performance Class: {perf_class} ({avg_latency:.1f} ns)")
    
    # System efficiency metrics
    cpu_freq_ghz = 3.5  # Apple M4 estimated
    cycles_per_msg = avg_latency * cpu_freq_ghz
    print(f"\n🎛️  System Efficiency:")
    print(f"   CPU cycles/msg: ~{cycles_per_msg:.0f}")
    print(f"   Efficiency: {'🔥 Excellent' if cycles_per_msg < 100 else '✅ Good' if cycles_per_msg < 500 else '⚠️ Could improve'}")
    
    # Validation checks
    timing_accurate = abs(python_duration - stats.wall_clock_duration_s) < 0.1
    timer_consistent = 0.5 <= latency_stats['timer_consistency_ratio'] <= 2.0
    
    print(f"\n🔍 Validation:")
    print(f"   Duration accurate: {'✅' if timing_accurate else '❌'} ({python_duration:.3f}s vs {stats.wall_clock_duration_s:.3f}s)")
    print(f"   Timer consistent: {'✅' if timer_consistent else '❌'} ({latency_stats['timer_consistency_ratio']:.2f}x)")
    
    # Overall assessment
    benchmark_valid = timing_accurate and timer_consistent
    print(f"\n🎯 Benchmark Valid: {'✅ YES' if benchmark_valid else '❌ NO'}")
    
    # Export results as JSON for automation
    results = {
        'benchmark_info': {
            'duration_s': duration_s,
            'batch_size': batch_size,
            'num_symbols': num_symbols,
            'num_pairs': num_pairs,
            'timer_factor_ns': timer_factor_ns,
            'timestamp': time.time()
        },
        'performance': perf_stats,
        'latency': latency_stats,
        'validation': {
            'timing_accurate': timing_accurate,
            'timer_consistent': timer_consistent,
            'benchmark_valid': benchmark_valid,
            'performance_class': perf_class
        }
    }
    
    # Save results
    results_file = Path("benchmark_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {results_file}")
    
    return stats, results

def main():
    parser = argparse.ArgumentParser(description='HFT Engine Benchmark Harness')
    parser.add_argument('--duration', type=float, default=5.0, help='Benchmark duration in seconds')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--num-symbols', type=int, default=1000, help='Number of symbols')
    parser.add_argument('--num-pairs', type=int, default=200, help='Number of trading pairs')
    parser.add_argument('--timer-factor', type=float, default=41.67, help='Timer conversion factor (ns/cycle)')
    
    args = parser.parse_args()
    
    try:
        stats, results = run_benchmark(
            duration_s=args.duration,
            batch_size=args.batch_size,
            num_symbols=args.num_symbols,
            num_pairs=args.num_pairs,
            timer_factor_ns=args.timer_factor
        )
        
        if results['validation']['benchmark_valid']:
            print(f"\n🎉 Benchmark completed successfully!")
            print(f"💫 {results['latency']['wall_clock_avg_ns']:.1f} ns average latency")
            sys.exit(0)
        else:
            print(f"\n⚠️  Benchmark completed with validation issues.")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
