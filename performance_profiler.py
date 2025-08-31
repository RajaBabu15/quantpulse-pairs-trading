#!/usr/bin/env python3
"""
QuantPulse Performance Profiler and Timing Analysis System
=========================================================

Advanced profiling system to analyze performance bottlenecks across:
- Function execution times
- API calls and data downloads
- Optimization algorithms
- Trading backtests
- Memory usage patterns
- CPU utilization

Author: QuantPulse Trading Systems
"""

import time
import functools
import psutil
import threading
import inspect
import tracemalloc
import cProfile
import pstats
import io
import json
from datetime import datetime, timedelta
from collections import defaultdict, deque
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import contextmanager
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PERFORMANCE DECORATORS
# ============================================================================

class PerformanceProfiler:
    """Comprehensive performance profiler with timing and resource monitoring"""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.call_counts = defaultdict(int)
        self.memory_usage = defaultdict(list)
        self.cpu_usage = defaultdict(list)
        self.function_stack = []
        self.start_time = time.time()
        self.active_timers = {}
        self.profiler_enabled = True
        
        # Resource monitoring
        self.process = psutil.Process()
        self.monitoring_active = False
        self.resource_history = deque(maxlen=1000)
        
        # Call graph tracking
        self.call_graph = defaultdict(lambda: defaultdict(int))
        self.call_hierarchy = []
        
    def enable_monitoring(self):
        """Enable continuous resource monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._resource_monitor, daemon=True)
            self.monitor_thread.start()
    
    def disable_monitoring(self):
        """Disable continuous resource monitoring"""
        self.monitoring_active = False
    
    def _resource_monitor(self):
        """Background thread for resource monitoring"""
        while self.monitoring_active:
            try:
                cpu_percent = self.process.cpu_percent()
                memory_info = self.process.memory_info()
                memory_percent = self.process.memory_percent()
                
                self.resource_history.append({
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_rss': memory_info.rss,
                    'memory_vms': memory_info.vms,
                    'memory_percent': memory_percent,
                    'threads': self.process.num_threads()
                })
                
                time.sleep(0.1)  # Monitor every 100ms
            except:
                break
    
    def timer(self, category="general", track_memory=False):
        """Decorator for timing function execution"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.profiler_enabled:
                    return func(*args, **kwargs)
                
                # Track call hierarchy
                caller_frame = inspect.currentframe().f_back
                caller_name = caller_frame.f_code.co_name if caller_frame else "unknown"
                
                func_name = f"{func.__module__}.{func.__qualname__}"
                full_name = f"[{category}] {func_name}"
                
                # Update call graph
                if self.call_hierarchy:
                    parent = self.call_hierarchy[-1]
                    self.call_graph[parent][full_name] += 1
                
                self.call_hierarchy.append(full_name)
                self.call_counts[full_name] += 1
                
                # Memory tracking
                if track_memory:
                    tracemalloc.start()
                    memory_before = self.process.memory_info().rss
                
                # Timing
                start_time = time.perf_counter()
                start_cpu_time = time.process_time()
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    success = False
                    error = str(e)
                    result = None
                finally:
                    # Calculate timing
                    end_time = time.perf_counter()
                    end_cpu_time = time.process_time()
                    
                    wall_time = end_time - start_time
                    cpu_time = end_cpu_time - start_cpu_time
                    
                    # Memory tracking
                    if track_memory:
                        memory_after = self.process.memory_info().rss
                        memory_delta = memory_after - memory_before
                        self.memory_usage[full_name].append(memory_delta)
                        tracemalloc.stop()
                    
                    # Store timing data
                    timing_data = {
                        'wall_time': wall_time,
                        'cpu_time': cpu_time,
                        'timestamp': start_time,
                        'success': success,
                        'error': error,
                        'args_count': len(args),
                        'kwargs_count': len(kwargs),
                        'caller': caller_name
                    }
                    
                    self.timings[full_name].append(timing_data)
                    
                    # Pop from hierarchy
                    if self.call_hierarchy:
                        self.call_hierarchy.pop()
                    
                    if not success:
                        raise
                    
                    return result
            
            return wrapper
        return decorator
    
    def context_timer(self, name, category="context"):
        """Context manager for timing code blocks"""
        return TimingContext(self, name, category)
    
    def get_timing_summary(self, top_n=20):
        """Get comprehensive timing summary"""
        summary = []
        
        for func_name, times in self.timings.items():
            if not times:
                continue
            
            wall_times = [t['wall_time'] for t in times]
            cpu_times = [t['cpu_time'] for t in times]
            successes = sum(1 for t in times if t['success'])
            failures = len(times) - successes
            
            summary.append({
                'function': func_name,
                'call_count': len(times),
                'total_wall_time': sum(wall_times),
                'avg_wall_time': np.mean(wall_times),
                'median_wall_time': np.median(wall_times),
                'max_wall_time': max(wall_times),
                'min_wall_time': min(wall_times),
                'std_wall_time': np.std(wall_times),
                'total_cpu_time': sum(cpu_times),
                'avg_cpu_time': np.mean(cpu_times),
                'success_rate': successes / len(times),
                'failures': failures,
                'cpu_efficiency': np.mean(cpu_times) / np.mean(wall_times) if np.mean(wall_times) > 0 else 0
            })
        
        # Sort by total time
        summary.sort(key=lambda x: x['total_wall_time'], reverse=True)
        return summary[:top_n]
    
    def get_bottlenecks(self, threshold_seconds=0.1, min_calls=5):
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        for func_name, times in self.timings.items():
            if len(times) < min_calls:
                continue
            
            wall_times = [t['wall_time'] for t in times]
            avg_time = np.mean(wall_times)
            
            if avg_time > threshold_seconds:
                bottlenecks.append({
                    'function': func_name,
                    'avg_time': avg_time,
                    'call_count': len(times),
                    'total_time': sum(wall_times),
                    'impact_score': avg_time * len(times)
                })
        
        bottlenecks.sort(key=lambda x: x['impact_score'], reverse=True)
        return bottlenecks
    
    def print_performance_report(self):
        """Print comprehensive performance report"""
        print("\n" + "="*80)
        print("🔍 QUANTPULSE PERFORMANCE ANALYSIS REPORT")
        print("="*80)
        
        total_runtime = time.time() - self.start_time
        print(f"⏱️  Total Runtime: {total_runtime:.2f}s")
        print(f"📊 Total Function Calls: {sum(self.call_counts.values())}")
        print(f"🔧 Unique Functions: {len(self.timings)}")
        
        # Top functions by total time
        print(f"\n🏆 TOP 10 FUNCTIONS BY TOTAL TIME")
        print("-" * 60)
        summary = self.get_timing_summary(10)
        
        for i, func in enumerate(summary, 1):
            print(f"{i:2d}. {func['function']}")
            print(f"    Total: {func['total_wall_time']:.3f}s | "
                  f"Calls: {func['call_count']} | "
                  f"Avg: {func['avg_wall_time']:.4f}s")
            print(f"    Max: {func['max_wall_time']:.4f}s | "
                  f"Min: {func['min_wall_time']:.4f}s | "
                  f"Success: {func['success_rate']:.1%}")
            print()
        
        # Performance bottlenecks
        bottlenecks = self.get_bottlenecks()
        if bottlenecks:
            print(f"\n⚠️  PERFORMANCE BOTTLENECKS")
            print("-" * 60)
            
            for i, bottleneck in enumerate(bottlenecks[:5], 1):
                print(f"{i}. {bottleneck['function']}")
                print(f"   Impact Score: {bottleneck['impact_score']:.3f}")
                print(f"   Average Time: {bottleneck['avg_time']:.4f}s")
                print(f"   Total Calls: {bottleneck['call_count']}")
                print()
        
        # Resource usage summary
        if self.resource_history:
            resources = list(self.resource_history)
            avg_cpu = np.mean([r['cpu_percent'] for r in resources])
            max_memory = max([r['memory_rss'] for r in resources])
            avg_memory = np.mean([r['memory_percent'] for r in resources])
            
            print(f"\n💻 RESOURCE USAGE SUMMARY")
            print("-" * 60)
            print(f"Average CPU: {avg_cpu:.1f}%")
            print(f"Peak Memory: {max_memory / 1024 / 1024:.1f} MB")
            print(f"Average Memory: {avg_memory:.1f}%")
            print(f"Thread Count: {resources[-1]['threads'] if resources else 'N/A'}")
        
        print("\n" + "="*80)
    
    def save_detailed_report(self, filename=None):
        """Save detailed performance report to JSON"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'performance_report_{timestamp}.json'
        
        report = {
            'metadata': {
                'total_runtime': time.time() - self.start_time,
                'total_calls': sum(self.call_counts.values()),
                'unique_functions': len(self.timings),
                'generated_at': datetime.now().isoformat(),
                'profiler_version': '1.0'
            },
            'timing_summary': self.get_timing_summary(50),
            'bottlenecks': self.get_bottlenecks(),
            'call_counts': dict(self.call_counts),
            'resource_history': list(self.resource_history)[-100:],  # Last 100 samples
            'call_graph': {k: dict(v) for k, v in self.call_graph.items()}
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Detailed report saved to: {filename}")
        return filename

class TimingContext:
    """Context manager for timing code blocks"""
    
    def __init__(self, profiler, name, category):
        self.profiler = profiler
        self.name = f"[{category}] {name}"
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        self.profiler.call_counts[self.name] += 1
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            wall_time = time.perf_counter() - self.start_time
            timing_data = {
                'wall_time': wall_time,
                'cpu_time': 0,
                'timestamp': self.start_time,
                'success': exc_type is None,
                'error': str(exc_val) if exc_val else None,
                'args_count': 0,
                'kwargs_count': 0,
                'caller': 'context_manager'
            }
            self.profiler.timings[self.name].append(timing_data)

# Global profiler instance
profiler = PerformanceProfiler()

# ============================================================================
# PROFILED VERSIONS OF TRADING COMPONENTS
# ============================================================================

def profile_data_download():
    """Profile data download performance"""
    from run import PairsTrader
    
    print("📊 Profiling data download operations...")
    
    # Test different data sources and methods
    test_symbols = [
        ('AAPL', 'MSFT'),
        ('GOOGL', 'META'),
        ('TSLA', 'NVDA'),
        ('JPM', 'BAC'),
        ('JNJ', 'PFE')
    ]
    
    profiler.enable_monitoring()
    
    for symbol1, symbol2 in test_symbols:
        print(f"  📥 Testing {symbol1}-{symbol2}...")
        
        with profiler.context_timer(f"data_download_{symbol1}_{symbol2}", "data"):
            try:
                trader = PairsTrader(symbol1, symbol2)
                
                with profiler.context_timer("get_data_call", "api"):
                    prices = trader.get_data('2020-01-01', '2024-12-31')
                
                with profiler.context_timer("data_processing", "computation"):
                    # Simulate some data processing
                    spread, z_score = trader.calculate_spread_stats(prices)
                    
                print(f"    ✅ Success: {len(prices)} data points")
                
            except Exception as e:
                print(f"    ❌ Failed: {e}")
    
    profiler.disable_monitoring()

@profiler.timer("optimization", track_memory=True)
def profile_optimization_algorithms():
    """Profile different optimization algorithms"""
    from scipy.optimize import minimize, differential_evolution
    import optuna
    
    print("🔧 Profiling optimization algorithms...")
    
    # Test function for optimization
    def test_function(params):
        x, y = params
        time.sleep(0.001)  # Simulate computation
        return (x - 2)**2 + (y - 3)**2 + 0.1 * np.random.random()
    
    bounds = [(-10, 10), (-10, 10)]
    
    # Test different optimizers
    optimizers = {
        'scipy_minimize': lambda: minimize(test_function, [0, 0], bounds=bounds, method='L-BFGS-B'),
        'differential_evolution': lambda: differential_evolution(test_function, bounds, maxiter=50),
    }
    
    for name, optimizer in optimizers.items():
        with profiler.context_timer(f"optimizer_{name}", "optimization"):
            try:
                result = optimizer()
                print(f"  ✅ {name}: {result.fun:.4f}")
            except Exception as e:
                print(f"  ❌ {name}: {e}")

@profiler.timer("backtest", track_memory=True)
def profile_backtesting():
    """Profile backtesting performance"""
    from run import PairsTrader
    
    print("📈 Profiling backtesting operations...")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    
    # Generate correlated price data
    n_days = len(dates)
    returns1 = np.random.normal(0.0005, 0.02, n_days)
    returns2 = 0.8 * returns1 + 0.6 * np.random.normal(0.0005, 0.02, n_days)
    
    price1 = 100 * np.exp(np.cumsum(returns1))
    price2 = 95 * np.exp(np.cumsum(returns2))
    
    prices = pd.DataFrame({
        'SYMBOL1': price1,
        'SYMBOL2': price2
    }, index=dates)
    
    # Test different backtest configurations
    configs = [
        {'lookback': 20, 'z_entry': 2.0, 'z_exit': 0.5},
        {'lookback': 30, 'z_entry': 2.5, 'z_exit': 0.3},
        {'lookback': 10, 'z_entry': 1.5, 'z_exit': 0.8},
    ]
    
    for i, config in enumerate(configs):
        with profiler.context_timer(f"backtest_config_{i}", "backtest"):
            trader = PairsTrader('SYMBOL1', 'SYMBOL2', **config)
            
            with profiler.context_timer("spread_calculation", "computation"):
                spread, z_score = trader.calculate_spread_stats(prices)
            
            with profiler.context_timer("trading_simulation", "computation"):
                # Simulate trading logic
                position = 0
                trades = []
                
                for j in range(len(z_score)):
                    if pd.isna(z_score.iloc[j]):
                        continue
                    
                    z = z_score.iloc[j]
                    
                    if position == 0:
                        if abs(z) > config['z_entry']:
                            position = 1 if z < 0 else -1
                            trades.append({'entry': j, 'z_entry': z})
                    else:
                        if abs(z) < config['z_exit']:
                            if trades:
                                trades[-1]['exit'] = j
                                trades[-1]['z_exit'] = z
                            position = 0
            
            print(f"  ✅ Config {i}: {len(trades)} trades simulated")

def profile_memory_usage():
    """Profile memory usage patterns"""
    import gc
    
    print("💾 Profiling memory usage...")
    
    tracemalloc.start()
    
    # Test memory-intensive operations
    with profiler.context_timer("large_array_creation", "memory"):
        large_arrays = []
        for i in range(10):
            arr = np.random.random((1000, 1000))
            large_arrays.append(arr)
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"  📊 Current memory: {current / 1024 / 1024:.2f} MB")
    print(f"  📈 Peak memory: {peak / 1024 / 1024:.2f} MB")
    
    with profiler.context_timer("garbage_collection", "cleanup"):
        del large_arrays
        gc.collect()
    
    tracemalloc.stop()

def create_performance_visualizations():
    """Create performance visualization plots"""
    print("📊 Creating performance visualizations...")
    
    if not profiler.timings:
        print("  ⚠️  No timing data available")
        return
    
    # Prepare data for plotting
    summary = profiler.get_timing_summary(15)
    
    if not summary:
        return
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('QuantPulse Performance Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Total time by function
    ax1 = axes[0, 0]
    functions = [s['function'].split(']')[-1].split('.')[-1][:20] for s in summary[:10]]
    times = [s['total_wall_time'] for s in summary[:10]]
    
    bars = ax1.barh(functions, times, color=sns.color_palette("viridis", len(functions)))
    ax1.set_xlabel('Total Time (seconds)')
    ax1.set_title('Top Functions by Total Time')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for bar, time_val in zip(bars, times):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{time_val:.3f}s', va='center', fontsize=9)
    
    # 2. Call count vs average time
    ax2 = axes[0, 1]
    call_counts = [s['call_count'] for s in summary]
    avg_times = [s['avg_wall_time'] for s in summary]
    
    scatter = ax2.scatter(call_counts, avg_times, 
                         s=[s['total_wall_time']*100 for s in summary],
                         c=range(len(summary)), 
                         cmap='plasma', alpha=0.7)
    ax2.set_xlabel('Call Count')
    ax2.set_ylabel('Average Time (seconds)')
    ax2.set_title('Call Count vs Average Time\n(Bubble size = Total Time)')
    ax2.grid(alpha=0.3)
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax2, label='Function Index')
    
    # 3. CPU vs Wall time efficiency
    ax3 = axes[1, 0]
    cpu_efficiency = [s['cpu_efficiency'] for s in summary if s['cpu_efficiency'] > 0]
    func_names = [s['function'].split(']')[-1].split('.')[-1][:15] 
                  for s in summary if s['cpu_efficiency'] > 0]
    
    if cpu_efficiency:
        bars = ax3.bar(range(len(cpu_efficiency)), cpu_efficiency, 
                      color=sns.color_palette("coolwarm", len(cpu_efficiency)))
        ax3.set_xlabel('Functions')
        ax3.set_ylabel('CPU Efficiency (CPU Time / Wall Time)')
        ax3.set_title('CPU Efficiency by Function')
        ax3.set_xticks(range(len(func_names)))
        ax3.set_xticklabels(func_names, rotation=45, ha='right')
        ax3.grid(axis='y', alpha=0.3)
        ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect Efficiency')
        ax3.legend()
    
    # 4. Resource usage over time (if available)
    ax4 = axes[1, 1]
    if profiler.resource_history:
        resources = list(profiler.resource_history)
        timestamps = [(r['timestamp'] - profiler.start_time) for r in resources]
        cpu_usage = [r['cpu_percent'] for r in resources]
        memory_usage = [r['memory_percent'] for r in resources]
        
        ax4_twin = ax4.twinx()
        
        line1 = ax4.plot(timestamps, cpu_usage, 'b-', label='CPU %', alpha=0.7)
        line2 = ax4_twin.plot(timestamps, memory_usage, 'r-', label='Memory %', alpha=0.7)
        
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('CPU Usage (%)', color='b')
        ax4_twin.set_ylabel('Memory Usage (%)', color='r')
        ax4.set_title('Resource Usage Over Time')
        ax4.grid(alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        ax4.text(0.5, 0.5, 'No resource monitoring data\navailable', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Resource Usage Over Time')
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'performance_analysis_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  📊 Performance plots saved to: {filename}")
    
    plt.show()

# ============================================================================
# COMPREHENSIVE PERFORMANCE TEST SUITE
# ============================================================================

def run_comprehensive_performance_analysis():
    """Run comprehensive performance analysis of the entire system"""
    print("🚀 STARTING COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("="*80)
    
    start_time = time.time()
    profiler.enable_monitoring()
    
    try:
        # 1. Profile data operations
        with profiler.context_timer("data_operations_suite", "suite"):
            profile_data_download()
        
        # 2. Profile optimization algorithms
        with profiler.context_timer("optimization_suite", "suite"):
            profile_optimization_algorithms()
        
        # 3. Profile backtesting
        with profiler.context_timer("backtesting_suite", "suite"):
            profile_backtesting()
        
        # 4. Profile memory usage
        with profiler.context_timer("memory_analysis_suite", "suite"):
            profile_memory_usage()
        
        # 5. Test HFT components if available
        with profiler.context_timer("hft_testing_suite", "suite"):
            try:
                from run import HFT_AVAILABLE
                if HFT_AVAILABLE:
                    print("⚡ Testing HFT components...")
                    # HFT-specific profiling would go here
                    time.sleep(0.1)  # Placeholder
                else:
                    print("⚠️  HFT components not available")
            except ImportError:
                print("⚠️  Could not test HFT components")
    
    finally:
        profiler.disable_monitoring()
        
        total_time = time.time() - start_time
        print(f"\n⏱️  Total analysis time: {total_time:.2f}s")
        
        # Generate reports
        profiler.print_performance_report()
        report_file = profiler.save_detailed_report()
        
        # Create visualizations
        try:
            create_performance_visualizations()
        except Exception as e:
            print(f"⚠️  Could not create visualizations: {e}")
        
        return report_file

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run the comprehensive performance analysis
    report_file = run_comprehensive_performance_analysis()
    
    print("\n🎉 PERFORMANCE ANALYSIS COMPLETE!")
    print(f"📄 Detailed report: {report_file}")
    print("📊 Check for performance visualization plots")
    print("\n💡 KEY RECOMMENDATIONS:")
    print("   1. Focus optimization on functions with high impact scores")
    print("   2. Consider caching for frequently called functions")
    print("   3. Profile memory usage in production environments")
    print("   4. Monitor resource usage during heavy operations")
