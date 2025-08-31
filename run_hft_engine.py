#!/usr/bin/env python3
"""
Production HFT Engine Launcher

This script automatically sets up the correct Python path and launches
the ultra-optimized HFT trading engine with the best available acceleration.
"""

import sys
import os

# Add current directory to Python path for compiled extensions
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import and run the main engine
from hft_engine.engine import run_hft_engine

if __name__ == "__main__":
    try:
        print("🚀 Launching Ultra-Optimized HFT Trading Engine...")
        print("🔧 Setting up environment...")
        
        # Check for compiled extensions
        try:
            import hft_core
            import hft_core_neon
            print("✅ NEON-accelerated extensions available")
        except ImportError:
            try:
                import hft_core
                print("✅ Baseline extensions available")
            except ImportError:
                print("❌ No compiled extensions found!")
                print("Run 'python setup.py build_ext --inplace' first")
                sys.exit(1)
        
        print("🏁 Starting engine...\n")
        
        # Run the engine
        stats = run_hft_engine()
        
        print(f"\n🎉 ENGINE COMPLETED SUCCESSFULLY!")
        print(f"🏆 Final Results:")
        print(f"   💫 Latency: {stats.wall_clock_avg_latency_ns:.1f} ns")
        print(f"   🚀 Throughput: {stats.throughput_msg_sec/1e6:.1f}M msg/sec") 
        print(f"   📊 Messages: {stats.total_messages:,}")
        
        # Performance class summary
        if stats.wall_clock_avg_latency_ns < 15:
            perf_class = "🔥 ULTRA-LOW LATENCY"
        elif stats.wall_clock_avg_latency_ns < 50:
            perf_class = "⚡ VERY LOW LATENCY"
        else:
            perf_class = "✅ LOW LATENCY"
            
        print(f"   🎯 Performance: {perf_class}")
        print(f"   🎛️  System: Apple M4 optimized")
        
    except KeyboardInterrupt:
        print("\n⚠️  Engine stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Engine error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
