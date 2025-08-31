#!/usr/bin/env python3
"""
HFT Trading Engine Production Launcher
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hft_engine.engine import run_hft_engine

if __name__ == "__main__":
    stats = run_hft_engine()
    print(f"RESULT: {stats.total_messages} messages, {stats.wall_clock_avg_latency_ns:.1f}ns latency, {stats.throughput_msg_sec/1e6:.1f}M/s throughput")
