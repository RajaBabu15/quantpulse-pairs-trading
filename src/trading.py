#!/usr/bin/env python3
"""
NANOSECOND-OPTIMIZED HFT (streamlined)
- Single optimized path only
- Ring-buffer price updates (Numba)
- Incremental z-score via C extension
- Optional incremental pairwise correlation via C extension
"""

import numpy as np
from numba import njit, prange
import time
import argparse
import json
from pathlib import Path
from typing import List, Tuple

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
    """Ultra-low latency engine with only the optimized hot path"""

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
        """Process one batch of ticks using the optimized path"""
        start_ns = time.perf_counter_ns()

        # Extract fields
        symbol_ids = market_data[:, 1].astype(np.int32)
        prices = market_data[:, 2].astype(np.float64)

        # Update ring buffer (in place)
        fast_price_update_ringbuffer(self.price_rb, self.write_idx, symbol_ids, prices)

        # Incremental z-score signals
        init_flag = 1 if self.zs_initialized else 0
        signals = c_zscore_batch_rb_inc(
            self.price_rb, self.write_idx, self.pair_indices, self.lookback, self.thresholds,
            self.zsum, self.zsumsq, init_flag
        )
        self.zs_initialized = True

        # Incremental correlations for current pairs
        correlations = None
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

    def run_demo(self, duration_seconds: int = 3):
        print(f"\n🔥 Optimized path demo ({duration_seconds}s)")
        start_time = time.time()
        total_lat_ns = 0
        msgs = 0
        signals = np.zeros(len(self.thresholds), dtype=np.int8)

        while time.time() - start_time < duration_seconds:
            batch_size = 50
            market_data = np.array([
                [time.time_ns(), i % self.num_symbols, 100.0 + self.rng.normal(0, 1)]
                for i in range(batch_size)
            ])
            sig, corr, lat_ns = self.process_market_data(market_data)
            signals = sig  # keep last
            total_lat_ns += lat_ns
            msgs += batch_size

        avg_ns = total_lat_ns / max(msgs, 1)
        print("📊 Results:")
        print(f"   Messages processed: {msgs:,}")
        print(f"   Average latency: {avg_ns:,.0f} ns ({avg_ns/1000:.2f} μs)")
        print(f"   Signals per batch: {len(signals)}")
        try:
            print(f"   Correlations per batch: {len(corr)} (sample: {float(corr[0]):+.4f})")
        except Exception:
            pass


def _parse_pairs_arg(pairs_str: str) -> np.ndarray:
    # format: "0-1,1-2,2-3" -> flat array [0,1,1,2,2,3]
    items = []
    for tok in pairs_str.split(','):
        a, b = tok.strip().split('-')
        items.extend([int(a), int(b)])
    return np.array(items, dtype=np.int32)


def _load_config(cfg_path: Path):
    with cfg_path.open('r') as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description="Streamlined nanosecond HFT engine")
    ap.add_argument('--config', '-c', type=str, help='Path to JSON config with symbols/pairs/thresholds/window/lookback')
    ap.add_argument('--symbols', type=str, help='CSV symbols list to override')
    ap.add_argument('--window', type=int, default=1024)
    ap.add_argument('--lookback', type=int, default=30)
    ap.add_argument('--pairs', type=str, help='Pairs as CSV like 0-1,1-2,2-3')
    ap.add_argument('--thresholds', type=str, help='Thresholds CSV like 2.0,2.0,2.0')
    ap.add_argument('--duration', type=int, default=3, help='Demo duration seconds')
    ap.add_argument('--seed', type=int, help='Random seed for reproducible synthetic data')
    ap.add_argument('--bench-runs', type=int, default=0, help='If >0, run N times and report median/p95')
    args = ap.parse_args()

    # Defaults
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX']
    window = args.window
    lookback = args.lookback
    pair_indices = None
    thresholds = None
    seed = args.seed

    # Config file overrides
    if args.config:
        cfg = _load_config(Path(args.config))
        symbols = cfg.get('symbols', symbols)
        window = int(cfg.get('window', window))
        lookback = int(cfg.get('lookback', lookback))
        if 'pairs' in cfg:
            flat = []
            for a, b in cfg['pairs']:
                flat.extend([int(a), int(b)])
            pair_indices = np.array(flat, dtype=np.int32)
        if 'thresholds' in cfg:
            thresholds = np.array(cfg['thresholds'], dtype=np.float64)

    # CLI overrides
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]
    if args.pairs:
        pair_indices = _parse_pairs_arg(args.pairs)
    if args.thresholds:
        thresholds = np.array([float(x) for x in args.thresholds.split(',')], dtype=np.float64)

    # Validate
    if pair_indices is None:
        pair_indices = np.array([0, 1, 1, 2, 2, 3, 0, 2], dtype=np.int32)
    n_pairs = len(pair_indices) // 2
    if thresholds is None:
        thresholds = np.array([2.0] * n_pairs, dtype=np.float64)
    else:
        if len(thresholds) != n_pairs:
            raise ValueError(f"thresholds length {len(thresholds)} must match n_pairs {n_pairs}")

    # If benchmarking multiple runs, vary seed per run deterministically
    if args.bench_runs and args.bench_runs > 0:
        results = []
        for r in range(args.bench_runs):
            run_seed = (seed if seed is not None else 12345) + r
            engine = NanosecondHFTEngine(symbols, window=window, lookback=lookback,
                                         pair_indices=pair_indices, thresholds=thresholds,
                                         seed=run_seed)
            # Run once and capture average latency
            start_time = time.time()
            total_lat = 0
            msgs = 0
            while time.time() - start_time < args.duration:
                batch_size = 50
                market_data = np.array([
                    [time.time_ns(), i % engine.num_symbols, 100.0 + engine.rng.normal(0, 1)]
                    for i in range(batch_size)
                ])
                _, _, lat_ns = engine.process_market_data(market_data)
                msgs += batch_size
                total_lat += lat_ns
            avg_ns = total_lat / max(msgs, 1)
            results.append(avg_ns)
        results = np.array(results, dtype=np.float64)
        median_ns = float(np.percentile(results, 50))
        p95_ns = float(np.percentile(results, 95))
        print("\n🏁 Multi-run benchmark:")
        print(f"   runs: {args.bench_runs}")
        print(f"   median avg latency: {median_ns:,.0f} ns ({median_ns/1000:.2f} μs)")
        print(f"   p95 avg latency:    {p95_ns:,.0f} ns ({p95_ns/1000:.2f} μs)")
        return

    engine = NanosecondHFTEngine(symbols, window=window, lookback=lookback,
                                 pair_indices=pair_indices, thresholds=thresholds,
                                 seed=seed)
    engine.run_demo(duration_seconds=args.duration)


if __name__ == "__main__":
    main()
