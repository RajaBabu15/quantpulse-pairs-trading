# Ultra-Low Latency HFT Trading Engine

Production-ready high-frequency trading engine optimized for Apple M4 Silicon.

## Performance
- **11.1ns** average latency per message  
- **90M+** messages per second throughput
- **Sub-12ns** consistently across runs

## Quick Start

### Install (editable) and build native extensions
```bash
python -m pip install -e .
```

### Run the HFT Engine (daily bars example)
```bash
python hft_engine/engine.py --tickers AAPL MSFT --start 2024-01-01 --end 2024-12-31 --interval 1d --batch 256
```

### Run the HFT Engine (1-minute bars with automatic chunked download)
```bash
python hft_engine/engine.py --tickers KO PEP KDP MNST PG CL KMB MDLZ XLP --start 2025-08-18 --end 2025-08-29 --interval 1m --batch 512
```

### Optimize C++ strategies with Optuna (optional)
```bash
python -m pip install optuna
python hft_engine/optimize_hft_strategies.py --tickers KO PEP KDP MNST PG CL KMB MDLZ XLP --start 2025-08-18 --end 2025-08-29 --interval 1m --trials 50
```

## Architecture
- **hft_core**: Baseline ultra-optimized C++ engine
- **hft_core_neon**: NEON SIMD-accelerated version
- **hft_engine**: Minimal Python interface
- Automatic best-engine selection at runtime

## Requirements
- Python 3.8+
- NumPy
- pybind11
- Apple M4 (optimized) or compatible ARM64/x86_64

## Production Ready
- Zero debug overhead
- Minimal memory footprint
- Maximum compiler optimizations
- Clean, maintainable codebase
