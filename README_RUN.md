# Run

Below are copy‑pastable commands to set up a conda environment, build/install the package (including the native C extension), run tests, and launch the streamlined demo.

## 0) Prerequisites (macOS)
- Ensure Command Line Tools are installed (for compilers):

```
# macOS only
xcode-select --install || true
```

## 1) Create and activate a conda environment

```
conda create -n quantpulse python=3.10 -y
conda activate quantpulse
```

## 2) Build and install the package (editable)

```
# From the repository root
python -m pip install -U pip setuptools wheel build numpy
python -m pip install -e . --no-deps --force-reinstall
```

## 3) Run sanity tests

```
python -m unittest -q tests/test_nanoext_inc.py
```

## 4) Run the streamlined demo (defaults)

```
python src/nanosecond_optimized.py
```

## 5) Run with a config file

```
python src/nanosecond_optimized.py --config config/sample_engine.json --duration 3
```

## 6) Override pairs/thresholds/lookback via CLI

```
python src/nanosecond_optimized.py \
  --pairs 0-1,1-2,2-3,0-2 \
  --thresholds 2.0,2.0,2.0,2.0 \
  --lookback 30 \
  --duration 3
```

## Notes
- The native extension is built automatically during the editable install.
- If you switch Python versions or compilers, re-run the editable install step.
- For FPGA development and kernel-bypass networking, use a Linux host.

