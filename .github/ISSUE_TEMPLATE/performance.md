---
name: ⚡ Performance Issue
about: Report performance problems or optimization opportunities
title: '[PERF] '
labels: ['performance', 'triage']
assignees: ''
---

## ⚡ Performance Issue
**Type of performance issue:**
- [ ] 🐌 Function/method is slower than expected
- [ ] 💾 High memory usage
- [ ] 🔄 Memory leak suspected
- [ ] ⏱️ Execution time regression
- [ ] 🏗️ Optimization opportunity identified

## 📊 Current Performance
**Measurements:**
- Execution time: [e.g., 5.2 seconds]
- Memory usage: [e.g., 1.2 GB peak]
- CPU usage: [e.g., 85% average]
- Dataset size: [e.g., 1M data points, 2 years]

**Benchmark code:**
```python
import time
import psutil
import quantpulse

# Your benchmarking code
start_time = time.time()
result = quantpulse.your_function(data)
execution_time = time.time() - start_time
print(f"Execution time: {execution_time:.2f} seconds")
```

## 🎯 Expected Performance
**What performance did you expect?**
- Expected execution time: [e.g., < 1 second]
- Expected memory usage: [e.g., < 500 MB]
- Based on: [e.g., documentation claims, previous versions, similar tools]

## 💻 Environment
**System Information:**
- OS: [e.g., macOS 13.0, Ubuntu 22.04]
- Python Version: [e.g., 3.11.5]
- QuantPulse Version: [e.g., 2.1.0]
- Hardware: 
  - CPU: [e.g., Apple M3 Pro, Intel i7-12700K]
  - RAM: [e.g., 32 GB]
  - Storage: [e.g., NVMe SSD]

## 📈 Profiling Data
**If you've profiled the code, include results:**
```
cProfile output, memory_profiler results, or other profiling data
```

**Bottlenecks identified:**
- Function/line taking most time
- Memory allocation hotspots
- I/O operations

## 📝 Code Sample
```python
# Minimal code that demonstrates the performance issue
import quantpulse
import numpy as np

# Setup
data = np.random.randn(1000000, 2)

# Slow operation
result = quantpulse.slow_function(data)
```

## 🔍 Investigation
**Have you investigated this issue?**
- [ ] Profiled the code
- [ ] Tested with different input sizes
- [ ] Compared with other implementations
- [ ] Checked for obvious bottlenecks

**Findings:**
- What did you discover?
- Where do you think the bottleneck is?

## 💡 Optimization Ideas
**Do you have suggestions for improvement?**
- [ ] Algorithm optimization
- [ ] Better data structures
- [ ] Caching/memoization
- [ ] Vectorization/SIMD
- [ ] Parallel processing
- [ ] Memory management
- [ ] C++ implementation

**Specific suggestions:**
```python
# Your optimization ideas
```

## 📊 Comparison
**How does this compare to:**
- Previous versions of QuantPulse
- Other similar libraries
- Pure Python implementations
- Theoretical limits

## 🎯 Success Criteria
**What would constitute a successful fix?**
- [ ] Execution time reduced by X%
- [ ] Memory usage reduced by Y MB
- [ ] No performance regression
- [ ] Scales well with input size

## 📋 Additional Context
- Does this happen with all datasets?
- Is it related to specific market conditions?
- Any patterns you've noticed?

## 🏷️ Priority
- [ ] 🔥 Critical - Blocks production use
- [ ] 📈 High - Significant impact
- [ ] 📋 Medium - Noticeable slowdown
- [ ] 📝 Low - Minor optimization
