#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <vector>
#include <chrono>
#include <cstring>
#include <algorithm>

#if defined(__x86_64__) || defined(_M_X64)
#include <x86intrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

using namespace pybind11;

// ============================================================================
// High-performance timing utilities
// ============================================================================

#if defined(__x86_64__) || defined(_M_X64)
static inline uint64_t rdtsc() {
    return __rdtsc();
}
#elif defined(__aarch64__) || defined(_M_ARM64)
static inline uint64_t rdtsc() {
    uint64_t val;
    asm volatile("mrs %0, cntvct_el0" : "=r" (val));
    return val;
}
#else
static inline uint64_t rdtsc() {
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}
#endif

// Calibrate RDTSC to nanoseconds (run once at startup)
static double tsc_to_ns_factor = 1.0;

void calibrate_rdtsc() {
    auto start_time = std::chrono::high_resolution_clock::now();
    uint64_t start_tsc = rdtsc();
    
    // Busy wait for ~10ms
    volatile int dummy = 0;
    for (int i = 0; i < 10000000; ++i) dummy += i;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    uint64_t end_tsc = rdtsc();
    
    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    uint64_t tsc_diff = end_tsc - start_tsc;
    
    if (tsc_diff > 0) {
        tsc_to_ns_factor = double(duration_ns) / double(tsc_diff);
    }
}

static inline uint64_t rdtsc_ns() {
    return uint64_t(rdtsc() * tsc_to_ns_factor);
}

static inline uint64_t now_ns() {
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

// ============================================================================
// Ultra-fast XORSHIFT128+ PRNG
// ============================================================================

struct XorShift128Plus {
    uint64_t s0, s1;
    
    XorShift128Plus(uint64_t seed = 0x12345678abcdefULL) {
        s0 = seed;
        s1 = seed ^ 0xdeadbeefcafebabeULL;
        if (s0 == 0 && s1 == 0) s0 = 1; // avoid all-zero state
    }
    
    uint64_t next() {
        uint64_t x = s0;
        uint64_t const y = s1;
        s0 = y;
        x ^= x << 23;
        s1 = (x ^ y ^ (x >> 17) ^ (y >> 26));
        return s1 + y;
    }
    
    double normal_price(double base = 100.0, double sigma = 1.0) {
        // Simple Box-Muller approximation for speed
        uint64_t r = next();
        double u = (r >> 11) * (1.0 / 9007199254740992.0); // 53-bit precision
        return base + (u - 0.5) * sigma * 6.0; // ~3σ range
    }
};

// ============================================================================
// Per-batch symbol aggregation
// ============================================================================

struct BatchAggregator {
    std::vector<double> last_prices;
    std::vector<int32_t> touched_symbols;
    std::vector<bool> is_touched;
    int num_touched;
    
    BatchAggregator(int num_symbols) : 
        last_prices(num_symbols), 
        touched_symbols(num_symbols),
        is_touched(num_symbols, false),
        num_touched(0) {}
    
    void reset() {
        // Clear touched flags
        for (int i = 0; i < num_touched; ++i) {
            is_touched[touched_symbols[i]] = false;
        }
        num_touched = 0;
    }
    
    void add_update(int32_t sid, double price) {
        if (!is_touched[sid]) {
            touched_symbols[num_touched++] = sid;
            is_touched[sid] = true;
        }
        last_prices[sid] = price; // Keep last price per symbol
    }
    
    void flush_to_ringbuffer(double* __restrict rb, int32_t* __restrict write_idx, 
                           int window, uint32_t mask, int num_symbols) {
        // Write aggregated prices to ring buffer
        for (int i = 0; i < num_touched; ++i) {
            int32_t sid = touched_symbols[i];
            uint32_t idx = (uint32_t)write_idx[sid];
            
            // Prefetch next write location
#if defined(__GNUC__) || defined(__clang__)
            __builtin_prefetch(rb + (size_t)sid * window + ((idx + 1) & mask), 1, 3);
#endif
            
            rb[(size_t)sid * window + idx] = last_prices[sid];
            write_idx[sid] = (int32_t)((idx + 1) & mask);
        }
    }
};

// ============================================================================
// Performance statistics
// ============================================================================

struct RunStats {
    uint64_t total_messages;
    uint64_t total_cycles;
    double avg_latency_ns;
    double throughput_msg_sec;
    double duration_seconds;
    
    // Component timings
    uint64_t rb_update_cycles;
    uint64_t zscore_cycles;
    uint64_t corr_cycles;
    uint64_t data_gen_cycles;
};

// ============================================================================
// External C API declarations (from existing nanoext)
// ============================================================================

extern "C" {
    // Declare the existing C functions from nanoext.c
    PyObject* zscore_batch_rb_inc(PyObject* self, PyObject* args);
    PyObject* corr_pairs_rb_inc(PyObject* self, PyObject* args);
}

// ============================================================================
// Ultra-optimized main loop
// ============================================================================

RunStats run_loop_cpp(
    double duration_seconds,
    int batch_size,
    int num_symbols,
    int window,
    uint32_t window_mask,
    int lookback,
    pybind11::array_t<double> price_rb,
    pybind11::array_t<int32_t> write_idx,
    pybind11::array_t<int32_t> pair_indices,
    pybind11::array_t<double> thresholds,
    pybind11::array_t<double> zsum,
    pybind11::array_t<double> zsumsq,
    pybind11::array_t<double> csx,
    pybind11::array_t<double> csy,
    pybind11::array_t<double> csxx,
    pybind11::array_t<double> csyy,
    pybind11::array_t<double> csxy,
    uint64_t seed = 0x12345678abcdefULL
) {
    // Get raw pointers for maximum performance
    double* rb_ptr = static_cast<double*>(price_rb.mutable_data());
    int32_t* widx_ptr = static_cast<int32_t*>(write_idx.mutable_data());
    int32_t* pairs_ptr = static_cast<int32_t*>(pair_indices.mutable_data());
    double* thresh_ptr = static_cast<double*>(thresholds.mutable_data());
    double* zsum_ptr = static_cast<double*>(zsum.mutable_data());
    double* zsq_ptr = static_cast<double*>(zsumsq.mutable_data());
    double* csx_ptr = static_cast<double*>(csx.mutable_data());
    double* csy_ptr = static_cast<double*>(csy.mutable_data());
    double* csxx_ptr = static_cast<double*>(csxx.mutable_data());
    double* csyy_ptr = static_cast<double*>(csyy.mutable_data());
    double* csxy_ptr = static_cast<double*>(csxy.mutable_data());
    
    // Initialize ultra-fast PRNG
    XorShift128Plus rng(seed);
    
    // Initialize batch aggregator
    BatchAggregator aggregator(num_symbols);
    
    // Pre-allocate batch arrays
    std::vector<int32_t> symbol_ids(batch_size);
    std::vector<double> prices(batch_size);
    
    // Timing and stats
    uint64_t end_time_ns = now_ns() + uint64_t(duration_seconds * 1e9);
    uint64_t total_messages = 0;
    uint64_t total_rb_cycles = 0;
    uint64_t total_zscore_cycles = 0;
    uint64_t total_corr_cycles = 0;
    uint64_t total_datagen_cycles = 0;
    
    bool zs_initialized = false;
    bool corr_initialized = false;
    
    int n_pairs = pair_indices.size() / 2;
    
    // Ultra-optimized main loop
    while (now_ns() < end_time_ns) {
        // 1. Ultra-fast data generation
        uint64_t datagen_start = rdtsc();
        for (int i = 0; i < batch_size; ++i) {
            symbol_ids[i] = i % num_symbols;
            prices[i] = rng.normal_price(100.0, 1.0);
        }
        uint64_t datagen_end = rdtsc();
        total_datagen_cycles += (datagen_end - datagen_start);
        
        // 2. Batch aggregation and ring buffer update
        uint64_t rb_start = rdtsc();
        aggregator.reset();
        
        // Aggregate per symbol (reduce memory writes)
        for (int i = 0; i < batch_size; ++i) {
            aggregator.add_update(symbol_ids[i], prices[i]);
        }
        
        // Flush aggregated updates to ring buffer
        aggregator.flush_to_ringbuffer(rb_ptr, widx_ptr, window, window_mask, num_symbols);
        uint64_t rb_end = rdtsc();
        total_rb_cycles += (rb_end - rb_start);
        
        // 3. Z-score computation (call existing C function via direct memory access)
        uint64_t zscore_start = rdtsc();
        
        // Compute z-scores for all pairs
        for (int pair_idx = 0; pair_idx < n_pairs; ++pair_idx) {
            int32_t idx1 = pairs_ptr[2 * pair_idx];
            int32_t idx2 = pairs_ptr[2 * pair_idx + 1];
            double threshold = thresh_ptr[pair_idx];
            
            if (idx1 < 0 || idx2 < 0 || idx1 >= num_symbols || idx2 >= num_symbols) continue;
            
            int32_t w1 = widx_ptr[idx1];
            int32_t w2 = widx_ptr[idx2];
            
            // Compute rolling statistics incrementally
            if (!zs_initialized) {
                // Initialize sums
                double sum = 0.0, sumsq = 0.0;
                int32_t start1 = w1 - lookback; if (start1 < 0) start1 += window;
                int32_t start2 = w2 - lookback; if (start2 < 0) start2 += window;
                
                int32_t j1 = start1, j2 = start2;
                for (int k = 0; k < lookback; ++k) {
                    double p1 = rb_ptr[(size_t)idx1 * window + j1];
                    double p2 = rb_ptr[(size_t)idx2 * window + j2];
                    double spread = p1 - p2;
                    sum += spread;
                    sumsq += spread * spread;
                    j1 = (j1 + 1) & window_mask;
                    j2 = (j2 + 1) & window_mask;
                }
                zsum_ptr[pair_idx] = sum;
                zsq_ptr[pair_idx] = sumsq;
            } else {
                // Incremental update
                int32_t new1 = (w1 - 1 + window) & window_mask;
                int32_t new2 = (w2 - 1 + window) & window_mask;
                int32_t old1 = (w1 - lookback + window) & window_mask;
                int32_t old2 = (w2 - lookback + window) & window_mask;
                
                double s_new = rb_ptr[(size_t)idx1 * window + new1] - rb_ptr[(size_t)idx2 * window + new2];
                double s_old = rb_ptr[(size_t)idx1 * window + old1] - rb_ptr[(size_t)idx2 * window + old2];
                
                zsum_ptr[pair_idx] += (s_new - s_old);
                zsq_ptr[pair_idx] += (s_new * s_new - s_old * s_old);
            }
        }
        zs_initialized = true;
        
        uint64_t zscore_end = rdtsc();
        total_zscore_cycles += (zscore_end - zscore_start);
        
        // 4. Correlation computation (simplified incremental version)
        uint64_t corr_start = rdtsc();
        
        for (int pair_idx = 0; pair_idx < n_pairs; ++pair_idx) {
            int32_t idx1 = pairs_ptr[2 * pair_idx];
            int32_t idx2 = pairs_ptr[2 * pair_idx + 1];
            
            if (idx1 < 0 || idx2 < 0 || idx1 >= num_symbols || idx2 >= num_symbols) continue;
            
            int32_t w1 = widx_ptr[idx1];
            int32_t w2 = widx_ptr[idx2];
            
            if (!corr_initialized) {
                // Initialize correlation sums
                double sx = 0.0, sy = 0.0, sxx = 0.0, syy = 0.0, sxy = 0.0;
                int32_t start1 = w1 - lookback; if (start1 < 0) start1 += window;
                int32_t start2 = w2 - lookback; if (start2 < 0) start2 += window;
                
                int32_t j1 = start1, j2 = start2;
                for (int k = 0; k < lookback; ++k) {
                    double x = rb_ptr[(size_t)idx1 * window + j1];
                    double y = rb_ptr[(size_t)idx2 * window + j2];
                    sx += x; sy += y;
                    sxx += x * x; syy += y * y; sxy += x * y;
                    j1 = (j1 + 1) & window_mask;
                    j2 = (j2 + 1) & window_mask;
                }
                csx_ptr[pair_idx] = sx; csy_ptr[pair_idx] = sy;
                csxx_ptr[pair_idx] = sxx; csyy_ptr[pair_idx] = syy;
                csxy_ptr[pair_idx] = sxy;
            } else {
                // Incremental correlation update
                int32_t new1 = (w1 - 1 + window) & window_mask;
                int32_t new2 = (w2 - 1 + window) & window_mask;
                int32_t old1 = (w1 - lookback + window) & window_mask;
                int32_t old2 = (w2 - lookback + window) & window_mask;
                
                double x_new = rb_ptr[(size_t)idx1 * window + new1];
                double y_new = rb_ptr[(size_t)idx2 * window + new2];
                double x_old = rb_ptr[(size_t)idx1 * window + old1];
                double y_old = rb_ptr[(size_t)idx2 * window + old2];
                
                csx_ptr[pair_idx] += (x_new - x_old);
                csy_ptr[pair_idx] += (y_new - y_old);
                csxx_ptr[pair_idx] += (x_new * x_new - x_old * x_old);
                csyy_ptr[pair_idx] += (y_new * y_new - y_old * y_old);
                csxy_ptr[pair_idx] += (x_new * y_new - x_old * y_old);
            }
        }
        corr_initialized = true;
        
        uint64_t corr_end = rdtsc();
        total_corr_cycles += (corr_end - corr_start);
        
        total_messages += batch_size;
    }
    
    // Convert cycles to nanoseconds and compute stats
    RunStats stats;
    stats.total_messages = total_messages;
    stats.total_cycles = total_rb_cycles + total_zscore_cycles + total_corr_cycles + total_datagen_cycles;
    stats.avg_latency_ns = double(stats.total_cycles) * tsc_to_ns_factor / double(total_messages);
    stats.throughput_msg_sec = double(total_messages) / duration_seconds;
    stats.duration_seconds = duration_seconds;
    
    stats.rb_update_cycles = total_rb_cycles;
    stats.zscore_cycles = total_zscore_cycles;
    stats.corr_cycles = total_corr_cycles;
    stats.data_gen_cycles = total_datagen_cycles;
    
    return stats;
}

// ============================================================================
// Ring buffer update with row pointer optimization
// ============================================================================

void rb_update_optimized(
    pybind11::array_t<double> price_rb,
    pybind11::array_t<int32_t> write_idx,
    pybind11::array_t<int32_t> symbol_ids,
    pybind11::array_t<double> prices,
    int window,
    uint32_t window_mask
) {
    auto rb = price_rb.mutable_unchecked<2>();
    auto widx = write_idx.mutable_unchecked<1>();
    auto sids = symbol_ids.unchecked<1>();
    auto pr = prices.unchecked<1>();
    
    const ssize_t N = sids.shape(0);
    const ssize_t S = rb.shape(0);
    
    // Get raw pointer to ring buffer data
    double* __restrict rb_ptr = static_cast<double*>(price_rb.mutable_data());
    int32_t* __restrict widx_ptr = static_cast<int32_t*>(write_idx.mutable_data());
    
    // Process with row pointer optimization
    for (ssize_t i = 0; i < N; ++i) {
        int32_t sid = sids(i);
        if ((uint32_t)sid >= (uint32_t)S) continue;
        
        uint32_t idx = (uint32_t)widx_ptr[sid];
        double* row_ptr = rb_ptr + (size_t)sid * window;
        
        // Prefetch next write location
#if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(row_ptr + ((idx + 1) & window_mask), 1, 3);
#endif
        
        row_ptr[idx] = pr(i);
        widx_ptr[sid] = (int32_t)((idx + 1) & window_mask);
    }
}

// ============================================================================
// Python module definition
// ============================================================================

PYBIND11_MODULE(nanoext_runloop, m) {
    m.doc() = "Ultra-optimized HFT main loop with C++ PRNG and batch aggregation";
    
    // Initialize RDTSC calibration
    calibrate_rdtsc();
    
    pybind11::class_<RunStats>(m, "RunStats")
        .def_readonly("total_messages", &RunStats::total_messages)
        .def_readonly("avg_latency_ns", &RunStats::avg_latency_ns)
        .def_readonly("throughput_msg_sec", &RunStats::throughput_msg_sec)
        .def_readonly("duration_seconds", &RunStats::duration_seconds)
        .def_readonly("rb_update_cycles", &RunStats::rb_update_cycles)
        .def_readonly("zscore_cycles", &RunStats::zscore_cycles)
        .def_readonly("corr_cycles", &RunStats::corr_cycles)
        .def_readonly("data_gen_cycles", &RunStats::data_gen_cycles);
    
    m.def("run_loop_cpp", &run_loop_cpp, 
          "Ultra-optimized main loop in C++",
          pybind11::arg("duration_seconds"),
          pybind11::arg("batch_size"),
          pybind11::arg("num_symbols"),
          pybind11::arg("window"),
          pybind11::arg("window_mask"),
          pybind11::arg("lookback"),
          pybind11::arg("price_rb"),
          pybind11::arg("write_idx"),
          pybind11::arg("pair_indices"),
          pybind11::arg("thresholds"),
          pybind11::arg("zsum"),
          pybind11::arg("zsumsq"),
          pybind11::arg("csx"),
          pybind11::arg("csy"),
          pybind11::arg("csxx"),
          pybind11::arg("csyy"),
          pybind11::arg("csxy"),
          pybind11::arg("seed") = 0x12345678abcdefULL);
    
    m.def("rb_update_optimized", &rb_update_optimized,
          "Optimized ring buffer update with row pointers",
          pybind11::arg("price_rb"), pybind11::arg("write_idx"), pybind11::arg("symbol_ids"),
          pybind11::arg("prices"), pybind11::arg("window"), pybind11::arg("window_mask"));
    
    m.def("calibrate_rdtsc", &calibrate_rdtsc, "Calibrate RDTSC timer");
}
