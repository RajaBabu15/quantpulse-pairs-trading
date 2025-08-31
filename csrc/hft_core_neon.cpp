#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <vector>
#include <chrono>
#include <cstring>
#include <algorithm>
#include "neon_kernels.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <x86intrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

using namespace pybind11;

// High-performance timing utilities
#if defined(__x86_64__) || defined(_M_X64)
static inline uint64_t rdtsc() {
    return __rdtsc();
}
static const double TIMER_TO_NS_FACTOR = 1.0/3.0;
#elif defined(__aarch64__) || defined(_M_ARM64)
static inline uint64_t rdtsc() {
    uint64_t val;
    asm volatile("mrs %0, cntvct_el0" : "=r" (val));
    return val;
}
static const double TIMER_TO_NS_FACTOR = 41.67;
#else
static inline uint64_t rdtsc() {
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}
static const double TIMER_TO_NS_FACTOR = 1.0;
#endif

static inline uint64_t cycles_to_ns(uint64_t cycles) {
    return (uint64_t)(cycles * TIMER_TO_NS_FACTOR);
}

static inline uint64_t now_ns() {
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

// NEON-enhanced PRNG
struct NeonXorShift128Plus {
    uint64_t s0, s1;
    
    NeonXorShift128Plus(uint64_t seed = 0x12345678abcdefULL) {
        s0 = seed;
        s1 = seed ^ 0xdeadbeefcafebabeULL;
        if (s0 == 0 && s1 == 0) s0 = 1;
    }
    
    uint64_t next() {
        uint64_t x = s0;
        uint64_t const y = s1;
        s0 = y;
        x ^= x << 23;
        s1 = (x ^ y ^ (x >> 17) ^ (y >> 26));
        return s1 + y;
    }
    
    void generate_batch(double* prices, int32_t* symbol_ids, int batch_size, int num_symbols) {
#if defined(__aarch64__) || defined(_M_ARM64)
        neon_generate_prices(prices, symbol_ids, batch_size, num_symbols, s0, s1);
#else
        for (int i = 0; i < batch_size; ++i) {
            uint64_t r = next();
            double u = (r >> 11) * (1.0 / 9007199254740992.0);
            prices[i] = 100.0 + (u - 0.5) * 6.0;
            symbol_ids[i] = i % num_symbols;
        }
#endif
    }
};

// NEON-enhanced batch aggregator
struct NeonBatchAggregator {
    std::vector<double> last_prices;
    std::vector<int32_t> touched_symbols;
    std::vector<bool> is_touched;
    int num_touched;
    
    NeonBatchAggregator(int num_symbols) : 
        last_prices(num_symbols), 
        touched_symbols(num_symbols),
        is_touched(num_symbols, false),
        num_touched(0) {}
    
    void reset() {
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
        last_prices[sid] = price;
    }
    
    void flush_to_ringbuffer(double* __restrict rb, int32_t* __restrict write_idx, 
                           int window, uint32_t mask, int num_symbols) {
        for (int i = 0; i < num_touched; ++i) {
            int32_t sid = touched_symbols[i];
            uint32_t idx = (uint32_t)write_idx[sid];
            
#if defined(__GNUC__) || defined(__clang__)
            __builtin_prefetch(rb + (size_t)sid * window + ((idx + 1) & mask), 1, 3);
            __builtin_prefetch(rb + (size_t)sid * window + ((idx + 2) & mask), 1, 2);
#endif
            
            rb[(size_t)sid * window + idx] = last_prices[sid];
            write_idx[sid] = (int32_t)((idx + 1) & mask);
        }
    }
};

// Minimal NEON statistics
struct NeonRunStats {
    uint64_t total_messages;
    double avg_latency_ns;
    double throughput_msg_sec;
    double wall_clock_avg_latency_ns;
    double wall_clock_duration_s;
    uint64_t neon_operations;
};

// NEON-accelerated main loop
NeonRunStats run_loop_neon(
    double duration_seconds,
    int batch_size,
    int num_symbols,
    int window,
    uint32_t window_mask,
    int lookback,
    pybind11::array_t<double> price_rb,
    pybind11::array_t<int32_t> write_idx,
    pybind11::array_t<int32_t> pair_indices,
    pybind11::array_t<double> zsum,
    pybind11::array_t<double> zsumsq,
    uint64_t seed = 0x12345678abcdefULL
) {
    // Get raw pointers
    double* rb_ptr = static_cast<double*>(price_rb.mutable_data());
    int32_t* widx_ptr = static_cast<int32_t*>(write_idx.mutable_data());
    int32_t* pairs_ptr = static_cast<int32_t*>(pair_indices.mutable_data());
    double* zsum_ptr = static_cast<double*>(zsum.mutable_data());
    double* zsq_ptr = static_cast<double*>(zsumsq.mutable_data());
    
    // Initialize NEON-enhanced components
    NeonXorShift128Plus rng(seed);
    NeonBatchAggregator aggregator(num_symbols);
    
    // Pre-allocate batch arrays
    std::vector<int32_t> symbol_ids(batch_size);
    std::vector<double> prices(batch_size);
    
    // Timing setup
    uint64_t wall_start_ns = now_ns();
    uint64_t end_time_ns = wall_start_ns + uint64_t(duration_seconds * 1e9);
    uint64_t total_messages = 0;
    uint64_t total_cycles = 0;
    uint64_t neon_operations = 0;
    
    bool zs_initialized = false;
    int n_pairs = pair_indices.size() / 2;
    
    // NEON-ACCELERATED MAIN LOOP
    while (now_ns() < end_time_ns) {
        uint64_t batch_start_cycles = rdtsc();
        
        // NEON-accelerated data generation
        rng.generate_batch(prices.data(), symbol_ids.data(), batch_size, num_symbols);
        
        // Enhanced ring buffer update
        aggregator.reset();
        for (int i = 0; i < batch_size; ++i) {
            aggregator.add_update(symbol_ids[i], prices[i]);
        }
        aggregator.flush_to_ringbuffer(rb_ptr, widx_ptr, window, window_mask, num_symbols);
        
        // NEON-accelerated Z-score computation
#if defined(__aarch64__) || defined(_M_ARM64)
        if (!zs_initialized) {
            neon_zscore_batch(rb_ptr, widx_ptr, pairs_ptr, zsum_ptr, zsq_ptr,
                            n_pairs, num_symbols, window, window_mask, lookback);
            zs_initialized = true;
            neon_operations++;
        } else {
            neon_zscore_incremental(rb_ptr, widx_ptr, pairs_ptr, zsum_ptr, zsq_ptr,
                                  n_pairs, num_symbols, window, window_mask, lookback);
            neon_operations++;
        }
#else
        // Scalar fallback for non-ARM64
        for (int pair_idx = 0; pair_idx < n_pairs; ++pair_idx) {
            int32_t idx1 = pairs_ptr[2 * pair_idx];
            int32_t idx2 = pairs_ptr[2 * pair_idx + 1];
            
            if (idx1 < 0 || idx2 < 0 || idx1 >= num_symbols || idx2 >= num_symbols) continue;
            
            int32_t w1 = widx_ptr[idx1];
            int32_t w2 = widx_ptr[idx2];
            
            if (!zs_initialized) {
                double sum = 0.0, sumsq = 0.0;
                for (int k = 0; k < lookback; ++k) {
                    int32_t j1 = (w1 - lookback + k + window) & window_mask;
                    int32_t j2 = (w2 - lookback + k + window) & window_mask;
                    
                    double spread = rb_ptr[(size_t)idx1 * window + j1] - rb_ptr[(size_t)idx2 * window + j2];
                    sum += spread;
                    sumsq += spread * spread;
                }
                zsum_ptr[pair_idx] = sum;
                zsq_ptr[pair_idx] = sumsq;
            } else {
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
#endif
        
        uint64_t batch_end_cycles = rdtsc();
        
        // Record measurements
        uint64_t batch_cycles = batch_end_cycles - batch_start_cycles;
        total_cycles += batch_cycles;
        total_messages += batch_size;
    }
    
    uint64_t wall_end_ns = now_ns();
    
    // Calculate final statistics
    NeonRunStats stats;
    stats.total_messages = total_messages;
    stats.avg_latency_ns = cycles_to_ns(total_cycles) / double(total_messages);
    stats.throughput_msg_sec = double(total_messages) / duration_seconds;
    stats.wall_clock_duration_s = (wall_end_ns - wall_start_ns) / 1e9;
    stats.wall_clock_avg_latency_ns = (wall_end_ns - wall_start_ns) / double(total_messages);
    stats.neon_operations = neon_operations;
    
    return stats;
}

// Python module definition
PYBIND11_MODULE(hft_core_neon, m) {
    m.doc() = "NEON-accelerated HFT trading engine core";
    
    pybind11::class_<NeonRunStats>(m, "NeonRunStats")
        .def_readonly("total_messages", &NeonRunStats::total_messages)
        .def_readonly("avg_latency_ns", &NeonRunStats::avg_latency_ns)
        .def_readonly("throughput_msg_sec", &NeonRunStats::throughput_msg_sec)
        .def_readonly("wall_clock_avg_latency_ns", &NeonRunStats::wall_clock_avg_latency_ns)
        .def_readonly("wall_clock_duration_s", &NeonRunStats::wall_clock_duration_s)
        .def_readonly("neon_operations", &NeonRunStats::neon_operations);
    
    m.def("run_loop_neon", &run_loop_neon, 
          "NEON-accelerated main loop",
          pybind11::arg("duration_seconds"),
          pybind11::arg("batch_size"),
          pybind11::arg("num_symbols"),
          pybind11::arg("window"),
          pybind11::arg("window_mask"),
          pybind11::arg("lookback"),
          pybind11::arg("price_rb"),
          pybind11::arg("write_idx"),
          pybind11::arg("pair_indices"),
          pybind11::arg("zsum"),
          pybind11::arg("zsumsq"),
          pybind11::arg("seed") = 0x12345678abcdefULL);
}
