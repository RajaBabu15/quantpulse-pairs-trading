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

// Ultra-fast XORSHIFT128+ PRNG
struct XorShift128Plus {
    uint64_t s0, s1;
    
    XorShift128Plus(uint64_t seed = 0x12345678abcdefULL) {
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
    
    double normal_price(double base = 100.0, double sigma = 1.0) {
        uint64_t r = next();
        double u = (r >> 11) * (1.0 / 9007199254740992.0);
        return base + (u - 0.5) * sigma * 6.0;
    }
};

// Per-batch symbol aggregation
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
#endif
            
            rb[(size_t)sid * window + idx] = last_prices[sid];
            write_idx[sid] = (int32_t)((idx + 1) & mask);
        }
    }
};

// Minimal statistics structure
struct RunStats {
    uint64_t total_messages;
    double avg_latency_ns;
    double throughput_msg_sec;
    double wall_clock_avg_latency_ns;
    double wall_clock_duration_s;
};

// Real-data HFTEngine class (processes external price/symbol-id batches)
struct HFTEngine {
    int num_symbols;
    int window;
    uint32_t window_mask;
    int lookback;

    std::vector<double> price_rb;
    std::vector<int32_t> write_idx;

    std::vector<int32_t> pairs;
    int n_pairs;

    std::vector<double> zsum;
    std::vector<double> zsq;
    bool zs_initialized;

    // stats
    uint64_t total_messages;
    double avg_latency_ns;
    double throughput_msg_sec;
    double wall_clock_avg_latency_ns;
    double wall_clock_duration_s;

    HFTEngine(int num_symbols_, int window_, int lookback_, uint64_t /*seed*/)
        : num_symbols(num_symbols_), window(window_), window_mask((uint32_t)(window_ - 1)), lookback(lookback_),
          price_rb((size_t)num_symbols_ * window_, 0.0), write_idx((size_t)num_symbols_, 0),
          n_pairs(0), zs_initialized(false), total_messages(0), avg_latency_ns(0.0), throughput_msg_sec(0.0),
          wall_clock_avg_latency_ns(0.0), wall_clock_duration_s(0.0) {}

    void set_pairs(pybind11::array_t<int32_t> pair_indices) {
        auto buf = pair_indices.request();
        int64_t size = buf.size;
        auto* data = static_cast<int32_t*>(buf.ptr);
        pairs.assign(data, data + size);
        n_pairs = (int)(pairs.size() / 2);
        zsum.assign((size_t)n_pairs, 0.0);
        zsq.assign((size_t)n_pairs, 0.0);
        zs_initialized = false;
    }

    // Fill zsum/zsumsq into provided arrays
    void fill_zstats(pybind11::array_t<double> zsum_out, pybind11::array_t<double> zsumsq_out) {
        auto zsum_buf = zsum_out.request();
        auto zsq_buf = zsumsq_out.request();
        if ((int)zsum_buf.size != n_pairs || (int)zsq_buf.size != n_pairs) {
            throw std::runtime_error("Output zstat sizes must equal n_pairs");
        }
        auto* zsum_ptr = static_cast<double*>(zsum_buf.ptr);
        auto* zsq_ptr = static_cast<double*>(zsq_buf.ptr);
        std::memcpy(zsum_ptr, zsum.data(), sizeof(double) * (size_t)n_pairs);
        std::memcpy(zsq_ptr, zsq.data(), sizeof(double) * (size_t)n_pairs);
    }

    // Process a batch of prices/symbol_ids from Python
    pybind11::dict process_batch(pybind11::array_t<double> prices, pybind11::array_t<int32_t> sids) {
        auto t0 = now_ns();

        auto p_buf = prices.request();
        auto s_buf = sids.request();
        if (p_buf.size != s_buf.size) throw std::runtime_error("prices and sids must be same length");
        int B = (int)p_buf.size;
        auto* p_ptr = static_cast<double*>(p_buf.ptr);
        auto* s_ptr = static_cast<int32_t*>(s_buf.ptr);

        // aggregate last update per symbol in this batch
        // reuse a small touched list to avoid clearing O(num_symbols) each batch
        std::vector<int32_t> touched; touched.reserve((size_t)B);
        std::vector<char> seen((size_t)num_symbols, 0);
        for (int i = 0; i < B; ++i) {
            int32_t sid = s_ptr[i];
            if ((uint32_t)sid >= (uint32_t)num_symbols) continue;
            if (!seen[(size_t)sid]) { touched.push_back(sid); seen[(size_t)sid] = 1; }
            // store last seen price in write buffer slot; we'll actually commit below
            // temporarily use price_rb to stage value (we will overwrite ring location below)
            // but better to store in a small map; here we directly commit incrementally
        }
        // flush to ring buffer: set last price for each touched
        for (int32_t sid : touched) {
            uint32_t idx = (uint32_t)write_idx[(size_t)sid];
            // find last occurrence price for sid by scanning backwards (cheap if few per sid)
            double lastp = 0.0; bool found=false;
            for (int i = B - 1; i >= 0; --i) {
                if (s_ptr[i] == sid) { lastp = p_ptr[i]; found = true; break; }
            }
            if (!found) continue;
            price_rb[(size_t)sid * window + idx] = lastp;
            write_idx[(size_t)sid] = (int32_t)((idx + 1) & window_mask);
        }

        // z-stats computation
        if (n_pairs > 0) {
            if (!zs_initialized) {
                for (int p = 0; p < n_pairs; ++p) {
                    int32_t idx1 = pairs[2 * p];
                    int32_t idx2 = pairs[2 * p + 1];
                    if ((uint32_t)idx1 >= (uint32_t)num_symbols || (uint32_t)idx2 >= (uint32_t)num_symbols) continue;
                    int32_t w1 = write_idx[(size_t)idx1];
                    int32_t w2 = write_idx[(size_t)idx2];
                    double s=0.0, ss=0.0;
                    for (int k = 0; k < lookback; ++k) {
                        int32_t j1 = (w1 - lookback + k + window) & window_mask;
                        int32_t j2 = (w2 - lookback + k + window) & window_mask;
                        double spread = price_rb[(size_t)idx1 * window + j1] - price_rb[(size_t)idx2 * window + j2];
                        s += spread; ss += spread * spread;
                    }
                    zsum[(size_t)p] = s; zsq[(size_t)p] = ss;
                }
                zs_initialized = true;
            } else {
                for (int p = 0; p < n_pairs; ++p) {
                    int32_t idx1 = pairs[2 * p];
                    int32_t idx2 = pairs[2 * p + 1];
                    int32_t w1 = write_idx[(size_t)idx1];
                    int32_t w2 = write_idx[(size_t)idx2];
                    int32_t new1 = (w1 - 1 + window) & window_mask;
                    int32_t new2 = (w2 - 1 + window) & window_mask;
                    int32_t old1 = (w1 - lookback + window) & window_mask;
                    int32_t old2 = (w2 - lookback + window) & window_mask;
                    double s_new = price_rb[(size_t)idx1 * window + new1] - price_rb[(size_t)idx2 * window + new2];
                    double s_old = price_rb[(size_t)idx1 * window + old1] - price_rb[(size_t)idx2 * window + old2];
                    zsum[(size_t)p] += (s_new - s_old);
                    zsq[(size_t)p] += (s_new * s_new - s_old * s_old);
                }
            }
        }

        auto t1 = now_ns();
        uint64_t batch_ns = (uint64_t)(t1 - t0);
        double prev_total = (double)total_messages;
        total_messages += (uint64_t)B;
        if (prev_total == 0.0) avg_latency_ns = (double)batch_ns;
        else avg_latency_ns = (avg_latency_ns * prev_total + (double)batch_ns) / (double)total_messages;

        pybind11::dict d;
        d["total_messages"] = total_messages;
        d["avg_latency_ns"] = avg_latency_ns;
        d["throughput_msg_sec"] = throughput_msg_sec; // filled by orchestrator if desired
        d["wall_clock_avg_latency_ns"] = wall_clock_avg_latency_ns;
        d["wall_clock_duration_s"] = wall_clock_duration_s;
        return d;
    }

    pybind11::dict get_stats() const {
        pybind11::dict d;
        d["total_messages"] = total_messages;
        d["avg_latency_ns"] = avg_latency_ns;
        d["throughput_msg_sec"] = throughput_msg_sec;
        d["wall_clock_avg_latency_ns"] = wall_clock_avg_latency_ns;
        d["wall_clock_duration_s"] = wall_clock_duration_s;
        return d;
    }

    // Expose internal buffers as NumPy arrays (read-only views)
    pybind11::array price_rb_view() {
        namespace py = pybind11;
        py::ssize_t shape[1] = { static_cast<py::ssize_t>(num_symbols * window) };
        py::ssize_t strides[1] = { static_cast<py::ssize_t>(sizeof(double)) };
        return py::array(py::buffer_info(
            price_rb.data(),                // ptr
            sizeof(double),                 // itemsize
            py::format_descriptor<double>::format(), // format
            1,                              // ndim
            { shape[0] },                   // shape
            { strides[0] }                  // strides
        ));
    }
    pybind11::array write_idx_view() {
        namespace py = pybind11;
        py::ssize_t shape[1] = { static_cast<py::ssize_t>(num_symbols) };
        py::ssize_t strides[1] = { static_cast<py::ssize_t>(sizeof(int32_t)) };
        return py::array(py::buffer_info(
            write_idx.data(),
            sizeof(int32_t),
            py::format_descriptor<int32_t>::format(),
            1,
            { shape[0] },
            { strides[0] }
        ));
    }
};

// Production main loop - minimal overhead
RunStats run_loop(
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
    
    // Initialize components
    XorShift128Plus rng(seed);
    BatchAggregator aggregator(num_symbols);
    
    // Pre-allocate batch arrays
    std::vector<int32_t> symbol_ids(batch_size);
    std::vector<double> prices(batch_size);
    
    // Timing setup
    uint64_t wall_start_ns = now_ns();
    uint64_t end_time_ns = wall_start_ns + uint64_t(duration_seconds * 1e9);
    uint64_t total_messages = 0;
    uint64_t total_cycles = 0;
    
    bool zs_initialized = false;
    int n_pairs = pair_indices.size() / 2;
    
    // Main processing loop
    while (now_ns() < end_time_ns) {
        uint64_t batch_start_cycles = rdtsc();
        
        // Data generation
        for (int i = 0; i < batch_size; ++i) {
            symbol_ids[i] = i % num_symbols;
            prices[i] = rng.normal_price(100.0, 1.0);
        }
        
        // Ring buffer update
        aggregator.reset();
        for (int i = 0; i < batch_size; ++i) {
            aggregator.add_update(symbol_ids[i], prices[i]);
        }
        aggregator.flush_to_ringbuffer(rb_ptr, widx_ptr, window, window_mask, num_symbols);
        
        // Z-score computation
        for (int pair_idx = 0; pair_idx < n_pairs; ++pair_idx) {
            int32_t idx1 = pairs_ptr[2 * pair_idx];
            int32_t idx2 = pairs_ptr[2 * pair_idx + 1];
            
            if (idx1 < 0 || idx2 < 0 || idx1 >= num_symbols || idx2 >= num_symbols) continue;
            
            int32_t w1 = widx_ptr[idx1];
            int32_t w2 = widx_ptr[idx2];
            
            if (!zs_initialized) {
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
        
        uint64_t batch_end_cycles = rdtsc();
        
        // Record measurements
        uint64_t batch_cycles = batch_end_cycles - batch_start_cycles;
        total_cycles += batch_cycles;
        total_messages += batch_size;
    }
    
    uint64_t wall_end_ns = now_ns();
    
    // Calculate final statistics
    RunStats stats;
    stats.total_messages = total_messages;
    stats.avg_latency_ns = cycles_to_ns(total_cycles) / double(total_messages);
    stats.throughput_msg_sec = double(total_messages) / duration_seconds;
    stats.wall_clock_duration_s = (wall_end_ns - wall_start_ns) / 1e9;
    stats.wall_clock_avg_latency_ns = (wall_end_ns - wall_start_ns) / double(total_messages);
    
    return stats;
}

// Python module definition
PYBIND11_MODULE(hft_core, m) {
    m.doc() = "Ultra-optimized HFT trading engine core";
    
    pybind11::class_<RunStats>(m, "RunStats")
        .def_readonly("total_messages", &RunStats::total_messages)
        .def_readonly("avg_latency_ns", &RunStats::avg_latency_ns)
        .def_readonly("throughput_msg_sec", &RunStats::throughput_msg_sec)
        .def_readonly("wall_clock_avg_latency_ns", &RunStats::wall_clock_avg_latency_ns)
        .def_readonly("wall_clock_duration_s", &RunStats::wall_clock_duration_s);
    
    // Expose real-data HFTEngine class
    pybind11::class_<HFTEngine>(m, "HFTEngine")
        .def(pybind11::init<int,int,int,uint64_t>(),
             pybind11::arg("num_symbols"), pybind11::arg("window"), pybind11::arg("lookback"), pybind11::arg("seed") = 0x12345678abcdefULL)
        .def("set_pairs", &HFTEngine::set_pairs)
        .def("process_batch", &HFTEngine::process_batch)
        .def("fill_zstats", &HFTEngine::fill_zstats)
        .def("get_stats", &HFTEngine::get_stats)
        .def_property_readonly("price_rb", &HFTEngine::price_rb_view)
        .def_property_readonly("write_idx", &HFTEngine::write_idx_view);

    m.def("run_loop", &run_loop, 
          "Production main loop",
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
