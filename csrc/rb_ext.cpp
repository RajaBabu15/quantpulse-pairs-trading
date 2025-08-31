#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>

using namespace pybind11;

void rb_update(
    pybind11::array_t<double, array::c_style|array::forcecast> price_rb, // (S, W)
    pybind11::array_t<int32_t, array::c_style|array::forcecast> write_idx, // (S,)
    pybind11::array_t<int32_t, array::c_style|array::forcecast> symbol_ids, // (N,)
    pybind11::array_t<double,  array::c_style|array::forcecast> prices,     // (N,)
    int width,              // must be power of two
    uint32_t width_mask     // == width-1
) {
    auto rb   = price_rb.mutable_unchecked<2>();
    auto widx = write_idx.mutable_unchecked<1>();
    auto sids = symbol_ids.unchecked<1>();
    auto pr   = prices.unchecked<1>();

    const ssize_t N = sids.shape(0);
    const ssize_t S = rb.shape(0);

    for (ssize_t i = 0; i < N; ++i) {
        int32_t sid = sids(i);
        if ((uint32_t)sid >= (uint32_t)S) continue;
        uint32_t idx = (uint32_t)widx(sid);

        // prefetch next line to reduce miss on scattered access
        // (safe even if idx+1 wraps; the mask corrects actual write)
#if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(&rb(sid, (idx+1) & width_mask), 1, 3);
#endif

        rb(sid, idx) = pr(i);
        widx(sid) = (int32_t)((idx + 1) & width_mask);
    }
}

// Batch version for better cache utilization when symbols are clustered
void rb_update_batched(
    pybind11::array_t<double, array::c_style|array::forcecast> price_rb,
    pybind11::array_t<int32_t, array::c_style|array::forcecast> write_idx,
    pybind11::array_t<int32_t, array::c_style|array::forcecast> symbol_ids,
    pybind11::array_t<double,  array::c_style|array::forcecast> prices,
    int width,
    uint32_t width_mask
) {
    auto rb   = price_rb.mutable_unchecked<2>();
    auto widx = write_idx.mutable_unchecked<1>();
    auto sids = symbol_ids.unchecked<1>();
    auto pr   = prices.unchecked<1>();

    const ssize_t N = sids.shape(0);
    const ssize_t S = rb.shape(0);

    // Process in blocks to improve cache locality
    constexpr ssize_t BLOCK_SIZE = 64; // Adjust based on cache line size
    
    for (ssize_t block_start = 0; block_start < N; block_start += BLOCK_SIZE) {
        ssize_t block_end = std::min(block_start + BLOCK_SIZE, N);
        
        for (ssize_t i = block_start; i < block_end; ++i) {
            int32_t sid = sids(i);
            if ((uint32_t)sid >= (uint32_t)S) continue;
            uint32_t idx = (uint32_t)widx(sid);

            // Prefetch for next iteration in block
            if (i + 1 < block_end) {
                int32_t next_sid = sids(i + 1);
                if ((uint32_t)next_sid < (uint32_t)S) {
#if defined(__GNUC__) || defined(__clang__)
                    __builtin_prefetch(&rb(next_sid, 0), 1, 1);
#endif
                }
            }

            rb(sid, idx) = pr(i);
            widx(sid) = (int32_t)((idx + 1) & width_mask);
        }
    }
}

PYBIND11_MODULE(nano_rb, m) {
    m.doc() = "Ultra-fast ring buffer updates for HFT";
    m.def("rb_update", &rb_update, "Branchless ring buffer update with bitmask",
          pybind11::arg("price_rb"), pybind11::arg("write_idx"), pybind11::arg("symbol_ids"), 
          pybind11::arg("prices"), pybind11::arg("width"), pybind11::arg("width_mask"));
    m.def("rb_update_batched", &rb_update_batched, "Batched ring buffer update with improved cache locality",
          pybind11::arg("price_rb"), pybind11::arg("write_idx"), pybind11::arg("symbol_ids"), 
          pybind11::arg("prices"), pybind11::arg("width"), pybind11::arg("width_mask"));
}
