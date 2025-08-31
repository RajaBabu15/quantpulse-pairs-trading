#pragma once

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>
#include <cstdint>
#include <cmath>
#include <algorithm>

// Minimal NEON kernel declarations + definitions
static inline void neon_generate_prices(double* prices, int32_t* symbol_ids, int batch_size,
                                       int num_symbols, uint64_t& s0, uint64_t& s1) {
    // Xorshift128+; vectorized generation could be added, but this is already fast under -O3
    for (int i = 0; i < batch_size; ++i) {
        uint64_t x = s0;
        uint64_t const y = s1;
        s0 = y;
        x ^= x << 23;
        s1 = (x ^ y ^ (x >> 17) ^ (y >> 26));
        uint64_t r = s1 + y;
        double u = (r >> 11) * (1.0 / 9007199254740992.0); // 2^53
        prices[i] = 100.0 + (u - 0.5) * 6.0;
        symbol_ids[i] = i % std::max(1, num_symbols);
    }
}

static inline void neon_zscore_batch(double* rb_ptr, int32_t* widx_ptr, int32_t* pairs_ptr,
                                     double* zsum_ptr, double* zsq_ptr, int n_pairs,
                                     int num_symbols, int window, uint32_t window_mask, int lookback) {
    for (int p = 0; p < n_pairs; ++p) {
        int32_t idx1 = pairs_ptr[2 * p];
        int32_t idx2 = pairs_ptr[2 * p + 1];
        if (idx1 < 0 || idx2 < 0 || idx1 >= num_symbols || idx2 >= num_symbols) continue;
        int32_t w1 = widx_ptr[idx1];
        int32_t w2 = widx_ptr[idx2];
        int32_t j1 = (w1 - lookback + window) & (int32_t)window_mask;
        int32_t j2 = (w2 - lookback + window) & (int32_t)window_mask;
        double sum = 0.0, sumsq = 0.0;
        for (int k = 0; k < lookback; ++k) {
            double p1 = rb_ptr[(size_t)idx1 * window + j1];
            double p2 = rb_ptr[(size_t)idx2 * window + j2];
            double spread = p1 - p2;
            sum += spread;
            sumsq += spread * spread;
            j1 = (j1 + 1) & (int32_t)window_mask;
            j2 = (j2 + 1) & (int32_t)window_mask;
        }
        zsum_ptr[p] = sum;
        zsq_ptr[p] = sumsq;
    }
}

static inline void neon_zscore_incremental(double* rb_ptr, int32_t* widx_ptr, int32_t* pairs_ptr,
                                           double* zsum_ptr, double* zsq_ptr, int n_pairs,
                                           int num_symbols, int window, uint32_t window_mask, int lookback) {
    for (int p = 0; p < n_pairs; ++p) {
        int32_t idx1 = pairs_ptr[2 * p];
        int32_t idx2 = pairs_ptr[2 * p + 1];
        if (idx1 < 0 || idx2 < 0 || idx1 >= num_symbols || idx2 >= num_symbols) continue;
        int32_t w1 = widx_ptr[idx1];
        int32_t w2 = widx_ptr[idx2];
        int32_t new1 = (w1 - 1 + window) & (int32_t)window_mask;
        int32_t new2 = (w2 - 1 + window) & (int32_t)window_mask;
        int32_t old1 = (w1 - lookback + window) & (int32_t)window_mask;
        int32_t old2 = (w2 - lookback + window) & (int32_t)window_mask;
        double s_new = rb_ptr[(size_t)idx1 * window + new1] - rb_ptr[(size_t)idx2 * window + new2];
        double s_old = rb_ptr[(size_t)idx1 * window + old1] - rb_ptr[(size_t)idx2 * window + old2];
        zsum_ptr[p] += (s_new - s_old);
        zsq_ptr[p] += (s_new * s_new - s_old * s_old);
    }
}

#endif
