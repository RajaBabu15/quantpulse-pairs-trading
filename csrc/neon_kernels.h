#pragma once

#include <cstdint>
#include <arm_neon.h>

// ============================================================================
// NEON-accelerated kernels for Apple M4 ARM64
// ============================================================================

#if defined(__aarch64__) || defined(_M_ARM64)

// Vectorized zscore computation using NEON intrinsics
static inline void neon_zscore_batch(
    const double* __restrict rb_data,
    const int32_t* __restrict write_indices,
    const int32_t* __restrict pair_indices,
    double* __restrict zsum,
    double* __restrict zsumsq,
    int num_pairs,
    int num_symbols,
    int window,
    uint32_t window_mask,
    int lookback
) {
    // Process pairs in groups of 2 (4 doubles per iteration with NEON)
    int pairs_simd = (num_pairs / 2) * 2;
    
    for (int pair_idx = 0; pair_idx < pairs_simd; pair_idx += 2) {
        // Load pair indices
        int32_t idx1_a = pair_indices[2 * pair_idx];
        int32_t idx2_a = pair_indices[2 * pair_idx + 1];
        int32_t idx1_b = pair_indices[2 * (pair_idx + 1)];
        int32_t idx2_b = pair_indices[2 * (pair_idx + 1) + 1];
        
        // Get write positions
        int32_t w1_a = write_indices[idx1_a];
        int32_t w2_a = write_indices[idx2_a];
        int32_t w1_b = write_indices[idx1_b];
        int32_t w2_b = write_indices[idx2_b];
        
        // Initialize NEON accumulators
        float64x2_t sum_vec = vdupq_n_f64(0.0);
        float64x2_t sumsq_vec = vdupq_n_f64(0.0);
        
        // Vectorized accumulation over lookback window
        for (int k = 0; k < lookback; k += 1) {
            // Calculate indices with wrap-around
            int32_t j1_a = (w1_a - lookback + k + window) & window_mask;
            int32_t j2_a = (w2_a - lookback + k + window) & window_mask;
            int32_t j1_b = (w1_b - lookback + k + window) & window_mask;
            int32_t j2_b = (w2_b - lookback + k + window) & window_mask;
            
            // Load prices
            double p1_a = rb_data[(size_t)idx1_a * window + j1_a];
            double p2_a = rb_data[(size_t)idx2_a * window + j2_a];
            double p1_b = rb_data[(size_t)idx1_b * window + j1_b];
            double p2_b = rb_data[(size_t)idx2_b * window + j2_b];
            
            // Compute spreads
            double temp_p1[2] = {p1_a, p1_b};
            double temp_p2[2] = {p2_a, p2_b};
            float64x2_t spreads = vsubq_f64(
                vld1q_f64(temp_p1),
                vld1q_f64(temp_p2)
            );
            
            // Accumulate sum and sum-of-squares
            sum_vec = vaddq_f64(sum_vec, spreads);
            sumsq_vec = vfmaq_f64(sumsq_vec, spreads, spreads);  // FMA: sumsq += spreads * spreads
        }
        
        // Store results
        vst1q_f64(&zsum[pair_idx], sum_vec);
        vst1q_f64(&zsumsq[pair_idx], sumsq_vec);
    }
    
    // Handle remaining pairs (scalar fallback)
    for (int pair_idx = pairs_simd; pair_idx < num_pairs; ++pair_idx) {
        int32_t idx1 = pair_indices[2 * pair_idx];
        int32_t idx2 = pair_indices[2 * pair_idx + 1];
        
        if (idx1 < 0 || idx2 < 0 || idx1 >= num_symbols || idx2 >= num_symbols) continue;
        
        int32_t w1 = write_indices[idx1];
        int32_t w2 = write_indices[idx2];
        
        double sum = 0.0, sumsq = 0.0;
        for (int k = 0; k < lookback; ++k) {
            int32_t j1 = (w1 - lookback + k + window) & window_mask;
            int32_t j2 = (w2 - lookback + k + window) & window_mask;
            
            double p1 = rb_data[(size_t)idx1 * window + j1];
            double p2 = rb_data[(size_t)idx2 * window + j2];
            double spread = p1 - p2;
            
            sum += spread;
            sumsq += spread * spread;
        }
        
        zsum[pair_idx] = sum;
        zsumsq[pair_idx] = sumsq;
    }
}

// Vectorized incremental zscore update using NEON
static inline void neon_zscore_incremental(
    const double* __restrict rb_data,
    const int32_t* __restrict write_indices,
    const int32_t* __restrict pair_indices,
    double* __restrict zsum,
    double* __restrict zsumsq,
    int num_pairs,
    int num_symbols,
    int window,
    uint32_t window_mask,
    int lookback
) {
    // Process pairs in groups of 2 for NEON vectorization
    int pairs_simd = (num_pairs / 2) * 2;
    
    for (int pair_idx = 0; pair_idx < pairs_simd; pair_idx += 2) {
        // Load pair indices
        int32_t idx1_a = pair_indices[2 * pair_idx];
        int32_t idx2_a = pair_indices[2 * pair_idx + 1];
        int32_t idx1_b = pair_indices[2 * (pair_idx + 1)];
        int32_t idx2_b = pair_indices[2 * (pair_idx + 1) + 1];
        
        // Get write positions
        int32_t w1_a = write_indices[idx1_a];
        int32_t w2_a = write_indices[idx2_a];
        int32_t w1_b = write_indices[idx1_b];
        int32_t w2_b = write_indices[idx2_b];
        
        // Calculate new and old indices
        int32_t new1_a = (w1_a - 1 + window) & window_mask;
        int32_t new2_a = (w2_a - 1 + window) & window_mask;
        int32_t old1_a = (w1_a - lookback + window) & window_mask;
        int32_t old2_a = (w2_a - lookback + window) & window_mask;
        
        int32_t new1_b = (w1_b - 1 + window) & window_mask;
        int32_t new2_b = (w2_b - 1 + window) & window_mask;
        int32_t old1_b = (w1_b - lookback + window) & window_mask;
        int32_t old2_b = (w2_b - lookback + window) & window_mask;
        
        // Load new prices
        double temp_new1[2] = {
            rb_data[(size_t)idx1_a * window + new1_a],
            rb_data[(size_t)idx1_b * window + new1_b]
        };
        double temp_new2[2] = {
            rb_data[(size_t)idx2_a * window + new2_a],
            rb_data[(size_t)idx2_b * window + new2_b]
        };
        float64x2_t new_prices1 = vld1q_f64(temp_new1);
        float64x2_t new_prices2 = vld1q_f64(temp_new2);
        
        // Load old prices
        double temp_old1[2] = {
            rb_data[(size_t)idx1_a * window + old1_a],
            rb_data[(size_t)idx1_b * window + old1_b]
        };
        double temp_old2[2] = {
            rb_data[(size_t)idx2_a * window + old2_a],
            rb_data[(size_t)idx2_b * window + old2_b]
        };
        float64x2_t old_prices1 = vld1q_f64(temp_old1);
        float64x2_t old_prices2 = vld1q_f64(temp_old2);
        
        // Calculate spreads
        float64x2_t new_spreads = vsubq_f64(new_prices1, new_prices2);
        float64x2_t old_spreads = vsubq_f64(old_prices1, old_prices2);
        float64x2_t spread_delta = vsubq_f64(new_spreads, old_spreads);
        
        // Load current sums
        float64x2_t current_sum = vld1q_f64(&zsum[pair_idx]);
        float64x2_t current_sumsq = vld1q_f64(&zsumsq[pair_idx]);
        
        // Update sums
        float64x2_t new_sum = vaddq_f64(current_sum, spread_delta);
        
        // Update sum-of-squares: sumsq += new_spread^2 - old_spread^2
        float64x2_t new_sq_delta = vfmsq_f64(
            vmulq_f64(new_spreads, new_spreads),
            old_spreads, old_spreads
        );
        float64x2_t new_sumsq = vaddq_f64(current_sumsq, new_sq_delta);
        
        // Store results
        vst1q_f64(&zsum[pair_idx], new_sum);
        vst1q_f64(&zsumsq[pair_idx], new_sumsq);
    }
    
    // Handle remaining pairs (scalar fallback)
    for (int pair_idx = pairs_simd; pair_idx < num_pairs; ++pair_idx) {
        int32_t idx1 = pair_indices[2 * pair_idx];
        int32_t idx2 = pair_indices[2 * pair_idx + 1];
        
        if (idx1 < 0 || idx2 < 0 || idx1 >= num_symbols || idx2 >= num_symbols) continue;
        
        int32_t w1 = write_indices[idx1];
        int32_t w2 = write_indices[idx2];
        
        int32_t new1 = (w1 - 1 + window) & window_mask;
        int32_t new2 = (w2 - 1 + window) & window_mask;
        int32_t old1 = (w1 - lookback + window) & window_mask;
        int32_t old2 = (w2 - lookback + window) & window_mask;
        
        double s_new = rb_data[(size_t)idx1 * window + new1] - rb_data[(size_t)idx2 * window + new2];
        double s_old = rb_data[(size_t)idx1 * window + old1] - rb_data[(size_t)idx2 * window + old2];
        
        zsum[pair_idx] += (s_new - s_old);
        zsumsq[pair_idx] += (s_new * s_new - s_old * s_old);
    }
}

// Vectorized correlation coefficient computation
static inline void neon_correlation_batch(
    const double* __restrict rb_data,
    const int32_t* __restrict write_indices,
    const int32_t* __restrict pair_indices,
    double* __restrict csx,
    double* __restrict csy,
    double* __restrict csxx,
    double* __restrict csyy,
    double* __restrict csxy,
    int num_pairs,
    int num_symbols,
    int window,
    uint32_t window_mask,
    int lookback
) {
    // Process pairs in groups of 2 for NEON
    int pairs_simd = (num_pairs / 2) * 2;
    
    for (int pair_idx = 0; pair_idx < pairs_simd; pair_idx += 2) {
        int32_t idx1_a = pair_indices[2 * pair_idx];
        int32_t idx2_a = pair_indices[2 * pair_idx + 1];
        int32_t idx1_b = pair_indices[2 * (pair_idx + 1)];
        int32_t idx2_b = pair_indices[2 * (pair_idx + 1) + 1];
        
        int32_t w1_a = write_indices[idx1_a];
        int32_t w2_a = write_indices[idx2_a];
        int32_t w1_b = write_indices[idx1_b];
        int32_t w2_b = write_indices[idx2_b];
        
        // Initialize NEON accumulators
        float64x2_t sx_vec = vdupq_n_f64(0.0);
        float64x2_t sy_vec = vdupq_n_f64(0.0);
        float64x2_t sxx_vec = vdupq_n_f64(0.0);
        float64x2_t syy_vec = vdupq_n_f64(0.0);
        float64x2_t sxy_vec = vdupq_n_f64(0.0);
        
        // Vectorized accumulation
        for (int k = 0; k < lookback; ++k) {
            int32_t j1_a = (w1_a - lookback + k + window) & window_mask;
            int32_t j2_a = (w2_a - lookback + k + window) & window_mask;
            int32_t j1_b = (w1_b - lookback + k + window) & window_mask;
            int32_t j2_b = (w2_b - lookback + k + window) & window_mask;
            
            // Load prices into NEON vectors
            double temp_x[2] = {
                rb_data[(size_t)idx1_a * window + j1_a],
                rb_data[(size_t)idx1_b * window + j1_b]
            };
            double temp_y[2] = {
                rb_data[(size_t)idx2_a * window + j2_a],
                rb_data[(size_t)idx2_b * window + j2_b]
            };
            float64x2_t x_vals = vld1q_f64(temp_x);
            float64x2_t y_vals = vld1q_f64(temp_y);
            
            // Accumulate correlation components
            sx_vec = vaddq_f64(sx_vec, x_vals);
            sy_vec = vaddq_f64(sy_vec, y_vals);
            sxx_vec = vfmaq_f64(sxx_vec, x_vals, x_vals);  // sxx += x * x
            syy_vec = vfmaq_f64(syy_vec, y_vals, y_vals);  // syy += y * y
            sxy_vec = vfmaq_f64(sxy_vec, x_vals, y_vals);  // sxy += x * y
        }
        
        // Store results
        vst1q_f64(&csx[pair_idx], sx_vec);
        vst1q_f64(&csy[pair_idx], sy_vec);
        vst1q_f64(&csxx[pair_idx], sxx_vec);
        vst1q_f64(&csyy[pair_idx], syy_vec);
        vst1q_f64(&csxy[pair_idx], sxy_vec);
    }
    
    // Handle remaining pairs (scalar)
    for (int pair_idx = pairs_simd; pair_idx < num_pairs; ++pair_idx) {
        int32_t idx1 = pair_indices[2 * pair_idx];
        int32_t idx2 = pair_indices[2 * pair_idx + 1];
        
        if (idx1 < 0 || idx2 < 0 || idx1 >= num_symbols || idx2 >= num_symbols) continue;
        
        int32_t w1 = write_indices[idx1];
        int32_t w2 = write_indices[idx2];
        
        double sx = 0.0, sy = 0.0, sxx = 0.0, syy = 0.0, sxy = 0.0;
        
        for (int k = 0; k < lookback; ++k) {
            int32_t j1 = (w1 - lookback + k + window) & window_mask;
            int32_t j2 = (w2 - lookback + k + window) & window_mask;
            
            double x = rb_data[(size_t)idx1 * window + j1];
            double y = rb_data[(size_t)idx2 * window + j2];
            
            sx += x;
            sy += y;
            sxx += x * x;
            syy += y * y;
            sxy += x * y;
        }
        
        csx[pair_idx] = sx;
        csy[pair_idx] = sy;
        csxx[pair_idx] = sxx;
        csyy[pair_idx] = syy;
        csxy[pair_idx] = sxy;
    }
}

// Fast synthetic data generation using NEON
static inline void neon_generate_prices(
    double* __restrict prices,
    int32_t* __restrict symbol_ids,
    int batch_size,
    int num_symbols,
    uint64_t& rng_state0,
    uint64_t& rng_state1
) {
    // Generate in groups of 2 doubles
    int batch_simd = (batch_size / 2) * 2;
    
    for (int i = 0; i < batch_simd; i += 2) {
        // XorShift128+ for 2 random numbers
        uint64_t x = rng_state0;
        uint64_t y = rng_state1;
        rng_state0 = y;
        x ^= x << 23;
        rng_state1 = (x ^ y ^ (x >> 17) ^ (y >> 26));
        uint64_t r1 = rng_state1 + y;
        
        x = rng_state0;
        y = rng_state1;
        rng_state0 = y;
        x ^= x << 23;
        rng_state1 = (x ^ y ^ (x >> 17) ^ (y >> 26));
        uint64_t r2 = rng_state1 + y;
        
        // Convert to [0,1) uniform
        uint64_t temp_rands[2] = {r1, r2};
        float64x2_t uniform = vmulq_f64(
            vcvtq_f64_u64(vshrq_n_u64(vld1q_u64(temp_rands), 11)),
            vdupq_n_f64(1.0 / 9007199254740992.0)
        );
        
        // Convert to normal(100, 1) prices
        float64x2_t normal_prices = vfmaq_f64(
            vdupq_n_f64(100.0),
            vsubq_f64(uniform, vdupq_n_f64(0.5)),
            vdupq_n_f64(6.0)  // 6-sigma range
        );
        
        // Store prices
        vst1q_f64(&prices[i], normal_prices);
        
        // Generate symbol IDs
        symbol_ids[i] = i % num_symbols;
        symbol_ids[i + 1] = (i + 1) % num_symbols;
    }
    
    // Handle remaining items (scalar)
    for (int i = batch_simd; i < batch_size; ++i) {
        uint64_t x = rng_state0;
        uint64_t y = rng_state1;
        rng_state0 = y;
        x ^= x << 23;
        rng_state1 = (x ^ y ^ (x >> 17) ^ (y >> 26));
        uint64_t r = rng_state1 + y;
        
        double u = (r >> 11) * (1.0 / 9007199254740992.0);
        prices[i] = 100.0 + (u - 0.5) * 6.0;
        symbol_ids[i] = i % num_symbols;
    }
}

#else
// Fallback implementations for non-ARM64 platforms
#define neon_zscore_batch(...) /* fallback to scalar */
#define neon_zscore_incremental(...) /* fallback to scalar */
#define neon_generate_prices(...) /* fallback to scalar */
#endif
