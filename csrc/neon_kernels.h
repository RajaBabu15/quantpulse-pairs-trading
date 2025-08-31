#pragma once

#if defined(__aarch64__) || defined(_M_ARM64)

// Minimal NEON kernel declarations
void neon_generate_prices(double* prices, int32_t* symbol_ids, int batch_size, 
                         int num_symbols, uint64_t& s0, uint64_t& s1);

void neon_zscore_batch(double* rb_ptr, int32_t* widx_ptr, int32_t* pairs_ptr,
                      double* zsum_ptr, double* zsq_ptr, int n_pairs, 
                      int num_symbols, int window, uint32_t window_mask, int lookback);

void neon_zscore_incremental(double* rb_ptr, int32_t* widx_ptr, int32_t* pairs_ptr,
                            double* zsum_ptr, double* zsq_ptr, int n_pairs,
                            int num_symbols, int window, uint32_t window_mask, int lookback);

#endif
