#include "quantpulse_core.h"
#include <cstring>

extern "C" {

namespace simd {

// AVX2-optimized vector subtraction
void vectorized_subtract(const double* a, const double* b, double* result, size_t n) {
    const size_t simd_width = 4; // AVX2 processes 4 doubles at once
    size_t simd_end = n - (n % simd_width);
    
    // Process SIMD-aligned chunks
    for(size_t i = 0; i < simd_end; i += simd_width) {
        __m256d va = _mm256_load_pd(&a[i]);
        __m256d vb = _mm256_load_pd(&b[i]);
        __m256d vr = _mm256_sub_pd(va, vb);
        _mm256_store_pd(&result[i], vr);
    }
    
    // Handle remaining elements
    for(size_t i = simd_end; i < n; ++i) {
        result[i] = a[i] - b[i];
    }
}

// AVX2-optimized vector multiplication
void vectorized_multiply(const double* a, const double* b, double* result, size_t n) {
    const size_t simd_width = 4;
    size_t simd_end = n - (n % simd_width);
    
    for(size_t i = 0; i < simd_end; i += simd_width) {
        __m256d va = _mm256_load_pd(&a[i]);
        __m256d vb = _mm256_load_pd(&b[i]);
        __m256d vr = _mm256_mul_pd(va, vb);
        _mm256_store_pd(&result[i], vr);
    }
    
    for(size_t i = simd_end; i < n; ++i) {
        result[i] = a[i] * b[i];
    }
}

// AVX2-optimized vector addition
void vectorized_add(const double* a, const double* b, double* result, size_t n) {
    const size_t simd_width = 4;
    size_t simd_end = n - (n % simd_width);
    
    for(size_t i = 0; i < simd_end; i += simd_width) {
        __m256d va = _mm256_load_pd(&a[i]);
        __m256d vb = _mm256_load_pd(&b[i]);
        __m256d vr = _mm256_add_pd(va, vb);
        _mm256_store_pd(&result[i], vr);
    }
    
    for(size_t i = simd_end; i < n; ++i) {
        result[i] = a[i] + b[i];
    }
}

// AVX2-optimized sum reduction
double vectorized_sum(const double* arr, size_t n) {
    const size_t simd_width = 4;
    size_t simd_end = n - (n % simd_width);
    
    __m256d sum_vec = _mm256_setzero_pd();
    
    // Process SIMD chunks
    for(size_t i = 0; i < simd_end; i += simd_width) {
        __m256d va = _mm256_load_pd(&arr[i]);
        sum_vec = _mm256_add_pd(sum_vec, va);
    }
    
    // Extract and sum the 4 components
    double sum_array[4];
    _mm256_store_pd(sum_array, sum_vec);
    double total_sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
    
    // Add remaining elements
    for(size_t i = simd_end; i < n; ++i) {
        total_sum += arr[i];
    }
    
    return total_sum;
}

// Vectorized mean calculation
double vectorized_mean(const double* arr, size_t n) {
    if(n == 0) return 0.0;
    return vectorized_sum(arr, n) / static_cast<double>(n);
}

// Vectorized standard deviation
double vectorized_std(const double* arr, size_t n, double mean) {
    if(n <= 1) return 0.0;
    
    const size_t simd_width = 4;
    size_t simd_end = n - (n % simd_width);
    
    __m256d mean_vec = _mm256_set1_pd(mean);
    __m256d sum_sq_vec = _mm256_setzero_pd();
    
    // Process SIMD chunks
    for(size_t i = 0; i < simd_end; i += simd_width) {
        __m256d va = _mm256_load_pd(&arr[i]);
        __m256d diff = _mm256_sub_pd(va, mean_vec);
        __m256d diff_sq = _mm256_mul_pd(diff, diff);
        sum_sq_vec = _mm256_add_pd(sum_sq_vec, diff_sq);
    }
    
    // Extract sum of squares
    double sum_sq_array[4];
    _mm256_store_pd(sum_sq_array, sum_sq_vec);
    double total_sum_sq = sum_sq_array[0] + sum_sq_array[1] + sum_sq_array[2] + sum_sq_array[3];
    
    // Add remaining elements
    for(size_t i = simd_end; i < n; ++i) {
        double diff = arr[i] - mean;
        total_sum_sq += diff * diff;
    }
    
    return std::sqrt(total_sum_sq / (n - 1));
}

} // namespace simd

// Cache-optimized RMSprop implementation
struct RMSpropState {
    CACHE_ALIGNED double squared_gradients[32];
    CACHE_ALIGNED double gradients[32];
    double learning_rate;
    double decay;
    double epsilon;
    int param_count;
};

// Thread-safe RMSprop step with vectorization
void rmsprop_step_vectorized(RMSpropState* state, const double* params, 
                           double* new_params, double (*objective_func)(const double*)) {
    const double delta = 1e-5;
    const int n = state->param_count;
    
    // Compute gradient numerically using SIMD
    #pragma omp parallel for
    for(int i = 0; i < n; ++i) {
        CACHE_ALIGNED double params_plus[32];
        CACHE_ALIGNED double params_minus[32];
        
        // Copy parameters
        std::memcpy(params_plus, params, n * sizeof(double));
        std::memcpy(params_minus, params, n * sizeof(double));
        
        params_plus[i] += delta;
        params_minus[i] -= delta;
        
        double f_plus = objective_func(params_plus);
        double f_minus = objective_func(params_minus);
        
        state->gradients[i] = (f_plus - f_minus) / (2.0 * delta);
    }
    
    // Update squared gradients and parameters using SIMD
    const size_t simd_width = 4;
    size_t simd_end = n - (n % simd_width);
    
    __m256d decay_vec = _mm256_set1_pd(state->decay);
    __m256d one_minus_decay_vec = _mm256_set1_pd(1.0 - state->decay);
    __m256d lr_vec = _mm256_set1_pd(state->learning_rate);
    __m256d eps_vec = _mm256_set1_pd(state->epsilon);
    
    for(size_t i = 0; i < simd_end; i += simd_width) {
        // Load current state
        __m256d sq_grad = _mm256_load_pd(&state->squared_gradients[i]);
        __m256d grad = _mm256_load_pd(&state->gradients[i]);
        __m256d param = _mm256_load_pd(&params[i]);
        
        // Update squared gradients
        __m256d grad_sq = _mm256_mul_pd(grad, grad);
        sq_grad = _mm256_add_pd(
            _mm256_mul_pd(decay_vec, sq_grad),
            _mm256_mul_pd(one_minus_decay_vec, grad_sq)
        );
        
        // Compute parameter update
        __m256d sqrt_sq_grad = _mm256_sqrt_pd(_mm256_add_pd(sq_grad, eps_vec));
        __m256d update = _mm256_div_pd(_mm256_mul_pd(lr_vec, grad), sqrt_sq_grad);
        
        // Update parameters
        __m256d new_param = _mm256_sub_pd(param, update);
        
        // Store results
        _mm256_store_pd(&state->squared_gradients[i], sq_grad);
        _mm256_store_pd(&new_params[i], new_param);
    }
    
    // Handle remaining parameters
    for(size_t i = simd_end; i < n; ++i) {
        double grad = state->gradients[i];
        state->squared_gradients[i] = state->decay * state->squared_gradients[i] + 
                                     (1.0 - state->decay) * grad * grad;
        
        double update = state->learning_rate * grad / 
                       (std::sqrt(state->squared_gradients[i]) + state->epsilon);
        
        new_params[i] = params[i] - update;
    }
}

// Batch-optimized backtest for multiple parameter sets
void batch_backtest_vectorized(const double* prices1, const double* prices2, size_t n,
                              const double* param_batches, int batch_size, int param_count,
                              double* results) {
    #pragma omp parallel for
    for(int batch = 0; batch < batch_size; ++batch) {
        const double* params = &param_batches[batch * param_count];
        
        TradingParameters tp{
            static_cast<int>(params[0]), params[1], params[2], 
            static_cast<int>(params[3]), params[4], params[5], params[6]
        };
        
        BacktestResult result = vectorized_backtest(prices1, prices2, n, tp);
        results[batch] = result.final_pnl;
    }
}

} // extern "C"
