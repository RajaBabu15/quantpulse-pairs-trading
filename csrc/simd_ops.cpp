#include "quantpulse_core.h"
#include <cstring>
#include <cmath>

extern "C" {

// Platform-specific SIMD functions
namespace simd {

// Vector addition with platform detection
void vectorized_add(const double* a, const double* b, double* result, size_t n) {
#ifdef HAVE_X86_SIMD
    // AVX2 implementation for x86_64
    const size_t simd_width = 4;  // AVX2 processes 4 doubles at once
    const size_t simd_end = (n / simd_width) * simd_width;
    
    // Process in SIMD chunks
    for (size_t i = 0; i < simd_end; i += simd_width) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        __m256d vresult = _mm256_add_pd(va, vb);
        _mm256_storeu_pd(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (size_t i = simd_end; i < n; ++i) {
        result[i] = a[i] + b[i];
    }
#elif defined(HAVE_ARM_NEON)
    // NEON implementation for ARM64
    const size_t simd_width = 2;  // NEON processes 2 doubles at once
    const size_t simd_end = (n / simd_width) * simd_width;
    
    // Process in SIMD chunks
    for (size_t i = 0; i < simd_end; i += simd_width) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vb = vld1q_f64(&b[i]);
        float64x2_t vresult = vaddq_f64(va, vb);
        vst1q_f64(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (size_t i = simd_end; i < n; ++i) {
        result[i] = a[i] + b[i];
    }
#else
    // Scalar fallback
    for (size_t i = 0; i < n; ++i) {
        result[i] = a[i] + b[i];
    }
#endif
}

// Vector subtraction
void vectorized_subtract(const double* a, const double* b, double* result, size_t n) {
#ifdef HAVE_X86_SIMD
    const size_t simd_width = 4;
    const size_t simd_end = (n / simd_width) * simd_width;
    
    for (size_t i = 0; i < simd_end; i += simd_width) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        __m256d vresult = _mm256_sub_pd(va, vb);
        _mm256_storeu_pd(&result[i], vresult);
    }
    
    for (size_t i = simd_end; i < n; ++i) {
        result[i] = a[i] - b[i];
    }
#elif defined(HAVE_ARM_NEON)
    const size_t simd_width = 2;
    const size_t simd_end = (n / simd_width) * simd_width;
    
    for (size_t i = 0; i < simd_end; i += simd_width) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vb = vld1q_f64(&b[i]);
        float64x2_t vresult = vsubq_f64(va, vb);
        vst1q_f64(&result[i], vresult);
    }
    
    for (size_t i = simd_end; i < n; ++i) {
        result[i] = a[i] - b[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        result[i] = a[i] - b[i];
    }
#endif
}

// Vector multiplication
void vectorized_multiply(const double* a, const double* b, double* result, size_t n) {
#ifdef HAVE_X86_SIMD
    const size_t simd_width = 4;
    const size_t simd_end = (n / simd_width) * simd_width;
    
    for (size_t i = 0; i < simd_end; i += simd_width) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        __m256d vresult = _mm256_mul_pd(va, vb);
        _mm256_storeu_pd(&result[i], vresult);
    }
    
    for (size_t i = simd_end; i < n; ++i) {
        result[i] = a[i] * b[i];
    }
#elif defined(HAVE_ARM_NEON)
    const size_t simd_width = 2;
    const size_t simd_end = (n / simd_width) * simd_width;
    
    for (size_t i = 0; i < simd_end; i += simd_width) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vb = vld1q_f64(&b[i]);
        float64x2_t vresult = vmulq_f64(va, vb);
        vst1q_f64(&result[i], vresult);
    }
    
    for (size_t i = simd_end; i < n; ++i) {
        result[i] = a[i] * b[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        result[i] = a[i] * b[i];
    }
#endif
}

// Vector sum
double vectorized_sum(const double* arr, size_t n) {
    if (n == 0) return 0.0;
    
#ifdef HAVE_X86_SIMD
    const size_t simd_width = 4;
    const size_t simd_end = (n / simd_width) * simd_width;
    
    __m256d sum_vec = _mm256_setzero_pd();
    
    for (size_t i = 0; i < simd_end; i += simd_width) {
        __m256d va = _mm256_loadu_pd(&arr[i]);
        sum_vec = _mm256_add_pd(sum_vec, va);
    }
    
    // Extract and sum the 4 components
    double sum_array[4];
    _mm256_storeu_pd(sum_array, sum_vec);
    double total_sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
    
    // Add remaining elements
    for (size_t i = simd_end; i < n; ++i) {
        total_sum += arr[i];
    }
    
    return total_sum;
#elif defined(HAVE_ARM_NEON)
    const size_t simd_width = 2;
    const size_t simd_end = (n / simd_width) * simd_width;
    
    float64x2_t sum_vec = vdupq_n_f64(0.0);
    
    for (size_t i = 0; i < simd_end; i += simd_width) {
        float64x2_t va = vld1q_f64(&arr[i]);
        sum_vec = vaddq_f64(sum_vec, va);
    }
    
    // Extract and sum the 2 components
    double sum_array[2];
    vst1q_f64(sum_array, sum_vec);
    double total_sum = sum_array[0] + sum_array[1];
    
    // Add remaining elements
    for (size_t i = simd_end; i < n; ++i) {
        total_sum += arr[i];
    }
    
    return total_sum;
#else
    double total_sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        total_sum += arr[i];
    }
    return total_sum;
#endif
}

// Vector mean
double vectorized_mean(const double* arr, size_t n) {
    if (n == 0) return 0.0;
    return vectorized_sum(arr, n) / static_cast<double>(n);
}

// Vector standard deviation
double vectorized_std(const double* arr, size_t n, double mean) {
    if (n <= 1) return 0.0;
    
    double sum_sq = 0.0;
    
#ifdef HAVE_X86_SIMD
    const size_t simd_width = 4;
    const size_t simd_end = (n / simd_width) * simd_width;
    
    __m256d mean_vec = _mm256_set1_pd(mean);
    __m256d sum_sq_vec = _mm256_setzero_pd();
    
    for (size_t i = 0; i < simd_end; i += simd_width) {
        __m256d va = _mm256_loadu_pd(&arr[i]);
        __m256d diff = _mm256_sub_pd(va, mean_vec);
        __m256d diff_sq = _mm256_mul_pd(diff, diff);
        sum_sq_vec = _mm256_add_pd(sum_sq_vec, diff_sq);
    }
    
    double sum_sq_array[4];
    _mm256_storeu_pd(sum_sq_array, sum_sq_vec);
    sum_sq = sum_sq_array[0] + sum_sq_array[1] + sum_sq_array[2] + sum_sq_array[3];
    
    for (size_t i = simd_end; i < n; ++i) {
        double diff = arr[i] - mean;
        sum_sq += diff * diff;
    }
#elif defined(HAVE_ARM_NEON)
    const size_t simd_width = 2;
    const size_t simd_end = (n / simd_width) * simd_width;
    
    float64x2_t mean_vec = vdupq_n_f64(mean);
    float64x2_t sum_sq_vec = vdupq_n_f64(0.0);
    
    for (size_t i = 0; i < simd_end; i += simd_width) {
        float64x2_t va = vld1q_f64(&arr[i]);
        float64x2_t diff = vsubq_f64(va, mean_vec);
        float64x2_t diff_sq = vmulq_f64(diff, diff);
        sum_sq_vec = vaddq_f64(sum_sq_vec, diff_sq);
    }
    
    double sum_sq_array[2];
    vst1q_f64(sum_sq_array, sum_sq_vec);
    sum_sq = sum_sq_array[0] + sum_sq_array[1];
    
    for (size_t i = simd_end; i < n; ++i) {
        double diff = arr[i] - mean;
        sum_sq += diff * diff;
    }
#else
    for (size_t i = 0; i < n; ++i) {
        double diff = arr[i] - mean;
        sum_sq += diff * diff;
    }
#endif
    
    return std::sqrt(sum_sq / (n - 1));
}

} // namespace simd

// Non-SIMD wrapper functions for the C interface
void simd_vector_add(const double* a, const double* b, double* result, size_t n) {
    simd::vectorized_add(a, b, result, n);
}

void simd_vector_subtract(const double* a, const double* b, double* result, size_t n) {
    simd::vectorized_subtract(a, b, result, n);
}

void simd_vector_multiply(const double* a, const double* b, double* result, size_t n) {
    simd::vectorized_multiply(a, b, result, n);
}

double simd_vector_sum(const double* arr, size_t n) {
    return simd::vectorized_sum(arr, n);
}

double simd_vector_mean(const double* arr, size_t n) {
    return simd::vectorized_mean(arr, n);
}

double simd_vector_std(const double* arr, size_t n, double mean) {
    return simd::vectorized_std(arr, n, mean);
}

} // extern "C"
