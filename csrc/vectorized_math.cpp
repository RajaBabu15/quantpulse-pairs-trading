#include "quantpulse_core.h"
#include <cstring>
#include <cmath>
#include <arm_neon.h>
namespace simd {
void vectorized_add(const double* a, const double* b, double* result, size_t n) {
    const size_t simd_width = 2;
    const size_t simd_end = (n / simd_width) * simd_width;
    for (size_t i = 0; i < simd_end; i += simd_width) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vb = vld1q_f64(&b[i]);
        float64x2_t vresult = vaddq_f64(va, vb);
        vst1q_f64(&result[i], vresult);
    }
    for (size_t i = simd_end; i < n; ++i) {
        result[i] = a[i] + b[i];
    }
}
void vectorized_subtract(const double* a, const double* b, double* result, size_t n) {
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
}
void vectorized_multiply(const double* a, const double* b, double* result, size_t n) {
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
}

double vectorized_sum(const double* arr, size_t n) {
    if (n == 0) return 0.0;
    const size_t simd_width = 2;
    const size_t simd_end = (n / simd_width) * simd_width;
    float64x2_t sum_vec = vdupq_n_f64(0.0);
    for (size_t i = 0; i < simd_end; i += simd_width) {
        float64x2_t va = vld1q_f64(&arr[i]);
        sum_vec = vaddq_f64(sum_vec, va);
    }
    double sum_array[2];
    vst1q_f64(sum_array, sum_vec);
    double total_sum = sum_array[0] + sum_array[1];
    for (size_t i = simd_end; i < n; ++i) {
        total_sum += arr[i];
    }
    return total_sum;
}
double vectorized_mean(const double* arr, size_t n) {
    if (n == 0) return 0.0;
    return vectorized_sum(arr, n) / static_cast<double>(n);
}
double vectorized_std(const double* arr, size_t n, double mean) {
    if (n <= 1) return 0.0;
    double sum_sq = 0.0;
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
    return std::sqrt(sum_sq / (n - 1));
}

}
extern "C" {
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
}
