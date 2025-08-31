#include <chrono>
#include <cstdio>
#include <cstdint>
#include <thread>
#include <vector>
#include <algorithm>

#if defined(__x86_64__) || defined(_M_X64)
#include <x86intrin.h>
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

struct CalibrationResult {
    double cycles_per_second;
    double cycles_per_ns;
    double ns_per_cycle;
    double stability_pct;
};

CalibrationResult calibrate_rdtsc() {
    printf("🔬 Calibrating RDTSC timer...\n");
    
    std::vector<double> samples;
    const int num_samples = 10;
    
    for (int i = 0; i < num_samples; ++i) {
        auto wall_start = std::chrono::high_resolution_clock::now();
        uint64_t cycles_start = rdtsc();
        
        // Sleep for 100ms for more accurate measurement
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        uint64_t cycles_end = rdtsc();
        auto wall_end = std::chrono::high_resolution_clock::now();
        
        auto wall_duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(wall_end - wall_start).count();
        uint64_t cycle_duration = cycles_end - cycles_start;
        
        double cycles_per_ns = double(cycle_duration) / double(wall_duration_ns);
        samples.push_back(cycles_per_ns);
        
        printf("   Sample %d: %.6f cycles/ns\n", i+1, cycles_per_ns);
    }
    
    // Calculate statistics
    std::sort(samples.begin(), samples.end());
    double median = samples[num_samples / 2];
    double min_val = samples[0];
    double max_val = samples[num_samples - 1];
    double range = max_val - min_val;
    double stability = (1.0 - range / median) * 100.0;
    
    CalibrationResult result;
    result.cycles_per_ns = median;
    result.cycles_per_second = median * 1e9;
    result.ns_per_cycle = 1.0 / median;
    result.stability_pct = stability;
    
    printf("\n📊 Calibration Results:\n");
    printf("   Median cycles/ns: %.6f\n", result.cycles_per_ns);
    printf("   CPU frequency: %.0f Hz (%.2f GHz)\n", result.cycles_per_second, result.cycles_per_second / 1e9);
    printf("   ns per cycle: %.3f\n", result.ns_per_cycle);
    printf("   Stability: %.1f%% (higher is better)\n", result.stability_pct);
    
    if (result.stability_pct < 95.0) {
        printf("   ⚠️  Low stability - CPU frequency scaling may be active\n");
    } else {
        printf("   ✅ Good stability - RDTSC measurements should be reliable\n");
    }
    
    return result;
}

// Test single operation timing
void test_single_operation_floor() {
    printf("\n🧪 Testing single operation timing floor...\n");
    
    const int iterations = 1000000;
    std::vector<uint64_t> timings;
    timings.reserve(iterations);
    
    volatile int dummy = 0;  // Prevent optimization
    
    for (int i = 0; i < iterations; ++i) {
        uint64_t start = rdtsc();
        
        // Minimal work - just a volatile write
        dummy = i;
        asm volatile("" ::: "memory");  // Memory barrier
        
        uint64_t end = rdtsc();
        timings.push_back(end - start);
    }
    
    std::sort(timings.begin(), timings.end());
    
    uint64_t min_cycles = timings[0];
    uint64_t median_cycles = timings[iterations / 2];
    uint64_t p95_cycles = timings[iterations * 95 / 100];
    uint64_t p99_cycles = timings[iterations * 99 / 100];
    
    printf("   Minimal operation (volatile write + barrier):\n");
    printf("   Min: %llu cycles\n", min_cycles);
    printf("   Median: %llu cycles\n", median_cycles);
    printf("   P95: %llu cycles\n", p95_cycles);
    printf("   P99: %llu cycles\n", p99_cycles);
    
    CalibrationResult cal = calibrate_rdtsc();
    printf("   In nanoseconds (using calibration):\n");
    printf("   Min: %.1f ns\n", min_cycles * cal.ns_per_cycle);
    printf("   Median: %.1f ns\n", median_cycles * cal.ns_per_cycle);
    printf("   P95: %.1f ns\n", p95_cycles * cal.ns_per_cycle);
    printf("   P99: %.1f ns\n", p99_cycles * cal.ns_per_cycle);
}

int main() {
    printf("🕒 RDTSC Timer Validation Tool\n");
    printf("===============================\n");
    
    CalibrationResult cal = calibrate_rdtsc();
    test_single_operation_floor();
    
    printf("\n💡 Key Insights:\n");
    printf("   - Any claimed latency below the minimal operation time is suspect\n");
    printf("   - Real HFT operations should be 10-100x the minimal operation time\n");
    printf("   - Use calibrated cycles_per_ns = %.6f for accurate conversion\n", cal.cycles_per_ns);
    
    return 0;
}
