#!/usr/bin/env python3

import time
import ctypes
import os

# Create a simple C extension to test the ARM64 timer
calibration_c_code = '''
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>

#if defined(__aarch64__) || defined(_M_ARM64)
static inline uint64_t get_arm64_cycles() {
    uint64_t val;
    asm volatile("mrs %0, cntvct_el0" : "=r" (val));
    return val;
}
#else
static inline uint64_t get_arm64_cycles() {
    return 0; // Not ARM64
}
#endif

uint64_t test_timer_frequency() {
    printf("🔍 Calibrating ARM64 system timer...\\n");
    
    uint64_t start_cycles = get_arm64_cycles();
    uint64_t start_ns = (uint64_t)(clock() * 1000000000ULL / CLOCKS_PER_SEC);
    
    // Sleep for 100ms
    usleep(100000);
    
    uint64_t end_cycles = get_arm64_cycles();
    uint64_t end_ns = (uint64_t)(clock() * 1000000000ULL / CLOCKS_PER_SEC);
    
    uint64_t cycles_elapsed = end_cycles - start_cycles;
    uint64_t ns_elapsed = end_ns - start_ns;
    
    printf("   Cycles elapsed: %llu\\n", cycles_elapsed);
    printf("   Nanoseconds elapsed: %llu\\n", ns_elapsed);
    
    if (ns_elapsed > 0) {
        double frequency_hz = (double)cycles_elapsed / ((double)ns_elapsed / 1e9);
        double ns_per_cycle = (double)ns_elapsed / (double)cycles_elapsed;
        
        printf("   Timer frequency: %.0f Hz\\n", frequency_hz);
        printf("   Nanoseconds per cycle: %.3f\\n", ns_per_cycle);
        
        return (uint64_t)(ns_per_cycle * 1000);  // Return as fixed-point (x1000)
    }
    
    return 1000; // Default to 1.0 ns/cycle
}

#include <time.h>
#include <sys/time.h>

uint64_t precise_timer_calibration() {
    printf("🎯 Precise timer calibration using gettimeofday...\\n");
    
    struct timeval start_tv, end_tv;
    
    uint64_t start_cycles = get_arm64_cycles();
    gettimeofday(&start_tv, NULL);
    
    // Sleep for a more precise duration
    usleep(200000);  // 200ms
    
    uint64_t end_cycles = get_arm64_cycles();
    gettimeofday(&end_tv, NULL);
    
    uint64_t cycles_elapsed = end_cycles - start_cycles;
    uint64_t us_elapsed = (end_tv.tv_sec - start_tv.tv_sec) * 1000000ULL + 
                         (end_tv.tv_usec - start_tv.tv_usec);
    uint64_t ns_elapsed = us_elapsed * 1000;
    
    printf("   Cycles elapsed: %llu\\n", cycles_elapsed);
    printf("   Nanoseconds elapsed: %llu\\n", ns_elapsed);
    
    if (ns_elapsed > 0 && cycles_elapsed > 0) {
        double ns_per_cycle = (double)ns_elapsed / (double)cycles_elapsed;
        double frequency_hz = 1e9 / ns_per_cycle;
        
        printf("   Timer frequency: %.0f Hz\\n", frequency_hz);
        printf("   Nanoseconds per cycle: %.6f\\n", ns_per_cycle);
        
        return (uint64_t)(ns_per_cycle * 1000000);  // Return as fixed-point (x1M)
    }
    
    return 1000000; // Default to 1.0 ns/cycle
}
'''

def compile_and_run_calibration():
    print("🔧 Compiling timer calibration utility...")
    
    # Write C code to file
    with open('timer_calibration.c', 'w') as f:
        f.write(calibration_c_code)
    
    # Compile
    os.system('clang -O2 -o timer_calibration timer_calibration.c')
    
    if os.path.exists('timer_calibration'):
        print("✅ Running calibration...")
        os.system('./timer_calibration')
        
        # Clean up
        os.remove('timer_calibration.c')
        os.remove('timer_calibration')
    else:
        print("❌ Failed to compile calibration utility")

if __name__ == "__main__":
    compile_and_run_calibration()
