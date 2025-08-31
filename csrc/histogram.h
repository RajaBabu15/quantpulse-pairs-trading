#pragma once
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cmath>

class LatencyHistogram {
private:
    std::vector<uint64_t> samples;
    bool sorted = false;
    
public:
    void record(uint64_t latency_ns) {
        samples.push_back(latency_ns);
        sorted = false;
    }
    
    void ensure_sorted() {
        if (!sorted) {
            std::sort(samples.begin(), samples.end());
            sorted = true;
        }
    }
    
    uint64_t count() const {
        return samples.size();
    }
    
    uint64_t percentile(double p) {
        ensure_sorted();
        if (samples.empty()) return 0;
        
        size_t index = (size_t)((p / 100.0) * (samples.size() - 1));
        return samples[index];
    }
    
    uint64_t min() {
        ensure_sorted();
        return samples.empty() ? 0 : samples.front();
    }
    
    uint64_t max() {
        ensure_sorted();
        return samples.empty() ? 0 : samples.back();
    }
    
    double mean() {
        if (samples.empty()) return 0.0;
        
        uint64_t sum = 0;
        for (uint64_t sample : samples) {
            sum += sample;
        }
        return double(sum) / samples.size();
    }
    
    void print_summary(const char* name = "Latency") {
        if (samples.empty()) {
            printf("%s: No samples collected\n", name);
            return;
        }
        
        ensure_sorted();
        
        printf("\n📊 %s Distribution (%llu samples):\n", name, count());
        printf("   Mean:   %8.1f ns\n", mean());
        printf("   Min:    %8llu ns\n", min());
        printf("   P50:    %8llu ns\n", percentile(50));
        printf("   P95:    %8llu ns\n", percentile(95));
        printf("   P99:    %8llu ns\n", percentile(99));
        printf("   P99.9:  %8llu ns\n", percentile(99.9));
        printf("   Max:    %8llu ns\n", max());
        
        // Show distribution buckets
        printf("   Buckets:\n");
        print_buckets();
    }
    
    void print_buckets() {
        ensure_sorted();
        if (samples.empty()) return;
        
        // Power-of-2 buckets
        std::vector<std::pair<uint64_t, uint64_t>> buckets; // {threshold, count}
        
        // Define bucket thresholds (ns)
        std::vector<uint64_t> thresholds = {
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 
            1000, 2000, 4000, 8000, 16000, 32000, 64000, 
            UINT64_MAX
        };
        
        for (uint64_t threshold : thresholds) {
            buckets.push_back({threshold, 0});
        }
        
        // Count samples in each bucket
        for (uint64_t sample : samples) {
            for (auto& bucket : buckets) {
                if (sample <= bucket.first) {
                    bucket.second++;
                    break;
                }
            }
        }
        
        // Print non-empty buckets
        for (size_t i = 0; i < buckets.size(); ++i) {
            if (buckets[i].second == 0) continue;
            
            uint64_t prev_threshold = (i == 0) ? 0 : buckets[i-1].first + 1;
            uint64_t curr_threshold = buckets[i].first;
            double pct = (double(buckets[i].second) / samples.size()) * 100.0;
            
            if (curr_threshold == UINT64_MAX) {
                printf("     >%llu ns: %6llu (%5.1f%%)\n", 
                       prev_threshold - 1, buckets[i].second, pct);
            } else {
                printf("     %llu-%llu ns: %6llu (%5.1f%%)\n", 
                       prev_threshold, curr_threshold, buckets[i].second, pct);
            }
        }
    }
    
    void clear() {
        samples.clear();
        sorted = false;
    }
};
