#include "quantpulse_core.h"
#include <unordered_map>
#include <list>
#include <mutex>
#include <shared_mutex>
#include <functional>
#include <string>
#include <sstream>
#include <iomanip>

extern "C" {

// Thread-safe LRU cache implementation
template<typename K, typename V>
class ThreadSafeLRUCache {
private:
    struct CacheItem {
        K key;
        V value;
        size_t access_count;
        std::chrono::steady_clock::time_point last_access;
        
        CacheItem(const K& k, const V& v) 
            : key(k), value(v), access_count(1), 
              last_access(std::chrono::steady_clock::now()) {}
    };
    
    using ListIterator = typename std::list<CacheItem>::iterator;
    
    std::list<CacheItem> items;
    std::unordered_map<K, ListIterator> cache_map;
    size_t max_size;
    mutable std::shared_mutex mutex;
    
    // Statistics
    std::atomic<size_t> hits{0};
    std::atomic<size_t> misses{0};
    std::atomic<size_t> evictions{0};
    
    void move_to_front(ListIterator it) {
        items.splice(items.begin(), items, it);
        it->access_count++;
        it->last_access = std::chrono::steady_clock::now();
    }
    
public:
    ThreadSafeLRUCache(size_t max_sz) : max_size(max_sz) {}
    
    bool get(const K& key, V& value) {
        std::shared_lock<std::shared_mutex> lock(mutex);
        
        auto map_it = cache_map.find(key);
        if (map_it == cache_map.end()) {
            misses++;
            return false;
        }
        
        auto list_it = map_it->second;
        value = list_it->value;
        
        // Update access info (requires unique lock)
        lock.unlock();
        std::unique_lock<std::shared_mutex> unique_lock(mutex);
        move_to_front(list_it);
        hits++;
        
        return true;
    }
    
    void put(const K& key, const V& value) {
        std::unique_lock<std::shared_mutex> lock(mutex);
        
        auto map_it = cache_map.find(key);
        if (map_it != cache_map.end()) {
            // Update existing item
            auto list_it = map_it->second;
            list_it->value = value;
            move_to_front(list_it);
            return;
        }
        
        // Add new item
        items.emplace_front(key, value);
        cache_map[key] = items.begin();
        
        // Evict if necessary
        if (items.size() > max_size) {
            auto last = items.end();
            --last;
            cache_map.erase(last->key);
            items.erase(last);
            evictions++;
        }
    }
    
    void clear() {
        std::unique_lock<std::shared_mutex> lock(mutex);
        items.clear();
        cache_map.clear();
    }
    
    size_t size() const {
        std::shared_lock<std::shared_mutex> lock(mutex);
        return items.size();
    }
    
    double hit_rate() const {
        size_t total = hits + misses;
        return total > 0 ? static_cast<double>(hits) / total : 0.0;
    }
    
    struct Stats {
        size_t hits, misses, evictions, size;
        double hit_rate;
    };
    
    Stats get_stats() const {
        return {hits.load(), misses.load(), evictions.load(), size(), hit_rate()};
    }
};

// Hash function for parameter arrays
struct ParameterHash {
    size_t operator()(const std::vector<double>& params) const {
        size_t hash = 0;
        for (size_t i = 0; i < params.size(); ++i) {
            // Convert double to uint64_t for hashing
            uint64_t bits;
            std::memcpy(&bits, &params[i], sizeof(double));
            hash ^= std::hash<uint64_t>{}(bits) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

// Global caches for different computation types
static ThreadSafeLRUCache<std::vector<double>, BacktestResult> backtest_cache(5000);
static ThreadSafeLRUCache<std::vector<double>, double> objective_cache(10000);
static ThreadSafeLRUCache<std::string, std::vector<double>> spread_cache(1000);

// Cache key generation for spread statistics
std::string generate_spread_cache_key(const double* prices1, const double* prices2, 
                                     size_t n, int lookback) {
    // Create a hash of the price data and parameters
    std::hash<double> hasher;
    size_t hash1 = 0, hash2 = 0;
    
    // Sample prices for hashing (every 10th element to balance speed vs uniqueness)
    for (size_t i = 0; i < n; i += 10) {
        hash1 ^= hasher(prices1[i]) + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2);
        hash2 ^= hasher(prices2[i]) + 0x9e3779b9 + (hash2 << 6) + (hash2 >> 2);
    }
    
    std::ostringstream oss;
    oss << "spread_" << hash1 << "_" << hash2 << "_" << n << "_" << lookback;
    return oss.str();
}

// Cached spread calculation
bool get_cached_spread_stats(const double* prices1, const double* prices2, 
                            size_t n, int lookback, 
                            double* spread, double* z_scores) {
    std::string key = generate_spread_cache_key(prices1, prices2, n, lookback);
    std::vector<double> cached_data;
    
    if (spread_cache.get(key, cached_data)) {
        // Data format: [spread_values..., z_score_values...]
        if (cached_data.size() == 2 * n) {
            std::copy(cached_data.begin(), cached_data.begin() + n, spread);
            std::copy(cached_data.begin() + n, cached_data.end(), z_scores);
            return true;
        }
    }
    
    return false;
}

// Cache spread calculation results
void cache_spread_stats(const double* prices1, const double* prices2, 
                       size_t n, int lookback,
                       const double* spread, const double* z_scores) {
    std::string key = generate_spread_cache_key(prices1, prices2, n, lookback);
    std::vector<double> cache_data;
    cache_data.reserve(2 * n);
    
    // Store spread values followed by z-score values
    cache_data.insert(cache_data.end(), spread, spread + n);
    cache_data.insert(cache_data.end(), z_scores, z_scores + n);
    
    spread_cache.put(key, cache_data);
}

// Cached backtest with automatic caching
BacktestResult cached_vectorized_backtest(const double* prices1, const double* prices2, 
                                        size_t n, const TradingParameters& params) {
    // Create cache key from parameters
    std::vector<double> key = {
        static_cast<double>(params.lookback), params.z_entry, params.z_exit,
        static_cast<double>(params.position_size), params.transaction_cost,
        params.profit_target, params.stop_loss
    };
    
    // Add price data hash to key for uniqueness
    std::hash<double> hasher;
    size_t price_hash = 0;
    for (size_t i = 0; i < std::min(n, size_t(100)); i += 10) { // Sample for performance
        price_hash ^= hasher(prices1[i]) + hasher(prices2[i]);
    }
    key.push_back(static_cast<double>(price_hash));
    key.push_back(static_cast<double>(n));
    
    BacktestResult result;
    if (backtest_cache.get(key, result)) {
        return result;
    }
    
    // Compute result and cache it
    result = vectorized_backtest(prices1, prices2, n, params);
    backtest_cache.put(key, result);
    
    return result;
}

// Cached objective function evaluation
double cached_objective_evaluation(const double* params, size_t param_count,
                                  const double* prices1, const double* prices2, size_t n,
                                  double l1_ratio, double alpha, double kl_weight) {
    // Create comprehensive cache key
    std::vector<double> key;
    key.reserve(param_count + 4);
    
    for (size_t i = 0; i < param_count; ++i) {
        key.push_back(params[i]);
    }
    key.push_back(l1_ratio);
    key.push_back(alpha);
    key.push_back(kl_weight);
    
    // Add price data fingerprint
    std::hash<double> hasher;
    size_t price_hash = 0;
    for (size_t i = 0; i < std::min(n, size_t(50)); i += 5) {
        price_hash ^= hasher(prices1[i]) + hasher(prices2[i]);
    }
    key.push_back(static_cast<double>(price_hash));
    
    double result;
    if (objective_cache.get(key, result)) {
        return result;
    }
    
    // Compute using parallel cross-validation and cache result
    result = parallel_cross_validation(prices1, prices2, n, params, 3, l1_ratio, alpha, kl_weight);
    objective_cache.put(key, result);
    
    return result;
}

// Cache statistics and management
void print_cache_statistics() {
    auto backtest_stats = backtest_cache.get_stats();
    auto objective_stats = objective_cache.get_stats();
    auto spread_stats = spread_cache.get_stats();
    
    printf("Cache Statistics:\n");
    printf("Backtest Cache - Size: %zu, Hit Rate: %.2f%%, Hits: %zu, Misses: %zu, Evictions: %zu\n",
           backtest_stats.size, backtest_stats.hit_rate * 100,
           backtest_stats.hits, backtest_stats.misses, backtest_stats.evictions);
    printf("Objective Cache - Size: %zu, Hit Rate: %.2f%%, Hits: %zu, Misses: %zu, Evictions: %zu\n",
           objective_stats.size, objective_stats.hit_rate * 100,
           objective_stats.hits, objective_stats.misses, objective_stats.evictions);
    printf("Spread Cache - Size: %zu, Hit Rate: %.2f%%, Hits: %zu, Misses: %zu, Evictions: %zu\n",
           spread_stats.size, spread_stats.hit_rate * 100,
           spread_stats.hits, spread_stats.misses, spread_stats.evictions);
}

void clear_all_caches() {
    backtest_cache.clear();
    objective_cache.clear();
    spread_cache.clear();
}

// Warm up cache with common parameter combinations
void warm_up_caches(const double* prices1, const double* prices2, size_t n) {
    // Common parameter combinations for warm-up
    double common_params[][7] = {
        {20, 2.0, 0.5, 10000, 0.001, 2.0, 1.0},
        {25, 2.5, 0.3, 15000, 0.001, 2.5, 1.5},
        {30, 1.5, 0.8, 20000, 0.0005, 1.8, 0.8},
        {15, 3.0, 0.2, 12000, 0.002, 3.0, 2.0},
        {35, 1.8, 0.6, 25000, 0.0015, 2.2, 1.2}
    };
    
    printf("Warming up caches with common parameter combinations...\n");
    
    #pragma omp parallel for
    for (int i = 0; i < 5; ++i) {
        TradingParameters params{
            static_cast<int>(common_params[i][0]), common_params[i][1], common_params[i][2],
            static_cast<int>(common_params[i][3]), common_params[i][4], 
            common_params[i][5], common_params[i][6]
        };
        
        // This will populate the backtest cache
        cached_vectorized_backtest(prices1, prices2, n, params);
        
        // This will populate the objective cache
        cached_objective_evaluation(common_params[i], 7, prices1, prices2, n, 0.7, 0.02, 0.15);
    }
    
    printf("Cache warm-up complete.\n");
    print_cache_statistics();
}

} // extern "C"
