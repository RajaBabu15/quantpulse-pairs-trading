#include "quantpulse_core.h"
#include <unordered_map>
#include <list>
#include <mutex>
#include <shared_mutex>
#include <functional>
#include <string>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <atomic>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <omp.h>
BacktestResult cpp_vectorized_backtest(const double* prices1, const double* prices2, size_t n, const TradingParameters& params);
template<typename K, typename V, typename Hash = std::hash<K>>
class ThreadSafeLRUCache {
private:
    struct CacheItem {
        K key;
        V value;
        size_t access_count;
        std::chrono::steady_clock::time_point last_access;
        CacheItem(const K& k, const V& v) : key(k), value(v), access_count(1), last_access(std::chrono::steady_clock::now()) {}
    };
    using ListIterator = typename std::list<CacheItem>::iterator;
    std::list<CacheItem> items;
    std::unordered_map<K, ListIterator, Hash> cache_map;
    size_t max_size;
    mutable std::shared_mutex mutex;
    CACHE_ALIGNED std::atomic<size_t> hits{0};
    CACHE_ALIGNED std::atomic<size_t> misses{0};
    CACHE_ALIGNED std::atomic<size_t> evictions{0};
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
            misses.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        auto list_it = map_it->second;
        value = list_it->value;
        lock.unlock();
        std::unique_lock<std::shared_mutex> unique_lock(mutex);
        move_to_front(list_it);
        hits.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    void put(const K& key, const V& value) {
        std::unique_lock<std::shared_mutex> lock(mutex);
        auto map_it = cache_map.find(key);
        if (map_it != cache_map.end()) {
            auto list_it = map_it->second;
            list_it->value = value;
            move_to_front(list_it);
            return;
        }
        items.emplace_front(key, value);
        cache_map[key] = items.begin();
        if (items.size() > max_size) {
            auto last = items.end();
            --last;
            cache_map.erase(last->key);
            items.erase(last);
            evictions.fetch_add(1, std::memory_order_relaxed);
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
        size_t total_hits = hits.load(std::memory_order_relaxed);
        size_t total_misses = misses.load(std::memory_order_relaxed);
        size_t total = total_hits + total_misses;
        return total > 0 ? static_cast<double>(total_hits) / total : 0.0;
    }
    struct Stats {
        size_t hits, misses, evictions, size;
        double hit_rate;
    };
    Stats get_stats() const {
        return {hits.load(std::memory_order_relaxed), misses.load(std::memory_order_relaxed), evictions.load(std::memory_order_relaxed), size(), hit_rate()};
    }
};
struct ParameterHash {
    size_t operator()(const std::vector<double>& params) const {
        size_t hash = 0;
        const size_t simd_width = 2;
        const size_t simd_end = (params.size() / simd_width) * simd_width;
        for (size_t i = 0; i < simd_end; i += simd_width) {
            uint64_t bits1, bits2;
            std::memcpy(&bits1, &params[i], sizeof(double));
            std::memcpy(&bits2, &params[i + 1], sizeof(double));
            hash ^= std::hash<uint64_t>{}(bits1) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            hash ^= std::hash<uint64_t>{}(bits2) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        for (size_t i = simd_end; i < params.size(); ++i) {
            uint64_t bits;
            std::memcpy(&bits, &params[i], sizeof(double));
            hash ^= std::hash<uint64_t>{}(bits) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

static ThreadSafeLRUCache<std::vector<double>, BacktestResult, ParameterHash> backtest_cache(5000);
static ThreadSafeLRUCache<std::vector<double>, double, ParameterHash> objective_cache(10000);
static ThreadSafeLRUCache<std::string, std::vector<double>> spread_cache(1000);
std::string generate_spread_cache_key(const double* prices1, const double* prices2, size_t n, int lookback) {
    std::hash<double> hasher;
    size_t hash1 = 0, hash2 = 0;
    const size_t step = 8;
    for (size_t i = 0; i < n; i += step) {
        hash1 ^= hasher(prices1[i]) + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2);
        hash2 ^= hasher(prices2[i]) + 0x9e3779b9 + (hash2 << 6) + (hash2 >> 2);
    }
    std::ostringstream oss;
    oss << "spread_" << hash1 << "_" << hash2 << "_" << n << "_" << lookback;
    return oss.str();
}
bool get_cached_spread_stats(const double* prices1, const double* prices2, size_t n, int lookback, double* spread, double* z_scores) {
    std::string key = generate_spread_cache_key(prices1, prices2, n, lookback);
    std::vector<double> cached_data;
    if (spread_cache.get(key, cached_data)) {
        if (cached_data.size() == 2 * n) {
            const size_t simd_width = 2;
            const size_t simd_end = (n / simd_width) * simd_width;
            for (size_t i = 0; i < simd_end; i += simd_width) {
                float64x2_t vdata = vld1q_f64(&cached_data[i]);
                vst1q_f64(&spread[i], vdata);
            }
            for (size_t i = simd_end; i < n; ++i) {
                spread[i] = cached_data[i];
            }
            for (size_t i = 0; i < simd_end; i += simd_width) {
                float64x2_t vdata = vld1q_f64(&cached_data[n + i]);
                vst1q_f64(&z_scores[i], vdata);
            }
            for (size_t i = simd_end; i < n; ++i) {
                z_scores[i] = cached_data[n + i];
            }
            return true;
        }
    }
    return false;
}
void cache_spread_stats(const double* prices1, const double* prices2, size_t n, int lookback, const double* spread, const double* z_scores) {
    std::string key = generate_spread_cache_key(prices1, prices2, n, lookback);
    std::vector<double> cache_data;
    cache_data.reserve(2 * n);
    cache_data.insert(cache_data.end(), spread, spread + n);
    cache_data.insert(cache_data.end(), z_scores, z_scores + n);
    spread_cache.put(key, cache_data);
}
BacktestResult cpp_cached_vectorized_backtest(const double* prices1, const double* prices2, size_t n, const TradingParameters& params) {
    std::vector<double> key = {
        static_cast<double>(params.lookback), params.z_entry, params.z_exit,
        static_cast<double>(params.position_size), params.transaction_cost,
        params.profit_target, params.stop_loss
    };
    std::hash<double> hasher;
    size_t price_hash = 0;
    const size_t sample_size = std::min(n, size_t(100));
    const size_t step = 10;
    for (size_t i = 0; i < sample_size; i += step) {
        uint64_t bits1, bits2;
        std::memcpy(&bits1, &prices1[i], sizeof(double));
        std::memcpy(&bits2, &prices2[i], sizeof(double));
        price_hash ^= bits1 + bits2;
    }
    key.push_back(static_cast<double>(price_hash));
    key.push_back(static_cast<double>(n));
    BacktestResult result;
    if (backtest_cache.get(key, result)) {
        return result;
    }
    result = cpp_vectorized_backtest(prices1, prices2, n, params);
    backtest_cache.put(key, result);
    return result;
}

double cpp_cached_objective_evaluation(const double* params, size_t param_count, const double* prices1, const double* prices2, size_t n, double l1_ratio, double alpha, double kl_weight) {
    std::vector<double> key;
    key.reserve(param_count + 4);
    for (size_t i = 0; i < param_count; ++i) {
        key.push_back(params[i]);
    }
    key.push_back(l1_ratio);
    key.push_back(alpha);
    key.push_back(kl_weight);
    std::hash<double> hasher;
    size_t price_hash = 0;
    const size_t sample_size = std::min(n, size_t(50));
    for (size_t i = 0; i < sample_size; i += 5) {
        price_hash ^= hasher(prices1[i]) + hasher(prices2[i]);
    }
    key.push_back(static_cast<double>(price_hash));
    double result;
    if (objective_cache.get(key, result)) {
        return result;
    }
    result = parallel_cross_validation(prices1, prices2, n, params, 3, l1_ratio, alpha, kl_weight);
    objective_cache.put(key, result);
    return result;
}
static void cpp_print_cache_statistics() {
    auto backtest_stats = backtest_cache.get_stats();
    auto objective_stats = objective_cache.get_stats();
    auto spread_stats = spread_cache.get_stats();
    printf("ARM64-Optimized Cache Statistics:\\n");
    printf("Backtest Cache - Size: %zu, Hit Rate: %.2f%%, Hits: %zu, Misses: %zu, Evictions: %zu\\n", backtest_stats.size, backtest_stats.hit_rate * 100, backtest_stats.hits, backtest_stats.misses, backtest_stats.evictions);
    printf("Objective Cache - Size: %zu, Hit Rate: %.2f%%, Hits: %zu, Misses: %zu, Evictions: %zu\\n", objective_stats.size, objective_stats.hit_rate * 100, objective_stats.hits, objective_stats.misses, objective_stats.evictions);
    printf("Spread Cache - Size: %zu, Hit Rate: %.2f%%, Hits: %zu, Misses: %zu, Evictions: %zu\\n", spread_stats.size, spread_stats.hit_rate * 100, spread_stats.hits, spread_stats.misses, spread_stats.evictions);
}
static void cpp_clear_all_caches() {
    backtest_cache.clear();
    objective_cache.clear();
    spread_cache.clear();
}
static void cpp_warm_up_caches(const double* prices1, const double* prices2, size_t n) {
    double common_params[][7] = {
        {20, 2.0, 0.5, 10000, 0.001, 2.0, 1.0},
        {25, 2.5, 0.3, 15000, 0.001, 2.5, 1.5},
        {30, 1.5, 0.8, 20000, 0.0005, 1.8, 0.8},
        {15, 3.0, 0.2, 12000, 0.002, 3.0, 2.0},
        {35, 1.8, 0.6, 25000, 0.0015, 2.2, 1.2}
    };
    printf("Warming up ARM64-optimized caches with common parameter combinations...\\n");
    #pragma omp parallel for
    for (int i = 0; i < 5; ++i) {
        TradingParameters params{static_cast<int>(common_params[i][0]), common_params[i][1], common_params[i][2], static_cast<int>(common_params[i][3]), common_params[i][4], common_params[i][5], common_params[i][6]};
        cpp_cached_vectorized_backtest(prices1, prices2, n, params);
        cpp_cached_objective_evaluation(common_params[i], 7, prices1, prices2, n, 0.7, 0.02, 0.15);
    }
    printf("ARM64 cache warm-up complete.\\n");
    cpp_print_cache_statistics();
}

static CBacktestResult to_c_result(const BacktestResult& r) {
    CBacktestResult c{};
    c.total_return = r.final_pnl;
    c.sharpe_ratio = r.sharpe_ratio;
    c.max_drawdown = r.max_drawdown;
    c.num_trades = r.num_trades;
    c.win_rate = r.win_rate;
    double avg = 0.0, vol = 0.0, gross_pos = 0.0, gross_neg = 0.0;
    if (!r.trade_returns.empty()) {
        avg = simd::vectorized_mean(r.trade_returns.data(), r.trade_returns.size());
        vol = simd::vectorized_std(r.trade_returns.data(), r.trade_returns.size(), avg);
        const size_t simd_width = 2;
        const size_t simd_end = (r.trade_returns.size() / simd_width) * simd_width;
        float64x2_t pos_vec = vdupq_n_f64(0.0);
        float64x2_t neg_vec = vdupq_n_f64(0.0);
        float64x2_t zero_vec = vdupq_n_f64(0.0);
        for (size_t i = 0; i < simd_end; i += simd_width) {
            float64x2_t vdata = vld1q_f64(&r.trade_returns[i]);
            uint64x2_t mask_pos = vcgeq_f64(vdata, zero_vec);
            uint64x2_t mask_neg = vcltq_f64(vdata, zero_vec);
            pos_vec = vaddq_f64(pos_vec, vbslq_f64(mask_pos, vdata, zero_vec));
            neg_vec = vaddq_f64(neg_vec, vbslq_f64(mask_neg, vnegq_f64(vdata), zero_vec));
        }
        double pos_array[2], neg_array[2];
        vst1q_f64(pos_array, pos_vec);
        vst1q_f64(neg_array, neg_vec);
        gross_pos = pos_array[0] + pos_array[1];
        gross_neg = neg_array[0] + neg_array[1];
        for (size_t i = simd_end; i < r.trade_returns.size(); ++i) {
            double v = r.trade_returns[i];
            if (v >= 0) gross_pos += v; else gross_neg += -v;
        }
    }
    c.avg_trade_return = avg;
    c.volatility = vol;
    c.profit_factor = (gross_neg > EPSILON) ? (gross_pos / gross_neg) : (gross_pos > 0 ? std::numeric_limits<double>::infinity() : 0.0);
    return c;
}
extern "C" {
CBacktestResult cached_vectorized_backtest(const double* prices1, const double* prices2, size_t n, TradingParameters params) {
    BacktestResult r = cpp_cached_vectorized_backtest(prices1, prices2, n, params);
    return to_c_result(r);
}
double cached_objective_evaluation(const double* params, size_t param_count, const double* prices1, const double* prices2, size_t n, double l1_ratio, double alpha, double kl_weight) {
    return cpp_cached_objective_evaluation(params, param_count, prices1, prices2, n, l1_ratio, alpha, kl_weight);
}
void print_cache_statistics() {
    cpp_print_cache_statistics();
}
void clear_all_caches() {
    cpp_clear_all_caches();
}
void warm_up_caches(const double* prices1, const double* prices2, size_t n) {
    cpp_warm_up_caches(prices1, prices2, n);
}
}
