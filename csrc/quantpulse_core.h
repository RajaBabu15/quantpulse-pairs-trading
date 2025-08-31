#ifndef QUANTPULSE_CORE_H
#define QUANTPULSE_CORE_H

#include <vector>
#include <array>
#include <memory>
#include <thread>
#include <future>
#include <atomic>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <immintrin.h>  // For SIMD/AVX
#include <omp.h>        // For OpenMP

// Common data structures
struct PriceData {
    std::vector<double> symbol1;
    std::vector<double> symbol2;
    size_t size;
    
    PriceData(size_t n) : symbol1(n), symbol2(n), size(n) {}
};

struct TradingParameters {
    int lookback;
    double z_entry;
    double z_exit;
    int position_size;
    double transaction_cost;
    double profit_target;
    double stop_loss;
};

struct BacktestResult {
    double final_pnl;
    double sharpe_ratio;
    double win_rate;
    double max_drawdown;
    std::vector<double> trade_returns;
    int num_trades;
};

struct CVFoldResult {
    double objective_score;
    BacktestResult train_result;
    BacktestResult val_result;
    double elasticnet_penalty;
    double kl_penalty;
    double stability_penalty;
};

// Thread pool for parallel processing
class ThreadPool {
private:
    std::vector<std::thread> workers;
    std::atomic<bool> stop;
    static const size_t hardware_threads;
    
public:
    ThreadPool(size_t threads = 0);
    ~ThreadPool();
    
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;
    
    void wait_all();
    size_t thread_count() const { return workers.size(); }
};

// Cache system for expensive calculations
class OptimizationCache {
private:
    std::unordered_map<std::string, std::shared_ptr<void>> cache;
    size_t max_size;
    std::atomic<size_t> current_size;
    
public:
    OptimizationCache(size_t max_sz = 10000) : max_size(max_sz), current_size(0) {}
    
    template<typename T>
    bool get(const std::string& key, T& result);
    
    template<typename T>
    void put(const std::string& key, const T& value);
    
    void clear();
    size_t size() const { return current_size.load(); }
};

// SIMD-optimized mathematical operations
namespace simd {
    void vectorized_subtract(const double* a, const double* b, double* result, size_t n);
    void vectorized_multiply(const double* a, const double* b, double* result, size_t n);
    void vectorized_add(const double* a, const double* b, double* result, size_t n);
    double vectorized_sum(const double* arr, size_t n);
    double vectorized_mean(const double* arr, size_t n);
    double vectorized_std(const double* arr, size_t n, double mean);
}

// Constants
constexpr double EPSILON = 1e-10;
constexpr size_t CACHE_LINE_SIZE = 64;
constexpr size_t MAX_THREADS = 32;

// Alignment macros for cache efficiency
#define CACHE_ALIGNED alignas(CACHE_LINE_SIZE)

#endif // QUANTPULSE_CORE_H
