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
#include <arm_neon.h>
#include <omp.h>
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

class ThreadPool {
private:
    std::vector<std::thread> workers;
    std::atomic<bool> stop;
    static const size_t hardware_threads;
public:
    ThreadPool(size_t threads = 0);
    ~ThreadPool();
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>;
    void wait_all();
    size_t thread_count() const { return workers.size(); }
};
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

extern "C" {
typedef struct {
    double total_return;
    double sharpe_ratio;
    double max_drawdown;
    int    num_trades;
    double win_rate;
    double profit_factor;
    double avg_trade_return;
    double volatility;
} CBacktestResult;
void simd_vector_subtract(const double* a, const double* b, double* result, size_t n);
void simd_vector_multiply(const double* a, const double* b, double* result, size_t n);
void simd_vector_add(const double* a, const double* b, double* result, size_t n);
double simd_vector_sum(const double* arr, size_t n);
double simd_vector_mean(const double* arr, size_t n);
double simd_vector_std(const double* arr, size_t n, double mean);
void calculate_spread_and_zscore(const double* prices1, const double* prices2, size_t n, int lookback, double* spread, double* z_scores);
CBacktestResult vectorized_backtest(const double* prices1, const double* prices2, size_t n, TradingParameters params);
CBacktestResult cached_vectorized_backtest(const double* prices1, const double* prices2, size_t n, TradingParameters params);
double parallel_cross_validation(const double* prices1, const double* prices2, size_t n, const double* param_array, int n_splits, double l1_ratio, double alpha, double kl_weight);
void batch_parameter_optimization(const double* prices1, const double* prices2, size_t n, const double** param_sets, int n_sets, int param_len, CBacktestResult* results);
size_t backtest_trade_returns(const double* prices1, const double* prices2, size_t n, TradingParameters params, double* out, size_t out_cap);
void print_cache_statistics();
void clear_all_caches();
void warm_up_caches(const double* prices1, const double* prices2, size_t n);
}

namespace simd {
    void vectorized_subtract(const double* a, const double* b, double* result, size_t n);
    void vectorized_multiply(const double* a, const double* b, double* result, size_t n);
    void vectorized_add(const double* a, const double* b, double* result, size_t n);
    double vectorized_sum(const double* arr, size_t n);
    double vectorized_mean(const double* arr, size_t n);
    double vectorized_std(const double* arr, size_t n, double mean);
}
constexpr double EPSILON = 1e-10;
constexpr size_t CACHE_LINE_SIZE = 64;
constexpr size_t MAX_THREADS = 32;
#define CACHE_ALIGNED alignas(CACHE_LINE_SIZE)
#endif
