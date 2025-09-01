#include "quantpulse_core.h"
#include <queue>
#include <condition_variable>
#include <mutex>
#include <functional>
#include <numeric>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <ctime>
#include <cstdio>

// Timing utility functions
inline std::string get_current_time() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    char buffer[9];
    strftime(buffer, sizeof(buffer), "%H:%M:%S", std::localtime(&time));
    return std::string(buffer);
}>
const size_t ThreadPool::hardware_threads = std::thread::hardware_concurrency();
ThreadPool::ThreadPool(size_t threads) : stop(false) {
    size_t num_threads = (threads == 0) ? hardware_threads : std::min(threads, MAX_THREADS);
    workers.reserve(num_threads);
    for(size_t i = 0; i < num_threads; ++i) {
        workers.emplace_back([this] {});
    }
}
ThreadPool::~ThreadPool() {
    stop = true;
    for(std::thread &worker: workers) {
        if(worker.joinable()) {
            worker.join();
        }
    }
}
void ThreadPool::wait_all() {}
struct ParallelCVContext {
    ThreadPool* pool;
    OptimizationCache* cache;
    const PriceData* prices;
    const TradingParameters* params;
    std::vector<std::pair<int, int>> cv_splits;
    std::atomic<int> completed_folds;
    std::mutex result_mutex;
    std::vector<CVFoldResult> results;
};
static void calculate_spread_vectorized(const PriceData& prices, int lookback, double* spread, double* z_scores) {
    const size_t n = prices.size;
    const double* s1 = prices.symbol1.data();
    const double* s2 = prices.symbol2.data();
    const size_t simd_width = 2;
    const size_t simd_end = (n / simd_width) * simd_width;
    for(size_t i = 0; i < simd_end; i += simd_width) {
        float64x2_t vs1 = vld1q_f64(&s1[i]);
        float64x2_t vs2 = vld1q_f64(&s2[i]);
        float64x2_t vspread = vsubq_f64(vs1, vs2);
        vst1q_f64(&spread[i], vspread);
    }
    for(size_t i = simd_end; i < n; ++i) {
        spread[i] = s1[i] - s2[i];
    }
    #pragma omp parallel for
    for(int i = lookback; i < static_cast<int>(n); ++i) {
        const double* window = &spread[i - lookback];
        double sum = simd::vectorized_sum(window, lookback);
        double mean = sum / lookback;
        double std_dev = simd::vectorized_std(window, lookback, mean);
        z_scores[i] = (spread[i] - mean) / (std_dev + EPSILON);
    }
}

BacktestResult cpp_vectorized_backtest(const double* prices1, const double* prices2, size_t n, const TradingParameters& params) {
    printf("🔄 ENTERING cpp_vectorized_backtest() at %s\n", get_current_time().c_str());
    BacktestResult result{0.0, 0.0, 0.0, 0.0, {}, 0};
    std::vector<double> spread(n), z_scores(n, 0.0);
    std::fill(z_scores.begin(), z_scores.begin() + params.lookback, 0.0);
    const size_t simd_width = 2;
    const size_t simd_end = (n / simd_width) * simd_width;
    for(size_t i = 0; i < simd_end; i += simd_width) {
        float64x2_t vp1 = vld1q_f64(&prices1[i]);
        float64x2_t vp2 = vld1q_f64(&prices2[i]);
        float64x2_t vspread = vsubq_f64(vp1, vp2);
        vst1q_f64(&spread[i], vspread);
    }
    for(size_t i = simd_end; i < n; ++i) {
        spread[i] = prices1[i] - prices2[i];
    }
    for(int i = params.lookback; i < static_cast<int>(n); ++i) {
        const double* window = &spread[i - params.lookback];
        double mean = simd::vectorized_mean(window, params.lookback);
        double std_dev = simd::vectorized_std(window, params.lookback, mean);
        z_scores[i] = (spread[i] - mean) / (std_dev + EPSILON);
    }
    double pnl = 0.0;
    int position = 0;
    double entry_price = 0.0;
    std::vector<double> trade_returns;
    double max_dd = 0.0, peak = 0.0;
    int winning_trades = 0;
    for(int i = params.lookback; i < static_cast<int>(n); ++i) {
        double current_z = z_scores[i];
        double current_spread = spread[i];
        if(std::isnan(current_z)) continue;
        if(position == 0) {
            if(current_z > params.z_entry) {
                position = -1;
                entry_price = current_spread;
                pnl -= params.position_size * params.transaction_cost;
            } else if(current_z < -params.z_entry) {
                position = 1;
                entry_price = current_spread;
                pnl -= params.position_size * params.transaction_cost;
            }
        } else {
            bool exit_signal = false;
            double spread_change = current_spread - entry_price;
            double unrealized_pnl = position * spread_change * params.position_size;
            if((position == 1 && spread_change > params.profit_target) || (position == -1 && spread_change < -params.profit_target)) {
                exit_signal = true;
            } else if((position == 1 && spread_change < -params.stop_loss) || (position == -1 && spread_change > params.stop_loss)) {
                exit_signal = true;
            } else if(std::abs(current_z) < params.z_exit) {
                exit_signal = true;
            }
            if(exit_signal) {
                double trade_pnl = unrealized_pnl;
                pnl += trade_pnl - params.position_size * params.transaction_cost;
                trade_returns.push_back(trade_pnl);
                if(trade_pnl > 0) winning_trades++;
                position = 0;
                entry_price = 0.0;
                result.num_trades++;
            }
        }
        peak = std::max(peak, pnl);
        max_dd = std::max(max_dd, peak - pnl);
    }
    result.final_pnl = pnl;
    result.max_drawdown = max_dd;
    result.win_rate = result.num_trades > 0 ? static_cast<double>(winning_trades) / result.num_trades : 0.0;
    result.trade_returns = std::move(trade_returns);
    if(!result.trade_returns.empty()) {
        double mean_return = simd::vectorized_mean(result.trade_returns.data(), result.trade_returns.size());
        double std_return = simd::vectorized_std(result.trade_returns.data(), result.trade_returns.size(), mean_return);
        result.sharpe_ratio = (std_return > EPSILON) ? mean_return / std_return : 0.0;
    }
    printf("✅ EXITING cpp_vectorized_backtest() at %s\n", get_current_time().c_str());
    return result;
}

CVFoldResult evaluate_cv_fold_parallel(const PriceData& prices, const TradingParameters& params, int train_start, int train_end, int val_start, int val_end, double l1_ratio, double alpha, double kl_weight) {
    CVFoldResult result;
    result.train_result = cpp_vectorized_backtest(&prices.symbol1[train_start], &prices.symbol2[train_start], train_end - train_start, params);
    result.val_result = cpp_vectorized_backtest(&prices.symbol1[val_start], &prices.symbol2[val_start], val_end - val_start, params);
    std::array<double, 7> normalized_params = {
        (params.lookback - 5.0) / 55.0,
        (params.z_entry - 0.5) / 3.5,
        (params.z_exit - 0.1) / 1.9,
        (params.position_size - 1000.0) / 49000.0,
        (params.transaction_cost - 0.0001) / 0.0049,
        (params.profit_target - 1.5) / 3.5,
        (params.stop_loss - 0.5) / 1.5
    };
    double l1_penalty = 0.0, l2_penalty = 0.0;
    const size_t param_simd_width = 2;
    const size_t param_simd_end = (7 / param_simd_width) * param_simd_width;
    float64x2_t l1_vec = vdupq_n_f64(0.0);
    float64x2_t l2_vec = vdupq_n_f64(0.0);
    for(size_t i = 0; i < param_simd_end; i += param_simd_width) {
        float64x2_t vparams = vld1q_f64(&normalized_params[i]);
        float64x2_t vabs = vabsq_f64(vparams);
        float64x2_t vsq = vmulq_f64(vparams, vparams);
        l1_vec = vaddq_f64(l1_vec, vabs);
        l2_vec = vaddq_f64(l2_vec, vsq);
    }
    double l1_array[2], l2_array[2];
    vst1q_f64(l1_array, l1_vec);
    vst1q_f64(l2_array, l2_vec);
    l1_penalty = l1_array[0] + l1_array[1];
    l2_penalty = l2_array[0] + l2_array[1];
    for(size_t i = param_simd_end; i < 7; ++i) {
        l1_penalty += std::abs(normalized_params[i]);
        l2_penalty += normalized_params[i] * normalized_params[i];
    }
    result.elasticnet_penalty = alpha * (l1_ratio * l1_penalty + (1 - l1_ratio) * l2_penalty);
    result.kl_penalty = kl_weight * 0.1;
    result.stability_penalty = std::abs(result.train_result.sharpe_ratio - result.val_result.sharpe_ratio) * 0.1;
    double primary_score = result.val_result.final_pnl / 100000.0 + result.val_result.sharpe_ratio;
    double secondary_score = result.val_result.win_rate * 0.2;
    double risk_adj = primary_score / (std::abs(result.val_result.max_drawdown) / 100000.0 + 1.0);
    double combined_score = risk_adj + secondary_score;
    result.objective_score = combined_score - result.elasticnet_penalty - result.kl_penalty - result.stability_penalty;
    return result;
}

static double parallel_cross_validation_impl(const double* prices1, const double* prices2, size_t n, const double* param_array, int n_splits, double l1_ratio, double alpha, double kl_weight) {
    printf("🔧 ENTERING parallel_cross_validation_impl() at %s\n", get_current_time().c_str());
    PriceData prices(n);
    std::copy(prices1, prices1 + n, prices.symbol1.begin());
    std::copy(prices2, prices2 + n, prices.symbol2.begin());
    TradingParameters params{
        static_cast<int>(param_array[0]),
        param_array[1],
        param_array[2],
        static_cast<int>(param_array[3]),
        param_array[4],
        param_array[5],
        param_array[6]
    };
    std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> splits;
    int fold_size = n / (n_splits + 1);
    for(int i = 0; i < n_splits; ++i) {
        int val_start = i * fold_size;
        int val_end = (i + 1) * fold_size;
        int train_start = val_end;
        int train_end = n;
        if(train_end - train_start < params.lookback + 10) continue;
        splits.push_back({{train_start, train_end}, {val_start, val_end}});
    }
    std::vector<CVFoldResult> results(splits.size());
    #pragma omp parallel for
    for(size_t i = 0; i < splits.size(); ++i) {
        auto& split = splits[i];
        results[i] = evaluate_cv_fold_parallel(prices, params, split.first.first, split.first.second, split.second.first, split.second.second, l1_ratio, alpha, kl_weight);
    }
    double total_score = 0.0;
    for(const auto& result : results) {
        total_score += result.objective_score;
    }
    printf("✅ EXITING parallel_cross_validation_impl() at %s\n", get_current_time().c_str());
    return total_score / results.size();
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
        for (double v : r.trade_returns) {
            if (v >= 0) gross_pos += v; else gross_neg += -v;
        }
    }
    c.avg_trade_return = avg;
    c.volatility = vol;
    c.profit_factor = (gross_neg > EPSILON) ? (gross_pos / gross_neg) : (gross_pos > 0 ? std::numeric_limits<double>::infinity() : 0.0);
    return c;
}

extern "C" {
void calculate_spread_and_zscore(const double* prices1, const double* prices2, size_t n, int lookback, double* spread, double* z_scores) {
    printf("📊 ENTERING calculate_spread_and_zscore() at %s\n", get_current_time().c_str());
    if (!prices1 || !prices2 || !spread || !z_scores || n == 0 || lookback <= 0) return;
    const size_t simd_width = 2;
    const size_t simd_end = (n / simd_width) * simd_width;
    for (size_t i = 0; i < simd_end; i += simd_width) {
        float64x2_t vp1 = vld1q_f64(&prices1[i]);
        float64x2_t vp2 = vld1q_f64(&prices2[i]);
        float64x2_t vspread = vsubq_f64(vp1, vp2);
        vst1q_f64(&spread[i], vspread);
    }
    for (size_t i = simd_end; i < n; ++i) {
        spread[i] = prices1[i] - prices2[i];
    }
    for (int i = 0; i < std::min(static_cast<size_t>(lookback), n); ++i) z_scores[i] = 0.0;
    for (int i = lookback; i < static_cast<int>(n); ++i) {
        const double* window = &spread[i - lookback];
        double mean = simd::vectorized_mean(window, lookback);
        double std_dev = simd::vectorized_std(window, lookback, mean);
        z_scores[i] = (spread[i] - mean) / (std_dev + EPSILON);
    }
    printf("✅ EXITING calculate_spread_and_zscore() at %s\n", get_current_time().c_str());
}
CBacktestResult vectorized_backtest(const double* prices1, const double* prices2, size_t n, TradingParameters params) {
    printf("🚀 ENTERING vectorized_backtest() at %s\n", get_current_time().c_str());
    BacktestResult r = cpp_vectorized_backtest(prices1, prices2, n, params);
    CBacktestResult result = to_c_result(r);
    printf("✅ EXITING vectorized_backtest() at %s\n", get_current_time().c_str());
    return result;
}
size_t backtest_trade_returns(const double* prices1, const double* prices2, size_t n, TradingParameters params, double* out, size_t out_cap) {
    printf("📈 ENTERING backtest_trade_returns() at %s\n", get_current_time().c_str());
    BacktestResult r = cpp_vectorized_backtest(prices1, prices2, n, params);
    size_t m = r.trade_returns.size();
    if (out && out_cap > 0) {
        size_t w = std::min(out_cap, m);
        std::copy(r.trade_returns.begin(), r.trade_returns.begin() + w, out);
        printf("✅ EXITING backtest_trade_returns() at %s\n", get_current_time().c_str());
        return w;
    }
    printf("✅ EXITING backtest_trade_returns() at %s\n", get_current_time().c_str());
    return m;
}
double parallel_cross_validation(const double* prices1, const double* prices2, size_t n, const double* param_array, int n_splits, double l1_ratio, double alpha, double kl_weight) {
    printf("🔧 ENTERING parallel_cross_validation() at %s\n", get_current_time().c_str());
    double result = parallel_cross_validation_impl(prices1, prices2, n, param_array, n_splits, l1_ratio, alpha, kl_weight);
    printf("✅ EXITING parallel_cross_validation() at %s\n", get_current_time().c_str());
    return result;
}
void batch_parameter_optimization(const double* prices1, const double* prices2, size_t n, const double** param_sets, int n_sets, int param_len, CBacktestResult* results) {
    printf("⚙️ ENTERING batch_parameter_optimization() at %s\n", get_current_time().c_str());
    if (!param_sets || !results || n_sets <= 0) return;
    #pragma omp parallel for
    for (int i = 0; i < n_sets; ++i) {
        const double* p = param_sets[i];
        TradingParameters params{static_cast<int>(p[0]), p[1], p[2], static_cast<int>(p[3]), p[4], p[5], p[6]};
        BacktestResult r = cpp_vectorized_backtest(prices1, prices2, n, params);
        results[i] = to_c_result(r);
    }
    printf("✅ EXITING batch_parameter_optimization() at %s\n", get_current_time().c_str());
}
}
