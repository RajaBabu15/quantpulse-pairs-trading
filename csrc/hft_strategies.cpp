#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <cstring>
#include <algorithm>

// Optional AVX2
#if defined(__AVX2__)
  #include <immintrin.h>
#endif

namespace py = pybind11;

// ================= USER STRATEGY STRUCTS =================

// 1. HMM REGIME SWITCHING
struct HMMRegime {
    double P[2][2] = {{0.95, 0.05}, {0.03, 0.97}};
    double mu[2] = {0.0008, -0.0003};
    double sigma[2] = {0.015, 0.025};
    double regime_prob[2] = {0.6, 0.4};
    int state = 0;
    // Tunable thresholds per regime
    double z_threshold_low = 1.2;
    double z_threshold_high = 2.8;
    void set_mu(double a, double b){ mu[0]=a; mu[1]=b; }
    void set_sigma(double a, double b){ sigma[0]=a; sigma[1]=b; }
    void set_P(double p00,double p01,double p10,double p11){ P[0][0]=p00;P[0][1]=p01;P[1][0]=p10;P[1][1]=p11; }
    void set_thresholds(double low,double high){ z_threshold_low=low; z_threshold_high=high; }
};

inline int hmm_regime_switch(const double* returns, int n, HMMRegime& hmm) {
    if (n <= 0) return 0;
    double obs = returns[n-1];
    double likelihood[2];
    for (int s=0;s<2;s++) {
        double z = (obs - hmm.mu[s]) / hmm.sigma[s];
        likelihood[s] = std::exp(-0.5 * z * z) / (hmm.sigma[s] * 2.50662827);
    }
    double forward[2];
    for (int s=0;s<2;s++) {
        forward[s] = likelihood[s] * (hmm.P[0][s] * hmm.regime_prob[0] + hmm.P[1][s] * hmm.regime_prob[1]);
    }
    double total = forward[0] + forward[1];
    if (total <= 0.0) total = 1e-12;
    hmm.regime_prob[0] = forward[0] / total;
    hmm.regime_prob[1] = forward[1] / total;
    hmm.state = (hmm.regime_prob[0] > 0.5) ? 0 : 1;

    double theta = (hmm.state == 0) ? 0.3 : 1.8;  // unused but kept for compatibility
    (void)theta;
    double z_threshold = (hmm.state == 0) ? hmm.z_threshold_low : hmm.z_threshold_high;

    double spread = obs; // using latest return as spread proxy
    double regime_mean = hmm.mu[hmm.state];
    double z_score = (spread - regime_mean) / hmm.sigma[hmm.state];
    return (z_score > z_threshold) ? -1 : (z_score < -z_threshold) ? 1 : 0;
}

// 2. KALMAN FILTER PAIRS
struct KalmanPair {
    double x[2] = {1.0, 0.0};
    double P[4] = {1.0, 0.0, 0.0, 1.0};
    double Q[4] = {1e-5, 0.0, 0.0, 1e-4};
    double R = 1e-3;
    double K[2] = {0.0, 0.0};
    void set_Q(double q_beta, double q_mu){ Q[0]=q_beta; Q[3]=q_mu; }
    void set_R(double r){ R=r; }
};

inline int kalman_pairs_filter(double price1, double price2, KalmanPair& kf) {
    // Predict: only covariance grows
    kf.P[0] += kf.Q[0];
    kf.P[3] += kf.Q[3];

    double predicted_price1 = kf.x[0] * price2 + kf.x[1];
    double innovation = price1 - predicted_price1;

    double S = price2*price2*kf.P[0] + 2.0*price2*kf.P[1] + kf.P[3] + kf.R;
    if (S <= 1e-12) S = 1e-12;
    kf.K[0] = (price2 * kf.P[0] + kf.P[1]) / S;
    kf.K[1] = (price2 * kf.P[1] + kf.P[3]) / S;

    kf.x[0] += kf.K[0] * innovation;
    kf.x[1] += kf.K[1] * innovation;

    double temp_P[4];
    temp_P[0] = kf.P[0] - kf.K[0] * price2 * kf.P[0] - kf.K[0] * kf.P[1];
    temp_P[1] = kf.P[1] - kf.K[0] * price2 * kf.P[1] - kf.K[0] * kf.P[3];
    temp_P[2] = kf.P[2] - kf.K[1] * price2 * kf.P[0] - kf.K[1] * kf.P[1];
    temp_P[3] = kf.P[3] - kf.K[1] * price2 * kf.P[1] - kf.K[1] * kf.P[3];
    std::memcpy(kf.P, temp_P, sizeof(temp_P));

    double innovation_std = std::sqrt(S);
    double z_innovation = innovation / innovation_std;
    return (z_innovation > 2.5) ? -1 : (z_innovation < -2.5) ? 1 : 0;
}

// 3. GARCH VOL (simplified state)
struct GARCHVolatility {
    double omega = 1e-6, alpha = 0.08, beta = 0.90, gamma = 0.05;
    double vol2_forecast = 0.0004;  // sigma^2 forecast
    double corr_ema = 0.5;
    double vol_ratio_ema = 1.0;
    void set_params(double o,double a,double b,double g){ omega=o; alpha=a; beta=b; gamma=g; }
};

inline int garch_vol_strategy(double ret1, double ret2, double vol1_realized,
                             double vol2_realized, GARCHVolatility& garch) {
    double asymmetry = (ret1 < 0.0) ? garch.gamma * ret1 * ret1 : 0.0;
    garch.vol2_forecast = garch.omega + garch.alpha * ret1 * ret1 + garch.beta * garch.vol2_forecast + asymmetry;
    if (garch.vol2_forecast <= 1e-12) garch.vol2_forecast = 1e-12;

    garch.corr_ema = 0.94 * garch.corr_ema + 0.06 * ret1 * ret2;

    double vol_ratio_current = vol1_realized / std::sqrt(garch.vol2_forecast);
    garch.vol_ratio_ema = 0.9 * garch.vol_ratio_ema + 0.1 * vol_ratio_current;

    double vol_dislocation = std::abs(garch.vol_ratio_ema - 1.0);
    double corr_regime = std::abs(garch.corr_ema);

    if (vol_dislocation > 0.25 && corr_regime < 0.4) {
        return (garch.vol_ratio_ema > 1.0) ? -1 : 1;
    }
    return 0;
}

// 4. MICROSTRUCTURE ORDER FLOW (simplified)
struct OrderFlow {
    double vpin = 0.0;
    double effective_spread = 0.0;
    double price_impact = 0.0;
    double flow_imbalance = 0.0;
    // Tunables
    double vpin_th = 0.3;
    double spread_th = 0.001;
    double flow_th = 0.15;
    void set_thresholds(double vpin_thr,double spread_thr,double flow_thr){ vpin_th=vpin_thr; spread_th=spread_thr; flow_th=flow_thr; }
};

inline int microstructure_signal(double bid, double ask, double volume,
                                double last_price, OrderFlow& flow) {
    double mid = 0.5 * (bid + ask);
    if (mid <= 0.0) mid = 1e-12;
    flow.effective_spread = 2.0 * std::abs(last_price - mid) / mid;

    double buy_volume = (last_price > mid) ? volume : 0.0;
    double sell_volume = (last_price < mid) ? volume : 0.0;

    double total_volume = buy_volume + sell_volume;
    flow.vpin = (total_volume > 0.0) ? std::abs(buy_volume - sell_volume) / total_volume : 0.0;

    flow.flow_imbalance = 0.95 * flow.flow_imbalance + 0.05 * ((buy_volume - sell_volume) / (total_volume + 1e-8));
    flow.price_impact = std::abs(last_price - mid) / (volume + 1e-8);

    bool favorable_microstructure = (flow.vpin < flow.vpin_th) && (flow.effective_spread < flow.spread_th);
    bool strong_flow = std::abs(flow.flow_imbalance) > flow.flow_th;

    if (favorable_microstructure && strong_flow) {
        return (flow.flow_imbalance > 0.0) ? 1 : -1;
    }
    return 0;
}

// 5. KELLY RISK
struct KellyRisk {
    double win_rate = 0.55, avg_win = 0.012, avg_loss = 0.008;
    double portfolio_dd = 0.0, max_dd = 0.0;
    double kelly_scale = 1.0;
    double heat_index = 0.0;
};

inline double fractional_kelly_size(double edge, double odds, double portfolio_value,
                                   KellyRisk& kelly) {
    if (odds <= 1e-12) odds = 1e-12;
    double kelly_fraction = edge / odds;
    if (kelly_fraction < 0.0) kelly_fraction = 0.0;
    if (kelly_fraction > 0.25) kelly_fraction = 0.25;

    kelly_fraction *= 0.25; // fractional kelly

    if (kelly.portfolio_dd > 0.02) kelly.kelly_scale *= 0.7;
    if (kelly.portfolio_dd > 0.05) kelly.kelly_scale *= 0.5;
    if (kelly.heat_index > 3.0) kelly.kelly_scale *= 0.8;
    if (kelly.portfolio_dd < 0.005) kelly.kelly_scale = std::min(1.0, kelly.kelly_scale * 1.05);

    return kelly_fraction * kelly.kelly_scale * portfolio_value;
}

// EXECUTION ENGINE
inline int execute_nanosecond_strategies(const double* prices1, const double* prices2,
                                        const double* returns1, const double* returns2,
                                        double bid, double ask, double volume,
                                        int n, HMMRegime& hmm, KalmanPair& kalman,
                                        GARCHVolatility& garch, OrderFlow& flow,
                                        KellyRisk& kelly, double* position_size) {
    if (n <= 0) { *position_size = 0.0; return 0; }
    int signals[4];
    signals[0] = hmm_regime_switch(returns1, n, hmm);
    signals[1] = kalman_pairs_filter(prices1[n-1], prices2[n-1], kalman);
    signals[2] = garch_vol_strategy(returns1[n-1], returns2[n-1],
                                    std::sqrt(std::max(1e-12, garch.vol2_forecast)),
                                    std::sqrt(std::max(1e-12, garch.vol2_forecast)), garch);
    signals[3] = microstructure_signal(bid, ask, volume, prices1[n-1], flow);

    int consensus = signals[0] + signals[1] + signals[2] + signals[3];
    if (std::abs(consensus) >= 2) {  // relaxed threshold to encourage trades during research
        double edge = kelly.win_rate * kelly.avg_win - (1 - kelly.win_rate) * kelly.avg_loss;
        double odds = kelly.avg_win / (kelly.avg_loss + 1e-12);
        *position_size = fractional_kelly_size(edge, odds, 1e6, kelly);
        return (consensus > 0) ? 1 : -1;
    }
    *position_size = 0.0;
    return 0;
}

// SIMD-like optimizer fallback (scalar if no AVX2)
inline void optimize_sharpe_scalar(const double* pnl_matrix, int n_periods, int n_params,
                                   double* sharpe_results) {
    for (int i=0;i<n_params;i++) {
        double mean=0.0, var=0.0;
        for (int t=0;t<n_periods;t++) mean += pnl_matrix[t*n_params + i];
        mean /= std::max(1, n_periods);
        for (int t=0;t<n_periods;t++) {
            double d = pnl_matrix[t*n_params + i] - mean;
            var += d*d;
        }
        var /= std::max(1, n_periods-1);
        double stdv = std::sqrt(std::max(0.0, var));
        sharpe_results[i] = (stdv>0.0) ? mean/stdv * 15.874507 : 0.0;
    }
}

// ============== PYBIND11 MODULE ==============
PYBIND11_MODULE(hft_strategies, m) {
    m.doc() = "Institutional-grade nanosecond HFT strategies (research-based)";

    py::class_<HMMRegime>(m, "HMMRegime")
        .def(py::init<>())
        .def("set_mu", &HMMRegime::set_mu)
        .def("set_sigma", &HMMRegime::set_sigma)
        .def("set_P", &HMMRegime::set_P)
        .def("set_thresholds", &HMMRegime::set_thresholds);

    py::class_<KalmanPair>(m, "KalmanPair")
        .def(py::init<>())
        .def("set_Q", &KalmanPair::set_Q)
        .def("set_R", &KalmanPair::set_R);

    py::class_<GARCHVolatility>(m, "GARCHVolatility")
        .def(py::init<>())
        .def("set_params", &GARCHVolatility::set_params);

    py::class_<OrderFlow>(m, "OrderFlow")
        .def(py::init<>())
        .def("set_thresholds", &OrderFlow::set_thresholds);

    py::class_<KellyRisk>(m, "KellyRisk")
        .def(py::init<>());

    m.def("execute_nanosecond_strategies",
        [](py::array_t<double> prices1, py::array_t<double> prices2,
           py::array_t<double> returns1, py::array_t<double> returns2,
           double bid, double ask, double volume,
           HMMRegime& hmm, KalmanPair& kalman, GARCHVolatility& garch, OrderFlow& flow, KellyRisk& kelly) {
            auto p1 = prices1.unchecked<1>();
            auto p2 = prices2.unchecked<1>();
            auto r1 = returns1.unchecked<1>();
            auto r2 = returns2.unchecked<1>();
            int n = (int)std::min({p1.shape(0), p2.shape(0), r1.shape(0), r2.shape(0)});
            if (n <= 0) return py::make_tuple(0, 0.0);
            double pos_size = 0.0;
            int sig = execute_nanosecond_strategies(&p1(0), &p2(0), &r1(0), &r2(0), bid, ask, volume, n,
                                                    hmm, kalman, garch, flow, kelly, &pos_size);
            return py::make_tuple(sig, pos_size);
        },
        py::arg("prices1"), py::arg("prices2"), py::arg("returns1"), py::arg("returns2"),
        py::arg("bid"), py::arg("ask"), py::arg("volume"),
        py::arg("hmm"), py::arg("kalman"), py::arg("garch"), py::arg("flow"), py::arg("kelly"));
}

