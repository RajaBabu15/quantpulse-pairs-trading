#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cstring>
#include "quantpulse_core.h"
namespace py = pybind11;
static inline py::array_t<double> make_array(std::size_t n) {
    return py::array_t<double>(py::array::ShapeContainer{ (py::ssize_t)n });
}
static TradingParameters make_params(const py::dict &d) {
    TradingParameters p{};
    p.lookback = d.contains("lookback") ? d["lookback"].cast<int>() : 20;
    p.z_entry = d.contains("z_entry") ? d["z_entry"].cast<double>() : 2.0;
    p.z_exit = d.contains("z_exit") ? d["z_exit"].cast<double>() : 0.5;
    p.position_size = d.contains("position_size") ? d["position_size"].cast<int>() : 10000;
    p.transaction_cost = d.contains("transaction_cost") ? d["transaction_cost"].cast<double>() : 0.001;
    p.profit_target = d.contains("profit_target") ? d["profit_target"].cast<double>() : 2.0;
    p.stop_loss = d.contains("stop_loss") ? d["stop_loss"].cast<double>() : 1.0;
    return p;
}
static py::dict to_dict(const CBacktestResult &res) {
    py::dict d;
    d["total_return"] = res.total_return;
    d["sharpe_ratio"] = res.sharpe_ratio;
    d["max_drawdown"] = res.max_drawdown;
    d["num_trades"] = res.num_trades;
    d["win_rate"] = res.win_rate;
    d["profit_factor"] = res.profit_factor;
    d["avg_trade_return"] = res.avg_trade_return;
    d["volatility"] = res.volatility;
    return d;
}

PYBIND11_MODULE(quantpulse_core_py, m) {
    m.def("simd_vector_add", [](py::array_t<double, py::array::c_style | py::array::forcecast> a, py::array_t<double, py::array::c_style | py::array::forcecast> b) {
        auto abuf = a.request(), bbuf = b.request();
        if (abuf.size != bbuf.size) throw std::runtime_error("a and b must match");
        py::array_t<double> out = make_array((std::size_t)abuf.size);
        auto obuf = out.request();
        simd_vector_add(static_cast<const double*>(abuf.ptr), static_cast<const double*>(bbuf.ptr), static_cast<double*>(obuf.ptr), (size_t)abuf.size);
        return out;
    });
    m.def("simd_vector_mean", [](py::array_t<double, py::array::c_style | py::array::forcecast> a) {
        auto buf = a.request();
        return simd_vector_mean(static_cast<const double*>(buf.ptr), (size_t)buf.size);
    });
    m.def("simd_vector_std", [](py::array_t<double, py::array::c_style | py::array::forcecast> a, double mean) {
        auto buf = a.request();
        return simd_vector_std(static_cast<const double*>(buf.ptr), (size_t)buf.size, mean);
    });
    m.def("calculate_spread_and_zscore", [](py::array_t<double, py::array::c_style | py::array::forcecast> p1, py::array_t<double, py::array::c_style | py::array::forcecast> p2, int lookback) {
        auto b1 = p1.request(), b2 = p2.request();
        if (b1.size != b2.size) throw std::runtime_error("prices must match");
        py::array_t<double> spread = make_array((size_t)b1.size), z = make_array((size_t)b1.size);
        {
            py::gil_scoped_release rel;
            calculate_spread_and_zscore(static_cast<const double*>(b1.ptr), static_cast<const double*>(b2.ptr), (size_t)b1.size, lookback, static_cast<double*>(spread.request().ptr), static_cast<double*>(z.request().ptr));
        }
        return py::make_tuple(spread, z);
    });

    m.def("vectorized_backtest", [](py::array_t<double, py::array::c_style | py::array::forcecast> p1, py::array_t<double, py::array::c_style | py::array::forcecast> p2, py::dict params, bool use_cache) {
        auto b1 = p1.request(), b2 = p2.request();
        if (b1.size != b2.size) throw std::runtime_error("prices must match");
        TradingParameters tp = make_params(params);
        CBacktestResult res{};
        {
            py::gil_scoped_release rel;
            res = use_cache ? cached_vectorized_backtest(static_cast<const double*>(b1.ptr), static_cast<const double*>(b2.ptr), (size_t)b1.size, tp) : vectorized_backtest(static_cast<const double*>(b1.ptr), static_cast<const double*>(b2.ptr), (size_t)b1.size, tp);
        }
        return to_dict(res);
    }, py::arg("p1"), py::arg("p2"), py::arg("params"), py::arg("use_cache") = true);
    m.def("parallel_cross_validation", [](py::array_t<double, py::array::c_style | py::array::forcecast> p1, py::array_t<double, py::array::c_style | py::array::forcecast> p2, py::array_t<double, py::array::c_style | py::array::forcecast> params, int n_folds, double l1_ratio, double alpha, double kl_weight) {
        auto b1 = p1.request(), b2 = p2.request(), bp = params.request();
        if (b1.size != b2.size) throw std::runtime_error("prices must match");
        double score;
        {
            py::gil_scoped_release rel;
            score = parallel_cross_validation(static_cast<const double*>(b1.ptr), static_cast<const double*>(b2.ptr), (size_t)b1.size, static_cast<const double*>(bp.ptr), n_folds, l1_ratio, alpha, kl_weight);
        }
        return score;
    }, py::arg("p1"), py::arg("p2"), py::arg("params"), py::arg("n_folds") = 3, py::arg("l1_ratio") = 0.7, py::arg("alpha") = 0.02, py::arg("kl_weight") = 0.15);

    m.def("batch_parameter_optimization", [](py::array_t<double, py::array::c_style | py::array::forcecast> p1, py::array_t<double, py::array::c_style | py::array::forcecast> p2, std::vector<py::array_t<double, py::array::c_style | py::array::forcecast>> param_sets) {
        auto b1 = p1.request(), b2 = p2.request();
        if (b1.size != b2.size) throw std::runtime_error("prices must match");
        int n_sets = (int)param_sets.size();
        if (n_sets == 0) return py::list();
        std::vector<std::unique_ptr<py::array_t<double>>> holders; holders.reserve(n_sets);
        std::vector<const double*> ptrs; ptrs.reserve(n_sets);
        int param_len = (int)param_sets[0].request().size;
        for (int i = 0; i < n_sets; ++i) {
            auto req = param_sets[i].request();
            if ((int)req.size != param_len) throw std::runtime_error("parameter length mismatch");
            ptrs.push_back(static_cast<const double*>(req.ptr));
        }
        std::vector<CBacktestResult> results((size_t)n_sets);
        {
            py::gil_scoped_release rel;
            batch_parameter_optimization(static_cast<const double*>(b1.ptr), static_cast<const double*>(b2.ptr), (size_t)b1.size, ptrs.data(), n_sets, param_len, results.data());
        }
        py::list out;
        for (int i = 0; i < n_sets; ++i) out.append(to_dict(results[(size_t)i]));
        return out;
    });
    m.def("backtest_trade_returns", [](py::array_t<double, py::array::c_style | py::array::forcecast> p1, py::array_t<double, py::array::c_style | py::array::forcecast> p2, py::dict params) {
        auto b1 = p1.request(), b2 = p2.request();
        if (b1.size != b2.size) throw std::runtime_error("prices must match");
        TradingParameters tp = make_params(params);
        py::array_t<double> out = make_array((size_t)b1.size);
        size_t count;
        {
            py::gil_scoped_release rel;
            count = backtest_trade_returns(static_cast<const double*>(b1.ptr), static_cast<const double*>(b2.ptr), (size_t)b1.size, tp, static_cast<double*>(out.request().ptr), (size_t)b1.size);
        }
        out.resize({ (py::ssize_t)count });
        return out;
    });
    m.def("warm_up_caches", [](py::array_t<double, py::array::c_style | py::array::forcecast> p1, py::array_t<double, py::array::c_style | py::array::forcecast> p2) {
        auto b1 = p1.request(), b2 = p2.request();
        if (b1.size != b2.size) throw std::runtime_error("prices must match");
        py::gil_scoped_release rel;
        warm_up_caches(static_cast<const double*>(b1.ptr), static_cast<const double*>(b2.ptr), (size_t)b1.size);
    });
    m.def("print_cache_statistics", [](){ py::gil_scoped_release rel; print_cache_statistics(); });
    m.def("clear_all_caches", [](){ py::gil_scoped_release rel; clear_all_caches(); });
}

