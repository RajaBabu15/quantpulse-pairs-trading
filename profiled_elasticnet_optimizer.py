#!/usr/bin/env python3
"""
Profiled ElasticNet + KL Divergence + RMSprop Optimization System
================================================================

This is the performance-profiled version of our advanced optimization system
to identify bottlenecks and optimize performance.

Author: QuantPulse Trading Systems
"""

import time
import numpy as np
import pandas as pd
from performance_profiler import profiler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import optimized components
from kl_elasticnet_optimizer import (
    ElasticNetKLTrader, 
    ElasticNetKLPortfolioOptimizer,
    KLDivergenceLoss,
    RMSpropOptimizer
)
from run import PairsTrader, HFTPairsTrader, HFT_AVAILABLE
from performance_analyzer import generate_random_pairs

# ============================================================================
# PROFILED OPTIMIZATION COMPONENTS
# ============================================================================

class ProfiledElasticNetKLTrader(ElasticNetKLTrader):
    """Profiled version of ElasticNetKLTrader with detailed timing"""
    
    @profiler.timer("elasticnet_optimization", track_memory=True)
    def optimize_with_rmsprop_kl(self, prices, n_splits=3, max_iterations=100):
        """Profiled optimization with detailed timing breakdown"""
        print(f"🔧 [PROFILED] Optimizing {self.symbol1}-{self.symbol2}...")
        
        with profiler.context_timer("time_series_cv_setup", "setup"):
            from sklearn.model_selection import TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=n_splits)
        
        with profiler.context_timer("parameter_initialization", "setup"):
            initial_params = np.array([0.5, 0.5, 0.25, 0.6, 0.3, 0.5, 0.5])
            current_params = initial_params.copy()
            self.rmsprop.reset(len(current_params))
            best_objective = float('inf')
            best_params = current_params.copy()
        
        # Main optimization loop with detailed profiling
        for iteration in range(max_iterations):
            with profiler.context_timer(f"optimization_iteration_{iteration}", "iteration"):
                # Define objective function for this iteration
                def iteration_objective(norm_params):
                    with profiler.context_timer("parameter_denormalization", "computation"):
                        params = self.denormalize_parameters(norm_params)
                    
                    cv_scores = []
                    with profiler.context_timer("cross_validation_loop", "cv"):
                        for fold, (train_idx, val_idx) in enumerate(tscv.split(prices)):
                            with profiler.context_timer(f"cv_fold_{fold}", "cv_fold"):
                                with profiler.context_timer("data_splitting", "data"):
                                    prices_train = prices.iloc[train_idx]
                                    prices_val = prices.iloc[val_idx]
                                
                                with profiler.context_timer("objective_function_eval", "optimization"):
                                    score = self.enhanced_objective_function(
                                        params, prices_train, prices_val
                                    )
                                    cv_scores.append(score)
                    
                    with profiler.context_timer("cv_score_aggregation", "computation"):
                        return np.mean(cv_scores)
                
                # RMSprop step with profiling
                try:
                    with profiler.context_timer("rmsprop_step", "optimization"):
                        new_params = self.rmsprop.step(current_params, iteration_objective)
                        new_params = np.clip(new_params, 0.0, 1.0)
                    
                    with profiler.context_timer("objective_evaluation", "optimization"):
                        current_objective = iteration_objective(new_params)
                    
                    with profiler.context_timer("best_parameter_update", "bookkeeping"):
                        if current_objective < best_objective:
                            best_objective = current_objective
                            best_params = new_params.copy()
                            print(f"✅ Iteration {iteration + 1}: New best objective: {-best_objective:.6f}")
                    
                    current_params = new_params
                    
                    # Early stopping check
                    with profiler.context_timer("convergence_check", "bookkeeping"):
                        if len(self.rmsprop.history) > 10:
                            recent_objectives = [h['objective'] for h in self.rmsprop.history[-10:]]
                            if np.std(recent_objectives) < 1e-6:
                                print(f"🎯 Converged after {iteration + 1} iterations")
                                break
                        
                except Exception as e:
                    print(f"⚠ RMSprop iteration {iteration + 1} failed: {e}")
                    continue
        
        # Final processing with profiling
        with profiler.context_timer("final_parameter_processing", "postprocessing"):
            final_params = self.denormalize_parameters(best_params)
            
            # Final cross-validation
            final_cv_scores = []
            for train_idx, val_idx in tscv.split(prices):
                prices_train = prices.iloc[train_idx]
                prices_val = prices.iloc[val_idx]
                
                result = self.enhanced_objective_function(
                    final_params, prices_train, prices_val, return_details=True
                )
                final_cv_scores.append(result)
        
        with profiler.context_timer("result_storage", "bookkeeping"):
            self.best_params = {
                'lookback': int(final_params[0]),
                'z_entry': final_params[1],
                'z_exit': final_params[2],
                'position_size': int(final_params[3]),
                'transaction_cost': final_params[4],
                'profit_target': final_params[5],
                'stop_loss': final_params[6]
            }
            
            self.cv_results = final_cv_scores
            self.optimization_history = self.rmsprop.history
        
        return self.best_params, self.cv_results

    @profiler.timer("objective_function", track_memory=True)
    def enhanced_objective_function(self, params, prices_train, prices_val, 
                                  return_details=False):
        """Profiled version of the enhanced objective function"""
        
        with profiler.context_timer("parameter_validation", "validation"):
            if len(params) < 7:
                params = np.append(params, [2.0, 1.0])
            
            lookback = max(5, min(60, int(params[0])))
            z_entry = max(0.5, min(4.0, abs(params[1])))
            z_exit = max(0.1, min(2.0, abs(params[2])))
            position_size = max(1000, min(50000, abs(params[3])))
            transaction_cost = max(0.0001, min(0.005, abs(params[4])))
            profit_target = max(1.5, min(5.0, abs(params[5])))
            stop_loss = max(0.5, min(2.0, abs(params[6])))
            
            if z_exit >= z_entry:
                z_exit = z_entry * 0.5
        
        try:
            with profiler.context_timer("trader_creation", "initialization"):
                if self.use_hft:
                    from kl_elasticnet_optimizer import EnhancedHFTTrader
                    trader = EnhancedHFTTrader(
                        self.symbol1, self.symbol2,
                        lookback=lookback, z_entry=z_entry, z_exit=z_exit,
                        position_size=position_size, transaction_cost=transaction_cost,
                        profit_target=profit_target, stop_loss=stop_loss
                    )
                else:
                    from kl_elasticnet_optimizer import EnhancedPairsTrader
                    trader = EnhancedPairsTrader(
                        self.symbol1, self.symbol2,
                        lookback=lookback, z_entry=z_entry, z_exit=z_exit,
                        position_size=position_size, transaction_cost=transaction_cost,
                        profit_target=profit_target, stop_loss=stop_loss
                    )
            
            with profiler.context_timer("training_backtest", "backtest"):
                train_results = trader.enhanced_backtest_with_data(prices_train)
            
            with profiler.context_timer("validation_backtest", "backtest"):
                val_results = trader.enhanced_backtest_with_data(prices_val)
            
            with profiler.context_timer("metric_extraction", "computation"):
                train_pnl = train_results.get('final_pnl', 0)
                val_pnl = val_results.get('final_pnl', 0)
                train_sharpe = train_results.get('sharpe_ratio', 0)
                val_sharpe = val_results.get('sharpe_ratio', 0)
                train_returns = train_results.get('trade_returns', [])
                val_returns = val_results.get('trade_returns', [])
                val_win_rate = val_results.get('win_rate', 0)
                val_max_dd = val_results.get('max_drawdown', 0)
            
            with profiler.context_timer("multiobjective_scoring", "computation"):
                if self.optimization_mode == 'profit':
                    primary_score = val_pnl / 100000
                    secondary_score = val_sharpe * 0.3
                elif self.optimization_mode == 'sharpe':
                    primary_score = val_sharpe * 2
                    secondary_score = max(0, val_pnl) / 100000 * 0.3
                else:  # hybrid
                    primary_score = val_pnl / 100000 + val_sharpe
                    secondary_score = val_win_rate * 0.2
                
                risk_adj_return = primary_score / (abs(val_max_dd) / 100000 + 1.0)
                combined_score = risk_adj_return + secondary_score
            
            with profiler.context_timer("elasticnet_regularization", "regularization"):
                normalized_params = self.normalize_parameters(params)
                l1_penalty = self.alpha * self.l1_ratio * np.sum(np.abs(normalized_params))
                l2_penalty = self.alpha * (1 - self.l1_ratio) * np.sum(normalized_params**2)
                elasticnet_penalty = l1_penalty + l2_penalty
            
            with profiler.context_timer("kl_divergence_computation", "regularization"):
                kl_penalty_train = self.kl_loss.compute_kl_loss(train_returns, self.optimization_mode)
                kl_penalty_val = self.kl_loss.compute_kl_loss(val_returns, self.optimization_mode)
                kl_penalty = (kl_penalty_train + kl_penalty_val) / 2
            
            with profiler.context_timer("stability_penalty", "computation"):
                stability_penalty = abs(train_sharpe - val_sharpe) * 0.1
            
            with profiler.context_timer("final_objective_computation", "computation"):
                objective = (combined_score - 
                            elasticnet_penalty - 
                            self.kl_weight * kl_penalty - 
                            stability_penalty)
            
            if return_details:
                return {
                    'objective': objective,
                    'val_pnl': val_pnl,
                    'val_sharpe': val_sharpe,
                    'val_win_rate': val_win_rate,
                    'val_max_dd': val_max_dd,
                    'train_pnl': train_pnl,
                    'train_sharpe': train_sharpe,
                    'combined_score': combined_score,
                    'elasticnet_penalty': elasticnet_penalty,
                    'kl_penalty': kl_penalty,
                    'stability_penalty': stability_penalty,
                    'l1_penalty': l1_penalty,
                    'l2_penalty': l2_penalty,
                    'params': {
                        'lookback': lookback,
                        'z_entry': z_entry,
                        'z_exit': z_exit,
                        'position_size': position_size,
                        'transaction_cost': transaction_cost,
                        'profit_target': profit_target,
                        'stop_loss': stop_loss
                    }
                }
            
            return -objective  # Minimize (negative of maximize)
            
        except Exception as e:
            print(f"⚠ Objective function error: {e}")
            return 1000.0

class ProfiledElasticNetKLPortfolioOptimizer(ElasticNetKLPortfolioOptimizer):
    """Profiled portfolio optimizer with detailed timing"""
    
    @profiler.timer("portfolio_optimization", track_memory=True)
    def optimize_all_pairs(self, max_iterations=30, n_splits=3):
        """Profiled portfolio optimization"""
        print("🚀 [PROFILED] ELASTICNET + KL DIVERGENCE + RMSPROP OPTIMIZATION")
        print("=" * 70)
        
        profiler.enable_monitoring()
        
        with profiler.context_timer("optimization_setup", "setup"):
            print(f"📊 Optimizing {len(self.pairs_data)} pairs")
            print(f"🔧 Mode: {self.optimization_mode.upper()}")
            print(f"⚡ ElasticNet Alpha: {self.alpha}, L1/L2: {self.l1_ratio}")
            print(f"🎯 KL Weight: {self.kl_weight}")
            print()
            
            start_time = time.time()
        
        for i, (pair_name, prices) in enumerate(self.pairs_data.items(), 1):
            with profiler.context_timer(f"pair_optimization_{pair_name}", "pair_optimization"):
                print(f"📈 {i}/{len(self.pairs_data)}: Optimizing {pair_name}")
                
                with profiler.context_timer("symbol_extraction", "preprocessing"):
                    symbol1, symbol2 = pair_name.split('-')
                
                try:
                    with profiler.context_timer("optimizer_creation", "setup"):
                        optimizer = ProfiledElasticNetKLTrader(
                            symbol1, symbol2,
                            l1_ratio=self.l1_ratio,
                            alpha=self.alpha,
                            kl_weight=self.kl_weight,
                            optimization_mode=self.optimization_mode
                        )
                    
                    with profiler.context_timer("rmsprop_optimization", "optimization"):
                        best_params, cv_scores = optimizer.optimize_with_rmsprop_kl(
                            prices, n_splits=n_splits, max_iterations=max_iterations
                        )
                    
                    with profiler.context_timer("result_aggregation", "postprocessing"):
                        self.optimized_pairs[pair_name] = {
                            'optimizer': optimizer,
                            'best_params': best_params,
                            'cv_scores': cv_scores,
                            'avg_val_pnl': np.mean([s['val_pnl'] for s in cv_scores]),
                            'avg_val_sharpe': np.mean([s['val_sharpe'] for s in cv_scores]),
                            'avg_val_win_rate': np.mean([s['val_win_rate'] for s in cv_scores]),
                            'avg_elasticnet_penalty': np.mean([s['elasticnet_penalty'] for s in cv_scores]),
                            'avg_kl_penalty': np.mean([s['kl_penalty'] for s in cv_scores])
                        }
                        
                        print(f"✅ {pair_name}: ${self.optimized_pairs[pair_name]['avg_val_pnl']:,.2f} avg P&L")
                    
                except Exception as e:
                    print(f"❌ {pair_name}: Failed - {e}")
                    continue
                
                print()
        
        with profiler.context_timer("final_analysis", "postprocessing"):
            total_time = time.time() - start_time
            
            print(f"✅ ElasticNet+KL+RMSprop optimization complete!")
            print(f"⏱️ Total time: {total_time:.2f}s")
            print(f"✅ Successfully optimized: {len(self.optimized_pairs)}/{len(self.pairs_data)} pairs")
            
            self.analyze_portfolio()
        
        profiler.disable_monitoring()
        return self.optimized_pairs

# ============================================================================
# PROFILED DATA OPERATIONS
# ============================================================================

@profiler.timer("data_collection", track_memory=True)
def profile_data_collection(num_pairs=5):
    """Profile data collection operations"""
    print("📊 [PROFILED] Data Collection Analysis")
    print("-" * 40)
    
    with profiler.context_timer("pair_generation", "setup"):
        pairs = generate_random_pairs(10, sector_bias=True)
    
    pairs_data = {}
    
    for i, (symbol1, symbol2) in enumerate(pairs[:num_pairs]):
        pair_name = f"{symbol1}-{symbol2}"
        
        with profiler.context_timer(f"data_download_{pair_name}", "data_download"):
            print(f"📥 {i+1}/{num_pairs}: Downloading {pair_name}")
            
            try:
                with profiler.context_timer("trader_initialization", "setup"):
                    temp_trader = PairsTrader(symbol1, symbol2)
                
                with profiler.context_timer("actual_data_fetch", "api_call"):
                    prices = temp_trader.get_data('2020-01-01', '2024-12-31')
                
                with profiler.context_timer("data_validation", "validation"):
                    if len(prices) > 100:
                        pairs_data[pair_name] = prices
                        print(f"✅ {pair_name}: {len(prices)} data points")
                    else:
                        print(f"❌ {pair_name}: Insufficient data")
                        
            except Exception as e:
                print(f"❌ {pair_name}: Failed - {e}")
                continue
    
    return pairs_data

# ============================================================================
# COMPREHENSIVE PROFILED OPTIMIZATION RUN
# ============================================================================

def run_profiled_elasticnet_optimization():
    """Run comprehensive profiled ElasticNet optimization"""
    print("🚀 COMPREHENSIVE PROFILED ELASTICNET OPTIMIZATION")
    print("=" * 80)
    
    profiler.enable_monitoring()
    total_start = time.time()
    
    try:
        # Phase 1: Data Collection
        with profiler.context_timer("data_collection_phase", "phase"):
            pairs_data = profile_data_collection(num_pairs=5)  # Reduced for detailed analysis
        
        if not pairs_data:
            print("❌ No valid pairs data collected")
            return None, None
        
        print(f"\n✅ Collected data for {len(pairs_data)} pairs")
        
        # Phase 2: Optimization Setup
        with profiler.context_timer("optimization_setup_phase", "phase"):
            optimizer = ProfiledElasticNetKLPortfolioOptimizer(
                pairs_data,
                l1_ratio=0.7,
                alpha=0.02,
                kl_weight=0.15,
                optimization_mode='hybrid'
            )
        
        # Phase 3: Main Optimization
        with profiler.context_timer("main_optimization_phase", "phase"):
            results = optimizer.optimize_all_pairs(
                max_iterations=15,  # Reduced for detailed analysis
                n_splits=3
            )
        
        # Phase 4: Results Processing
        with profiler.context_timer("results_processing_phase", "phase"):
            results_file = optimizer.save_results()
        
        total_time = time.time() - total_start
        
        print(f"\n🎉 PROFILED OPTIMIZATION COMPLETE!")
        print(f"⏱️ Total Runtime: {total_time:.2f}s")
        print(f"📄 Results: {results_file}")
        
        return optimizer, results
        
    finally:
        profiler.disable_monitoring()
        
        # Generate comprehensive performance report
        print(f"\n📊 GENERATING PERFORMANCE ANALYSIS...")
        profiler.print_performance_report()
        
        performance_report = profiler.save_detailed_report()
        print(f"📄 Performance report: {performance_report}")
        
        # Create visualizations
        try:
            from performance_profiler import create_performance_visualizations
            create_performance_visualizations()
        except Exception as e:
            print(f"⚠️ Could not create visualizations: {e}")

if __name__ == "__main__":
    # Run the profiled optimization
    optimizer, results = run_profiled_elasticnet_optimization()
    
    print("\n💡 PERFORMANCE OPTIMIZATION RECOMMENDATIONS:")
    print("=" * 50)
    print("1. Data Download: Consider implementing caching")
    print("2. Objective Function: Most expensive operation - optimize hot paths")
    print("3. Cross-Validation: Parallelize fold evaluation")
    print("4. Backtest: Consider vectorized operations")
    print("5. Parameter Updates: Optimize RMSprop implementation")
